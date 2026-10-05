r"""Run LLM interpretation on per-gene evidence records.

Reads ``{records_dir}/{gene}.json`` (produced by ``evidence.py``), composes a
prompt from ``system.txt`` + the evidence JSON + ``output_schema.json``, calls
the Anthropic API, validates + writes the structured output to ``{out}/{gene}.json``.

Single-threaded; idempotent (skips genes whose output already exists unless
``force``). ``dry_run`` prints prompt-size diagnostics and exits without any
network calls; ``save_prompts`` writes each assembled prompt in full to
``data/llm_inputs/{run}/`` and composes with ``dry_run``.

Every invocation mints a run id (``_begin_run``) that is stamped into both the
captured prompts and a ``.meta.json`` sidecar beside each output artifact, so a
prompt can be tied to the verdict it produced — and a corpus that describes a
different run is reported rather than silently trusted.

The prompts directory is parameterized: ``main`` accepts ``prompts_dir`` (the
thin CLI supplies ``scripts/site/prompts/``). When not given, the module-level
``SYSTEM_PROMPT_PATH`` / ``OUTPUT_SCHEMA_PATH`` defaults are used (tests
monkeypatch these).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# Default prompts location — the thin CLI overrides this with its own
# scripts/site/prompts/ path. Resolves relative to the repo root so the
# module remains importable without a CLI.
ROOT = Path(__file__).resolve().parents[3]
PROMPTS_DIR = ROOT / "scripts" / "site" / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system.txt"
OUTPUT_SCHEMA_PATH = PROMPTS_DIR / "output_schema.json"

DEFAULT_MODEL = "claude-sonnet-5"
# Optional per-run override: the schema a TOOL LOOP validates its verdict
# against, when it differs from the one the single-shot path decodes against.
TOOL_VERDICT_SCHEMA = Path("output_schemas") / "category_read_tools.json"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 4000

# Standard (non-batch) token pricing, USD per token, by model id. The Message
# Batches API bills the SAME token counts at HALF these rates — batch mode saves
# no tokens (identical prompts), only cost (the 50% discount).
# NOTE: claude-sonnet-5 uses a new tokenizer (~30% more tokens for the same text
# than sonnet-4-6), so per-isoform token counts and the usage-report cost rise
# ~30% at these same sticker rates; introductory pricing ($2/$10 per MTok through
# 2026-08-31) means these figures currently over-estimate the real Sonnet-5 cost.
_PRICING: dict[str, tuple[float, float]] = {  # model -> (input $/tok, output $/tok)
    "claude-sonnet-4-6": (3.0e-6, 15.0e-6),
    "claude-sonnet-5": (3.0e-6, 15.0e-6),
    "claude-opus-4-8": (5.0e-6, 25.0e-6),
    "claude-haiku-4-5": (1.0e-6, 5.0e-6),
}
_BATCH_DISCOUNT = 0.5

# Models that reject non-default sampling params (temperature/top_p/top_k) with an
# HTTP 400, and run adaptive thinking by default when `thinking` is omitted. For
# these we omit temperature (sending 0.0 — a non-default value — would 400) and
# explicitly disable thinking: this pipeline wants one deterministic JSON verdict,
# and a leading `thinking` block would also break text extraction. Older models
# (sonnet-4-6 and earlier) keep the temperature knob and are thinking-off by
# default, so their behavior is unchanged.
_NO_SAMPLING_MODELS = frozenset(
    {"claude-sonnet-5", "claude-opus-4-8", "claude-opus-4-7", "claude-fable-5"}
)

# Synthetic evidence record used by --dry-run when no llm_evidence dir exists.
_SYNTHETIC_RECORD: dict[str, Any] = {
    "gene": {
        "name": "TRNT1",
        "uniprot_id": "Q96Q11",
        "function": "CCA-adding enzyme; adds the 3' CCA terminus to tRNAs.",
        "subcellular_location": "Mitochondrion; Cytoplasm; Nucleus",
    },
    "isoforms": [
        {
            "tis_id": "chr3:3129127:+:ATG:ENST00000434583.5",
            "orf_type": "truncated",
            "alt_start_codon": "ATG",
            "isoform_length_aa": 405,
            "canonical_length_aa": 434,
            "differential_sequence": "MLRCLYHWHRPVLNRRWSRLCLPKQYLFT",
            "diff_space": "canonical",
            "kozak_context": "CTATTCACAATGA",
            "scoring": {
                "existence_score": 5,
                "existence_evaluable": 6,
                "existence_high_confidence": True,
                "functional_score": 5,
                "functional_evaluable": 6,
                "functional_high_confidence": True,
                "criteria": {},
            },
            "key_metrics": {},
            "pathogenic_variants_in_unique": [],
        }
    ],
}


# ── Defensive imports ─────────────────────────────────────────────────────


def _try_import_mozzarellm():
    """Try to import mozzarellm's AnthropicClient. Returns class or None."""
    try:
        from mozzarellm.clients.llm_api_clients import AnthropicClient

        return AnthropicClient
    except Exception:
        return None


def _try_import_anthropic():
    """Try to import the anthropic SDK directly. Returns module or None."""
    try:
        import anthropic

        return anthropic
    except Exception:
        return None


def _try_import_jsonschema():
    """Try to import jsonschema for output validation. Returns module or None."""
    try:
        import jsonschema

        return jsonschema
    except Exception:
        return None


# ── Prompt assembly ───────────────────────────────────────────────────────


@dataclass
class Prompt:
    """A composed prompt ready for an LLM API call."""

    system: str
    user: str

    @property
    def estimated_input_tokens(self) -> int:
        """Rough char/4 heuristic for input token count."""
        return (len(self.system) + len(self.user)) // 4


# Splice markers in the tracked category prompts (figures/prompt_variants/
# assemble.py varies the blocks they delimit). They are authoring structure, not
# instructions, so they must never reach a model — production included.
PROMPT_BLOCK_RE = re.compile(r"^<!-- @block:([a-z_]+) -->$")
PROMPT_END_RE = re.compile(r"^<!-- @end -->$")


def collapse_blank_lines(lines: list[str]) -> str:
    """Join *lines*, squeezing runs of blank lines and trimming trailing ones."""
    kept: list[str] = []
    for line in lines:
        if not line.strip() and kept and not kept[-1].strip():
            continue
        kept.append(line)
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept)


def strip_prompt_markers(text: str) -> str:
    """*text* with every ``@block`` / ``@end`` marker line removed.

    The one stripping rule both readers share: production through
    :func:`load_system_prompt`, and the prompt-variant harness through
    ``assemble.render``. Two copies is how the harness's "status quo" arm and the
    prompt production actually sends drift apart.
    """
    lines = [
        line
        for line in text.splitlines()
        if not (PROMPT_BLOCK_RE.match(line.strip()) or PROMPT_END_RE.match(line.strip()))
    ]
    return collapse_blank_lines(lines)


def load_system_prompt(path: Path | None = None) -> str:
    """Load the system prompt, erroring early if missing.

    Reads ``SYSTEM_PROMPT_PATH`` at call time (so tests can monkeypatch it).
    Splice markers are stripped (:func:`strip_prompt_markers`).
    """
    path = path or SYSTEM_PROMPT_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"System prompt not found at {path}. "
            "Agent A is responsible for producing scripts/site/prompts/system.txt."
        )
    return strip_prompt_markers(path.read_text(encoding="utf-8")).strip()


def load_output_schema(path: Path | None = None) -> dict[str, Any]:
    """Load the output JSON schema, erroring early if missing or malformed.

    Reads ``OUTPUT_SCHEMA_PATH`` at call time (so tests can monkeypatch it).
    """
    path = path or OUTPUT_SCHEMA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Output schema not found at {path}. "
            "Agent A is responsible for producing scripts/site/prompts/output_schema.json."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Output schema at {path} is not valid JSON: {e}") from e


# Keywords the structured-output decoder 400s on. `minLength`/`maxLength` are
# accepted and silently ignored, so they stay — which is also why the reasoning
# cap still needs checking after the fact.
_UNSUPPORTED_SCHEMA_KEYWORDS = frozenset(
    {"uniqueItems", "minItems", "maxItems", "minimum", "maximum", "multipleOf"}
)

# Values that are instance DATA, not subschemas. Filtering them rewrites what the
# schema asserts — a `const: {"minimum": 3}` would reach the API missing a key,
# constraining the model to a different constant than the schema declares.
_DATA_VALUED_KEYWORDS = frozenset({"enum", "const", "default", "examples"})

# Values that are name -> subschema MAPS. Recurse into the values only: the names
# are the author's field names, and one that happens to read like a keyword
# ("maximum") must not be filtered out of `properties` while `required` still
# names it — with additionalProperties:false that schema cannot be satisfied.
_SCHEMA_MAP_KEYWORDS = frozenset(
    {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
)


def _output_format_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Copy of ``schema`` safe to send as ``output_config.format``.

    Wire copy only — ``build_prompt`` and ``_emit_schema_warnings`` keep the full
    schema, which is the only thing communicating the constraints the decoder
    ignores.

    Every value is classified before it is walked, because "is this a dict?" does
    not answer the question that matters: a subschema is filtered, a name-to-
    subschema map has only its values filtered, and instance data is left alone.
    """
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _UNSUPPORTED_SCHEMA_KEYWORDS:
            continue
        if key in _DATA_VALUED_KEYWORDS:
            out[key] = value
        elif key in _SCHEMA_MAP_KEYWORDS and isinstance(value, dict):
            out[key] = {name: _output_format_schema(sub) for name, sub in value.items()}
        elif isinstance(value, list):
            # anyOf/oneOf/allOf and tuple-form items. String lists (required,
            # multi-type) pass through untouched — the non-dict guard returns
            # each element as-is.
            out[key] = [_output_format_schema(item) for item in value]
        else:
            out[key] = _output_format_schema(value) if isinstance(value, dict) else value
    return out


def build_prompt(
    record: dict[str, Any], system_prompt: str, output_schema: dict[str, Any]
) -> Prompt:
    """Compose the prompt from an evidence record + system prompt + output schema.

    The user message is the evidence-record JSON, then a blank line, then
    ``Respond with valid JSON matching this schema:`` followed by the schema JSON.
    Returns a :class:`Prompt` containing both the system and user messages.
    """
    record_json = json.dumps(record, indent=2, ensure_ascii=False)
    schema_json = json.dumps(output_schema, indent=2, ensure_ascii=False)
    user = f"{record_json}\n\nRespond with valid JSON matching this schema:\n{schema_json}"
    return Prompt(system=system_prompt, user=user)


# ── Record loading ────────────────────────────────────────────────────────


def load_records(records_dir: Path, gene: str | None = None) -> dict[str, dict[str, Any]]:
    """Load evidence records from ``records_dir``.

    Returns a dict mapping gene_name -> record. If ``gene`` is given, only that
    one is returned. Errors if ``records_dir`` is missing OR if ``gene`` is
    requested but its file is absent.
    """
    if not records_dir.exists() or not records_dir.is_dir():
        raise FileNotFoundError(
            f"Evidence-records directory not found: {records_dir}. "
            "Run scripts/site/build_evidence_records.py first."
        )

    if gene is not None:
        path = records_dir / f"{gene}.json"
        if not path.exists():
            raise FileNotFoundError(f"No evidence record for gene {gene!r} at {path}")
        return {gene: json.loads(path.read_text(encoding="utf-8"))}

    records: dict[str, dict[str, Any]] = {}
    for path in sorted(records_dir.glob("*.json")):
        records[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    if not records:
        raise FileNotFoundError(
            f"No *.json evidence records found in {records_dir}. "
            "Run scripts/site/build_evidence_records.py first."
        )
    return records


# ── Pass registry ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PassSpec:
    """Static description of one LLM pass.

    Drives prompt loading, per-isoform input construction, and output file
    placement. New passes register here without touching call_llm or main.
    """

    name: str
    system_prompt_filename: str
    output_schema_filename: str
    output_filename_template: str  # e.g. "{gene}.json", "{tis_slug}/categories.json"
    iterates_categories: bool = False  # True for the per-category pass
    requires_prereq: tuple[str, ...] = ()  # other pass names that must have produced output


PASS_REGISTRY: dict[str, PassSpec] = {
    "default": PassSpec(
        name="default",
        system_prompt_filename="system.txt",
        output_schema_filename="output_schema.json",
        output_filename_template="{gene}.json",
    ),
    "category": PassSpec(
        name="category",
        system_prompt_filename="category-pass.txt",
        output_schema_filename="output_schemas/category_read.json",
        output_filename_template="{tis_slug}/categories.json",
        iterates_categories=True,
    ),
    "synthesis": PassSpec(
        name="synthesis",
        system_prompt_filename="synthesis-pass.txt",
        output_schema_filename="output_schemas/synthesis.json",
        output_filename_template="{tis_slug}/synthesis.json",
        requires_prereq=("category",),
    ),
}


# Categories that read their own data through a multi-turn tool loop instead of
# a single-shot call, mapped to the system prompt that describes their tools.
# These two aggregate the most away: M collapses several hundred variants into
# ~20 scalars, P collapses a per-residue array and an L×L matrix into 26. Each
# has its own data precondition and dispatch factory — see _tool_setup.
# Multi-turn means they cannot go through the Message Batches API; see
# _run_category_pass_batch. Everything not listed here keeps the single-shot
# path and stays batchable.
TOOL_CATEGORY_PROMPTS: dict[str, str] = {
    "M": "category-pass-M.txt",
    "P": "category-pass-P.txt",
}


# ── LLM call dispatch ─────────────────────────────────────────────────────


# Per-call token usage, appended by call_llm and drained by the pass runners so
# they can attribute cost per isoform. Single-threaded, so a module list is safe.
_USAGE_EVENTS: list[dict[str, int]] = []
_USAGE_KEYS = ("input", "output", "cache_read", "cache_creation")


def _record_usage(usage: Any) -> None:
    """Append one call's token counts (zeros when the backend gives no usage)."""
    if usage is None:
        _USAGE_EVENTS.append(dict.fromkeys(_USAGE_KEYS, 0))
        return
    _USAGE_EVENTS.append(
        {
            "input": getattr(usage, "input_tokens", 0) or 0,
            "output": getattr(usage, "output_tokens", 0) or 0,
            "cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0,
            "cache_creation": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        }
    )


def _drain_usage() -> dict[str, int]:
    """Pop the most recent recorded call usage (zeros if none)."""
    return _USAGE_EVENTS.pop() if _USAGE_EVENTS else dict.fromkeys(_USAGE_KEYS, 0)


def _add_usage(
    acc: dict[str, dict[str, int]], key: str, ev: dict[str, int], *, calls: int = 1
) -> None:
    """Fold one usage event into the per-isoform accumulator.

    ``calls`` is the number of API round trips the event covers — 1 for a
    single-shot call, N for an N-turn tool loop, so the report's call count
    stays proportional to what was actually billed.
    """
    slot = acc.setdefault(key, {**dict.fromkeys(_USAGE_KEYS, 0), "calls": 0})
    for k in _USAGE_KEYS:
        slot[k] += ev.get(k, 0)
    slot["calls"] += calls


def _write_usage_report(
    out_dir: Path, pass_name: str, model: str, per_isoform: dict, *, batch: bool = False
) -> None:
    """Write a per-isoform + total token/cost report JSON and print a summary.

    Cost is computed from ``_PRICING[model]``. When ``batch`` is True the reported
    cost applies the 50% Message-Batches discount — token counts are unchanged
    (batch saves no tokens), so ``batch_savings_usd`` is purely the price delta.
    """
    pin, pout = _PRICING.get(model, (0.0, 0.0))
    mult = _BATCH_DISCOUNT if batch else 1.0

    total = {**dict.fromkeys(_USAGE_KEYS, 0), "calls": 0}
    for slug, slot in per_isoform.items():
        for k in _USAGE_KEYS + ("calls",):
            total[k] += slot.get(k, 0)
        slot["cost_usd"] = round((slot["input"] * pin + slot["output"] * pout) * mult, 4)

    direct = total["input"] * pin + total["output"] * pout
    cost = direct * mult
    total["cost_usd"] = round(cost, 4)
    total["direct_cost_usd"] = round(direct, 4)
    total["batch_savings_usd"] = round(direct - cost, 4)

    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": _RUN_ID,
        **_RUN_PROVENANCE,
        "pass": pass_name,
        "model": model,
        "batch": batch,
        "pricing_usd_per_mtok": {"input": pin * 1e6, "output": pout * 1e6},
        "note": "batch bills identical tokens at 50% price; no token savings, only cost",
        "total": total,
        "per_isoform": per_isoform,
    }
    (out_dir / f"_usage_{pass_name}.json").write_text(json.dumps(report, indent=2))

    tag = "batch" if batch else "direct"
    line = (
        f"[usage] {pass_name} ({tag}): {total['calls']} calls, "
        f"{total['input']:,} in + {total['output']:,} out tokens, ${cost:.3f}"
    )
    if batch:
        line += f"  (direct = ${direct:.3f}; batch saves ${direct - cost:.3f} = 50%, same tokens)"
    print(line + f"  -> {out_dir / f'_usage_{pass_name}.json'}")


# ── Run identity ──────────────────────────────────────────────────────────

# One id per invocation, stamped into BOTH the prompt corpus and a sidecar next
# to every output artifact. Without it the two sides cannot be joined: the
# corpus is keyed by isoform and overwritten in place, outputs are reused across
# runs, and --dry-run --save-prompts captures prompts for calls that never
# happened — so a .txt and a categories.json that look like a pair routinely are
# not one. The id does not prevent that; it makes it checkable from the files.
_RUN_ID: str = ""
_RUN_MODE: str = "live"
# What the run was built from, stamped beside every output and into the usage
# report: the code (commit + dirty flag), the source parquet behind the evidence
# records, and whatever the caller adds (an arm's grounding, hint level and
# reference versions). Without it, "same commit, same corpus" between two arms
# was unverifiable from the files (audit38_P).
_RUN_PROVENANCE: dict[str, Any] = {}


def _begin_run(dry_run: bool) -> str:
    """Mint this invocation's run id. Called once, at the top of :func:`main`."""
    global _RUN_ID, _RUN_MODE
    _RUN_ID = f"{_utc_now().strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(3)}"
    _RUN_MODE = "dry_run" if dry_run else "live"
    _RUN_PROVENANCE.clear()
    _RUN_PROVENANCE["code"] = _code_provenance()
    return _RUN_ID


def _code_provenance() -> dict[str, Any]:
    """``{commit, dirty}`` for the checkout this module runs from; ``None``s if unknown.

    ``dirty`` counts tracked changes only: an untracked scratch file does not
    change what ran, an edited prompt or module does.
    """

    def vcs(*cmd: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", "-C", str(ROOT), *cmd], capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout if done.returncode == 0 else None

    commit = vcs("rev-parse", "HEAD")
    status = vcs("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": commit.strip() if commit else None,
        "dirty": bool(status.strip()) if status is not None else None,
    }


def _source_provenance(records_dir: Path) -> dict[str, Any]:
    """The evidence records' source parquet and its sha256, when it sits beside them.

    The standard layout is ``data/output/{run}/llm_evidence/`` next to
    ``data/output/{run}/all_paired.parquet``; anything else records the records
    path alone rather than guessing.
    """
    from swissisoform.setup._common import rel_to_root, sha256_file

    records_dir = Path(records_dir).resolve()
    out: dict[str, Any] = {"records": rel_to_root(records_dir)}
    parquet = records_dir.parent / "all_paired.parquet"
    if parquet.is_file():
        out["source_parquet"] = rel_to_root(parquet)
        out["source_parquet_sha256"] = sha256_file(parquet)
    return out


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _write_run_stamp(out_path: Path, *, pass_name: str, model: str, **extra: Any) -> None:
    """Record which invocation wrote ``out_path``, in a sidecar beside it.

    The artifact cannot carry this itself: ``categories.json`` is keyed by
    category name and its consumers iterate those keys, so an extra entry would
    read as a seventh category. A sidecar nothing reads keeps the shape intact —
    the same move as ``categories.partial.json``.
    """
    stamp = {
        "run_id": _RUN_ID,
        "run_mode": _RUN_MODE,
        "written_utc": _utc_now().isoformat(timespec="seconds"),
        "pass": pass_name,
        "model": model,
        "prompts_captured": _PROMPT_DIR is not None,
        **_RUN_PROVENANCE,
        **extra,
    }
    out_path.with_suffix(".meta.json").write_text(
        json.dumps(stamp, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


# ── Prompt capture ────────────────────────────────────────────────────────

# Opt-in, write-only dump of every assembled prompt, one .txt per API call,
# under its own top-level data/ folder with one subdir per run. Single-threaded,
# like _USAGE_EVENTS above.
#
# Deliberately NOT under data/cache/: nothing ever reads these files back and no
# step is skipped because one exists, so they are an audit artifact rather than a
# results cache and do not bend the "Execution Contract — fresh reruns" rule in
# CLAUDE.md. Every run rewrites them from scratch.
DEFAULT_PROMPT_DIR = ROOT / "data" / "llm_inputs"

_PROMPT_DIR: Path | None = None
_PROMPT_PASS: str = "prompts"
_PROMPT_INDEX: list[dict[str, Any]] = []
# Where this run's outputs go — held so the flush can say whether the corpus
# actually describes them (see _report_corpus_drift).
_PROMPT_OUT_DIR: Path | None = None


def _default_prompt_dir(out: Path) -> Path:
    """Where captured prompts go when ``--save-prompts-dir`` is not given.

    ``data/llm_inputs/{run}/``, where ``run`` comes from ``--out``: the standard layout
    is ``data/output/{run}/llm``, so the run name is the parent directory's. A
    non-standard ``--out`` falls back to its own basename rather than guessing.

    The per-run subdir is load-bearing, not cosmetic: ``index_{pass}.json`` sits
    at the root of the prompt dir, so two presets sharing one directory would
    clobber each other's index while their .txt files merged silently.
    """
    run = out.parent.name if out.name == "llm" else out.name
    return DEFAULT_PROMPT_DIR / (run or "unnamed")


def enable_prompt_capture(
    directory: Path, pass_name: str = "prompts", *, out_dir: Path | None = None
) -> None:
    """Start capturing assembled prompts under ``directory``.

    ``pass_name`` scopes the index file, because each pass runs as its own
    process against the same directory — a single ``index.json`` would be
    truncated to whichever pass ran last while its ``.txt`` files remained.
    Mirrors the ``_usage_{pass}.json`` naming next door.

    ``out_dir`` is where the pass writes its outputs; kept so the flush can
    compare their run stamps against this run's.
    """
    global _PROMPT_DIR, _PROMPT_PASS, _PROMPT_OUT_DIR
    _PROMPT_DIR = directory
    _PROMPT_PASS = pass_name
    _PROMPT_OUT_DIR = out_dir
    _PROMPT_INDEX.clear()
    directory.mkdir(parents=True, exist_ok=True)


def record_prompt(
    rel: str,
    prompt: Prompt,
    params: dict[str, Any],
    *,
    meta: dict[str, Any],
    tools: list[dict[str, Any]] | None = None,
) -> None:
    """Write one call's fully assembled prompt to ``{dir}/{rel}.txt``.

    No-op unless :func:`enable_prompt_capture` has run, so the normal path pays
    one ``if``.

    ``prompt`` supplies the system and user text verbatim — both are already
    rendered strings by this point, so nothing is re-serialised. ``params``
    (from :func:`_build_request_params` or :func:`_build_tool_request_params`)
    supplies the model and the sampling/thinking gate, so the header cannot
    drift from what is actually sent. ``meta`` becomes the leading header lines,
    in the order given.

    The header is strict ``# key: value``, one key per line, so the exact call
    is reconstructible from the file alone and a diff between two runs points at
    the single field that changed.

    ``run_id`` and ``run_mode`` lead the header: they are what ties this prompt
    to the output it produced (via that output's ``.meta.json`` sidecar), and
    what marks a ``--dry-run`` capture as describing no call at all. Distinct
    from ``meta["mode"]``, which describes the call's shape (batch, tool_loop).
    """
    if _PROMPT_DIR is None:
        return

    code = _RUN_PROVENANCE.get("code") or {}
    fields: dict[str, Any] = {
        "run_id": _RUN_ID,
        "run_mode": _RUN_MODE,
        "code_commit": f"{code.get('commit')}{'+dirty' if code.get('dirty') else ''}",
    }
    fields.update({k: v for k, v in meta.items() if v is not None})
    fields["model"] = params.get("model")
    fields["max_tokens"] = params.get("max_tokens")
    # Structured decoding and the transport it forces are part of the call, so a
    # reader can tell a schema-constrained response from a free-form one.
    fields["structured"] = "output_config" in params
    fields["transport"] = _transport_for(
        params.get("model") or "",
        structured=fields["structured"],
        tools=bool(tools),
    )
    # Exactly one of these is present — see _NO_SAMPLING_MODELS.
    if "thinking" in params:
        fields["thinking"] = (params["thinking"] or {}).get("type")
    if "temperature" in params:
        fields["temperature"] = params["temperature"]
    fields["system_chars"] = len(prompt.system)
    fields["user_chars"] = len(prompt.user)
    fields["est_input_tokens"] = prompt.estimated_input_tokens

    body = [
        "\n".join(f"# {k}: {v}" for k, v in fields.items()),
        "",
        "=== SYSTEM ===",
        prompt.system,
        "",
        "=== USER ===",
        prompt.user,
    ]
    if tools:
        body += ["", "=== TOOLS ===", json.dumps(tools, indent=2, ensure_ascii=False)]

    path = _PROMPT_DIR / f"{rel}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    _PROMPT_INDEX.append({"rel": f"{rel}.txt", **fields})


def flush_prompt_index() -> None:
    """Write ``index_{pass}.json`` listing this pass's prompts, then audit the corpus.

    Every field the index carries also sits in the corresponding ``.txt`` header,
    so the index is a convenience for sorting/filtering the corpus, not the record.

    The drift report runs even when this pass captured nothing — that is the case
    worth hearing about, not a no-op: a rerun that skipped every isoform leaves
    the whole corpus stale while writing no new file to hint at it.
    """
    if _PROMPT_DIR is None:
        return
    if _PROMPT_INDEX:
        index_path = _PROMPT_DIR / f"index_{_PROMPT_PASS}.json"
        index_path.write_text(
            json.dumps(_merged_index(index_path), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    print(f"[prompts] captured {len(_PROMPT_INDEX)} prompts (run {_RUN_ID}) -> {_PROMPT_DIR}")
    _report_corpus_drift()


def _merged_index(index_path: Path) -> list[dict[str, Any]]:
    """This run's index entries merged over the existing index, keyed by file.

    The .txt files are overwritten per isoform, so after a partial rerun
    (``--gene``, ``--only-category``) the corpus is mostly the earlier run's
    files. Rewriting the index from this run alone dropped those from it — the
    Sep-25 reruns left three arms' indexes listing 1-12 prompts out of 300.
    Entries whose file is gone are dropped; every entry keeps its own run_id.
    """
    merged: dict[str, dict[str, Any]] = {}
    if index_path.exists():
        try:
            prior = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a corrupt index is rebuilt from this run
            prior = []
        for entry in prior if isinstance(prior, list) else []:
            rel = entry.get("rel") if isinstance(entry, dict) else None
            if rel and (index_path.parent / rel).exists():
                merged[rel] = entry
    for entry in _PROMPT_INDEX:
        merged[entry["rel"]] = entry
    return list(merged.values())


def _carries_other_runs(stamp_path: Path) -> bool:
    """Whether a stamp's per-category provenance names any run but this one."""
    try:
        runs = json.loads(stamp_path.read_text(encoding="utf-8")).get("category_runs") or {}
    except Exception:  # noqa: BLE001 - unreadable is reported elsewhere
        return False
    return any((r or {}).get("run_id") != _RUN_ID for r in runs.values())


def _read_run_id(stamp_path: Path) -> str | None:
    """The run id recorded in a ``.meta.json`` sidecar, or None if unreadable."""
    try:
        return json.loads(stamp_path.read_text(encoding="utf-8")).get("run_id")
    except Exception:  # noqa: BLE001 - an unreadable stamp is "unknown", not fatal
        return None


def _report_corpus_drift() -> None:
    """Say whether this corpus describes the outputs sitting in ``--out``.

    The prompt files are keyed by isoform and overwritten in place, so the corpus
    is the union of the most recent capture per isoform rather than a snapshot of
    one run. Comparing each output's run stamp against this run's is what turns
    that from an invisible property into a printed line.
    """
    if _PROMPT_OUT_DIR is None:
        return
    spec = PASS_REGISTRY.get(_PROMPT_PASS)
    if spec is None or "{tis_slug}" not in spec.output_filename_template:
        return  # the per-gene default pass has no per-isoform layout to check
    artifact = Path(spec.output_filename_template).name

    outputs = sorted(_PROMPT_OUT_DIR.glob(f"*/{artifact}"))
    if _RUN_MODE == "dry_run":
        if outputs:
            print(
                f"[prompts] run_mode=dry_run: no API calls were made, so this corpus "
                f"describes none of the {len(outputs)} {artifact} in {_PROMPT_OUT_DIR}",
                file=sys.stderr,
            )
        return

    foreign, unknown, mixed = [], [], []
    for out_file in outputs:
        stamp = out_file.with_suffix(".meta.json")
        run_id = _read_run_id(stamp) if stamp.exists() else None
        if run_id is None:
            unknown.append(out_file.parent.name)
        elif run_id != _RUN_ID:
            foreign.append(out_file.parent.name)
        elif _carries_other_runs(stamp):
            mixed.append(out_file.parent.name)

    def _warn(kind: str, slugs: list[str]) -> None:
        if not slugs:
            return
        shown = ", ".join(slugs[:3]) + (f", +{len(slugs) - 3} more" if len(slugs) > 3 else "")
        print(
            f"[prompts] WARNING: {len(slugs)} isoform(s) in {_PROMPT_OUT_DIR} have "
            f"{artifact} {kind} — the prompts here do not describe them: {shown}",
            file=sys.stderr,
        )

    _warn("from a different run", foreign)
    _warn("with no run stamp (written before stamping, or by hand)", unknown)
    _warn("carrying categories from an earlier run (partial rerun; see category_runs)", mixed)

    captured = {entry["rel"] for entry in _PROMPT_INDEX}
    stale = [
        p for p in _PROMPT_DIR.rglob("*.txt") if str(p.relative_to(_PROMPT_DIR)) not in captured
    ]
    if stale:
        print(
            f"[prompts] {len(stale)} prompt file(s) in {_PROMPT_DIR} are left over from "
            "earlier runs and were not rewritten by this one",
            file=sys.stderr,
        )


def _extract_text(content: Any) -> str | None:
    """Return the first ``text`` block's text from an SDK content list, or None.

    Robust to a leading non-text block (e.g. a ``thinking`` block emitted by newer
    models) — unlike ``content[0].text``. Shared by the direct and batch paths.
    """
    return next((b.text for b in content if getattr(b, "type", None) == "text"), None)


def _build_request_params(
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    system: str,
    user: str,
    warn: bool = True,
    output_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble Messages-API request params, gating sampling/thinking by model.

    One place decides temperature/thinking so the direct (:func:`call_llm`) and
    batch (:func:`call_llm_batch`) paths stay in sync. For models in
    :data:`_NO_SAMPLING_MODELS`, ``temperature`` is omitted and thinking is
    disabled (see that constant's rationale); other models are unchanged.

    ``output_schema`` turns on structured outputs, constraining decoding so the
    response parses by construction. It does not cover a response cut short by
    ``max_tokens`` or stopped by ``refusal`` — see :func:`_raise_on_incomplete`.

    ``warn=False`` suppresses the ignored-temperature notice. Prompt capture
    calls this a second time purely to read the gate back out, and would
    otherwise print the same warning twice per call.
    """
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": [{"type": "text", "text": user}]}],
    }
    if output_schema is not None:
        params["output_config"] = {
            "format": {"type": "json_schema", "schema": _output_format_schema(output_schema)}
        }
    if model in _NO_SAMPLING_MODELS:
        params["thinking"] = {"type": "disabled"}
        if warn and temperature != DEFAULT_TEMPERATURE:
            print(
                f"[warn] {model} rejects non-default sampling params; "
                f"ignoring temperature={temperature}",
                file=sys.stderr,
            )
    else:
        params["temperature"] = temperature
    return params


def _transport_for(model: str, *, structured: bool, tools: bool = False) -> str:
    """Which client a call will go through: ``mozzarellm`` or the raw ``sdk``.

    The decision :func:`call_llm` makes at its head, lifted so prompt capture can
    record it without restating (and drifting from) the condition. Structured
    outputs and tool loops both force the raw SDK — mozzarellm's ``query()`` has
    nowhere to put ``output_config`` or ``tools``.
    """
    if tools or structured or model in _NO_SAMPLING_MODELS:
        return "sdk"
    return "mozzarellm" if _try_import_mozzarellm() is not None else "sdk"


def _capture_single_shot(
    rel: str,
    prompt: Prompt,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    meta: dict[str, Any],
    output_schema: dict[str, Any] | None = None,
) -> None:
    """Capture a single-shot prompt alongside the params its call will use.

    Rebuilds the params rather than threading them out of :func:`call_llm`:
    :func:`_build_request_params` is pure, so the second call is byte-identical
    to the one that goes on the wire, and capture stays at the runner level
    where the gene/isoform/category identity is in scope. ``output_schema`` must
    be passed for that to hold — it is what adds ``output_config`` to the wire
    params, and the header reports it as ``structured``.
    """
    if _PROMPT_DIR is None:
        return
    record_prompt(
        rel,
        prompt,
        _build_request_params(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=prompt.system,
            user=prompt.user,
            warn=False,
            output_schema=output_schema,
        ),
        meta=meta,
    )


def call_llm(
    prompt: Prompt,
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str,
    output_schema: dict[str, Any] | None = None,
) -> str:
    """Call the Anthropic API and return the response text.

    Prefers mozzarellm's ``AnthropicClient`` if importable (gets retry logic for
    free); falls back to the official ``anthropic`` SDK otherwise. Raises
    :class:`RuntimeError` if neither is available, or if the call fails.

    mozzarellm passes ``temperature`` unconditionally in its constructor, which a
    :data:`_NO_SAMPLING_MODELS` model rejects with a 400, so for those models we
    skip mozzarellm and use the raw SDK (where :func:`_build_request_params` omits
    temperature).

    ``output_schema`` also forces the raw SDK — mozzarellm's ``query()`` has
    nowhere to put ``output_config``, so it would drop the guarantee silently.
    """
    mozz = _try_import_mozzarellm()
    if _transport_for(model, structured=output_schema is not None) == "mozzarellm":
        client = mozz(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )
        response_text, error = client.query(
            system_prompt=prompt.system,
            user_prompt=prompt.user,
        )
        if error is not None:
            raise RuntimeError(f"mozzarellm AnthropicClient.query failed: {error}")
        if not response_text:
            raise RuntimeError("mozzarellm AnthropicClient.query returned empty response")
        _record_usage(None)  # mozzarellm.query does not surface token usage
        return response_text

    anthropic = _try_import_anthropic()
    if anthropic is None:
        if mozz is not None:
            raise RuntimeError(
                f"{model} requires the official `anthropic` SDK (it rejects "
                "mozzarellm's unconditional temperature, or structured outputs are "
                "in use); install `anthropic`."
            )
        raise RuntimeError(
            "Neither mozzarellm nor anthropic SDK is importable. "
            "Install one of them in the swissisoform-v2 env (e.g. `pip install anthropic`)."
        )

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        **_build_request_params(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=prompt.system,
            user=prompt.user,
            output_schema=output_schema,
        )
    )
    _raise_on_incomplete(getattr(response, "stop_reason", None))
    if not response.content:
        raise RuntimeError("anthropic SDK returned empty content list")
    _record_usage(getattr(response, "usage", None))
    text = _extract_text(response.content)
    if text is None:
        raise RuntimeError("anthropic SDK response had no text block")
    return text


def _empty_usage() -> dict[str, int]:
    return dict.fromkeys(_USAGE_KEYS, 0)


# ── Tool loop (multi-turn categories) ─────────────────────────────────────

# A category read that queries its own data runs several API turns instead of
# one. Defaults are deliberately tight: the readers answer in one call each, so a
# well-behaved loop finishes in 3-5 turns and the cap only catches a model that
# is going in circles.
DEFAULT_MAX_TOOL_TURNS = 8
MIN_DATA_TOOL_CALLS = 2
# Hard cap on one serialised tool result. The readers already bound their own
# output; this is the backstop that keeps a pathological isoform from blowing the
# context window mid-loop.
MAX_TOOL_RESULT_CHARS = 60_000

# Category-neutral on purpose: this is sent to every tool loop, and M's readers
# query variants while P's read a structure. The category's own vocabulary is
# already in front of the model twice — its system prompt's ## Tools section and
# the tool schemas on every request — so naming it here would only be a way to
# get it wrong.
_TOOL_NUDGE = (
    "You did not call a tool. Use the reader tools to inspect the underlying data, "
    "then call emit_verdict exactly once with your reasoning and evidence_used."
)


class ToolLoopError(RuntimeError):
    """A tool loop that ended without a read, carrying its partial trace.

    The trace is attached because a loop that failed is precisely the one worth
    inspecting; the caller persists it alongside successful ones.
    """

    def __init__(self, message: str, trace: dict[str, Any]):
        """Store the failure message plus the transcript captured so far."""
        super().__init__(message)
        self.trace = trace


def _drain_all_usage() -> tuple[dict[str, int], int]:
    """Pop and sum every recorded call usage, with the number of events drained.

    :func:`_drain_usage` pops a single event, which is right for a one-shot call
    but under-counts a multi-turn loop that records one event per turn. The count
    is returned so the usage report can attribute one "call" per round trip
    rather than one per category.
    """
    acc = dict.fromkeys(_USAGE_KEYS, 0)
    n = 0
    while _USAGE_EVENTS:
        ev = _USAGE_EVENTS.pop()
        n += 1
        for k in _USAGE_KEYS:
            acc[k] += ev.get(k, 0)
    return acc, n


def _build_tool_request_params(
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    system: str,
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """Request params for a tool-enabled call, minus ``messages``.

    Same sampling/thinking gate as :func:`_build_request_params` (see
    :data:`_NO_SAMPLING_MODELS`); ``messages`` is supplied per turn because it
    grows as the conversation does.
    """
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "tools": tools,
        "tool_choice": {"type": "auto"},
    }
    if model in _NO_SAMPLING_MODELS:
        params["thinking"] = {"type": "disabled"}
    else:
        params["temperature"] = temperature
    return params


def _block_to_trace(block: Any) -> dict[str, Any]:
    """Serialise one SDK content block for the persisted transcript."""
    kind = getattr(block, "type", None)
    if kind == "text":
        return {"type": "text", "text": getattr(block, "text", "")}
    if kind == "thinking":
        return {"type": "thinking", "thinking": getattr(block, "thinking", "")}
    if kind == "tool_use":
        return {
            "type": "tool_use",
            "id": getattr(block, "id", None),
            "name": getattr(block, "name", None),
            "input": dict(getattr(block, "input", None) or {}),
        }
    return {"type": str(kind)}


def run_tool_loop(
    *,
    system: str,
    user: str,
    tools: list[dict[str, Any]],
    dispatch: Callable[[str, dict[str, Any]], dict[str, Any]],
    model: str,
    max_tokens: int,
    api_key: str,
    temperature: float = DEFAULT_TEMPERATURE,
    terminal_tool: str = "emit_verdict",
    min_data_calls: int = MIN_DATA_TOOL_CALLS,
    max_turns: int = DEFAULT_MAX_TOOL_TURNS,
    verdict_schema: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Drive a multi-turn tool conversation to a terminal read.

    The model receives ``user`` as its opening context (for the category passes,
    the same precomputed slice the single-shot path uses) plus ``tools``. On each
    ``tool_use`` block the corresponding reader runs locally via ``dispatch`` and
    its JSON result is fed back as a ``tool_result``. The loop ends when the model
    calls ``terminal_tool``, whose input becomes the returned verdict.

    Requires the official ``anthropic`` SDK — mozzarellm's client has no tool
    support, so unlike :func:`call_llm` there is no fallback.

    Args:
        system: System prompt describing the tools and the expected verdict.
        user: Opening user message — the precomputed category slice.
        tools: Anthropic tool definitions, including the terminal tool.
        dispatch: ``(name, input) -> result`` executor for the reader tools.
        model: Anthropic model id.
        max_tokens: Per-turn output cap.
        api_key: Anthropic API key.
        temperature: Sampling temperature, omitted for models that reject it.
        terminal_tool: Name of the tool whose input becomes the verdict.
        min_data_calls: Terminal calls made before this many successful data-tool
            calls are rejected with an error ``tool_result``, and the loop
            continues. Stops the model from restating the aggregate it was handed
            without ever looking at the underlying rows.
        max_turns: Backstop on API round trips.
        verdict_schema: Validated against the terminal tool's input before it is
            accepted; a failing payload is rejected with an error ``tool_result``
            and the loop continues. Omit to accept whatever the model emits.

    Returns:
        ``(verdict, trace)``. ``verdict`` is the terminal tool's input dict;
        ``trace`` is the full transcript, written out by the caller as an audit
        artifact.

    Raises:
        ToolLoopError: No verdict within ``max_turns``, or the model refused to
            use tools twice running. Carries the partial trace.
        RuntimeError: The ``anthropic`` SDK is unavailable.
    """
    anthropic = _try_import_anthropic()
    if anthropic is None:
        raise RuntimeError(
            "Tool-augmented category reads require the official `anthropic` SDK "
            "(mozzarellm's client does not support tool use). Install it with "
            "`pip install anthropic`, or pass --no-tools to run every category "
            "through the single-shot path."
        )

    client = anthropic.Anthropic(api_key=api_key)
    params = _build_tool_request_params(
        model=model, max_tokens=max_tokens, temperature=temperature, system=system, tools=tools
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": [{"type": "text", "text": user}]}]
    trace: dict[str, Any] = {
        "model": model,
        "max_turns": max_turns,
        "min_data_calls": min_data_calls,
        "turns": [],
    }
    n_data_calls = 0
    nudged = False
    bad_verdicts: list[dict[str, Any]] = []

    for turn in range(1, max_turns + 1):
        response = client.messages.create(**params, messages=messages)
        _record_usage(getattr(response, "usage", None))
        content = list(response.content or [])
        turn_record: dict[str, Any] = {
            "turn": turn,
            "stop_reason": getattr(response, "stop_reason", None),
            "assistant": [_block_to_trace(b) for b in content],
            "tool_results": [],
        }
        trace["turns"].append(turn_record)

        # Append the assistant turn wholesale so any thinking/redacted blocks
        # survive intact — reconstructing content block-by-block breaks models
        # that require their thinking blocks echoed back verbatim.
        messages.append({"role": "assistant", "content": content})

        tool_uses = [b for b in content if getattr(b, "type", None) == "tool_use"]
        if not tool_uses:
            # No tool call. A verdict the model wrote as plain JSON is still a
            # verdict, so accept it — but through the same gates as the terminal
            # tool below. `strict: true` constrains tool INPUTS, so a payload
            # arriving as text is unpoliced by construction; taking it on the
            # truthiness of one key let an off-enum verdict, an over-long
            # reasoning, or leaked tool-call markup reach categories.json, where
            # nothing downstream rejects it either.
            text = _extract_text(content)
            payload: Any = None
            if text:
                try:
                    payload = parse_response(text)
                except json.JSONDecodeError:
                    payload = None

            reason: str | None = None  # None means "accept"
            # Tier 1: is this an answer at all? The verdict enum used to serve as
            # the format marker separating a real payload from chatter; with it
            # gone, non-empty reasoning is what says "the model answered". Keep
            # this distinct from the _verdict_violations branch below — that one
            # means "an answer, but a bad one", and collects for salvage. Merging
            # them would let arbitrary JSON reach the salvage path and ship.
            #
            # Deliberately NOT derived from verdict_schema["required"]: the tags
            # arms list tags_fired there, whose legitimate value is [], so a
            # truthiness rule over required keys would nudge a valid payload.
            if not (
                isinstance(payload, dict)
                and isinstance(payload.get("reasoning"), str)
                and payload["reasoning"].strip()
            ):
                reason = _TOOL_NUDGE
            elif n_data_calls < min_data_calls:
                # Premature, not corrupt — and deliberately NOT added to
                # bad_verdicts, or answering in prose would be a way around the
                # read-first rule via the salvage path.
                reason = _premature_verdict_msg(
                    min_data_calls=min_data_calls,
                    n_data_calls=n_data_calls,
                    terminal_tool=terminal_tool,
                )
            else:
                violations = _verdict_violations(payload, verdict_schema) if verdict_schema else []
                if violations:
                    bad_verdicts.append(payload)
                    reason = (
                        "Rejected: " + "; ".join(violations) + f". Re-emit via {terminal_tool} "
                        "with plain-text reasoning only."
                    )

            if reason is None:
                trace["outcome"] = "text_verdict"
                trace["n_data_calls"] = n_data_calls
                return payload, trace
            if nudged:
                # The rejection shares the nudge's one-shot budget rather than
                # earning its own; a payload rejected on content is still worth
                # salvaging, one rejected as premature is not.
                if bad_verdicts:
                    return _salvaged(bad_verdicts, trace, n_data_calls=n_data_calls, turns=turn)
                trace["outcome"] = "no_tool_call"
                raise ToolLoopError(
                    f"model made no tool call and gave no usable verdict after "
                    f"{turn} turns: {reason}",
                    trace,
                )
            nudged = True
            turn_record["nudged"] = True
            turn_record["nudge_reason"] = reason
            messages.append({"role": "user", "content": [{"type": "text", "text": reason}]})
            continue

        results: list[dict[str, Any]] = []
        for block in tool_uses:
            name = getattr(block, "name", "")
            tool_input = dict(getattr(block, "input", None) or {})

            if name == terminal_tool:
                if n_data_calls < min_data_calls:
                    msg = _premature_verdict_msg(
                        min_data_calls=min_data_calls,
                        n_data_calls=n_data_calls,
                        terminal_tool=terminal_tool,
                    )
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": getattr(block, "id", None),
                            "is_error": True,
                            "content": msg,
                        }
                    )
                    turn_record["tool_results"].append({"name": name, "rejected": msg})
                    continue
                # strict:true guarantees the keys and types, not what is inside a
                # string — so validate before accepting. A turn may carry several
                # terminal calls (the model correcting itself); take the first
                # that passes rather than the first that appears.
                violations = (
                    _verdict_violations(tool_input, verdict_schema) if verdict_schema else []
                )
                if not violations:
                    trace["outcome"] = "emit_verdict"
                    trace["n_data_calls"] = n_data_calls
                    return tool_input, trace
                bad_verdicts.append(tool_input)
                msg = (
                    "Rejected: " + "; ".join(violations) + f". Re-emit {terminal_tool} "
                    "with plain-text reasoning only."
                )
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": getattr(block, "id", None),
                        "is_error": True,
                        "content": msg,
                    }
                )
                turn_record["tool_results"].append({"name": name, "rejected": msg})
                continue

            result = dispatch(name, tool_input)
            # Only a call that returned data counts toward min_data_calls —
            # a rejected argument or an unknown tool name is not evidence.
            if not (isinstance(result, dict) and "error" in result):
                n_data_calls += 1
            body = json.dumps(result, ensure_ascii=False, default=str)
            if len(body) > MAX_TOOL_RESULT_CHARS:
                body = body[:MAX_TOOL_RESULT_CHARS] + "\n... [tool result truncated]"
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": getattr(block, "id", None),
                    "content": body,
                }
            )
            turn_record["tool_results"].append(
                {"name": name, "input": tool_input, "result": result}
            )

        messages.append({"role": "user", "content": results})

    if bad_verdicts:
        return _salvaged(bad_verdicts, trace, n_data_calls=n_data_calls, turns=max_turns)

    trace["outcome"] = "max_turns_exhausted"
    trace["n_data_calls"] = n_data_calls
    raise ToolLoopError(
        f"tool loop exhausted {max_turns} turns without calling {terminal_tool}", trace
    )


def call_llm_batch(
    items: list[tuple[str, Prompt]],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str,
    poll_interval: int = 15,
    output_schema: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run ``items`` through the Anthropic Message Batches API (50% token price).

    ``items`` is a list of ``(custom_id, Prompt)``. Submits one batch, polls until
    it ends (usually minutes, up to 24h), and returns
    ``{custom_id: {"text": str|None, "usage": {...}, "error": str|None}}``.
    Results arrive in any order — keyed by ``custom_id`` (must be ``[A-Za-z0-9_-]``,
    ≤64 chars; callers pass index-based ids like ``c0``/``s3``). Same tokens as
    direct calls; the saving is the batch price discount, applied at report time.

    ``output_schema`` enables structured outputs on every request in the batch;
    the Batches API supports them, so the discount is unaffected.
    """
    anthropic = _try_import_anthropic()
    if anthropic is None:
        raise RuntimeError("anthropic SDK required for --batch mode (pip install anthropic).")
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic(api_key=api_key)
    requests = [
        Request(
            custom_id=cid,
            params=MessageCreateParamsNonStreaming(
                **_build_request_params(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=p.system,
                    user=p.user,
                    output_schema=output_schema,
                )
            ),
        )
        for cid, p in items
    ]
    batch = client.messages.batches.create(requests=requests)
    print(f"[batch] submitted {len(requests)} requests -> {batch.id} ({batch.processing_status})")
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        time.sleep(poll_interval)

    out: dict[str, dict[str, Any]] = {}
    for r in client.messages.batches.results(batch.id):
        cid = r.custom_id
        if r.result.type == "succeeded":
            msg = r.result.message
            text = _extract_text(msg.content)
            u = getattr(msg, "usage", None)
            usage = (
                {
                    "input": getattr(u, "input_tokens", 0) or 0,
                    "output": getattr(u, "output_tokens", 0) or 0,
                    "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
                    "cache_creation": getattr(u, "cache_creation_input_tokens", 0) or 0,
                }
                if u is not None
                else _empty_usage()
            )
            # "succeeded" means the HTTP call worked, not that generation did.
            try:
                _raise_on_incomplete(getattr(msg, "stop_reason", None))
            except IncompleteResponse as e:
                out[cid] = {"text": text, "usage": usage, "error": str(e)}
                continue
            out[cid] = {"text": text, "usage": usage, "error": None if text else "empty response"}
        else:
            err = getattr(r.result, "error", None)
            out[cid] = {"text": None, "usage": _empty_usage(), "error": f"{r.result.type}: {err}"}
    return out


# ── Output validation ────────────────────────────────────────────────────


class IncompleteResponse(RuntimeError):
    """The model stopped before finishing — the text is not a failed generation."""


def _raise_on_incomplete(stop_reason: Any) -> None:
    """Raise when the model stopped before finishing.

    Structured outputs guarantee parseable JSON only if generation completed;
    otherwise both cases surface as a generic ``JSONDecodeError`` and look like a
    malformed response.
    """
    if stop_reason == "max_tokens":
        raise IncompleteResponse(
            "response hit max_tokens before completing; the JSON is truncated. "
            "Raise --max-tokens rather than treating this as a bad response."
        )
    if stop_reason == "refusal":
        raise IncompleteResponse(
            "model declined the request (stop_reason=refusal); output does not "
            "follow the schema."
        )


def _save_failed_response(
    out_dir: Path, tis_slug: str, label: str, text: str | None, err: Exception
) -> None:
    """Persist a response that would not parse, before it is lost.

    Best-effort — a failure here must never mask the original error.
    """
    if not text:
        return
    try:
        d = out_dir / "_failures"
        d.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{tis_slug}__{label}")
        (d / f"{safe}.txt").write_text(
            f"# {type(err).__name__}: {err}\n\n{text}", encoding="utf-8"
        )
    except Exception:  # noqa: BLE001 - diagnostics must never break the run
        pass


def _category_runs(
    out_path: Path, results: dict[str, Any], *, model: str, regenerated: set[str] | None
) -> dict[str, dict[str, Any]]:
    """Which run produced each category in *results*.

    A partial (``--only-category``) run writes ``categories.json`` whole, the
    carried-forward verdicts included — so a single file-level ``run_id`` would
    claim all six for this run, prompt and model and all. Regenerated categories
    get this run; the rest keep what the previous stamp recorded for them (its
    own ``category_runs`` entry, or its file-level fields for a stamp that
    predates per-category provenance).
    """
    stamp = out_path.with_suffix(".meta.json")
    try:
        prior = json.loads(stamp.read_text(encoding="utf-8")) if stamp.exists() else {}
    except Exception:  # noqa: BLE001 - an unreadable stamp is "unknown", not fatal
        prior = {}
    prior_runs = prior.get("category_runs") or {}
    this_run = {"run_id": _RUN_ID, "model": model, "code": _RUN_PROVENANCE.get("code")}
    runs: dict[str, dict[str, Any]] = {}
    for name in results:
        if regenerated is None or name in regenerated:
            runs[name] = this_run
        else:
            runs[name] = prior_runs.get(name) or {
                "run_id": prior.get("run_id"),
                "model": prior.get("model"),
                "code": prior.get("code"),
            }
    return runs


def _write_category_results(
    out_path: Path,
    tis_slug: str,
    results: dict[str, Any],
    *,
    model: str,
    regenerated: set[str] | None = None,
) -> bool:
    """Write ``categories.json`` only when every category produced a read.

    The skip check downstream is bare file existence, so a file carrying an
    ``{"error": ...}`` entry would block its own retry: the rerun skips the
    isoform, reports ``0/0 successful`` and exits 0 with the errored read
    staged. Holding the write back leaves the isoform genuinely absent, so the
    next run regenerates it. Partial results land in ``categories.partial.json``
    — nothing reads that name — so the good reads stay auditable.

    ``regenerated`` names the categories this run produced, for a partial run;
    ``None`` means all of them. The stamp records it per category.

    Returns True when the real file was written.
    """
    errored = sorted(k for k, v in results.items() if isinstance(v, dict) and "error" in v)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_suffix(".partial.json")
    if errored:
        partial.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        # Under --force the previous, complete file is still there; not overwriting
        # it with a partial one is the point, but say so — it is now stale.
        kept = " previous output KEPT (now stale)," if out_path.exists() else ""
        print(
            f"[hold] {tis_slug}: {', '.join(errored)} errored — {out_path.name} not written,"
            f"{kept} partial in {partial.name}; rerun to retry this isoform",
            file=sys.stderr,
        )
        return False
    # Read the previous stamp before anything overwrites it.
    runs = _category_runs(out_path, results, model=model, regenerated=regenerated)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    partial.unlink(missing_ok=True)  # a retry succeeded; don't leave the old partial behind
    extra: dict[str, Any] = {"categories": sorted(results), "category_runs": runs}
    if regenerated is not None:
        extra["regenerated"] = sorted(regenerated)
    _write_run_stamp(out_path, pass_name="category", model=model, **extra)
    return True


def parse_response(response_text: str) -> dict[str, Any]:
    """Parse the response text as JSON. May raise ``json.JSONDecodeError``.

    Strips Markdown ``json`` fences if the model wraps the JSON in a code block.
    """
    text = response_text.strip()
    if text.startswith("```"):
        # Strip ```json ... ``` fences
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
    return json.loads(text)


def validate_against_schema(payload: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Validate ``payload`` against ``schema`` if jsonschema is importable.

    Returns a list of human-readable validation messages (empty = clean / not
    validated). Never raises — schema validation is best-effort.
    """
    jsonschema = _try_import_jsonschema()
    if jsonschema is None:
        return []
    try:
        validator = jsonschema.Draft202012Validator(schema)
        return [
            f"{'.'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
            for e in validator.iter_errors(payload)
        ]
    except Exception as e:
        return [f"validator init failed: {e}"]


# Tool-call syntax the model sometimes writes as prose. `strict: true` cannot
# catch it: inside a string value the only rules are escaping and termination.
_VERDICT_MARKUP = re.compile(r"</reasoning>|<parameter\s+name=|<invoke|<antml")


def _tidy_violation(msg: str, limit: int = 140) -> str:
    """Make a violation safe to send back to the model.

    jsonschema embeds the offending value, so a too-long ``reasoning`` echoes
    itself — markup included — into the rejection. Feeding that back would
    re-inject the syntax we are asking it to stop writing.
    """
    msg = _VERDICT_MARKUP.sub("<markup>", msg)
    if len(msg) <= limit:
        return msg
    return f"{msg[: limit - 60]}… {msg[-50:]}"


def _top_level_violations(payload: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """``required``, ``enum`` and string-length checks on top-level fields only."""
    out = [
        f"<root>: {k!r} is a required property"
        for k in schema.get("required", [])
        if k not in payload
    ]
    for name, sub in (schema.get("properties") or {}).items():
        if name not in payload or not isinstance(sub, dict):
            continue
        value = payload[name]
        if "enum" in sub and value not in sub["enum"]:
            out.append(f"{name}: {value!r} is not one of {sub['enum']}")
        if isinstance(value, str):
            if "maxLength" in sub and len(value) > sub["maxLength"]:
                # Worded as jsonschema words it, so the rejection reads the same.
                out.append(f"{name}: is too long ({len(value)} > maxLength {sub['maxLength']})")
            if "minLength" in sub and len(value) < sub["minLength"]:
                out.append(f"{name}: shorter than minLength {sub['minLength']}")
    return out


def _verdict_violations(payload: Any, schema: dict[str, Any]) -> list[str]:
    """Why this verdict payload is unusable, or ``[]`` if it is fine.

    jsonschema honours ``maxLength`` even though the decoder ignores it, so the
    reasoning cap is enforced here rather than at generation time.
    """
    if not isinstance(payload, dict):
        return [f"payload is {type(payload).__name__}, expected object"]
    out = [_tidy_violation(v) for v in validate_against_schema(payload, schema)]
    if _try_import_jsonschema() is None:
        # validate_against_schema is best-effort and returns [] without
        # jsonschema — which silently switched the reasoning cap off in an env
        # missing that (declared) dependency. The verdict checks that matter are
        # top-level, so hold them here regardless.
        out += _top_level_violations(payload, schema)
    for field, value in payload.items():
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        if _VERDICT_MARKUP.search(text):
            out.append(f"{field}: contains tool-call markup, expected plain text")
    return out


def _checked_single_shot(
    prompt: Prompt,
    response_text: str,
    *,
    output_schema: dict[str, Any],
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str,
) -> tuple[dict[str, Any], int]:
    """Parse a single-shot verdict and hold it to the schema; return ``(payload, retries)``.

    The structured-output decoder accepts ``maxLength`` and ignores it, so the
    reasoning cap was enforced only on the tool-loop path (by
    :func:`_verdict_violations`) — the four single-shot categories kept
    over-long reads, up to 88% of one arm's outputs (audit38_synthesis). Same
    check here: a violating verdict is re-asked once, with the rejection and its
    own text, so the model shortens it rather than starting over (an identical
    re-send at temperature 0 would return the same text). A second violation
    raises, which the caller records as that category's error — except a read
    that is still only too long, which the caller cuts with :func:`_cap_reasoning`.

    Raises:
        ValueError: The verdict still violates the schema after one retry.
    """
    payload = _parse_category_read(response_text, output_schema)
    violations = _verdict_violations(payload, output_schema)
    if not violations:
        return payload, 0
    retry = Prompt(
        system=prompt.system,
        user=(
            f"{prompt.user}\n\nYour previous response was rejected: "
            f"{'; '.join(violations)}. Re-emit the complete JSON verdict, fixing only "
            f"that — keep the verdict and the substance, cut the wording.\n\n"
            f"Previous response:\n{response_text}"
        ),
    )
    payload = _parse_category_read(
        call_llm(
            retry,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            output_schema=output_schema,
        ),
        output_schema,
    )
    # A read that is only still too long is not lost: the caller's
    # _cap_reasoning cuts it back to a whole sentence and flags it. Anything
    # else wrong after the retry is an error.
    violations = [
        v for v in _verdict_violations(payload, output_schema) if "is too long" not in v
    ]
    if violations:
        raise ValueError("verdict failed the schema after one retry: " + "; ".join(violations))
    return payload, 1


def _premature_verdict_msg(*, min_data_calls: int, n_data_calls: int, terminal_tool: str) -> str:
    """Rejection for a verdict offered before the data was read.

    Shared by both doors a verdict can arrive through — the terminal tool call
    and the plain-JSON text fallback — so the read-first rule reads the same
    either way and cannot be reworded in one place only. Category-neutral for
    the same reason as :data:`_TOOL_NUDGE`.
    """
    return (
        f"Rejected: call at least {min_data_calls} reader tools before "
        f"{terminal_tool}. You have made {n_data_calls} successful "
        "data call(s). Inspect the underlying data first, then emit your verdict."
    )


def _salvaged(
    bad_verdicts: list[dict[str, Any]], trace: dict[str, Any], *, n_data_calls: int, turns: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ship the last rejected read with its markup cut off. Last resort.

    A rejected payload is usually good prose with the markup at the tail, so
    truncating beats losing the turn's work. Shared by both doors — only content
    rejections reach here; a premature read is never collected.

    Raises:
        ToolLoopError: Truncation left no reasoning. This mattered less when a
            verdict label rode alongside — an empty string still shipped a
            usable category. Now the record would be *contentless*, and the only
            write gate is ``"error" in v`` (:func:`_write_category_results`), so
            it would land in categories.json as a success.
    """
    salvaged = _strip_verdict_markup(bad_verdicts[-1])
    if not str(salvaged.get("reasoning") or "").strip():
        trace["outcome"] = "salvage_empty"
        trace["n_data_calls"] = n_data_calls
        raise ToolLoopError(
            f"no read validated in {turns} turns, and truncating the last payload "
            "left no reasoning — nothing to salvage",
            trace,
        )
    trace["outcome"] = "emit_verdict_salvaged"
    trace["n_data_calls"] = n_data_calls
    print(
        f"    [salvage] no read validated in {turns} turns; "
        "truncating the last payload at the markup marker",
        file=sys.stderr,
    )
    return salvaged, trace


def _strip_verdict_markup(payload: dict[str, Any]) -> dict[str, Any]:
    """Salvage: cut each string field at the first markup marker. Last resort."""
    out = dict(payload)
    for field, value in payload.items():
        if isinstance(value, str):
            m = _VERDICT_MARKUP.search(value)
            if m:
                out[field] = value[: m.start()].rstrip()
    return out


# Appended to a reasoning cut back to fit the schema's cap. The record also gets
# ``reasoning_truncated: True`` so the cut is machine-visible, not just visible.
_TRUNCATION_MARK = " […]"
_SENTENCE_END = re.compile(r"[.!?][\"')\]]*(?=\s|$)")


def _reasoning_limit(schema: dict[str, Any] | None) -> int | None:
    """The ``reasoning`` maxLength the schema declares, if any."""
    prop = ((schema or {}).get("properties") or {}).get("reasoning") or {}
    limit = prop.get("maxLength")
    return limit if isinstance(limit, int) and limit > len(_TRUNCATION_MARK) else None


def _cap_reasoning(
    payload: Any, schema: dict[str, Any] | None, *, mid_sentence: bool = False
) -> Any:
    """Cut an over-long ``reasoning`` back to its last whole sentence, marked.

    The decoder ignores ``maxLength`` and the single-shot path only warns, so
    without this an over-cap read was shipped as-is on most paths. Deterministic
    by construction: the same text always yields the same cut. Falls back to the
    last word break when no sentence ends inside the budget.

    Args:
        payload: A category read; anything else is returned unchanged.
        schema: The output schema whose ``reasoning.maxLength`` is the cap.
        mid_sentence: The text is known to stop mid-sentence (a repaired cut-off
            response), so cut back to a sentence boundary even under the cap.
    """
    limit = _reasoning_limit(schema)
    text = payload.get("reasoning") if isinstance(payload, dict) else None
    if limit is None or not isinstance(text, str):
        return payload
    text = text.rstrip()
    ends_cleanly = bool(_SENTENCE_END.search(text[-4:]))
    if len(text) <= limit and (ends_cleanly or not mid_sentence):
        return payload
    head = text[: limit - len(_TRUNCATION_MARK)]
    ends = [m.end() for m in _SENTENCE_END.finditer(head)]
    cut = ends[-1] if ends else (head.rfind(" ") if " " in head else len(head))
    kept = head[:cut].rstrip()
    if not kept:
        return payload
    print(
        f"    [truncated] reasoning cut from {len(text)} to {len(kept)} chars at a "
        "sentence boundary",
        file=sys.stderr,
    )
    return {**payload, "reasoning": kept + _TRUNCATION_MARK, "reasoning_truncated": True}


def _parse_category_read(text: str, schema: dict[str, Any] | None) -> dict[str, Any]:
    """Parse a single-shot category read, recovering a response cut off after ``reasoning``.

    The one recoverable break is an object closed early with the ``reasoning``
    string itself intact — ``{"reasoning": "…",`` — which is what an over-long
    read has produced. That is re-closed and its reasoning cut back to a whole
    sentence. Anything else (an unterminated string, a non-object, no reasoning)
    raises the original ``JSONDecodeError``.
    """
    try:
        return parse_response(text)
    except json.JSONDecodeError as original:
        stub = text.strip()
        if stub.startswith("```"):  # same fence handling as parse_response
            stub = stub.split("\n", 1)[1] if "\n" in stub else stub
        stub = stub.removesuffix("```").rstrip().rstrip(",").rstrip()
        try:
            payload = json.loads(stub if stub.endswith("}") else stub + "}")
        except json.JSONDecodeError:
            raise original from None
        if not (isinstance(payload, dict) and str(payload.get("reasoning") or "").strip()):
            raise original from None
        payload = _cap_reasoning(payload, schema, mid_sentence=True)
        print("    [repaired] category read was cut off; re-closed the object", file=sys.stderr)
        return payload


def _emit_schema_warnings(
    payload: dict[str, Any], schema: dict[str, Any], label: str, *, verbose: bool
) -> None:
    """Validate ``payload`` and print any warnings to stderr.

    Real violations print unconditionally; ``verbose`` only adds the clean-pass
    line. The reverse — gating violations behind ``--verbose`` — is how the
    ``evidence_used`` type drift went unnoticed across a full run. Never raises:
    a schema mismatch on a cosmetic field should not kill a batch.
    """
    warnings = validate_against_schema(payload, schema)
    for w in warnings:
        print(f"    schema warning [{label}]: {w}", file=sys.stderr)
    if verbose and not warnings:
        print(f"    schema ok [{label}]", file=sys.stderr)


# ── Per-gene runner ───────────────────────────────────────────────────────


@dataclass
class GeneResult:
    """Outcome of running the LLM on a single gene."""

    gene: str
    ok: bool
    elapsed_s: float
    error: str | None = None
    output_path: Path | None = None


def run_one_gene(
    gene: str,
    record: dict[str, Any],
    *,
    system_prompt: str,
    output_schema: dict[str, Any],
    out_dir: Path,
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str,
    force: bool,
    verbose: bool,
) -> GeneResult:
    """Run the LLM on one gene, write output, return the outcome.

    Idempotent: skips if ``out_dir/{gene}.json`` already exists unless
    ``force=True``. On JSON parse failure, dumps the raw response to
    ``out_dir/_failures/{gene}.txt`` and returns ``GeneResult(ok=False, ...)``.
    """
    out_path = out_dir / f"{gene}.json"
    if out_path.exists() and not force:
        return GeneResult(
            gene=gene,
            ok=True,
            elapsed_s=0.0,
            error="skipped (already exists; use --force to overwrite)",
            output_path=out_path,
        )

    prompt = build_prompt(record, system_prompt, output_schema)
    _capture_single_shot(
        f"default/{gene}",
        prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        meta={"pass": "default", "gene": gene},
    )
    t0 = time.time()
    try:
        response_text = call_llm(
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )
    except Exception as e:
        return GeneResult(
            gene=gene,
            ok=False,
            elapsed_s=time.time() - t0,
            error=f"API call failed: {type(e).__name__}: {e}",
        )

    elapsed = time.time() - t0
    try:
        payload = parse_response(response_text)
    except json.JSONDecodeError as e:
        failures_dir = out_dir / "_failures"
        failures_dir.mkdir(parents=True, exist_ok=True)
        (failures_dir / f"{gene}.txt").write_text(response_text, encoding="utf-8")
        return GeneResult(
            gene=gene,
            ok=False,
            elapsed_s=elapsed,
            error=f"JSON parse error: {e}",
        )

    warnings = validate_against_schema(payload, output_schema)
    if warnings and verbose:
        for w in warnings:
            print(f"    schema warning: {w}", file=sys.stderr)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return GeneResult(gene=gene, ok=True, elapsed_s=elapsed, output_path=out_path)


# ── CLI ───────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the LLM-interpretation CLI."""
    parser = argparse.ArgumentParser(
        description="Run LLM interpretation on per-gene SwissIsoform evidence records.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=ROOT / "data/output/cheeseman_13gene/llm_evidence",
        help="Directory of per-gene evidence record JSON files (default: %(default)s).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/output/cheeseman_13gene/llm",
        help="Directory for LLM output JSON files (default: %(default)s).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Anthropic model id.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--gene", help="Run only this gene.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompt assembly diagnostics and exit; no API calls.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs in --out.",
    )
    parser.add_argument(
        "--only-category",
        action="append",
        default=None,
        metavar="LETTER",
        help=(
            "Regenerate only these category letters (repeatable); the rest are "
            "carried forward from the existing categories.json, keeping their own "
            "provenance in the run stamp. Implies --force for the named categories. "
            "Isoforms with no categories.json yet are skipped."
        ),
    )
    parser.add_argument(
        "--only-isoform",
        action="append",
        default=None,
        metavar="TIS_SLUG",
        help=(
            "Restrict to these isoforms by tis_slug (repeatable). The sibling of "
            "--only-category: together they retry one failed cell without touching "
            "the rest of an arm that cost 45 minutes to produce."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Submit the category/synthesis pass via the Message Batches API "
        "(50%% token price, same tokens, results usually minutes / <24h). For the "
        "full offline runs; leave off for the interactive single-gene spot-checks.",
    )
    parser.add_argument(
        "--pass",
        dest="pass_name",
        default="default",
        choices=sorted(PASS_REGISTRY),
        help="Which LLM pass to run (default = V1 single-pass).",
    )
    parser.add_argument(
        "--variants-long",
        type=Path,
        default=None,
        help="variants_long.parquet backing the M-category reader tools. Default: "
        "a sibling of --records (i.e. {run_dir}/variants_long.parquet), which is "
        "where the evidence stage writes it.",
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Run every category through the single-shot path, including the ones "
        "that would otherwise query their own data in a tool loop. Use to A/B "
        "against the pre-tool behaviour, or when the variants table is unavailable.",
    )
    parser.add_argument(
        "--max-tool-turns",
        type=int,
        default=DEFAULT_MAX_TOOL_TURNS,
        help="API round-trip cap per tool-loop category (default: %(default)s).",
    )
    parser.add_argument(
        "--save-prompts",
        action="store_true",
        help="Write every assembled prompt to data/llm_inputs/{run}/ as one .txt per "
        "call (system + user verbatim, plus a header with the run id, model and "
        "sampling gate), so a prompt can be read or diffed without reassembling it "
        "from the evidence record, the prompt file and the schema. Combines with "
        "--dry-run to dump the whole corpus, M/P tool-loop openings included, "
        "at zero API cost. The header's run id matches the output's .meta.json "
        "sidecar only when this run produced that output; mismatches are reported "
        "at the end of the run.",
    )
    parser.add_argument(
        "--save-prompts-dir",
        type=Path,
        default=None,
        help="Override where --save-prompts writes "
        "(default: data/llm_inputs/{run}/, run taken from --out).",
    )
    return parser


def _load_records_with_synthetic_fallback(
    records_dir: Path, gene: str | None, dry_run: bool
) -> dict[str, dict[str, Any]]:
    """Like :func:`load_records` but falls back to a synthetic stub for dry-run only."""
    try:
        return load_records(records_dir, gene=gene)
    except FileNotFoundError:
        if not dry_run:
            raise
        # Synthetic fallback — dry-run with no evidence records yet.
        synthetic_gene = gene or _SYNTHETIC_RECORD["gene"]["name"]
        record = dict(_SYNTHETIC_RECORD)
        record["gene"] = {**record["gene"], "name": synthetic_gene}
        return {synthetic_gene: record}


def _print_dry_run(gene: str, prompt: Prompt, model: str) -> None:
    print(f"=== {gene} ===")
    print(f"model:                 {model}")
    print(f"system prompt chars:   {len(prompt.system)}")
    print(f"user prompt chars:     {len(prompt.user)}")
    print(f"estimated input tokens: {prompt.estimated_input_tokens}")
    print(f"user prompt[:80]:      {prompt.user[:80]!r}")
    print()


def main(
    argv: list[str] | None = None,
    *,
    prompts_dir: Path | None = None,
    run_meta: dict[str, Any] | None = None,
) -> int:
    """CLI entry point. Returns process exit code.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv``).
        prompts_dir: Prompts directory supplied by the thin CLI. When given, the
            non-default passes resolve their prompt/schema files relative to it.
            When ``None``, the module-level ``SYSTEM_PROMPT_PATH`` /
            ``OUTPUT_SCHEMA_PATH`` defaults are used.
        run_meta: Extra provenance to stamp beside every output and into the
            usage report — the prompt-variant driver passes the arm here.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    spec = PASS_REGISTRY[args.pass_name]
    if args.only_category:
        if not spec.iterates_categories:
            parser.error("--only-category applies to --pass category only")
        try:
            _selected_categories(args.only_category)
        except ValueError as e:
            parser.error(str(e))
        # Naming a category IS asking for it to be regenerated. Without this the
        # existence skip fired first and the run printed "0/0 successful, N
        # reused" — a silent no-op. Every isoform it touches is a merge into an
        # existing file, so forcing cannot clobber the categories not named.
        args.force = True
    _begin_run(dry_run=getattr(args, "dry_run", False))
    _RUN_PROVENANCE["source"] = _source_provenance(args.records)
    if run_meta:
        _RUN_PROVENANCE["run_meta"] = run_meta

    # The evidence slices this pass builds carry threshold language ("qualifies
    # when length >= 6 aa AND plddt_mean >= 0.70") that must match the run whose
    # records it is reading — same sibling-of-records trick as --variants-long.
    from swissisoform.site.evidence import use_scoring_config

    use_scoring_config(Path(args.records).parent)

    if getattr(args, "save_prompts", False):
        enable_prompt_capture(
            args.save_prompts_dir or _default_prompt_dir(args.out),
            pass_name=spec.name,
            out_dir=args.out,
        )

    # V1 default-pass reads SYSTEM_PROMPT_PATH / OUTPUT_SCHEMA_PATH directly so tests
    # that monkeypatch those constants keep working bit-identically. Other passes
    # resolve their files relative to the prompts dir.
    prompts_root = prompts_dir or SYSTEM_PROMPT_PATH.parent
    if spec.name == "default":
        system_prompt = load_system_prompt()
        output_schema = load_output_schema()
    else:
        system_prompt = load_system_prompt(prompts_root / spec.system_prompt_filename)
        output_schema = load_output_schema(prompts_root / spec.output_schema_filename)

    records = _load_records_with_synthetic_fallback(args.records, args.gene, args.dry_run)
    if getattr(args, "only_isoform", None):
        records = _filter_isoforms(records, args.only_isoform)

    if spec.requires_prereq:
        missing = _check_prereqs(records, args.out, spec.requires_prereq)
        if missing:
            hint = ", ".join(f"--pass {p}" for p in spec.requires_prereq)
            print(
                f"{spec.name}: missing prereq outputs for {len(missing)} isoform(s); "
                f"run {hint} first.",
                file=sys.stderr,
            )
            return 2

    # finally, not a trailing call: the index must be written even when a pass
    # raises partway through, since a partial corpus is still worth having.
    try:
        if spec.iterates_categories:
            return _run_category_pass(
                records, spec, args, system_prompt, output_schema, prompts_root=prompts_root
            )

        if spec.name == "synthesis":
            return _run_synthesis_pass(records, spec, args, system_prompt, output_schema)

        # spec.name == "default" — V1 single-pass behavior
        return _run_default_pass(records, spec, args, system_prompt, output_schema)
    finally:
        flush_prompt_index()


def _run_default_pass(records, spec, args, system_prompt, output_schema) -> int:
    """V1 single-pass per-gene loop. Preserved bit-identically from V1 main."""
    if args.dry_run:
        for gene_name, record in records.items():
            prompt = build_prompt(record, system_prompt, output_schema)
            _capture_single_shot(
                f"default/{gene_name}",
                prompt,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                meta={"pass": "default", "gene": gene_name},
            )
            _print_dry_run(gene_name, prompt, args.model)
        return 0

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it before running, or pass --dry-run."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    total = len(records)
    results: list[GeneResult] = []
    for i, (gene_name, record) in enumerate(records.items(), start=1):
        result = run_one_gene(
            gene_name,
            record,
            system_prompt=system_prompt,
            output_schema=output_schema,
            out_dir=args.out,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            api_key=api_key,
            force=args.force,
            verbose=args.verbose,
        )
        results.append(result)
        glyph = "OK" if result.ok else "FAIL"
        suffix = f"({result.error})" if result.error else ""
        print(f"[{i}/{total}] {gene_name} ... {glyph} {result.elapsed_s:.1f}s {suffix}".rstrip())

    n_ok = sum(1 for r in results if r.ok)
    n_fail = total - n_ok
    print()
    print(f"{total} genes processed, {n_ok} successful, {n_fail} failed", end="")
    if n_fail:
        failures = ", ".join(f"{r.gene}: {r.error}" for r in results if not r.ok)
        print(f" ({failures})")
        return 1
    print()
    return 0


def _tis_slug(tis_id: str | None) -> str:
    """URL-safe form of tis_id (matches website slugify filter)."""
    return re.sub(r"[:.]+", "-", tis_id or "unknown")


def _resolve_variants_long(args) -> Path:
    """Path to variants_long.parquet for the tool readers.

    Defaults to a sibling of ``--records``: the evidence stage writes
    ``{OUT}/llm_evidence/`` and ``{OUT}/variants_long.parquet`` together, so the
    parent of the records dir is the run directory.
    """
    explicit = getattr(args, "variants_long", None)
    return Path(explicit) if explicit else Path(args.records).parent / "variants_long.parquet"


def _first_isoform(records) -> dict[str, Any]:
    """Any one isoform record, for validating run-wide preconditions up front."""
    for gene_record in (records or {}).values():
        for iso in gene_record.get("isoforms", []) or []:
            return iso
    return {}


def _tool_schemas(letter: str) -> list[dict[str, Any]]:
    """Tool definitions for one category, with no data preconditions.

    Split out of :func:`_tool_setup` because the schemas are static constants
    while the dispatch needs the variants parquet / fold cache. Capturing a tool
    loop's opening request needs the schemas only — the dispatch is what *runs*
    the loop, and a dry run never does.
    """
    if letter == "M":
        from swissisoform.site import tools as m_tools

        return m_tools.M_TOOLS
    if letter == "P":
        from swissisoform.site import structure_tools as p_tools

        return p_tools.P_TOOLS
    raise KeyError(f"no tool schemas registered for category {letter!r}")


def _tool_setup(letter: str, args, records) -> tuple[list[dict[str, Any]], Any]:
    """Validate one tool category's data precondition and build its factory.

    Each category reads a different artifact, so each validates its own and
    returns ``(tool_schemas, dispatch_for)`` where ``dispatch_for(iso)`` binds the
    readers to a single isoform. Preconditions are checked HERE, once, so a
    misconfigured run fails before the first API call rather than after it — and
    loudly rather than degrading, since a verdict reached with the readers and one
    reached without are not comparable.
    """
    if letter == "M":
        from swissisoform.site import tools as m_tools

        variants_long = m_tools.require_variants_long(_resolve_variants_long(args))
        return m_tools.M_TOOLS, (
            lambda iso: m_tools.make_m_dispatch(
                variants_long, iso.get("tis_id"), iso.get("orf_type")
            )
        )

    if letter == "P":
        from swissisoform.site import structure_tools as p_tools

        # The fold cache is keyed by protein-sequence hash, and those hashes only
        # reach the LLM pass via StructureModule's columns. A record from before
        # they existed cannot address the cache at all.
        p_tools.require_structure_hashes(_first_isoform(records).get("_raw") or {})
        return p_tools.P_TOOLS, (
            lambda iso: p_tools.make_p_dispatch(iso.get("_raw") or {})
        )

    raise KeyError(f"no tool setup registered for category {letter!r}")


def _tool_categories(args, prompts_root: Path, records=None) -> dict[str, dict[str, Any]]:
    """Per-letter tool config for categories that run as a loop, or ``{}``.

    Returns an empty mapping — meaning every category takes the single-shot path
    — for ``--no-tools``, and for a plain ``--dry-run`` (which makes no API
    calls, so there is nothing to loop).

    ``--dry-run --save-prompts`` is the exception: it builds a *capture-only*
    config carrying the system prompt and the tool schemas but no dispatch. A
    tool loop's opening request is fully determined client-side — system prompt,
    stripped record, static tool schemas — so it can be captured verbatim
    without an API call. Only turns 2+ depend on model responses. The data
    preconditions in :func:`_tool_setup` are skipped along with the dispatch,
    since the opening context is derived from the evidence record, not from the
    variants parquet or the fold cache.
    """
    if getattr(args, "no_tools", False):
        return {}
    dry_run = getattr(args, "dry_run", False)
    capture_only = dry_run and getattr(args, "save_prompts", False)
    if dry_run and not capture_only:
        return {}

    out: dict[str, dict[str, Any]] = {}
    for letter, prompt_filename in TOOL_CATEGORY_PROMPTS.items():
        prompt_path = prompts_root / prompt_filename
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Tool-loop system prompt for category {letter} not found at "
                f"{prompt_path}. Restore it, or pass --no-tools."
            )
        if capture_only:
            tools, dispatch_for = _tool_schemas(letter), None
        else:
            tools, dispatch_for = _tool_setup(letter, args, records)
        out[letter] = {
            "system": load_system_prompt(prompt_path),
            "tools": tools,
            "dispatch_for": dispatch_for,
            # A tool loop may validate its verdict against a different schema than
            # the single-shot path decodes against. They are not the same problem:
            # the single-shot schema drives constrained decoding, so anything it
            # declares is a slot the model will fill, while the tool loop only ever
            # has its payload checked with jsonschema afterwards. A pass that gives
            # its loop an extra field therefore cannot express that in the shared
            # file without also offering the field to every single-shot category.
            # Optional: absent, the shared schema is used, which is today's behaviour.
            "verdict_schema": _optional_schema(prompts_root / TOOL_VERDICT_SCHEMA),
        }
    return out


def _optional_schema(path: Path) -> dict[str, Any] | None:
    """Load a schema file if it exists, else ``None``."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# Evidence columns a category's readers SUPERSEDE, dropped from that category's
# opening context when it runs as a tool loop. These are pre-computed answers to
# questions the model can now ask itself over the real arrays, so shipping them
# is both redundant and anchoring — measured on a live run, 10 of 18 P reasonings
# cited the pre-computed PAE block means rather than the values they had queried.
#
# P: ``pae_region_blocks`` partitions the structure into diff/body and reports the
# three block means — every one of which ``pae_block()`` recomputes from the same
# matrix, the diagonal diff-vs-diff on a bare call and the other two by naming
# ranges, so the loop was shipping the answers to its own tool calls on every
# turn. ``pae_status`` is deliberately RETAINED — it is
# availability metadata, not a measurement, and it saves a wasted call
# discovering that no PAE matrix exists.
SUPERSEDED_BY_TOOLS: dict[str, tuple[str, ...]] = {
    "P": (
        "isoform_structure_pae_diff_vs_diff",
        "isoform_structure_pae_body_vs_body",
        "isoform_structure_pae_diff_vs_body",
    ),
}


_SUPERSEDED_NOTE = (
    "Pre-computed PAE block means are omitted here: query pae_block "
    "over the ranges you care about instead of reading a fixed "
    "diff/body partition."
)


def _drop_keys(value: Any, keys: frozenset[str]) -> tuple[Any, int]:
    """Copy of *value* with *keys* removed at every depth, and how many went."""
    if isinstance(value, dict):
        out, n = {}, 0
        for k, v in value.items():
            if k in keys:
                n += 1
                continue
            out[k], m = _drop_keys(v, keys)
            n += m
        return out, n
    if isinstance(value, list):
        items = [_drop_keys(v, keys) for v in value]
        return [v for v, _ in items], sum(m for _, m in items)
    return value, 0


def _strip_superseded_evidence(
    category_record: dict[str, Any], letter: str
) -> dict[str, Any]:
    """Drop evidence columns this category's readers replace. Returns a copy.

    Every grounding goes through here, not only the ``members``-shaped one. The
    alternative payloads carry the same columns under other keys — a tag's
    ``metrics``, the raw arm's ``evidence``, the dist arm's ``fields`` — and an
    early return on "no members" handed exactly those arms the answers the
    criteria arm is denied. So a foreign shape is walked whole and the note
    lands once at its top level; the ``members`` shape keeps its per-member note.
    """
    cols = SUPERSEDED_BY_TOOLS.get(letter)
    if not cols:
        return category_record
    if "members" not in category_record:
        stripped, n_dropped = _drop_keys(category_record, frozenset(cols))
        if n_dropped:
            stripped["_superseded_note"] = _SUPERSEDED_NOTE
        return stripped
    members = []
    for member in category_record.get("members") or []:
        evidence = member.get("evidence")
        if not isinstance(evidence, dict):
            members.append(member)
            continue
        dropped = [c for c in cols if c in evidence]
        kept = {k: v for k, v in evidence.items() if k not in cols}
        if dropped:
            kept["_superseded_note"] = _SUPERSEDED_NOTE
        members.append({**member, "evidence": kept})
    return {**category_record, "members": members}


# Categories whose hit rows the readers SUPERSEDE, dropped from that category's
# opening context. Keyed by letter like SUPERSEDED_BY_TOOLS, and for the same
# reason: what a reader replaces is a property of that category's data.
#
# M only — its hits are a pre-filtered 2% sample of a table the readers query in
# full, and ~93% of the payload. P's are P3's complete SSE scan and the exact
# list its verdict was computed from, so nothing supersedes them. Same
# lossy-vs-full line _strip_superseded_evidence draws for pae_status.
STRIP_HITS_FOR_TOOLS: frozenset[str] = frozenset({"M"})


def _selected_categories(only: list[str] | None) -> list[dict[str, Any]]:
    """``CATEGORIES``, or just the requested letters, in registry order."""
    from swissisoform.site.evidence import CATEGORIES

    if not only:
        return list(CATEGORIES)
    want = {c.upper() for c in only}
    picked = [c for c in CATEGORIES if c["letter"] in want]
    missing = want - {c["letter"] for c in picked}
    if missing:
        raise ValueError(
            f"unknown category letter(s): {', '.join(sorted(missing))}; "
            f"have {', '.join(c['letter'] for c in CATEGORIES)}"
        )
    return picked


def _regenerated(only: list[str] | None) -> set[str] | None:
    """Category names a partial run produces, or ``None`` for a full run."""
    return {c["name"] for c in _selected_categories(only)} if only else None


def _nothing_to_merge_into(out_path: Path, args, gene_name: str, tis_slug: str) -> bool:
    """Whether a ``--only-category`` run must skip this isoform.

    A partial run merges into an existing ``categories.json``. With none there,
    it would write a file holding only the named categories — and the skip check
    downstream is bare existence, so that one-category file would pass for a
    complete one forever after.
    """
    if not args.only_category or out_path.exists():
        return False
    print(
        f"[skip] {gene_name} {tis_slug}: --only-category needs an existing "
        f"{out_path.name} to merge into; run the full pass for this isoform first",
        file=sys.stderr,
    )
    return True


def _seed_results(out_path: Path, only: list[str] | None) -> dict[str, Any]:
    """Prior verdicts for the categories this run is not regenerating.

    A partial run rewrites the same ``categories.json``, and the writer emits
    ``results`` wholesale — so without this the five categories we did not ask
    for would be dropped on disk. Returns empty for a full run, where every
    category is about to be recomputed anyway.
    """
    if not only or not out_path.exists():
        return {}
    prior = json.loads(out_path.read_text())
    regenerating = {c["name"] for c in _selected_categories(only)}
    return {k: v for k, v in prior.items() if k not in regenerating}


def _strip_hits_for_tools(category_record: dict[str, Any]) -> dict[str, Any]:
    """Drop the truncated hit rows from a tool-loop category's opening context.

    For a single-shot read that ``MAX_HITS`` sample IS the evidence; for a tool
    loop it is the thing the readers replace, and keeping it is doubly wrong. The
    API is stateless so the opening context is re-sent every turn, and on M those
    rows are ~93% of the payload (78,655 chars with, 5,810 without — serialised
    twice, since both members declare the same ``evidence_hits_col``). They are
    also a pre-filtered 2% sample competing with honest query access to the table.

    Scalars, reasons and hints stay — that is the intended starting context. Each
    member keeps ``n_hits_total`` plus a note pointing at the tools. Returns a
    copy; the input is not mutated.

    Applied only to the categories in :data:`STRIP_HITS_FOR_TOOLS`, so the note's
    "variant records" wording describes every record it can reach.
    """
    from swissisoform.site.evidence import hits_omitted_note

    # An alternative grounding may not use ``members`` at all — the tags arm keys
    # its payload on ``tags`` — and inventing an empty one here made the strip
    # look like it had run when it had not. Leave a foreign shape untouched; that
    # arm strips at its own builder (``grounding.hits_for``).
    if "members" not in category_record:
        return category_record
    members = []
    for member in category_record.get("members") or []:
        members.append(
            {
                **member,
                "hits": [],
                "n_hits_shown": 0,
                "hits_note": hits_omitted_note(member.get("n_hits_total") or 0),
            }
        )
    return {**category_record, "members": members}


def _capture_tool_opening(*, config, iso, category_record, args, letter: str) -> str:
    """Build a tool loop's opening user message, capturing it when enabled.

    The opening request is fully determined client-side — system prompt,
    stripped record, static tool schemas — so this runs identically whether or
    not the loop is about to be executed, and a dry run can capture it for free.

    Worth capturing because ``{letter}_trace.json`` records only
    ``opening_context_chars``: never the system prompt, the opening text, or the
    tool schemas. The trace is what the model *did*; this is what it was *told*.
    """
    opening = _strip_superseded_evidence(category_record, letter)
    if letter in STRIP_HITS_FOR_TOOLS:
        opening = _strip_hits_for_tools(opening)
    prompt_user = json.dumps(opening, indent=2, ensure_ascii=False)
    if _PROMPT_DIR is not None:
        record_prompt(
            f"category/{_tis_slug(iso.get('tis_id'))}/{letter}_tools",
            Prompt(system=config["system"], user=prompt_user),
            _build_tool_request_params(
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                system=config["system"],
                tools=config["tools"],
            ),
            meta={
                "pass": "category",
                "gene": (iso.get("gene") or {}).get("name"),
                "tis_id": iso.get("tis_id"),
                "category": letter,
                "mode": "tool_loop",
                "max_tool_turns": getattr(args, "max_tool_turns", DEFAULT_MAX_TOOL_TURNS),
            },
            tools=config["tools"],
        )
    return prompt_user


def _run_tool_category(
    *, config, iso, category_record, args, api_key, out_dir: Path, letter: str,
    output_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one category as a tool loop and persist its transcript.

    The trace is written whether the loop succeeded or failed — a loop that ran
    out of turns is exactly the one worth inspecting. Returns the verdict payload,
    or re-raises so the caller's per-category error handling applies.
    """
    dispatch = config["dispatch_for"](iso)
    prompt_user = _capture_tool_opening(
        config=config, iso=iso, category_record=category_record, args=args, letter=letter
    )
    trace_path = out_dir / f"{letter}_trace.json"

    def _persist(trace: dict[str, Any]) -> None:
        # Recorded so the cost of the multi-turn path can be measured from the
        # transcripts alone (the opening context is billed once per turn).
        trace["opening_context_chars"] = len(prompt_user)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False, default=str))

    try:
        verdict, trace = run_tool_loop(
            system=config["system"],
            user=prompt_user,
            tools=config["tools"],
            dispatch=dispatch,
            model=args.model,
            max_tokens=args.max_tokens,
            api_key=api_key,
            temperature=args.temperature,
            max_turns=getattr(args, "max_tool_turns", DEFAULT_MAX_TOOL_TURNS),
            verdict_schema=config.get("verdict_schema") or output_schema,
        )
    except ToolLoopError as e:
        _persist(e.trace)
        raise
    _persist(trace)
    return verdict


def _filter_isoforms(records: dict, wanted: list[str]) -> dict:
    """Keep only ``wanted`` tis_slugs, dropping genes left with none.

    Raises on a slug that matches nothing: a targeted retry that silently
    selects zero isoforms looks exactly like a successful one.
    """
    want = set(wanted)
    kept: dict = {}
    seen: set[str] = set()
    for gene, record in records.items():
        isos = [i for i in (record.get("isoforms") or []) if _tis_slug(i.get("tis_id")) in want]
        seen.update(_tis_slug(i.get("tis_id")) for i in isos)
        if isos:
            kept[gene] = {**record, "isoforms": isos}
    if missing := sorted(want - seen):
        raise SystemExit(f"--only-isoform matched no record for: {', '.join(missing)}")
    return kept


def _check_prereqs(records, out_dir: Path, prereqs: tuple[str, ...]) -> list[str]:
    """Return tis_slugs missing any prereq output file."""
    missing: list[str] = []
    for _, gene_record in records.items():
        for iso in gene_record.get("isoforms", []) or []:
            tis_slug = _tis_slug(iso.get("tis_id"))
            for prereq in prereqs:
                # Resolve the prereq's ACTUAL output path from its PassSpec — the
                # pass name ("category") differs from its filename ("categories.json"),
                # so f"{prereq}.json" would look for the wrong file.
                rel = PASS_REGISTRY[prereq].output_filename_template.format(tis_slug=tis_slug)
                if not (out_dir / rel).exists():
                    missing.append(tis_slug)
                    break
    return missing


def _run_category_pass(
    records, spec, args, system_prompt, output_schema, *, prompts_root: Path
) -> int:
    """Per-(isoform, category) dispatch — one call per CDLMPS category.

    Each call bundles all of the category's members (all first-class scored
    criteria, including S2 biophysics + S3 SAE) into one slice and asks the model
    for a single ``{reasoning}``. Writes ``{tis_slug}/categories.json`` as
    a dict keyed by category name (the shape ``category_verdicts_for_isoform``
    consumes) — but only once every category produced a verdict; a run with any
    errored category holds the file back (:func:`_write_category_results`).

    Categories in :data:`TOOL_CATEGORY_PROMPTS` instead run a multi-turn tool
    loop, reading their own underlying data before emitting the same
    ``{reasoning}`` shape, and additionally write ``{letter}_trace.json``.
    """
    tool_configs = _tool_categories(args, prompts_root, records)

    if getattr(args, "batch", False) and not args.dry_run:
        return _run_category_pass_batch(
            records, spec, args, system_prompt, output_schema, tool_configs=tool_configs
        )

    from swissisoform.site.evidence import slice_category

    api_key = os.environ.get("ANTHROPIC_API_KEY") if not args.dry_run else "dry"
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it before running, or pass --dry-run."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    n_calls = 0
    n_ok = 0
    n_reused = 0
    usage_by_slug: dict[str, dict[str, int]] = {}
    tool_usage_by_slug: dict[str, dict[str, int]] = {}

    for gene_name, gene_record in records.items():
        for iso in gene_record.get("isoforms", []) or []:
            tis_slug_val = _tis_slug(iso.get("tis_id"))
            out_filename = spec.output_filename_template.format(tis_slug=tis_slug_val)
            out_path = args.out / out_filename
            # Idempotency: skip the entire isoform if its output already exists
            # (and not in --force). This applies in both real and dry-run modes —
            # dry-run prints already-exist skips so the user can audit them.
            if out_path.exists() and not args.force:
                if args.dry_run:
                    print(f"[skip] {gene_name} {tis_slug_val}: {out_path.name} exists")
                n_reused += 1
                continue
            if _nothing_to_merge_into(out_path, args, gene_name, tis_slug_val):
                continue

            out_dir = args.out / tis_slug_val
            if not args.dry_run:
                out_dir.mkdir(parents=True, exist_ok=True)

            iso_with_gene = {**iso, "gene": {"name": gene_name}}
            # ``categories.json`` is written whole from ``results``, so a
            # --only-category run must carry the untouched categories forward or
            # it silently deletes them.
            results: dict[str, Any] = _seed_results(out_path, args.only_category)
            for category in _selected_categories(args.only_category):
                letter = category["letter"]
                tool_config = tool_configs.get(letter)
                category_record = slice_category(iso_with_gene, category)
                prompt = build_prompt(category_record, system_prompt, output_schema)
                # Tool-loop categories send their own opening context, not this
                # single-shot prompt, so recording it would put a counterfactual
                # in the corpus. Capture the real opening instead — it needs no
                # API call, so a dry run gets it for free.
                if tool_config is not None and args.dry_run:
                    _capture_tool_opening(
                        config=tool_config,
                        iso=iso_with_gene,
                        category_record=category_record,
                        args=args,
                        letter=letter,
                    )
                if tool_config is None:
                    _capture_single_shot(
                        f"category/{tis_slug_val}/{letter}",
                        prompt,
                        model=args.model,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        meta={
                            "pass": "category",
                            "gene": gene_name,
                            "tis_id": iso.get("tis_id"),
                            "category": f"{letter} ({category['name']})",
                        },
                        output_schema=output_schema,
                    )
                n_calls += 1
                if args.dry_run:
                    print(
                        f"[{n_calls}] {gene_name} {tis_slug_val} category: "
                        f"{letter} ({category['name']}) input chars: {len(prompt.user)}"
                    )
                    continue
                response_text: str | None = None
                try:
                    if tool_config is not None:
                        payload = _run_tool_category(
                            config=tool_config,
                            iso=iso_with_gene,
                            category_record=category_record,
                            args=args,
                            api_key=api_key,
                            out_dir=out_dir,
                            letter=letter,
                            output_schema=output_schema,
                        )
                        # One usage event per turn, so drain them all.
                        _tool_usage, _tool_turns = _drain_all_usage()
                        _add_usage(tool_usage_by_slug, tis_slug_val, _tool_usage, calls=_tool_turns)
                    else:
                        response_text = call_llm(
                            prompt,
                            model=args.model,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                            api_key=api_key,
                            output_schema=output_schema,
                        )
                        try:
                            payload, _retries = _checked_single_shot(
                                prompt,
                                response_text,
                                output_schema=output_schema,
                                model=args.model,
                                temperature=args.temperature,
                                max_tokens=args.max_tokens,
                                api_key=api_key,
                            )
                        finally:
                            # The first call's usage, plus a retry's if one ran.
                            _usage, _calls = _drain_all_usage()
                            _add_usage(usage_by_slug, tis_slug_val, _usage, calls=max(_calls, 1))
                    _emit_schema_warnings(
                        payload, output_schema,
                        f"{tis_slug_val}/{category['name']}",
                        verbose=getattr(args, "verbose", False),
                    )
                    results[category["name"]] = _cap_reasoning(payload, output_schema)
                    n_ok += 1
                except Exception as e:
                    if tool_config is not None:
                        _tool_usage, _tool_turns = _drain_all_usage()
                        _add_usage(tool_usage_by_slug, tis_slug_val, _tool_usage, calls=_tool_turns)
                    print(f"[{n_calls}] {letter} FAIL: {e}", file=sys.stderr)
                    results[category["name"]] = {"error": str(e)}
                    # None for a tool-loop failure (no single-shot text) and for a
                    # call that raised before returning; _save_failed_response no-ops.
                    _save_failed_response(
                        args.out, tis_slug_val, category["name"], response_text, e
                    )

            if args.dry_run:
                continue
            _write_category_results(
                out_path,
                tis_slug_val,
                results,
                model=args.model,
                regenerated=_regenerated(args.only_category),
            )

    if args.dry_run:
        return 0
    _write_usage_report(args.out, "category", args.model, usage_by_slug)
    if tool_usage_by_slug:
        # Separate report: tool categories are multi-turn and never batched, so
        # folding them into the batch-priced total would misstate both.
        _write_usage_report(args.out, "category_tools", args.model, tool_usage_by_slug)
    # Reuse is silent otherwise, so a run that regenerated nothing reads as a
    # clean 0/0 — say what was reused and how to override it.
    print(
        f"{spec.name}: {n_ok}/{n_calls} successful"
        + (f", {n_reused} isoform(s) reused (--force to regenerate)" if n_reused else "")
    )
    return 0 if n_ok == n_calls else 1


def _run_category_pass_batch(
    records, spec, args, system_prompt, output_schema, *, tool_configs=None
) -> int:
    """Category pass via the Message Batches API.

    One batch of all (isoform, category) calls at 50% token price; identical
    prompts/outputs to the sequential path.

    Tool-loop categories are excluded from the batch and run interactively after
    it. A batch request is fire-and-forget — there is no way to intercept a
    ``tool_use`` mid-flight and feed a ``tool_result`` back — so multi-turn
    categories are structurally unbatchable, not merely slower. They pay full
    price and are reported separately.
    """
    from swissisoform.site.evidence import slice_category

    tool_configs = tool_configs or {}
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    args.out.mkdir(parents=True, exist_ok=True)

    items: list[tuple[str, Prompt]] = []  # (custom_id, prompt)
    meta: dict[str, tuple[str, str]] = {}  # custom_id -> (tis_slug, category_name)
    iso_results: dict[str, dict[str, Any]] = {}  # tis_slug -> {category_name: payload}
    iso_out: dict[str, Path] = {}  # tis_slug -> categories.json path
    # tis_slug -> (iso_with_gene, [(category, sliced_record), ...]) for the
    # interactive tool pass below.
    tool_work: dict[str, tuple[dict[str, Any], list[tuple[dict[str, Any], dict[str, Any]]]]] = {}
    n_reused = 0
    for gene_name, gene_record in records.items():
        for iso in gene_record.get("isoforms", []) or []:
            tis_slug_val = _tis_slug(iso.get("tis_id"))
            out_path = args.out / spec.output_filename_template.format(tis_slug=tis_slug_val)
            if out_path.exists() and not args.force:
                n_reused += 1
                continue
            if _nothing_to_merge_into(out_path, args, gene_name, tis_slug_val):
                continue
            iso_with_gene = {**iso, "gene": {"name": gene_name}}
            iso_results.setdefault(tis_slug_val, _seed_results(out_path, args.only_category))
            iso_out[tis_slug_val] = out_path
            for category in _selected_categories(args.only_category):
                record = slice_category(iso_with_gene, category)
                if category["letter"] in tool_configs:
                    entry = tool_work.setdefault(tis_slug_val, (iso_with_gene, []))
                    entry[1].append((category, record))
                    continue
                cid = f"c{len(items)}"
                prompt = build_prompt(record, system_prompt, output_schema)
                _capture_single_shot(
                    f"category/{tis_slug_val}/{category['letter']}",
                    prompt,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    meta={
                        "pass": "category",
                        "gene": gene_name,
                        "tis_id": iso.get("tis_id"),
                        "category": f"{category['letter']} ({category['name']})",
                        "mode": "batch",
                        "custom_id": cid,
                    },
                    output_schema=output_schema,
                )
                items.append((cid, prompt))
                meta[cid] = (tis_slug_val, category["name"])

    if not items and not tool_work:
        print("category: nothing to do (all outputs exist; use --force to rebuild).")
        return 0

    responses = (
        call_llm_batch(
            items,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            api_key=api_key,
            output_schema=output_schema,
        )
        if items
        else {}
    )

    usage_by_slug: dict[str, dict[str, int]] = {}
    retry_usage_by_slug: dict[str, dict[str, int]] = {}
    prompts_by_cid = dict(items)
    n_ok = 0
    for cid, (tis_slug_val, cat_name) in meta.items():
        r = responses.get(cid) or {"text": None, "usage": _empty_usage(), "error": "missing result"}
        _add_usage(usage_by_slug, tis_slug_val, r["usage"])
        if r["error"] or not r["text"]:
            print(f"[{cid}] {cat_name} FAIL: {r['error']}", file=sys.stderr)
            iso_results[tis_slug_val][cat_name] = {"error": r["error"] or "empty"}
            continue
        try:
            try:
                # A retry cannot ride the batch (it is already back), so it is a
                # direct call — billed at full price and reported separately.
                payload, _retries = _checked_single_shot(
                    prompts_by_cid[cid],
                    r["text"],
                    output_schema=output_schema,
                    model=args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    api_key=api_key,
                )
            finally:
                _usage, _calls = _drain_all_usage()
                if _calls:
                    _add_usage(retry_usage_by_slug, tis_slug_val, _usage, calls=_calls)
            _emit_schema_warnings(
                payload, output_schema, f"{tis_slug_val}/{cat_name}",
                verbose=getattr(args, "verbose", False),
            )
            iso_results[tis_slug_val][cat_name] = _cap_reasoning(payload, output_schema)
            n_ok += 1
        except Exception as e:
            print(f"[{cid}] {cat_name} parse FAIL: {e}", file=sys.stderr)
            iso_results[tis_slug_val][cat_name] = {"error": str(e)}
            _save_failed_response(args.out, tis_slug_val, cat_name, r["text"], e)

    # Tool categories, interactively, merged into the same categories.json.
    tool_usage_by_slug: dict[str, dict[str, int]] = {}
    n_tool_calls = 0
    n_tool_ok = 0
    for tis_slug_val, (iso_with_gene, categories) in tool_work.items():
        out_dir = args.out / tis_slug_val
        out_dir.mkdir(parents=True, exist_ok=True)
        for category, record in categories:
            letter = category["letter"]
            n_tool_calls += 1
            try:
                payload = _run_tool_category(
                    config=tool_configs[letter],
                    iso=iso_with_gene,
                    category_record=record,
                    args=args,
                    api_key=api_key,
                    out_dir=out_dir,
                    letter=letter,
                    output_schema=output_schema,
                )
                _tool_usage, _tool_turns = _drain_all_usage()
                _add_usage(tool_usage_by_slug, tis_slug_val, _tool_usage, calls=_tool_turns)
                _emit_schema_warnings(
                    payload,
                    output_schema,
                    f"{tis_slug_val}/{category['name']}",
                    verbose=getattr(args, "verbose", False),
                )
                iso_results[tis_slug_val][category["name"]] = _cap_reasoning(payload, output_schema)
                n_tool_ok += 1
            except Exception as e:
                _tool_usage, _tool_turns = _drain_all_usage()
                _add_usage(tool_usage_by_slug, tis_slug_val, _tool_usage, calls=_tool_turns)
                print(f"[{tis_slug_val}] {letter} FAIL: {e}", file=sys.stderr)
                iso_results[tis_slug_val][category["name"]] = {"error": str(e)}

    for tis_slug_val, results in iso_results.items():
        _write_category_results(
            iso_out[tis_slug_val],
            tis_slug_val,
            results,
            model=args.model,
            regenerated=_regenerated(args.only_category),
        )

    if items:
        _write_usage_report(args.out, "category", args.model, usage_by_slug, batch=True)
    if tool_usage_by_slug:
        _write_usage_report(args.out, "category_tools", args.model, tool_usage_by_slug)
    if retry_usage_by_slug:
        _write_usage_report(args.out, "category_retry", args.model, retry_usage_by_slug)
    total = len(items) + n_tool_calls
    print(
        f"category: {n_ok}/{len(items)} successful (batch)"
        + (f" + {n_tool_ok}/{n_tool_calls} tool-loop (direct)" if n_tool_calls else "")
        + (f", {n_reused} isoform(s) reused (--force to regenerate)" if n_reused else "")
    )
    return 0 if (n_ok + n_tool_ok) == total else 1


def _build_synthesis_record(
    isoform: dict, gene_name: str, isoform_out_dir: Path, gene: dict | None = None
) -> dict:
    """Build the synthesis-pass input from the disk-cached categories.json output.

    Carries the gene's ESTABLISHED function (``gene`` — Affinage function / keywords
    / localization, the baseline the isoform diverges from), the digested per-category
    reads (``category_reads`` — the ``{reasoning}`` per CDLMPS category), and
    the raw underlying evidence (``criteria_evidence``, one ``slice_criterion`` payload
    per criterion, all 16 incl. P3/S2/S3) so the model can weigh actual numbers.

    ``gene`` is the gene block from the evidence record (``build_gene_record``); when
    absent, only ``{name}`` is carried (older records / dry-run stubs).
    """
    from swissisoform.site.evidence import CRITERIA, _diff_region_location, slice_criterion

    category_reads: dict[str, Any] = {}
    pp = isoform_out_dir / "categories.json"
    if pp.exists():
        category_reads = json.loads(pp.read_text(encoding="utf-8"))

    iso_with_gene = {**isoform, "gene": {"name": gene_name}}
    # Drop each criterion's own identity block: the record's top-level ``isoform``
    # block below states the same thing (and adds differential_region_location).
    # Keeping them repeated the same seven fields 15 times, once per criterion.
    criteria_evidence: dict[str, Any] = {}
    for cid in CRITERIA:
        entry = slice_criterion(iso_with_gene, cid)
        entry.pop("isoform", None)
        criteria_evidence[cid] = entry

    gene_block = {
        "name": gene_name,
        "function": (gene or {}).get("function"),
        "keywords": (gene or {}).get("keywords"),
        "subcellular_location": (gene or {}).get("subcellular_location"),
    }

    # Localization is the one category with a machine-readable KNOWN value
    # (Affinage subcellular_location) to compare a structured prediction against,
    # so surface the calibration triad explicitly: literature vs DeepLoc-on-
    # canonical (+confidence) vs DeepLoc-on-isoform (+confidence). The model uses
    # the canonical-vs-literature agreement to decide whether DeepLoc is calibrated
    # for THIS protein before trusting its isoform call. All values already live in
    # the L1 criterion evidence — no new data source.
    l1_ev = (criteria_evidence.get("L1_localization_change") or {}).get("evidence") or {}
    localization_block = {
        "known_from_literature": gene_block["subcellular_location"],
        "predicted_canonical": {
            "compartment": l1_ev.get("canonical_localization_deeploc_prediction"),
            "top_prob": l1_ev.get("canonical_localization_deeploc_top_prob"),
        },
        "predicted_isoform": {
            "compartment": l1_ev.get("isoform_localization_deeploc_prediction"),
            "top_prob": l1_ev.get("isoform_localization_deeploc_top_prob"),
        },
    }

    return {
        "gene": gene_block,
        "localization": localization_block,
        "isoform": {
            "tis_id": isoform.get("tis_id"),
            "gene_name": gene_name,
            "orf_type": isoform.get("orf_type"),
            "differential_region_location": _diff_region_location(isoform.get("orf_type")),
            "differential_sequence": isoform.get("differential_sequence"),
            "diff_space": isoform.get("diff_space"),
            "isoform_length_aa": isoform.get("isoform_length_aa"),
            "canonical_length_aa": isoform.get("canonical_length_aa"),
        },
        "scoring": isoform.get("scoring") or {},
        "key_metrics": isoform.get("key_metrics") or {},
        "category_reads": category_reads,
        "criteria_evidence": criteria_evidence,
    }


def _run_synthesis_pass(records, spec, args, system_prompt, output_schema) -> int:
    if getattr(args, "batch", False) and not args.dry_run:
        return _run_synthesis_pass_batch(records, spec, args, system_prompt, output_schema)

    api_key = os.environ.get("ANTHROPIC_API_KEY") if not args.dry_run else "dry"
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Export it or pass --dry-run.")

    args.out.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_calls = 0
    n_reused = 0
    usage_by_slug: dict[str, dict[str, int]] = {}
    for gene_name, gene_record in records.items():
        for iso in gene_record.get("isoforms", []) or []:
            tis_slug = _tis_slug(iso.get("tis_id"))
            iso_dir = args.out / tis_slug
            out_path = iso_dir / "synthesis.json"
            # Idempotency: skip if output exists and not forced (matches the
            # hoisted check in _run_modality_pass).
            if out_path.exists() and not args.force:
                if args.dry_run:
                    print(f"[skip] {gene_name} {tis_slug}: synthesis.json exists")
                n_reused += 1
                continue
            synthesis_record = _build_synthesis_record(
                iso, gene_name, iso_dir, gene_record.get("gene")
            )
            prompt = build_prompt(synthesis_record, system_prompt, output_schema)
            _capture_single_shot(
                f"synthesis/{tis_slug}",
                prompt,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                meta={
                    "pass": "synthesis",
                    "gene": gene_name,
                    "tis_id": iso.get("tis_id"),
                },
                output_schema=output_schema,
            )
            n_calls += 1
            if args.dry_run:
                print(
                    f"[{n_calls}] {gene_name} {tis_slug} synthesis input chars: {len(prompt.user)}"
                )
                continue
            response_text: str | None = None
            try:
                response_text = call_llm(
                    prompt,
                    model=args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    api_key=api_key,
                    output_schema=output_schema,
                )
                _add_usage(usage_by_slug, tis_slug, _drain_usage())
                payload = parse_response(response_text)
                _emit_schema_warnings(
                    payload, output_schema, f"{tis_slug}/synthesis",
                    verbose=getattr(args, "verbose", False),
                )
                iso_dir.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
                _write_run_stamp(out_path, pass_name="synthesis", model=args.model)
                n_ok += 1
                print(f"[{n_calls}] {tis_slug} OK")
            except Exception as e:
                print(f"[{n_calls}] {tis_slug} FAIL: {e}", file=sys.stderr)
                _save_failed_response(args.out, tis_slug, "synthesis", response_text, e)

    if args.dry_run:
        return 0
    _write_usage_report(args.out, "synthesis", args.model, usage_by_slug)
    print(
        f"synthesis: {n_ok}/{n_calls} successful"
        + (f", {n_reused} isoform(s) reused (--force to regenerate)" if n_reused else "")
    )
    return 0 if n_ok == n_calls else 1


def _run_synthesis_pass_batch(records, spec, args, system_prompt, output_schema) -> int:
    """Synthesis pass via the Message Batches API.

    One batch, one request per isoform, at 50% token price. Runs after the
    category batch (it reads each isoform's categories.json).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    args.out.mkdir(parents=True, exist_ok=True)

    items: list[tuple[str, Prompt]] = []  # (custom_id, prompt)
    meta: dict[str, str] = {}  # custom_id -> tis_slug
    out_by_slug: dict[str, Path] = {}  # tis_slug -> synthesis.json path
    n_reused = 0
    for gene_name, gene_record in records.items():
        for iso in gene_record.get("isoforms", []) or []:
            tis_slug = _tis_slug(iso.get("tis_id"))
            iso_dir = args.out / tis_slug
            out_path = iso_dir / "synthesis.json"
            if out_path.exists() and not args.force:
                n_reused += 1
                continue
            record = _build_synthesis_record(
                iso, gene_name, iso_dir, gene_record.get("gene")
            )
            cid = f"s{len(items)}"
            prompt = build_prompt(record, system_prompt, output_schema)
            _capture_single_shot(
                f"synthesis/{tis_slug}",
                prompt,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                meta={
                    "pass": "synthesis",
                    "gene": gene_name,
                    "tis_id": iso.get("tis_id"),
                    "mode": "batch",
                    "custom_id": cid,
                },
                output_schema=output_schema,
            )
            items.append((cid, prompt))
            meta[cid] = tis_slug
            out_by_slug[tis_slug] = out_path

    if not items:
        print("synthesis: nothing to do (all outputs exist; use --force to rebuild).")
        return 0

    responses = call_llm_batch(
        items, model=args.model, temperature=args.temperature,
        max_tokens=args.max_tokens, api_key=api_key,
        output_schema=output_schema,
    )

    usage_by_slug: dict[str, dict[str, int]] = {}
    n_ok = 0
    for cid, tis_slug in meta.items():
        r = responses.get(cid) or {"text": None, "usage": _empty_usage(), "error": "missing result"}
        _add_usage(usage_by_slug, tis_slug, r["usage"])
        if r["error"] or not r["text"]:
            print(f"[{cid}] {tis_slug} FAIL: {r['error']}", file=sys.stderr)
            continue
        try:
            payload = parse_response(r["text"])
        except Exception as e:
            print(f"[{cid}] {tis_slug} parse FAIL: {e}", file=sys.stderr)
            _save_failed_response(args.out, tis_slug, "synthesis", r["text"], e)
            continue
        _emit_schema_warnings(
            payload, output_schema, f"{tis_slug}/synthesis",
            verbose=getattr(args, "verbose", False),
        )
        out_path = out_by_slug[tis_slug]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        _write_run_stamp(out_path, pass_name="synthesis", model=args.model)
        n_ok += 1

    _write_usage_report(args.out, "synthesis", args.model, usage_by_slug, batch=True)
    print(
        f"synthesis: {n_ok}/{len(items)} successful (batch)"
        + (f", {n_reused} isoform(s) reused (--force to regenerate)" if n_reused else "")
    )
    return 0 if n_ok == len(items) else 1
