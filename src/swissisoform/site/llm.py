r"""Run LLM interpretation on per-gene evidence records.

Reads ``{records_dir}/{gene}.json`` (produced by ``evidence.py``), composes a
prompt from ``system.txt`` + the evidence JSON + ``output_schema.json``, calls
the Anthropic API, validates + writes the structured output to ``{out}/{gene}.json``.

Single-threaded; idempotent (skips genes whose output already exists unless
``force``). ``dry_run`` prints prompt-size diagnostics and exits without any
network calls; ``save_prompts`` writes each assembled prompt in full to
``data/llm/{run}/`` and composes with ``dry_run``.

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
import sys
import time
from dataclasses import dataclass
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


def load_system_prompt(path: Path | None = None) -> str:
    """Load the system prompt, erroring early if missing.

    Reads ``SYSTEM_PROMPT_PATH`` at call time (so tests can monkeypatch it).
    """
    path = path or SYSTEM_PROMPT_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"System prompt not found at {path}. "
            "Agent A is responsible for producing scripts/site/prompts/system.txt."
        )
    return path.read_text(encoding="utf-8").strip()


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


# ── Prompt capture ────────────────────────────────────────────────────────

# Opt-in, write-only dump of every assembled prompt, one .txt per API call,
# under its own top-level data/ folder with one subdir per run. Single-threaded,
# like _USAGE_EVENTS above.
#
# Deliberately NOT under data/cache/: nothing ever reads these files back and no
# step is skipped because one exists, so they are an audit artifact rather than a
# results cache and do not bend the "Execution Contract — fresh reruns" rule in
# CLAUDE.md. Every run rewrites them from scratch.
DEFAULT_PROMPT_DIR = ROOT / "data" / "llm"

_PROMPT_DIR: Path | None = None
_PROMPT_PASS: str = "prompts"
_PROMPT_INDEX: list[dict[str, Any]] = []


def _default_prompt_dir(out: Path) -> Path:
    """Where captured prompts go when ``--save-prompts-dir`` is not given.

    ``data/llm/{run}/``, where ``run`` comes from ``--out``: the standard layout
    is ``data/output/{run}/llm``, so the run name is the parent directory's. A
    non-standard ``--out`` falls back to its own basename rather than guessing.

    The per-run subdir is load-bearing, not cosmetic: ``index_{pass}.json`` sits
    at the root of the prompt dir, so two presets sharing one directory would
    clobber each other's index while their .txt files merged silently.
    """
    run = out.parent.name if out.name == "llm" else out.name
    return DEFAULT_PROMPT_DIR / (run or "unnamed")


def enable_prompt_capture(directory: Path, pass_name: str = "prompts") -> None:
    """Start capturing assembled prompts under ``directory``.

    ``pass_name`` scopes the index file, because each pass runs as its own
    process against the same directory — a single ``index.json`` would be
    truncated to whichever pass ran last while its ``.txt`` files remained.
    Mirrors the ``_usage_{pass}.json`` naming next door.
    """
    global _PROMPT_DIR, _PROMPT_PASS
    _PROMPT_DIR = directory
    _PROMPT_PASS = pass_name
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
    """
    if _PROMPT_DIR is None:
        return

    fields: dict[str, Any] = {k: v for k, v in meta.items() if v is not None}
    fields["model"] = params.get("model")
    fields["max_tokens"] = params.get("max_tokens")
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
    """Write ``index_{pass}.json`` listing this pass's prompts. No-op when empty.

    Every field it carries also sits in the corresponding ``.txt`` header, so
    the index is a convenience for sorting/filtering the corpus, not the record.
    """
    if _PROMPT_DIR is None or not _PROMPT_INDEX:
        return
    out = _PROMPT_DIR / f"index_{_PROMPT_PASS}.json"
    out.write_text(
        json.dumps(_PROMPT_INDEX, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"[prompts] captured {len(_PROMPT_INDEX)} prompts -> {_PROMPT_DIR}")


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
) -> dict[str, Any]:
    """Assemble Messages-API request params, gating sampling/thinking by model.

    One place decides temperature/thinking so the direct (:func:`call_llm`) and
    batch (:func:`call_llm_batch`) paths stay in sync. For models in
    :data:`_NO_SAMPLING_MODELS`, ``temperature`` is omitted and thinking is
    disabled (see that constant's rationale); other models are unchanged.

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


def _capture_single_shot(
    rel: str,
    prompt: Prompt,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    meta: dict[str, Any],
) -> None:
    """Capture a single-shot prompt alongside the params its call will use.

    Rebuilds the params rather than threading them out of :func:`call_llm`:
    :func:`_build_request_params` is pure, so the second call is byte-identical
    to the one that goes on the wire, and capture stays at the runner level
    where the gene/isoform/category identity is in scope.
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
) -> str:
    """Call the Anthropic API and return the response text.

    Prefers mozzarellm's ``AnthropicClient`` if importable (gets retry logic for
    free); falls back to the official ``anthropic`` SDK otherwise. Raises
    :class:`RuntimeError` if neither is available, or if the call fails.

    mozzarellm passes ``temperature`` unconditionally in its constructor, which a
    :data:`_NO_SAMPLING_MODELS` model rejects with a 400, so for those models we
    skip mozzarellm and use the raw SDK (where :func:`_build_request_params` omits
    temperature).
    """
    mozz = _try_import_mozzarellm()
    if mozz is not None and model not in _NO_SAMPLING_MODELS:
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
                "mozzarellm's unconditional temperature); install `anthropic`."
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
        )
    )
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

_TOOL_NUDGE = (
    "You did not call a tool. Use the reader tools to inspect the variant data, "
    "then call emit_verdict exactly once with your verdict and reasoning."
)


class ToolLoopError(RuntimeError):
    """A tool loop that ended without a verdict, carrying its partial trace.

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
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Drive a multi-turn tool conversation to a terminal verdict.

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
            # No tool call. Accept a verdict the model wrote as plain JSON, else
            # nudge once before giving up.
            text = _extract_text(content)
            payload: Any = None
            if text:
                try:
                    payload = parse_response(text)
                except json.JSONDecodeError:
                    payload = None
            if isinstance(payload, dict) and payload.get("verdict"):
                trace["outcome"] = "text_verdict"
                trace["n_data_calls"] = n_data_calls
                return payload, trace
            if nudged:
                trace["outcome"] = "no_tool_call"
                raise ToolLoopError(
                    f"model made no tool call and returned no parseable verdict after {turn} turns",
                    trace,
                )
            nudged = True
            turn_record["nudged"] = True
            messages.append({"role": "user", "content": [{"type": "text", "text": _TOOL_NUDGE}]})
            continue

        results: list[dict[str, Any]] = []
        for block in tool_uses:
            name = getattr(block, "name", "")
            tool_input = dict(getattr(block, "input", None) or {})

            if name == terminal_tool:
                if n_data_calls < min_data_calls:
                    msg = (
                        f"Rejected: call at least {min_data_calls} reader tools before "
                        f"{terminal_tool}. You have made {n_data_calls} successful "
                        "data call(s). Inspect the variant data first, then emit "
                        "your verdict."
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
                # Taken verbatim. The terminal tool declares strict:true, so the
                # API constrains sampling to its schema and the input arrives with
                # the declared keys and types or not at all — no client-side
                # coercion, which would only hide a contract the model broke.
                # A payload that is still wrong surfaces downstream through
                # _emit_schema_warnings against output_schemas/category_read.json.
                trace["outcome"] = "emit_verdict"
                trace["n_data_calls"] = n_data_calls
                return tool_input, trace

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
) -> dict[str, dict[str, Any]]:
    """Run ``items`` through the Anthropic Message Batches API (50% token price).

    ``items`` is a list of ``(custom_id, Prompt)``. Submits one batch, polls until
    it ends (usually minutes, up to 24h), and returns
    ``{custom_id: {"text": str|None, "usage": {...}, "error": str|None}}``.
    Results arrive in any order — keyed by ``custom_id`` (must be ``[A-Za-z0-9_-]``,
    ≤64 chars; callers pass index-based ids like ``c0``/``s3``). Same tokens as
    direct calls; the saving is the batch price discount, applied at report time.
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
            out[cid] = {"text": text, "usage": usage, "error": None if text else "empty response"}
        else:
            err = getattr(r.result, "error", None)
            out[cid] = {"text": None, "usage": _empty_usage(), "error": f"{r.result.type}: {err}"}
    return out


# ── Output validation ────────────────────────────────────────────────────


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
        help="Write every assembled prompt to data/llm/{run}/ as one .txt per "
        "call (system + user verbatim, plus a header with the model and sampling "
        "gate), so a prompt can be read or diffed without reassembling it from "
        "the evidence record, the prompt file and the schema. Combines with "
        "--dry-run to dump the whole corpus, M/P tool-loop openings included, "
        "at zero API cost.",
    )
    parser.add_argument(
        "--save-prompts-dir",
        type=Path,
        default=None,
        help="Override where --save-prompts writes "
        "(default: data/llm/{run}/, run taken from --out).",
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


def main(argv: list[str] | None = None, *, prompts_dir: Path | None = None) -> int:
    """CLI entry point. Returns process exit code.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv``).
        prompts_dir: Prompts directory supplied by the thin CLI. When given, the
            non-default passes resolve their prompt/schema files relative to it.
            When ``None``, the module-level ``SYSTEM_PROMPT_PATH`` /
            ``OUTPUT_SCHEMA_PATH`` defaults are used.
    """
    args = build_parser().parse_args(argv)
    spec = PASS_REGISTRY[args.pass_name]

    if getattr(args, "save_prompts", False):
        enable_prompt_capture(
            args.save_prompts_dir or _default_prompt_dir(args.out), pass_name=spec.name
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
            "system": prompt_path.read_text(encoding="utf-8").strip(),
            "tools": tools,
            "dispatch_for": dispatch_for,
        }
    return out


# Evidence columns a category's readers SUPERSEDE, dropped from that category's
# opening context when it runs as a tool loop. These are pre-computed answers to
# questions the model can now ask itself over the real arrays, so shipping them
# is both redundant and anchoring — measured on a live run, 10 of 18 P reasonings
# cited the pre-computed PAE block means rather than the values they had queried.
#
# P: ``pae_region_blocks`` partitions the structure into diff/body and reports the
# three block means. ``pae_block()`` with default arguments computes exactly the
# diff-vs-body value, so the loop was shipping the answer to one of its own tool
# calls on every turn. ``pae_status`` is deliberately RETAINED — it is
# availability metadata, not a measurement, and it saves a wasted call
# discovering that no PAE matrix exists.
SUPERSEDED_BY_TOOLS: dict[str, tuple[str, ...]] = {
    "P": (
        "isoform_structure_pae_diff_vs_diff",
        "isoform_structure_pae_body_vs_body",
        "isoform_structure_pae_diff_vs_body",
    ),
}


def _strip_superseded_evidence(
    category_record: dict[str, Any], letter: str
) -> dict[str, Any]:
    """Drop evidence columns this category's readers replace. Returns a copy."""
    cols = SUPERSEDED_BY_TOOLS.get(letter)
    if not cols:
        return category_record
    members = []
    for member in category_record.get("members") or []:
        evidence = member.get("evidence")
        if not isinstance(evidence, dict):
            members.append(member)
            continue
        dropped = [c for c in cols if c in evidence]
        kept = {k: v for k, v in evidence.items() if k not in cols}
        if dropped:
            kept["_superseded_note"] = (
                "Pre-computed PAE block means are omitted here: query pae_block "
                "over the ranges you care about instead of reading a fixed "
                "diff/body partition."
            )
        members.append({**member, "evidence": kept})
    return {**category_record, "members": members}


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
    """
    members = []
    for member in category_record.get("members") or []:
        n_total = member.get("n_hits_total") or 0
        members.append(
            {
                **member,
                "hits": [],
                "n_hits_shown": 0,
                "hits_note": (
                    f"{n_total} variant records exist for this isoform. Example rows "
                    "are deliberately omitted here: query them with the reader tools "
                    "so you choose the filter, rather than reasoning from a fixed "
                    "sample."
                ),
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
    opening = _strip_superseded_evidence(_strip_hits_for_tools(category_record), letter)
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
    *, config, iso, category_record, args, api_key, out_dir: Path, letter: str
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
        )
    except ToolLoopError as e:
        _persist(e.trace)
        raise
    _persist(trace)
    return verdict


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
    for a single ``{verdict, reasoning}``. Writes ``{tis_slug}/categories.json`` as
    a dict keyed by category name (the shape ``category_verdicts_for_isoform``
    consumes).

    Categories in :data:`TOOL_CATEGORY_PROMPTS` instead run a multi-turn tool
    loop, reading their own underlying data before emitting the same
    ``{verdict, reasoning}`` shape, and additionally write ``{letter}_trace.json``.
    """
    tool_configs = _tool_categories(args, prompts_root, records)

    if getattr(args, "batch", False) and not args.dry_run:
        return _run_category_pass_batch(
            records, spec, args, system_prompt, output_schema, tool_configs=tool_configs
        )

    from swissisoform.site.evidence import CATEGORIES, slice_category

    api_key = os.environ.get("ANTHROPIC_API_KEY") if not args.dry_run else "dry"
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it before running, or pass --dry-run."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    n_calls = 0
    n_ok = 0
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
                continue

            out_dir = args.out / tis_slug_val
            if not args.dry_run:
                out_dir.mkdir(parents=True, exist_ok=True)

            iso_with_gene = {**iso, "gene": {"name": gene_name}}
            results: dict[str, Any] = {}
            for category in CATEGORIES:
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
                    )
                n_calls += 1
                if args.dry_run:
                    print(
                        f"[{n_calls}] {gene_name} {tis_slug_val} category: "
                        f"{letter} ({category['name']}) input chars: {len(prompt.user)}"
                    )
                    continue
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
                        )
                        _add_usage(usage_by_slug, tis_slug_val, _drain_usage())
                        payload = parse_response(response_text)
                    _emit_schema_warnings(
                        payload, output_schema,
                        f"{tis_slug_val}/{category['name']}",
                        verbose=getattr(args, "verbose", False),
                    )
                    results[category["name"]] = payload
                    n_ok += 1
                except Exception as e:
                    if tool_config is not None:
                        _tool_usage, _tool_turns = _drain_all_usage()
                        _add_usage(tool_usage_by_slug, tis_slug_val, _tool_usage, calls=_tool_turns)
                    print(f"[{n_calls}] {letter} FAIL: {e}", file=sys.stderr)
                    results[category["name"]] = {"error": str(e)}

            if args.dry_run:
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    if args.dry_run:
        return 0
    _write_usage_report(args.out, "category", args.model, usage_by_slug)
    if tool_usage_by_slug:
        # Separate report: tool categories are multi-turn and never batched, so
        # folding them into the batch-priced total would misstate both.
        _write_usage_report(args.out, "category_tools", args.model, tool_usage_by_slug)
    print(f"{spec.name}: {n_ok}/{n_calls} successful")
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
    from swissisoform.site.evidence import CATEGORIES, slice_category

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
    for gene_name, gene_record in records.items():
        for iso in gene_record.get("isoforms", []) or []:
            tis_slug_val = _tis_slug(iso.get("tis_id"))
            out_path = args.out / spec.output_filename_template.format(tis_slug=tis_slug_val)
            if out_path.exists() and not args.force:
                continue
            iso_with_gene = {**iso, "gene": {"name": gene_name}}
            iso_results.setdefault(tis_slug_val, {})
            iso_out[tis_slug_val] = out_path
            for category in CATEGORIES:
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
        )
        if items
        else {}
    )

    usage_by_slug: dict[str, dict[str, int]] = {}
    n_ok = 0
    for cid, (tis_slug_val, cat_name) in meta.items():
        r = responses.get(cid) or {"text": None, "usage": _empty_usage(), "error": "missing result"}
        _add_usage(usage_by_slug, tis_slug_val, r["usage"])
        if r["error"] or not r["text"]:
            print(f"[{cid}] {cat_name} FAIL: {r['error']}", file=sys.stderr)
            iso_results[tis_slug_val][cat_name] = {"error": r["error"] or "empty"}
            continue
        try:
            payload = parse_response(r["text"])
            _emit_schema_warnings(
                payload, output_schema, f"{tis_slug_val}/{cat_name}",
                verbose=getattr(args, "verbose", False),
            )
            iso_results[tis_slug_val][cat_name] = payload
            n_ok += 1
        except Exception as e:
            print(f"[{cid}] {cat_name} parse FAIL: {e}", file=sys.stderr)
            iso_results[tis_slug_val][cat_name] = {"error": str(e)}

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
                )
                _tool_usage, _tool_turns = _drain_all_usage()
                _add_usage(tool_usage_by_slug, tis_slug_val, _tool_usage, calls=_tool_turns)
                _emit_schema_warnings(
                    payload,
                    output_schema,
                    f"{tis_slug_val}/{category['name']}",
                    verbose=getattr(args, "verbose", False),
                )
                iso_results[tis_slug_val][category["name"]] = payload
                n_tool_ok += 1
            except Exception as e:
                _tool_usage, _tool_turns = _drain_all_usage()
                _add_usage(tool_usage_by_slug, tis_slug_val, _tool_usage, calls=_tool_turns)
                print(f"[{tis_slug_val}] {letter} FAIL: {e}", file=sys.stderr)
                iso_results[tis_slug_val][category["name"]] = {"error": str(e)}

    for tis_slug_val, results in iso_results.items():
        out_path = iso_out[tis_slug_val]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    if items:
        _write_usage_report(args.out, "category", args.model, usage_by_slug, batch=True)
    if tool_usage_by_slug:
        _write_usage_report(args.out, "category_tools", args.model, tool_usage_by_slug)
    total = len(items) + n_tool_calls
    print(
        f"category: {n_ok}/{len(items)} successful (batch)"
        + (f" + {n_tool_ok}/{n_tool_calls} tool-loop (direct)" if n_tool_calls else "")
    )
    return 0 if (n_ok + n_tool_ok) == total else 1


def _build_synthesis_record(
    isoform: dict, gene_name: str, isoform_out_dir: Path, gene: dict | None = None
) -> dict:
    """Build the synthesis-pass input from the disk-cached categories.json output.

    Carries the gene's ESTABLISHED function (``gene`` — Affinage function / keywords
    / localization, the baseline the isoform diverges from), the digested per-category
    reads (``category_reads`` — the ``{verdict, reasoning}`` per CDLMPS category), and
    the raw underlying evidence (``criteria_evidence``, one ``slice_criterion`` payload
    per criterion, all 15 incl. S2/S3) so the model can weigh actual numbers.

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
            )
            n_calls += 1
            if args.dry_run:
                print(
                    f"[{n_calls}] {gene_name} {tis_slug} synthesis input chars: {len(prompt.user)}"
                )
                continue
            try:
                response_text = call_llm(
                    prompt,
                    model=args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    api_key=api_key,
                )
                _add_usage(usage_by_slug, tis_slug, _drain_usage())
                payload = parse_response(response_text)
                _emit_schema_warnings(
                    payload, output_schema, f"{tis_slug}/synthesis",
                    verbose=getattr(args, "verbose", False),
                )
                iso_dir.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
                n_ok += 1
                print(f"[{n_calls}] {tis_slug} OK")
            except Exception as e:
                print(f"[{n_calls}] {tis_slug} FAIL: {e}", file=sys.stderr)

    if args.dry_run:
        return 0
    _write_usage_report(args.out, "synthesis", args.model, usage_by_slug)
    print(f"synthesis: {n_ok}/{n_calls} successful")
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
    for gene_name, gene_record in records.items():
        for iso in gene_record.get("isoforms", []) or []:
            tis_slug = _tis_slug(iso.get("tis_id"))
            iso_dir = args.out / tis_slug
            out_path = iso_dir / "synthesis.json"
            if out_path.exists() and not args.force:
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
            continue
        _emit_schema_warnings(
            payload, output_schema, f"{tis_slug}/synthesis",
            verbose=getattr(args, "verbose", False),
        )
        out_path = out_by_slug[tis_slug]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        n_ok += 1

    _write_usage_report(args.out, "synthesis", args.model, usage_by_slug, batch=True)
    print(f"synthesis: {n_ok}/{len(items)} successful (batch)")
    return 0 if n_ok == len(items) else 1
