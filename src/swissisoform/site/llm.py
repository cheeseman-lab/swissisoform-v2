r"""Run LLM interpretation on per-gene evidence records.

Reads ``{records_dir}/{gene}.json`` (produced by ``evidence.py``), composes a
prompt from ``system.txt`` + the evidence JSON + ``output_schema.json``, calls
the Anthropic API, validates + writes the structured output to ``{out}/{gene}.json``.

Single-threaded; idempotent (skips genes whose output already exists unless
``force``). ``dry_run`` prints the assembled prompt and exits without any
network calls.

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
from typing import Any

# Default prompts location — the thin CLI overrides this with its own
# scripts/site/prompts/ path. Resolves relative to the repo root so the
# module remains importable without a CLI.
ROOT = Path(__file__).resolve().parents[3]
PROMPTS_DIR = ROOT / "scripts" / "site" / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system.txt"
OUTPUT_SCHEMA_PATH = PROMPTS_DIR / "output_schema.json"

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 4000

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


# ── LLM call dispatch ─────────────────────────────────────────────────────


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
    """
    mozz = _try_import_mozzarellm()
    if mozz is not None:
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
        return response_text

    anthropic = _try_import_anthropic()
    if anthropic is None:
        raise RuntimeError(
            "Neither mozzarellm nor anthropic SDK is importable. "
            "Install one of them in the swissisoform-v2 env (e.g. `pip install anthropic`)."
        )

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=prompt.system,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt.user}]}],
    )
    if not response.content:
        raise RuntimeError("anthropic SDK returned empty content list")
    return response.content[0].text


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
        "--pass",
        dest="pass_name",
        default="default",
        choices=sorted(PASS_REGISTRY),
        help="Which LLM pass to run (default = V1 single-pass).",
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

    # V1 default-pass reads SYSTEM_PROMPT_PATH / OUTPUT_SCHEMA_PATH directly so tests
    # that monkeypatch those constants keep working bit-identically. Other passes
    # resolve their files relative to the prompts dir.
    if spec.name == "default":
        system_prompt = load_system_prompt()
        output_schema = load_output_schema()
    else:
        prompts_root = prompts_dir or SYSTEM_PROMPT_PATH.parent
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

    if spec.iterates_categories:
        return _run_category_pass(records, spec, args, system_prompt, output_schema)

    if spec.name == "synthesis":
        return _run_synthesis_pass(records, spec, args, system_prompt, output_schema)

    # spec.name == "default" — V1 single-pass behavior
    return _run_default_pass(records, spec, args, system_prompt, output_schema)


def _run_default_pass(records, spec, args, system_prompt, output_schema) -> int:
    """V1 single-pass per-gene loop. Preserved bit-identically from V1 main."""
    if args.dry_run:
        for gene_name, record in records.items():
            prompt = build_prompt(record, system_prompt, output_schema)
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


def _check_prereqs(records, out_dir: Path, prereqs: tuple[str, ...]) -> list[str]:
    """Return tis_slugs missing any prereq output file."""
    missing: list[str] = []
    for _, gene_record in records.items():
        for iso in gene_record.get("isoforms", []) or []:
            tis_slug = _tis_slug(iso.get("tis_id"))
            for prereq in prereqs:
                pp = out_dir / tis_slug / f"{prereq}.json"
                if not pp.exists():
                    missing.append(tis_slug)
                    break
    return missing


def _run_category_pass(records, spec, args, system_prompt, output_schema) -> int:
    """Per-(isoform, category) dispatch — one call per CDLMPS category.

    Each call bundles all of the category's members (scored criteria + the
    descriptive biophysics/sae cards, incl. F7) into one slice and asks the model
    for a single ``{verdict, reasoning}``. Writes ``{tis_slug}/categories.json`` as
    a dict keyed by category name (the shape ``category_verdicts_for_isoform``
    consumes).
    """
    from swissisoform.site.evidence import CATEGORIES, slice_category

    api_key = os.environ.get("ANTHROPIC_API_KEY") if not args.dry_run else "dry"
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it before running, or pass --dry-run."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    n_calls = 0
    n_ok = 0

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
                category_record = slice_category(iso_with_gene, category)
                prompt = build_prompt(category_record, system_prompt, output_schema)
                n_calls += 1
                if args.dry_run:
                    print(
                        f"[{n_calls}] {gene_name} {tis_slug_val} category: "
                        f"{category['letter']} ({category['name']}) input chars: {len(prompt.user)}"
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
                    payload = parse_response(response_text)
                    results[category["name"]] = payload
                    n_ok += 1
                except Exception as e:
                    print(f"[{n_calls}] {category['letter']} FAIL: {e}", file=sys.stderr)
                    results[category["name"]] = {"error": str(e)}

            if args.dry_run:
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    if args.dry_run:
        return 0
    print(f"{spec.name}: {n_ok}/{n_calls} successful")
    return 0 if n_ok == n_calls else 1


def _build_synthesis_record(isoform: dict, gene_name: str, isoform_out_dir: Path) -> dict:
    """Build the synthesis-pass input from the disk-cached categories.json output.

    Carries both the digested per-category reads (``category_reads`` — the
    ``{verdict, reasoning}`` per CDLMPS category) and the raw underlying evidence
    (``criteria_evidence``, one ``slice_criterion`` payload per criterion, all 13
    incl. F7) so the model can weigh actual numbers, not just the category verdicts.
    """
    from swissisoform.site.evidence import CRITERIA, slice_criterion

    category_reads: dict[str, Any] = {}
    pp = isoform_out_dir / "categories.json"
    if pp.exists():
        category_reads = json.loads(pp.read_text(encoding="utf-8"))

    iso_with_gene = {**isoform, "gene": {"name": gene_name}}
    criteria_evidence: dict[str, Any] = {
        cid: slice_criterion(iso_with_gene, cid) for cid in CRITERIA
    }

    return {
        "isoform": {
            "tis_id": isoform.get("tis_id"),
            "gene_name": gene_name,
            "orf_type": isoform.get("orf_type"),
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
    api_key = os.environ.get("ANTHROPIC_API_KEY") if not args.dry_run else "dry"
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Export it or pass --dry-run.")

    args.out.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_calls = 0
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
            synthesis_record = _build_synthesis_record(iso, gene_name, iso_dir)
            prompt = build_prompt(synthesis_record, system_prompt, output_schema)
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
                payload = parse_response(response_text)
                iso_dir.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
                n_ok += 1
                print(f"[{n_calls}] {tis_slug} OK")
            except Exception as e:
                print(f"[{n_calls}] {tis_slug} FAIL: {e}", file=sys.stderr)

    if args.dry_run:
        return 0
    print(f"synthesis: {n_ok}/{n_calls} successful")
    return 0 if n_ok == n_calls else 1
