r"""Run LLM interpretation on per-gene evidence records.

Reads `data/output/cheeseman_12gene/llm_evidence/{gene}.json` (produced by
`build_evidence_records.py`), composes a prompt from `prompts/system.txt` +
the evidence JSON + `prompts/output_schema.json`, calls the Anthropic API,
validates + writes the structured output to `llm/{gene}.json`.

Single-threaded; idempotent (skips genes whose output already exists unless
``--force``). ``--dry-run`` prints the assembled prompt and exits without any
network calls.

Usage:
    python scripts/site/run_llm_interpretation.py \
      --records data/output/cheeseman_12gene/llm_evidence/ \
      --out data/output/cheeseman_12gene/llm/

    python scripts/site/run_llm_interpretation.py --dry-run --gene TRNT1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
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
    output_filename_template: str  # e.g. "{gene}.json", "{tis_slug}/existence.json"
    iterates_modalities: bool = False  # True for existence + functional passes
    requires_prereq: tuple[str, ...] = ()  # other pass names that must have produced output


PASS_REGISTRY: dict[str, PassSpec] = {
    "default": PassSpec(
        name="default",
        system_prompt_filename="system.txt",
        output_schema_filename="output_schema.json",
        output_filename_template="{gene}.json",
    ),
    "existence": PassSpec(
        name="existence",
        system_prompt_filename="existence-pass.txt",
        output_schema_filename="output_schemas/modality_read.json",
        output_filename_template="{tis_slug}/existence.json",
        iterates_modalities=True,
    ),
    "functional": PassSpec(
        name="functional",
        system_prompt_filename="functional-pass.txt",
        output_schema_filename="output_schemas/modality_read.json",
        output_filename_template="{tis_slug}/functional.json",
        iterates_modalities=True,
    ),
    "synthesis": PassSpec(
        name="synthesis",
        system_prompt_filename="synthesis-pass.txt",
        output_schema_filename="output_schemas/synthesis.json",
        output_filename_template="{tis_slug}/synthesis.json",
        requires_prereq=("existence", "functional"),
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LLM interpretation on per-gene SwissIsoform evidence records.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=ROOT / "data/output/cheeseman_12gene/llm_evidence",
        help="Directory of per-gene evidence record JSON files (default: %(default)s).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/output/cheeseman_12gene/llm",
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


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns process exit code."""
    args = _build_parser().parse_args(argv)

    system_prompt = load_system_prompt()
    output_schema = load_output_schema()
    records = _load_records_with_synthetic_fallback(args.records, args.gene, args.dry_run)

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


if __name__ == "__main__":
    raise SystemExit(main())
