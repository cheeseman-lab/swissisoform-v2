r"""Thin CLI for the per-gene LLM interpretation runner.

Logic lives in ``swissisoform.site.llm``. Reads
`data/output/cheeseman_13gene/llm_evidence/{gene}.json` (produced by
`build_evidence_records.py`), composes a prompt from `prompts/system.txt` +
the evidence JSON + `prompts/output_schema.json`, calls the Anthropic API,
validates + writes the structured output to `llm/{gene}.json`.

Single-threaded; idempotent (skips genes whose output already exists unless
``--force``). ``--dry-run`` prints prompt-size diagnostics and exits without any
network calls; ``--save-prompts`` writes every assembled prompt in full to
``data/llm_inputs/{run}/`` as one ``.txt`` per call, and composes with ``--dry-run``.

Usage:
    python scripts/site/run_llm_interpretation.py \
      --records data/output/cheeseman_13gene/llm_evidence/ \
      --out data/output/cheeseman_13gene/llm/

    python scripts/site/run_llm_interpretation.py --dry-run --gene TRNT1

    # Dump the whole prompt corpus without any API calls:
    python scripts/site/run_llm_interpretation.py --pass category \
      --dry-run --save-prompts \
      --records data/output/cheeseman_test/llm_evidence \
      --out data/output/cheeseman_test/llm
"""

from __future__ import annotations

from pathlib import Path

from swissisoform.site import llm

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# Point the default-pass prompt/schema constants at this script's prompts dir
# so the V1 single-pass reads the work-in-progress prompts that live alongside
# this CLI (not the package-relative defaults).
llm.PROMPTS_DIR = PROMPTS_DIR
llm.SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system.txt"
llm.OUTPUT_SCHEMA_PATH = PROMPTS_DIR / "output_schema.json"


def main(argv: list[str] | None = None) -> int:
    return llm.main(argv, prompts_dir=PROMPTS_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
