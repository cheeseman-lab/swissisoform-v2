"""Thin CLI for the per-gene LLM evidence-record builder.

Logic lives in ``swissisoform.site.evidence``. Consumes a paired
``all_paired.parquet`` and emits one ``{gene}.json`` per gene matching the
"Per-gene evidence record" schema in ``docs/site_and_llm_plan.md``.

This is pure DataFrame → dict conversion: no LLM calls, no network.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from swissisoform.site.evidence import (
    CRITERIA,
    CRITERIA_METRIC_LABELS,
    format_metric,
    slice_criterion,
    summarise,
    write_evidence_records,
    write_variants_long,
)

__all__ = [
    "CRITERIA",
    "CRITERIA_METRIC_LABELS",
    "format_metric",
    "slice_criterion",
    "summarise",
    "write_evidence_records",
    "write_variants_long",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--parquet",
        type=Path,
        required=True,
        help="Path to all_paired.parquet",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for per-gene JSON files",
    )
    p.add_argument(
        "--variants-long-out",
        type=Path,
        default=None,
        help="If set, also write a flat variants_long parquet to this path.",
    )
    args = p.parse_args()

    counts = write_evidence_records(args.parquet, args.out)
    print(f"Wrote {counts['genes']} gene files ({counts['isoforms']} isoforms) to {args.out}")
    summarise(args.out)

    if args.variants_long_out is not None:
        n = write_variants_long(args.parquet, args.variants_long_out)
        print(f"Wrote {n} variant rows → {args.variants_long_out}")


if __name__ == "__main__":
    main()
