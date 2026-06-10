"""Thin CLI for the transcript-skeleton parquet builder.

Logic lives in ``swissisoform.site.skeletons``. The transcript subset is driven
by the ``transcript_id`` column of an ``all_paired.parquet``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq

from swissisoform.site.skeletons import write_skeletons_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gtf", required=True, type=Path, help="GTF used by the upstream pipeline.")
    parser.add_argument(
        "--parquet", required=True, type=Path,
        help="all_paired.parquet (transcript_id column drives the subset).",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output parquet path.")
    args = parser.parse_args()

    if not args.gtf.exists():
        raise SystemExit(f"GTF not found: {args.gtf}")
    if not args.parquet.exists():
        raise SystemExit(f"Parquet not found: {args.parquet}")

    schema_names = pq.read_schema(args.parquet).names
    if "transcript_id" not in schema_names:
        raise SystemExit(
            f"Parquet at {args.parquet} has no 'transcript_id' column "
            f"(found: {schema_names[:5]}{'...' if len(schema_names) > 5 else ''})"
        )

    import pandas as pd

    df = pd.read_parquet(args.parquet, columns=["transcript_id"])
    transcript_ids = set(df["transcript_id"].dropna().astype(str).tolist())
    n = write_skeletons_parquet(args.gtf, transcript_ids, args.out)
    print(f"Wrote {n} transcript skeletons → {args.out}")


if __name__ == "__main__":
    main()
