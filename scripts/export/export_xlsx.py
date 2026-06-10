"""Thin CLI for the collaborator-facing Excel export.

Logic lives in ``swissisoform.export.xlsx``. Writes a .xlsx with three sheets
(isoforms, data_dictionary, key_columns) from a paired-TIS parquet.

Usage:
    python scripts/export/export_xlsx.py --run cheeseman_12gene
    python scripts/export/export_xlsx.py --parquet path/to/all_paired.parquet --out out.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from swissisoform.export.xlsx import write_workbook

ROOT = Path(__file__).resolve().parent.parent.parent


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="Run name under data/output/ (uses all_paired.parquet)")
    g.add_argument("--parquet", type=Path, help="Path to a paired parquet")
    p.add_argument(
        "--out", type=Path, default=None, help="Output .xlsx (default: alongside the parquet)"
    )
    args = p.parse_args(argv)

    if args.run:
        parquet = ROOT / "data" / "output" / args.run / "all_paired.parquet"
    else:
        parquet = args.parquet
    if not parquet.exists():
        p.error(f"parquet not found: {parquet}")

    out = args.out or parquet.with_suffix(".xlsx").with_name(parquet.parent.name + ".xlsx")
    df = pd.read_parquet(parquet)
    write_workbook(df, out)
    print(f"wrote {out}  ({df.shape[0]} rows, {df.shape[1]} cols → 3 sheets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
