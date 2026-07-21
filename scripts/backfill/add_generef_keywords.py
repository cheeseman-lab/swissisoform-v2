"""Backfill the ``generef_keywords`` column into existing paired parquets.

The UniProt-keyword field was added to the generef path after these parquets
were written, so this splices it in without a full pipeline rerun. Keywords are
gene-level provisioned reference data (``data/reference/generef/generef.json``),
so a plain join on ``gene_name`` is exact and deterministic — the same value a
fresh run would compute (see the execution contract in CLAUDE.md).

For each target directory it rewrites ``all_paired.parquet`` plus any per-gene
``<GENE>_paired.parquet`` siblings. Re-fetch generef first so the JSON carries
keywords::

    python scripts/setup/fetch_generef.py --combined      # (or --genes ...)
    python scripts/backfill/add_generef_keywords.py website/data

Usage:
    python scripts/backfill/add_generef_keywords.py [DIR ...]   # default: website/data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GENEREF_JSON = ROOT / "data" / "reference" / "generef" / "generef.json"
COL = "generef_keywords"


def _load_keyword_map(generef_json: Path) -> dict[str, str | None]:
    """gene symbol -> ``"; "``-joined keyword string (or None)."""
    blob = json.loads(generef_json.read_text())
    return {gene: (rec or {}).get("keywords") for gene, rec in blob.items()}


def _backfill_parquet(path: Path, kw_map: dict[str, str | None]) -> tuple[int, int]:
    """Add/refresh ``generef_keywords`` on one parquet. Returns (rows, matched)."""
    df = pd.read_parquet(path)
    if "gene_name" not in df.columns:
        return (len(df), 0)
    df[COL] = df["gene_name"].map(kw_map)
    df.to_parquet(path, index=False)
    matched = int(df[COL].notna().sum())
    return (len(df), matched)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dirs",
        nargs="*",
        type=Path,
        default=[ROOT / "website" / "data"],
        help="Output dir(s) holding all_paired.parquet (default: website/data).",
    )
    parser.add_argument("--generef-json", type=Path, default=GENEREF_JSON)
    args = parser.parse_args(argv)

    if not args.generef_json.exists():
        print(f"generef.json not found: {args.generef_json}", file=sys.stderr)
        return 2
    kw_map = _load_keyword_map(args.generef_json)
    n_with_kw = sum(1 for v in kw_map.values() if v)
    print(f"generef.json: {len(kw_map)} genes, {n_with_kw} with keywords")

    for d in args.dirs:
        d = Path(d)
        targets = []
        combined = d / "all_paired.parquet"
        if combined.exists():
            targets.append(combined)
        # Per-gene siblings (full_catalog layout): <GENE>_paired.parquet.
        targets += sorted(p for p in d.glob("*_paired.parquet") if p != combined)
        if not targets:
            print(f"  {d}: no *_paired.parquet found — skipped", file=sys.stderr)
            continue
        for p in targets:
            rows, matched = _backfill_parquet(p, kw_map)
            print(f"  {p.relative_to(ROOT) if ROOT in p.parents else p}: "
                  f"{matched}/{rows} rows got keywords")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
