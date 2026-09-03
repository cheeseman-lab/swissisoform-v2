"""Thin CLI for the tag-candidate sweep.

Logic lives in ``swissisoform.tags.candidates``. Proposes every mechanically
derivable tag, cuts each against the frozen distributions, measures what it would
actually fire on, and writes a reviewable table with an empty ``decision`` column.

Outputs (default ``figures/tag_vocab/``):
    tag_candidates.csv    one row per surviving candidate — fill in `decision`
    percentile_chips.csv  candidates with no in-band cutoff: range-filter these
    tag_review.md         the funnel, per-category counts, blockers, deferrals

Usage:
    python scripts/tags/propose_candidates.py --run full_catalog --version v2
    python scripts/tags/propose_candidates.py --run cheeseman50 --version v2 \
        --out /tmp/$USER/tags_smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from swissisoform import distributions as dist_mod
from swissisoform.setup.distributions import DEFAULT_CATALOG, load_catalog, load_run
from swissisoform.tags import candidates as C

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "figures" / "tag_vocab"
DEFAULT_CORESET = (
    ROOT / "figures" / "clustering_dims" / "principled_sampling" / "coreset_50.csv"
)


def main(argv: list[str] | None = None) -> int:
    """Run the sweep and write the review artifacts."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="Run name under data/output/")
    g.add_argument("--parquet", type=Path, help="Explicit all_paired.parquet path")
    p.add_argument("--version", default="v2", help="Distributions version (default: v2)")
    p.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    p.add_argument("--coreset", type=Path, default=DEFAULT_CORESET)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--band",
        default="10,60",
        help="Acceptable fire-rate window as LO,HI percent (default: 10,60)",
    )
    p.add_argument("--jaccard-max", type=float, default=C.JACCARD_MAX)
    args = p.parse_args(argv)

    parquet = args.parquet or (ROOT / "data" / "output" / args.run / "all_paired.parquet")
    if not parquet.exists():
        p.error(f"parquet not found: {parquet}")
    lo, hi = (float(x) for x in str(args.band).split(","))

    dist = dist_mod.load(args.version)
    catalog = load_catalog(args.catalog)
    df = load_run([parquet])
    coreset = pd.read_csv(args.coreset) if args.coreset and args.coreset.exists() else None

    table, chips, funnel = C.build_table(
        df, catalog, dist, coreset=coreset, band=(lo, hi), jaccard_max=args.jaccard_max
    )
    C.write_outputs(table, chips, funnel, args.out, version=args.version)

    print(
        f"{funnel.proposed} proposals → {funnel.kept} candidates, "
        f"{len(chips)} chips  ({args.out})"
    )
    for reason, n in sorted(funnel.dropped.items(), key=lambda kv: -kv[1]):
        print(f"  dropped {n:>4}  {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
