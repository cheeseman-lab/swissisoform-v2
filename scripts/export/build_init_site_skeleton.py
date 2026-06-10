r"""Build the per-init-site genome-LM training skeleton.

One row per genomic initiation site (``chrom:gstart:strand:codon``) — the
natural unit for a DNA foundation model (Evo2 / AlphaGenome). Carries
per-condition usage labels (``max_norm_{sample}`` + ``present_{sample}``),
reproducibility, full ORF/protein/gene provenance, and transcription context,
but **no DNA sequence**: window extraction is the consuming repo's concern.

This is the contract file a sibling embedding repo reads. See
``docs/training-skeleton-plan.md``.

Usage:
    python scripts/export/build_init_site_skeleton.py \\
        [--combined data/output/filtered/all_samples_combined.parquet] \\
        [--out data/output/init_site_skeleton.parquet]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from swissisoform.combine import dedupe_unique_init_sites

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = ROOT / "data" / "output" / "filtered" / "all_samples_combined.parquet"
DEFAULT_OUT = ROOT / "data" / "output" / "init_site_skeleton.parquet"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--combined", type=Path, default=DEFAULT_IN, help="Combined catalog parquet.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output skeleton parquet.")
    args = ap.parse_args()

    if not args.combined.exists():
        raise SystemExit(f"Combined catalog not found: {args.combined}")

    df = pd.read_parquet(args.combined)
    sites = dedupe_unique_init_sites(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sites.to_parquet(args.out, index=False)
    print(f"Wrote {len(sites)} init sites × {sites.shape[1]} cols → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
