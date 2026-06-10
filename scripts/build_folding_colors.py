"""Thin CLI for the folding-colour precompute.

Logic lives in ``swissisoform.export.folding_colors``. Precomputes the
per-residue palette the website's 3Dmol.js viewer applies (whole protein by
pLDDT, differential region on a diverging ramp).

Usage:
    python scripts/build_folding_colors.py \
        --parquet data/output/cheeseman_13gene/all_paired.parquet \
        --structures data/output/cheeseman_13gene/structures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from swissisoform.export.folding_colors import build_folding_colors


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--structures", type=Path, required=True)
    p.add_argument(
        "--colors", type=Path, default=None, help="Output dir (default: <structures>/colors)"
    )
    args = p.parse_args(argv)

    written = build_folding_colors(args.parquet, args.structures, args.colors)
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
