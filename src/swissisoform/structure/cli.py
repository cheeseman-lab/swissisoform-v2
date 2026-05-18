"""CLI entrypoint: precompute Boltz-2 / Chai-1 structures for a FASTA.

Invoked from ``scripts/run_fold.sbatch`` on a GPU node. Reads a FASTA,
dedupes by sequence hash, runs the chosen backend, writes per-protein
cache directories under ``data/cache/structure/<backend>/<hash>/``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from swissisoform.structure.fold import (
    DEFAULT_BACKEND,
    DEFAULT_CACHE_DIR,
    DEFAULT_MAX_SEQ_LEN,
    precompute_fold,
)


def _read_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    label: str | None = None
    parts: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if label is not None:
                    seqs[label] = "".join(parts)
                label = line[1:].split()[0]
                parts = []
            else:
                parts.append(line)
        if label is not None:
            seqs[label] = "".join(parts)
    return seqs


def main(argv: list[str] | None = None) -> int:
    """Run structure precompute over a FASTA and populate the cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fasta", type=Path, help="FASTA of proteins to fold.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Root cache directory (per-backend subdirs underneath).",
    )
    parser.add_argument(
        "--backend",
        choices=["boltz", "chai"],
        default=DEFAULT_BACKEND,
        help="Folding backend.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        help="Skip sequences longer than this (status=too_long).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if not args.fasta.exists():
        print(f"FASTA not found: {args.fasta}", file=sys.stderr)
        return 2

    seqs = _read_fasta(args.fasta)
    if not seqs:
        print(f"No sequences in {args.fasta}", file=sys.stderr)
        return 2

    res = precompute_fold(
        seqs,
        backend=args.backend,
        cache_dir=args.cache_dir,
        max_seq_len=args.max_seq_len,
        inline=True,
    )
    print(f"Wrote {len(res)} cache entries to {args.cache_dir}/{args.backend}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
