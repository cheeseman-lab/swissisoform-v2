"""CLI entrypoint: precompute PLM embeddings + LLR for a FASTA of proteins.

Invoked from ``scripts/slurm/run_plm_embed.sbatch`` on a GPU node. Reads a
FASTA, dedupes by sequence hash, runs ESM-2 inline, and writes per-
protein ``.npz`` cache files under ``data/cache/plm_esm2/``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from swissisoform.plm.embed import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MODEL_ID,
    precompute_plm_esm2,
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
    """Run ESM-2 precompute over a FASTA and populate the cache directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fasta", type=Path, help="FASTA of proteins to embed.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory to write <hash>.npz files to.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument(
        "--no-require-aa-logprobs",
        dest="require_aa_logprobs",
        action="store_false",
        help="Accept legacy caches lacking the per-position aa_logprobs distribution "
        "instead of recomputing them.",
    )
    parser.set_defaults(require_aa_logprobs=True)
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

    res = precompute_plm_esm2(
        seqs,
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        device=args.device,
        dtype=args.dtype,
        inline=True,
        require_aa_logprobs=args.require_aa_logprobs,
    )
    print(f"Wrote {len(res)} cache entries to {args.cache_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
