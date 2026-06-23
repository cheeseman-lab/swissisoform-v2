"""CLI entrypoint: precompute PLM embeddings + LLR for a FASTA of proteins.

Invoked from ``scripts/slurm/run_plm_embed.sbatch`` on a GPU node. Reads a
FASTA, dedupes by sequence hash, runs ESM-C inline, and writes per-
protein ``.npz`` cache files under ``data/cache/plm_esmc/``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from swissisoform.plm.embed import (
    DEFAULT_MODEL_SIZE,
    ESMC_MODEL_IDS,
    plm_cache_dir,
    precompute_plm,
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
    """Run ESM-C precompute over a FASTA and populate the cache directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fasta", type=Path, help="FASTA of proteins to embed.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory to write <hash>.npz files to. Default: the per-size dir "
        "data/cache/plm_esmc/<SIZE> derived from --model-size (so 600m never "
        "writes into the 6B cache).",
    )
    parser.add_argument(
        "--model-size",
        choices=sorted(ESMC_MODEL_IDS),
        default=DEFAULT_MODEL_SIZE,
        help="ESM-C model size (default %(default)s).",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Explicit HuggingFace repo id; overrides --model-size.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", default=None,
        help="Inference precision; default is size-aware (bf16 for 6B, fp16 else).",
    )
    parser.add_argument(
        "--no-require-aa-logprobs",
        dest="require_aa_logprobs",
        action="store_false",
        help="Accept legacy caches lacking the per-position aa_logprobs distribution "
        "instead of recomputing them.",
    )
    parser.set_defaults(require_aa_logprobs=True)
    parser.add_argument(
        "--no-require-embedding-sae",
        dest="require_embedding_sae",
        action="store_false",
        help="Accept caches lacking the SAE-target layer embedding (embedding_sae) "
        "instead of recomputing them.",
    )
    parser.set_defaults(require_embedding_sae=True)
    parser.add_argument(
        "--no-llr",
        dest="compute_llr",
        action="store_false",
        help="Skip the masked-marginal LLR sweep (Pass 2); compute only the "
        "single embedding forward (embeddings + SAE layer). Much faster — one "
        "forward per protein instead of 1+L.",
    )
    parser.set_defaults(compute_llr=True)
    args = parser.parse_args(argv)

    # LLR off ⇒ don't also demand aa_logprobs in cache validation (contradictory).
    if not args.compute_llr:
        args.require_aa_logprobs = False

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

    res = precompute_plm(
        seqs,
        model_size=args.model_size,
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        device=args.device,
        dtype=args.dtype,
        inline=True,
        require_aa_logprobs=args.require_aa_logprobs,
        require_embedding_sae=args.require_embedding_sae,
        compute_llr=args.compute_llr,
    )
    resolved_dir = args.cache_dir or plm_cache_dir(args.model_size)
    print(f"Wrote {len(res)} cache entries to {resolved_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
