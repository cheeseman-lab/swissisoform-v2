#!/usr/bin/env python
"""Split the full-catalog protein FASTA into GPU chunks + genes into annotate shards.

Reads the deduped protein FASTA (from ``run.py --all --emit-fasta``) and the
combined catalog's unique gene ``Symbol`` column, writing:

    <outdir>/chunks/chunk_<i>.fa   — ~chunk_size proteins each (GPU embed/fold arrays)
    <outdir>/shards/shard_<i>.txt  — ~shard_size gene symbols each (annotate array)
    <outdir>/split_manifest.txt    — n_proteins / n_chunks / n_genes / n_shards

Chunk/shard indices are plain (0,1,2,…) to match ``$SLURM_ARRAY_TASK_ID``.
Idempotent: clears any existing chunks/ + shards/ first so a re-split is clean.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def read_fasta_records(p: Path) -> list[tuple[str, str]]:
    """Return [(header_line, sequence), …] preserving the ``>hash`` headers."""
    recs: list[tuple[str, str]] = []
    name: str | None = None
    seq: list[str] = []
    for line in p.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                recs.append((name, "".join(seq)))
            name, seq = line, []
        else:
            seq.append(line)
    if name is not None:
        recs.append((name, "".join(seq)))
    return recs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proteins", type=Path, required=True, help="Deduped proteins.fa")
    ap.add_argument("--combined", type=Path, required=True, help="all_samples_combined.parquet")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--chunk-size", type=int, default=1500, help="Proteins per GPU chunk.")
    ap.add_argument("--shard-size", type=int, default=100, help="Genes per annotate shard.")
    a = ap.parse_args()

    recs = read_fasta_records(a.proteins)
    chunks_dir = a.outdir / "chunks"
    if chunks_dir.exists():
        shutil.rmtree(chunks_dir)
    chunks_dir.mkdir(parents=True)
    n_chunks = 0
    for i in range(0, len(recs), a.chunk_size):
        block = recs[i : i + a.chunk_size]
        (chunks_dir / f"chunk_{n_chunks}.fa").write_text(
            "".join(f"{name}\n{seq}\n" for name, seq in block)
        )
        n_chunks += 1

    genes = sorted(g for g in pd.read_parquet(a.combined, columns=["Symbol"])["Symbol"].dropna().unique())
    shards_dir = a.outdir / "shards"
    if shards_dir.exists():
        shutil.rmtree(shards_dir)
    shards_dir.mkdir(parents=True)
    n_shards = 0
    for i in range(0, len(genes), a.shard_size):
        block = genes[i : i + a.shard_size]
        (shards_dir / f"shard_{n_shards}.txt").write_text("\n".join(block) + "\n")
        n_shards += 1

    (a.outdir / "split_manifest.txt").write_text(
        f"n_proteins\t{len(recs)}\n"
        f"n_chunks\t{n_chunks}\n"
        f"chunk_size\t{a.chunk_size}\n"
        f"n_genes\t{len(genes)}\n"
        f"n_shards\t{n_shards}\n"
        f"shard_size\t{a.shard_size}\n"
    )
    print(f"proteins={len(recs)} chunks={n_chunks} genes={len(genes)} shards={n_shards}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
