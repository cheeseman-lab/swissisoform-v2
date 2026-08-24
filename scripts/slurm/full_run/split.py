#!/usr/bin/env python
"""Split a campaign's protein FASTA into GPU chunks + genes into annotate shards.

Reads the deduped protein FASTA (from ``run.py --all --emit-fasta``) and the
combined catalog's unique gene ``Symbol`` column, writing:

    <outdir>/chunks/chunk_<i>.fa   — ~chunk_size proteins each (GPU embed/fold arrays)
    <outdir>/shards/shard_<i>.txt  — ~shard_size gene symbols each (annotate array)
    <outdir>/split_manifest.txt    — campaign / n_proteins / n_chunks / n_genes /
                                     n_shards / embed_split

Chunk/shard indices are plain (0,1,2,…) to match ``$SLURM_ARRAY_TASK_ID``.

Every path is campaign-scoped: ``--outdir`` is ``data/output/<campaign>`` and the
annotate array writes ``data/output/<campaign>_shard_<k>``. Re-splitting a
campaign in place is the one way old and new results can still blend, so this
script *fences* it: existing shard-output dirs are checked against the new split
(via the ``shard_meta.json`` each annotate task leaves behind) and any that no
longer match abort the split. Start a new campaign, or pass
``--allow-stale-outputs`` if you have already cleaned up by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

OUTPUT_ROOT = Path("data/output")


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


def embed_split(n_chunks: int) -> int:
    """Chunks to route to the A6000 embed queue; the rest go to the A100 queue.

    The two queues take disjoint ranges — A6000 ``[0, k)``, A100 ``[k, n_chunks)``
    — split ~3/7 by effective throughput (A100 is ~2x per GPU on the 6B model).
    ``k`` is clamped to ``[1, n_chunks]``, so a single-chunk run puts its one
    chunk on the A6000 and leaves the A100 range empty rather than producing the
    inverted ``--array=0--1`` that sbatch rejects.
    """
    if n_chunks < 1:
        raise ValueError(f"n_chunks must be >= 1, got {n_chunks}")
    return max(1, min(n_chunks, n_chunks * 3 // 7))


def shard_fingerprint(path: Path) -> str:
    """SHA1 of a shard's gene-list file — the identity merge.py checks against."""
    return hashlib.sha1(path.read_bytes()).hexdigest()


def check_existing_outputs(campaign: str, shards_dir: Path, n_shards: int) -> list[str]:
    """Return complaints about shard-output dirs that the new split invalidates."""
    problems: list[str] = []
    for d in sorted(OUTPUT_ROOT.glob(f"{campaign}_shard_*")):
        suffix = d.name.rsplit("_shard_", 1)[1]
        if not suffix.isdigit():
            continue
        k = int(suffix)
        if k >= n_shards:
            problems.append(f"{d.name}: shard {k} is outside the new range 0..{n_shards - 1}")
            continue
        if not (d / "all_paired.parquet").exists():
            continue  # no result to go stale
        meta_path = d / "shard_meta.json"
        if not meta_path.exists():
            problems.append(
                f"{d.name}: has results but no shard_meta.json — membership unverifiable"
            )
            continue
        meta = json.loads(meta_path.read_text())
        expected = shard_fingerprint(shards_dir / f"shard_{k}.txt")
        if meta.get("shard_list_sha1") != expected:
            problems.append(f"{d.name}: gene membership changed since it ran")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", required=True, help="Campaign name, e.g. full_catalog_20260821")
    ap.add_argument("--proteins", type=Path, required=True, help="Deduped proteins.fa")
    ap.add_argument("--combined", type=Path, required=True, help="all_samples_combined.parquet")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--chunk-size", type=int, default=1500, help="Proteins per GPU chunk.")
    ap.add_argument("--shard-size", type=int, default=100, help="Genes per annotate shard.")
    ap.add_argument(
        "--allow-stale-outputs", action="store_true",
        help="Split even if existing shard outputs no longer match the new split.",
    )
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

    symbols = pd.read_parquet(a.combined, columns=["Symbol"])["Symbol"]
    genes = sorted(g for g in symbols.dropna().unique())
    shards_dir = a.outdir / "shards"
    if shards_dir.exists():
        shutil.rmtree(shards_dir)
    shards_dir.mkdir(parents=True)
    n_shards = 0
    for i in range(0, len(genes), a.shard_size):
        block = genes[i : i + a.shard_size]
        (shards_dir / f"shard_{n_shards}.txt").write_text("\n".join(block) + "\n")
        n_shards += 1

    problems = check_existing_outputs(a.campaign, shards_dir, n_shards)
    if problems:
        print(f"ERROR: {len(problems)} existing shard output(s) do not match this split:")
        for p in problems[:20]:
            print(f"  {p}")
        if len(problems) > 20:
            print(f"  … and {len(problems) - 20} more")
        if not a.allow_stale_outputs:
            print(
                "\nMerging these with fresh shards would blend campaigns. Either run under a "
                f"new SWISSISO_CAMPAIGN, remove data/output/{a.campaign}_shard_*, or re-run "
                "with --allow-stale-outputs if you have already cleaned up.",
            )
            return 1
        print("\n--allow-stale-outputs: continuing anyway.")

    (a.outdir / "split_manifest.txt").write_text(
        f"campaign\t{a.campaign}\n"
        f"n_proteins\t{len(recs)}\n"
        f"n_chunks\t{n_chunks}\n"
        f"chunk_size\t{a.chunk_size}\n"
        f"n_genes\t{len(genes)}\n"
        f"n_shards\t{n_shards}\n"
        f"shard_size\t{a.shard_size}\n"
        f"embed_split\t{embed_split(n_chunks)}\n"
    )
    print(
        f"campaign={a.campaign} proteins={len(recs)} chunks={n_chunks} "
        f"genes={len(genes)} shards={n_shards}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
