#!/usr/bin/env python
"""Merge the per-shard annotate outputs into data/output/full_catalog/.

Concatenates every ``full_catalog_shard_<k>/all_paired.parquet`` into one
``full_catalog/all_paired.parquet`` and gathers the per-gene ``<gene>_paired.parquet``
files alongside. Gene-sharding is safe for scoring (every criterion is
per-TIS/per-gene; ``min_cell_lines`` reads ``present_*`` columns already present
in each row), so no cross-shard recompute is needed.

Idempotent: rebuilds the merged parquet from whatever shard outputs currently
exist. Reports how many shards were missing so a partial merge is explicit.
"""

from __future__ import annotations

import glob
import shutil
from pathlib import Path

import pandas as pd

OUT = Path("data/output/full_catalog")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    shard_dirs = sorted(
        glob.glob("data/output/full_catalog_shard_*"),
        key=lambda d: int(d.rsplit("_", 1)[1]),
    )
    frames: list[pd.DataFrame] = []
    n_missing = 0
    n_gene_files = 0
    for d in shard_dirs:
        paired = Path(d) / "all_paired.parquet"
        if not paired.exists():
            n_missing += 1
            continue
        frames.append(pd.read_parquet(paired))
        for g in Path(d).glob("*_paired.parquet"):
            if g.name == "all_paired.parquet":
                continue
            shutil.copyfile(g, OUT / g.name)
            n_gene_files += 1

    if not frames:
        print(f"no shard outputs found among {len(shard_dirs)} shard dirs — nothing merged")
        return 1

    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(OUT / "all_paired.parquet")
    print(
        f"merged {len(frames)}/{len(shard_dirs)} shards → {len(df)} rows, "
        f"{df['gene_name'].nunique() if 'gene_name' in df else '?'} genes, "
        f"{n_gene_files} per-gene parquets copied; {n_missing} shard(s) missing"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
