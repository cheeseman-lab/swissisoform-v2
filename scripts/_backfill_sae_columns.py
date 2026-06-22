"""Merge the SAE per-pair scalar summary into all_paired.parquet (website-ready).

Reads the already-built ``sae_comparison.parquet`` (produced by
``scripts/export/export_sae_comparison.py``) and splices its scalar summary
columns into ``all_paired.parquet`` as ``isoform_sae_*`` columns — joined on
``tis_id`` — then re-splits the per-gene ``{gene}_paired.parquet`` files. Mirrors
``scripts/_backfill_cols.py``; no genome/GTF assembly and no model inference.

The website reads ``all_paired.parquet`` directly, so after this the per-isoform
SAE feature counts (canonical/isoform totals + uniques) are available with no
website code change. The exhaustive per-feature detail stays in
``sae_comparison.parquet`` / ``sae_top_terms_by_category.parquet``.

Idempotent: existing ``isoform_sae_*`` columns are dropped before re-merging.
Backs up ``all_paired.parquet`` → ``all_paired_pre_sae.parquet`` first.

Usage: python scripts/_backfill_sae_columns.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

OUT = Path("data/output/cheeseman_13gene")
ALL_PAIRED = OUT / "all_paired.parquet"
SAE = OUT / "sae_comparison.parquet"
BACKUP = OUT / "all_paired_pre_sae.parquet"
KEY = "tis_id"

# Scalar summary columns from sae_comparison.parquet to surface in all_paired
# (the 4 user-requested counts first, then useful extras). List columns
# (features_*_only, shared_feature_deltas) are deliberately NOT merged — that
# detail lives in sae_comparison.parquet / sae_top_terms_by_category.parquet.
SCALARS = [
    "status",
    "n_isoform_total",
    "n_canonical_total",
    "n_isoform_only",
    "n_canonical_only",
    "n_shared",
    "mean_abs_delta_shared",
    "top_gained_feature_index",
    "top_gained_feature_label",
    "top_gained_delta_max",
    "top_lost_feature_index",
    "top_lost_feature_label",
    "top_lost_delta_max",
]


def main() -> int:
    """Splice isoform_sae_* scalar columns into all_paired.parquet + per-gene."""
    all_paired = pd.read_parquet(ALL_PAIRED)
    sae = pd.read_parquet(SAE)

    cols = [c for c in SCALARS if c in sae.columns]
    sae_sel = sae[[KEY] + cols].drop_duplicates(KEY).rename(
        columns={c: f"isoform_sae_{c}" for c in cols}
    )
    new_cols = [f"isoform_sae_{c}" for c in cols]
    print(f"merging {len(new_cols)} isoform_sae_* columns for {len(sae_sel)} pairs")

    # Back up once (don't clobber an existing pre-SAE backup).
    if not BACKUP.exists():
        shutil.copy2(ALL_PAIRED, BACKUP)
        print(f"backed up → {BACKUP}")

    drop = [c for c in all_paired.columns if c.startswith("isoform_sae_")]
    merged = all_paired.drop(columns=drop).merge(sae_sel, on=KEY, how="left")

    n_filled = int(merged["isoform_sae_n_isoform_total"].notna().sum())
    print(f"rows: {len(merged)} (isoform_sae populated on {n_filled})")

    merged.to_parquet(ALL_PAIRED, index=False)
    print(f"wrote {ALL_PAIRED} ({len(merged)} rows x {len(merged.columns)} cols)")
    for g, sub in merged.groupby("gene_name"):
        sub.to_parquet(OUT / f"{g}_paired.parquet", index=False)
    print(f"re-split {merged['gene_name'].nunique()} per-gene parquets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
