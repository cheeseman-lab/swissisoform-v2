"""One-off: backfill DeepLoc(localization) + SignalP columns from the ESM-2/Boltz
snapshot into the freshly regenerated ESM-C/ESMFold2 parquet.

Those two modules need dedicated conda envs (swissisoform-v2-{deeploc,signalp})
that aren't set up here (and InterProScan, whose nextflow run failed exit 1),
so the pipeline run leaves them empty. They are pure sequence-based predictions
and the proteins are byte-identical between runs, so the snapshot's values are
still valid — we just splice them back in (join on tis_id), then re-split the
per-gene parquets.

Usage: python scripts/_backfill_cols.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path("data/output/cheeseman_13gene")
NEW = OUT / "all_paired.parquet"
SNAP = OUT / "all_paired_esm2boltz.parquet"
KEY = "tis_id"
TAGS = ("localization", "signalp", "interproscan")


def main() -> int:
    new = pd.read_parquet(NEW)
    snap = pd.read_parquet(SNAP)

    # Sanity: confirm NEW really is the ESMFold2 regen, not the old parquet.
    bcol = next((c for c in new.columns if c.endswith("structure_backend")), None)
    backends = sorted(set(new[bcol].dropna())) if bcol else []
    print(f"NEW structure backends: {backends}  (rows={len(new)})")

    cols = [c for c in snap.columns if any(t in c for t in TAGS)]
    missing = [c for c in cols if c not in snap.columns]
    assert not missing, f"snapshot missing {missing}"
    print(f"backfilling {len(cols)} cols ({sum('localization' in c for c in cols)} localization, "
          f"{sum('signalp' in c for c in cols)} signalp)")

    # Null-count in NEW before (expect ~all null), valid count in SNAP.
    before = int(new[cols[0]].notna().sum()) if cols[0] in new.columns else 0
    print(f"  {cols[0]}: non-null in NEW before={before}, in SNAP={int(snap[cols[0]].notna().sum())}")

    drop = [c for c in new.columns if any(t in c for t in TAGS)]
    merged = new.drop(columns=drop).merge(
        snap[[KEY] + cols].drop_duplicates(KEY), on=KEY, how="left"
    )

    # Restore the snapshot's column order if the sets match (clean, consistent schema).
    if set(merged.columns) == set(snap.columns):
        merged = merged[snap.columns]

    after = int(merged[cols[0]].notna().sum())
    print(f"  {cols[0]}: non-null in NEW after={after}")

    merged.to_parquet(NEW, index=False)
    print(f"wrote {NEW} ({len(merged)} rows x {len(merged.columns)} cols)")
    for g, sub in merged.groupby("gene_name"):
        sub.to_parquet(OUT / f"{g}_paired.parquet", index=False)
    print(f"re-split {merged['gene_name'].nunique()} per-gene parquets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
