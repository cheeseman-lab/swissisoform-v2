"""Top-5 SAE functional terms per cached protein → CSV (placeholder labels).

For each protein in the SAE feature cache (``data/cache/sae_esmc/``), ranks
features by peak activation across the whole protein and attaches the ESM-Atlas
term (label/description) for each. Proteins are identified by their sequence
hash + length (the per-gene/per-isoform differential view comes from the wired
pipeline, which carries gene/tis ids).

PLACEHOLDER: terms come from the 6B-layer60 Atlas, not the 600M SAE dictionary
our features are from — see swissisoform.plm.atlas.ATLAS_PROVENANCE.

Usage: python scripts/_sae_top_terms.py
"""

from __future__ import annotations

import csv
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np

from swissisoform.plm.atlas import ATLAS_PROVENANCE, load_atlas

SAE_CACHE = Path("data/cache/sae_esmc")
OUT = Path("data/output/cheeseman_13gene/sae_top_terms.csv")
TOP_N = 5


def main() -> int:
    """Write the top-N functional terms per protein to a CSV."""
    atlas = load_atlas()
    print(f"atlas terms loaded: {len(atlas)}")

    rows: list[dict] = []
    for f in sorted(glob.glob(str(SAE_CACHE / "*.npz"))):
        h = Path(f).stem
        with np.load(f) as z:
            idx, val = z["idx"], z["val"]
        if idx.shape[0] == 0:
            continue
        # Per-feature peak activation + prevalence across the whole protein.
        peak: dict[int, float] = defaultdict(float)
        prev: dict[int, int] = defaultdict(int)
        for r in range(idx.shape[0]):
            for fi, fv in zip(idx[r], val[r]):
                fi = int(fi)
                fv = float(fv)
                if fv > peak[fi]:
                    peak[fi] = fv
                prev[fi] += 1
        top = sorted(peak, key=lambda k: peak[k], reverse=True)[:TOP_N]
        for rank, fi in enumerate(top, 1):
            term = atlas.get(fi) or {}
            rows.append({
                "protein_hash": h[:12],
                "protein_len": int(idx.shape[0]),
                "rank": rank,
                "feature_index": fi,
                "max_activation": round(peak[fi], 3),
                "prevalence": prev[fi],
                "label": term.get("label"),
                "description": term.get("description"),
                "label_source": ATLAS_PROVENANCE if term else None,
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows ({len(rows) // TOP_N} proteins) -> {OUT}")
    # Quick eyeball: first protein's top terms.
    for r in rows[:TOP_N]:
        print(f"  {r['protein_hash']}  #{r['feature_index']:<6} {r['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
