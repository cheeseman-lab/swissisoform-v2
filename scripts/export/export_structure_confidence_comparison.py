"""Compare per-residue confidence (pLDDT) between the Boltz-2 and ESMFold2
structure caches, one row per protein folded by BOTH backends.

For each sequence hash present in both data/cache/structure/{boltz,esmfold2}/:
  - pull the per-residue pLDDT vector from each confidence.json
  - cosine similarity of the two vectors (+ Pearson r and MAE as more
    discriminative companions — see notes below)
  - side-by-side + delta of the metrics.json scalars (plddt_mean, plddt_std, ptm)

Reads the caches read-only; writes one new CSV. Does not touch the pipeline.

Usage:
    python scripts/export/export_structure_confidence_comparison.py
    # → data/output/cheeseman_13gene/structure_confidence_comparison.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache" / "structure"
A_BACKEND = "boltz"
B_BACKEND = "esmfold2"
# Whether to write the full per-residue vectors into the CSV (semicolon-joined,
# rounded). They make cells very wide; set False to keep the CSV scalar-only.
INCLUDE_VECTORS = True
VEC_ROUND = 4


def _load(backend: str, h: str) -> tuple[list[float] | None, dict]:
    base = CACHE / backend / h
    conf_p, met_p = base / "confidence.json", base / "metrics.json"
    plddt = None
    if conf_p.exists():
        plddt = json.loads(conf_p.read_text()).get("plddt")
    metrics = json.loads(met_p.read_text()) if met_p.exists() else {}
    return plddt, metrics


def main(argv: list[str] | None = None) -> int:
    """Build the Boltz-vs-ESMFold2 per-residue confidence comparison CSV."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path,
                   default=ROOT / "data" / "output" / "cheeseman_13gene"
                   / "structure_confidence_comparison.csv")
    args = p.parse_args(argv)

    a_hashes = {d.name for d in (CACHE / A_BACKEND).iterdir() if d.is_dir()}
    b_hashes = {d.name for d in (CACHE / B_BACKEND).iterdir() if d.is_dir()}
    both = sorted(a_hashes & b_hashes)
    print(f"{A_BACKEND}={len(a_hashes)} {B_BACKEND}={len(b_hashes)} "
          f"in_both={len(both)}  ({A_BACKEND}_only={len(a_hashes - b_hashes)}, "
          f"{B_BACKEND}_only={len(b_hashes - a_hashes)})")

    rows: list[dict] = []
    for h in both:
        a_vec, a_met = _load(A_BACKEND, h)
        b_vec, b_met = _load(B_BACKEND, h)

        cos = pear = mae = None
        len_a = len(a_vec) if a_vec else None
        len_b = len(b_vec) if b_vec else None
        if a_vec and b_vec:
            n = min(len(a_vec), len(b_vec))  # align to overlap if lengths differ
            x, y = np.asarray(a_vec[:n], float), np.asarray(b_vec[:n], float)
            if n and np.linalg.norm(x) and np.linalg.norm(y):
                cos = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
                pear = float(np.corrcoef(x, y)[0, 1]) if n > 1 else None
                mae = float(np.mean(np.abs(x - y)))

        row = {
            "hash": h,
            f"len_{A_BACKEND}": len_a,
            f"len_{B_BACKEND}": len_b,
            "len_mismatch": (len_a != len_b),
            "cosine_similarity": cos,
            "pearson_r": pear,
            "mae": mae,
            f"status_{A_BACKEND}": a_met.get("status"),
            f"status_{B_BACKEND}": b_met.get("status"),
        }
        for field in ("plddt_mean", "plddt_std", "ptm"):
            av, bv = a_met.get(field), b_met.get(field)
            row[f"{A_BACKEND}_{field}"] = av
            row[f"{B_BACKEND}_{field}"] = bv
            row[f"delta_{field}"] = (
                (bv - av) if isinstance(av, (int, float)) and isinstance(bv, (int, float))
                else None
            )
        if INCLUDE_VECTORS:
            row[f"plddt_{A_BACKEND}"] = (
                ";".join(f"{v:.{VEC_ROUND}f}" for v in a_vec) if a_vec else ""
            )
            row[f"plddt_{B_BACKEND}"] = (
                ";".join(f"{v:.{VEC_ROUND}f}" for v in b_vec) if b_vec else ""
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows x {len(df.columns)} cols → {args.out}")
    if len(df):
        print(f"  cosine: mean={df['cosine_similarity'].mean():.4f} "
              f"min={df['cosine_similarity'].min():.4f}")
        print(f"  pearson_r: mean={df['pearson_r'].mean():.4f} "
              f"min={df['pearson_r'].min():.4f}")
        print(f"  delta_plddt_mean (esmfold2-boltz): mean={df['delta_plddt_mean'].mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
