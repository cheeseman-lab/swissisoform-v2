"""Regression snapshot of the 13-gene paired output.

Captures a stable, order-independent fingerprint of ``all_paired.parquet`` so any
refactor can prove the 13-gene run is unchanged. Run ``capture`` once to write the
baseline, then ``compare`` after each refactor task.

Usage:
    python tests/regression/snapshot_paired.py capture   # write baseline (pre-refactor)
    python tests/regression/snapshot_paired.py compare    # assert current == baseline
    python tests/regression/snapshot_paired.py compare --paired <path>  # compare a scratch run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAIRED = ROOT / "data" / "output" / "cheeseman_13gene" / "all_paired.parquet"
BASELINE = ROOT / "tests" / "regression" / "baseline" / "cheeseman_13gene.parquet"
SORT_KEYS = ["gene_name", "tis_id"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(SORT_KEYS).reset_index(drop=True)
    return df[sorted(df.columns)]


def capture(paired: Path = PAIRED) -> None:
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    _normalize(pd.read_parquet(paired)).to_parquet(BASELINE, index=False)
    print(f"baseline written: {BASELINE}")


def compare(paired: Path = PAIRED) -> bool:
    if not BASELINE.exists():
        print(f"no baseline at {BASELINE} — run `capture` first")
        return False
    cur = _normalize(pd.read_parquet(paired))
    base = _normalize(pd.read_parquet(BASELINE))
    if list(cur.columns) != list(base.columns):
        only_cur = sorted(set(cur.columns) - set(base.columns))
        only_base = sorted(set(base.columns) - set(cur.columns))
        print(f"COLUMNS differ — added {only_cur} / removed {only_base}")
        return False
    if cur.shape != base.shape:
        print(f"SHAPE differs: current {cur.shape} vs baseline {base.shape}")
        return False
    match = cur.equals(base)
    if not match:
        for col in cur.columns:
            if not cur[col].equals(base[col]):
                print(f"  column changed: {col}")
    print("MATCH" if match else "MISMATCH")
    return match


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["capture", "compare"], nargs="?", default="compare")
    ap.add_argument("--paired", type=Path, default=PAIRED)
    args = ap.parse_args()
    if args.cmd == "capture":
        capture(args.paired)
        return 0
    return 0 if compare(args.paired) else 1


if __name__ == "__main__":
    sys.exit(main())
