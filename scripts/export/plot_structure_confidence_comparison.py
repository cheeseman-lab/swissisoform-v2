"""Plots for the Boltz-vs-ESMFold2 structure-confidence comparison.

Reads structure_confidence_comparison.csv and writes:
  - scatter_<metric>.png for metric in {plddt_mean, plddt_std, ptm}:
    x=Boltz, y=ESMFold2, one point per protein, with an OLS regression line
    annotated with R / R².
  - barchart_R2_by_metric.png: one bar per metric (plddt_mean, plddt_std,
    ptm), height = R² of that metric's ESMFold2-vs-Boltz regression line.

Usage: python scripts/export/plot_structure_confidence_comparison.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
METRICS = ["plddt_mean", "plddt_std", "ptm"]


def main(argv: list[str] | None = None) -> int:
    """Generate the metric scatters (+y=x) and the per-protein R histogram."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path,
                   default=ROOT / "data" / "output" / "cheeseman_13gene"
                   / "structure_confidence_comparison.csv")
    p.add_argument("--outdir", type=Path,
                   default=ROOT / "data" / "output" / "cheeseman_13gene"
                   / "structure_confidence_plots")
    args = p.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    print(f"loaded {len(df)} proteins from {args.csv.name}\n")

    # --- per-metric scatter: Boltz (x) vs ESMFold2 (y), with y=x + regression ---
    metric_R = {}
    for m in METRICS:
        x = df[f"boltz_{m}"].to_numpy(float)
        y = df[f"esmfold2_{m}"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        lr = stats.linregress(x, y)
        metric_R[m] = lr.rvalue

        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        pad = 0.05 * (hi - lo or 1)
        lim = (lo - pad, hi + pad)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(x, y, s=40, alpha=0.75, edgecolor="k", linewidth=0.4, zorder=3)
        xs = np.linspace(*lim, 100)
        ax.plot(xs, lr.slope * xs + lr.intercept, "-", color="C3",
                label=f"fit: y={lr.slope:.2f}x+{lr.intercept:.2f}", zorder=2)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.set_xlabel(f"Boltz-2  {m}")
        ax.set_ylabel(f"ESMFold2  {m}")
        ax.set_title(f"{m}: Boltz vs ESMFold2  (n={len(x)})")
        ax.text(0.05, 0.95, f"R = {lr.rvalue:.3f}\nR² = {lr.rvalue**2:.3f}",
                transform=ax.transAxes, va="top", fontsize=11,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        out = args.outdir / f"scatter_{m}.png"
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"  {m:11}: R={lr.rvalue:.3f}  R²={lr.rvalue**2:.3f}  "
              f"slope={lr.slope:.3f}  -> {out.name}")

    # --- bar chart: R^2 of the regression line for each metric ---
    r2 = [metric_R[m] ** 2 for m in METRICS]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(METRICS, r2, color=["C0", "C1", "C2"], edgecolor="k", alpha=0.85)
    for b, v in zip(bars, r2):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R²  (ESMFold2 vs Boltz regression)")
    ax.set_title("Boltz-vs-ESMFold2 agreement by metric")
    fig.tight_layout()
    out = args.outdir / "barchart_R2_by_metric.png"
    fig.savefig(out, dpi=150); plt.close(fig)

    print("\nR² by metric: " + "  ".join(f"{m}={v:.3f}" for m, v in zip(METRICS, r2)))
    print(f"bar chart -> {out.name}")
    print(f"all plots -> {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
