#!/usr/bin/env python
"""Map the isoform feature space — one figure per matrix.

**No clustering.** A k-sweep (k = 5..100, both matrices) established that this
space has no discrete groups: silhouette peaked at 0.21 and fell monotonically to
0.15, cost showed no elbow, and a matched multivariate Gaussian — structureless
by construction — scored 0.12-0.16 on the same measure. Swapping rank-to-normal
for a multimodality-preserving robust z-score changed nothing (0.238/0.171/0.153
vs 0.211/0.168/0.154), so the blob is not an artifact of the transform. Isoform
feature space is a continuum, and any k-way partition of it is an arbitrary
slicing that shifts with the seed.

What remains is the map itself, plus the one measurement selection actually needs:
**how far each isoform sits from the nearest curated anchor.** That is what the
38 remaining `cheeseman50` genes are there to reduce, and unlike silhouette it
is well-defined on a continuum.

The 22 anchor isoforms are ordinary points here — they do not influence the MFA,
which is fit on all 6,462 isoforms. They are overlaid as labels only.

All components are kept — no truncation, no permutation-null cut, so distance in
score space equals distance in the weighted feature space exactly.

Outputs (alongside this script):
  - feature_space_<matrix>.png   ORF type on PC1-PC2 and PC3-PC4, block
                                 contributions per component
  - feature_space_report.md      the written read-out

Usage:
    python figures/clustering_dims/plot_feature_space.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import featurespace as fs
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

HERE = Path(__file__).resolve().parent

ORF_COLORS = {
    "extended": "#2563eb",
    "truncated": "#dc2626",
    "uorf": "#16a34a",
    "uoorf": "#65a30d",
    "internal_oof": "#9333ea",
    "3utr_orf": "#ea580c",
}


def anchor_mask(meta: pd.DataFrame, anchors: set[str]) -> np.ndarray:
    """Boolean mask of rows belonging to an anchor gene."""
    return meta["gene_name"].isin(anchors).to_numpy()


def _mark_anchors(ax, xy: np.ndarray, is_anchor: np.ndarray, label: bool = False) -> None:
    """Overlay the anchor isoforms on a panel."""
    ax.scatter(
        xy[is_anchor, 0],
        xy[is_anchor, 1],
        facecolors="none",
        edgecolors="black",
        s=42,
        linewidths=1.1,
        zorder=5,
        label="cheeseman50 anchors" if label else None,
    )


def plot_matrix(result: fs.MFAResult, anchors: set[str], path: Path) -> None:
    """One figure: ORF type on PC1-PC2 and PC3-PC4, block contributions."""
    scores = result.scores
    meta = result.matrix.meta
    is_anchor = anchor_mask(meta, anchors)
    r = scores.shape[1]

    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.22, height_ratios=[1, 0.68])

    # --- ORF type, the one real organizing structure ------------------------
    for pos, (a, b) in ((gs[0, 0], (0, 1)), (gs[0, 1], (2, 3))):
        ax = fig.add_subplot(pos)
        if b >= r:
            ax.axis("off")
            continue
        orf = meta["orf_type"].to_numpy()
        for t, c in ORF_COLORS.items():
            m = orf == t
            if m.any():
                ax.scatter(
                    scores[m, a],
                    scores[m, b],
                    c=c,
                    s=3,
                    alpha=0.5,
                    linewidths=0,
                    label=f"{t} ({m.sum()})",
                    rasterized=True,
                )
        _mark_anchors(ax, scores[:, [a, b]], is_anchor, label=True)
        if (a, b) == (0, 1):
            ax.legend(frameon=False, fontsize=7, markerscale=2.5)
        ax.set_xlabel(f"PC{a + 1}")
        ax.set_ylabel(f"PC{b + 1}")
        ax.set_title(f"ORF type — PC{a + 1} vs PC{b + 1}", fontsize=10)

    # --- block contributions ------------------------------------------------
    ax = fig.add_subplot(gs[1, :])
    ev = result.eigenvalues
    n_show = min(20, len(ev))
    contrib = result.block_contributions().iloc[:, :n_show]
    im = ax.imshow(contrib.values, aspect="auto", cmap="magma", vmin=0)
    ax.set_yticks(range(len(contrib.index)), contrib.index)
    ax.set_xticks(range(contrib.shape[1]), contrib.columns, rotation=45, ha="right")
    for i in range(contrib.shape[0]):
        for j in range(contrib.shape[1]):
            v = contrib.values[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if v < contrib.values.max() * 0.6 else "black",
            )
    fig.colorbar(im, ax=ax, shrink=0.85)
    ax.set_title(
        f"block contribution per component, PC1-PC{n_show} of {r}  "
        "(Σ V² over each block; columns sum to 1)",
        fontsize=10,
    )

    fig.suptitle(
        f"Isoform feature space — {result.matrix.name}   "
        f"({result.matrix.X.shape[0]:,} isoforms × {result.matrix.X.shape[1]} features, "
        f"MFA over C/D/L/M/P/S)",
        fontsize=13,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_report(results: dict[str, fs.MFAResult], anchors: set[str], path: Path) -> str:
    """Write the markdown read-out and return it."""
    lines = [
        "# Isoform feature space — MFA maps",
        "",
        f"Anchors: {len(anchors)} curated genes from `presets/cheeseman50.toml`.",
        "",
        "**No clustering.** A k-sweep over k = 5..100 found no discrete structure: "
        "silhouette peaked at 0.21 and fell monotonically to 0.15, cost had no elbow, "
        "and a matched multivariate Gaussian scored 0.12-0.16 on the same measure. "
        "Substituting a multimodality-preserving robust z-score for rank-to-normal "
        "changed nothing. The space is a continuum; selection must cover it rather "
        "than partition it.",
        "",
    ]
    for name, r in results.items():
        n_anchor = int(anchor_mask(r.matrix.meta, anchors).sum())
        lines += [
            f"## {name}",
            "",
            "```",
            fs.summarize(r),
            f"  anchor isoforms present: {n_anchor}",
            f"  verify: {fs.verify(r) or 'OK'}",
            f"  imputation bias (corr |score| vs observed): {fs.imputation_bias(r):+.3f}",
            "```",
            "",
            "Component correlation with the evidence scores — a check that the "
            "embedding did not merely rediscover the ~19 scored levers:",
            "",
            "```",
            fs.score_correlation(r).round(3).to_string(),
            "```",
            "",
            "Block contribution, PC1-PC6 (columns sum to 1):",
            "",
            "```",
            r.block_contributions().iloc[:, :6].round(3).to_string(),
            "```",
            "",
            "",
        ]
    text = "\n".join(lines)
    path.write_text(text)
    return text


def load_anchors() -> set[str]:
    """The 12 hand-curated anchor genes."""
    return set(fs.ANCHOR_GENES)


def main() -> None:
    """Fit both embeddings and draw one map each."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default=None, help="all_paired.parquet path or glob")
    args = ap.parse_args()

    anchors = load_anchors()
    print(f"anchors: {len(anchors)} genes")

    results = {}
    for name, matrix in fs.build_matrices(args.parquet).items():
        print(f"\nfitting MFA — {name}")
        r = fs.fit_mfa(matrix)
        results[name] = r
        print(fs.summarize(r))
        print("  verify:", fs.verify(r) or "OK")
        slug = name.replace("-", "_").lower()
        plot_matrix(r, anchors, HERE / f"feature_space_{slug}.png")

    print()
    print(write_report(results, anchors, HERE / "feature_space_report.md"))
    print(f"\nwrote figures + feature_space_report.md to {HERE}")


if __name__ == "__main__":
    main()
