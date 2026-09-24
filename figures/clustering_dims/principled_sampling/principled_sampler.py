#!/usr/bin/env python
"""Principled Sampler v1 over the MFA components — exploratory pass.

Selects a diverse subset of isoforms for the `cheeseman50` LLM-tuning coreset.
The coreset holds 12 curated genes / 22 isoforms against a target of 50, so 28
more are needed; with the sampler returning ``3n`` picks that puts ``n = 10``.

The algorithm (Algorithm 1 of the source paper) projects onto the top ``n``
principal components and takes three sets:

    Y   the row with the LARGEST projection on each component      (n picks)
    Z   the row with the SMALLEST (most negative) projection       (n picks)
    W   the n rows with the smallest infinity norm across those n components

Y and Z sit at the extremes of the dominant axes of variation — the "common
patterns" the data spreads along. W sits near the origin: rows that are extreme
on nothing, and therefore qualitatively unlike everything in Y and Z.

``X_pca`` in the algorithm is exactly our MFA score matrix. MFA is a
block-weighted PCA — components ordered by variance, orthonormal loadings, and
scores left unwhitened (``F = UΣ``), which matters because the algorithm reads
raw projection magnitudes. Substituting MFA scores for plain PCA scores is the
right adaptation here: it corrects the block imbalance (S carries 158 columns, P
carries 11) that plain PCA on the concatenated features would inherit.

**Exploratory: no constraints are enforced.** The curation rules for the coreset
are (1) 50 isoforms, (2) at least 3 per ORF type, (3) one isoform per gene except
the 22 anchors. This pass deliberately ignores all three so we can see what the
method returns on its own, and how far that is from the rules.

The sampler is deterministic — no seed, no restarts, nothing to tune but ``n``.

Outputs (alongside this script):
  - principled_sample.csv        one row per pick, every n
  - principled_sample_all_orf.png   the picks drawn on PC1-PC2 and PC3-PC4
  - principled_sample_report.md  composition, collisions, overlap, n-sensitivity

Usage:
    python figures/clustering_dims/principled_sampler.py
    python figures/clustering_dims/principled_sampler.py --n 5 10 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The MFA embedding lives in the sibling figure folder; these are standalone
# scripts run directly, not an installed package, so the path is added by hand.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "feature_space"))

import featurespace as fs  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent

DEFAULT_N = 10
SENSITIVITY_N = (5, 10, 15)

SET_STYLE = {
    "Y": ("#2563eb", "^", "Y — largest projection"),
    "Z": ("#dc2626", "v", "Z — most negative projection"),
    "W": ("#ca8a04", "s", "W — smallest infinity norm"),
}

ORF_COLORS = {
    "extended": "#93c5fd",
    "truncated": "#fca5a5",
    "uorf": "#86efac",
    "uoorf": "#bef264",
    "internal_oof": "#d8b4fe",
    "3utr_orf": "#fdba74",
}


# ---------------------------------------------------------------------------
# Algorithm 1
# ---------------------------------------------------------------------------


def principled_sample(X_pca: np.ndarray, n: int) -> dict[str, np.ndarray]:
    """Return the Y / Z / W index sets of Principled Sampler v1.

    Args:
        X_pca: ``(N, M)`` component scores; only the first ``n`` columns are read.
        n: Number of components to sample along. Yields ``3n`` picks.

    Returns:
        Dict with keys ``"Y"``, ``"Z"``, ``"W"``, each an array of ``n`` row
        indices into ``X_pca``.
    """
    ranked = ranked_candidates(X_pca, n)
    return {
        "Y": np.array([col[0] for col in ranked["Y"]]),
        "Z": np.array([col[0] for col in ranked["Z"]]),
        "W": ranked["W"][0][:n],
    }


def ranked_candidates(X_pca: np.ndarray, n: int) -> dict[str, list[np.ndarray]]:
    """Full orderings behind each of Algorithm 1's picks.

    ``principled_sample`` is the rank-0 case of this — it takes the first entry of
    each ordering. Keeping both on one implementation means a constrained selector
    that has to skip past an unusable candidate walks the *same* ranking the
    unconstrained algorithm would have read, rather than a parallel one that could
    drift from it.

    Returns:
        ``{"Y": [order_pc1, ...], "Z": [...], "W": [order]}`` — each Y/Z entry
        ranks every row on one component (descending for Y, ascending for Z); W is
        a single ordering of every row by ascending infinity norm.
    """
    if n > X_pca.shape[1]:
        raise ValueError(f"n={n} exceeds the {X_pca.shape[1]} available components")
    sub = X_pca[:, :n]
    return {
        "Y": [np.argsort(-sub[:, i], kind="stable") for i in range(n)],
        "Z": [np.argsort(sub[:, i], kind="stable") for i in range(n)],
        # Infinity norm of each row over the first n components: a row is near
        # the origin only if it is unremarkable on EVERY one of them.
        "W": [np.argsort(np.abs(sub).max(axis=1), kind="stable")],
    }


def verify_sample(X_pca: np.ndarray, n: int, picks: dict[str, np.ndarray]) -> list[str]:
    """Check the picks against a direct recomputation of the definitions."""
    problems: list[str] = []
    sub = X_pca[:, :n]

    for i in range(n):
        col = sub[:, i]
        if picks["Y"][i] != int(np.argmax(col)):
            problems.append(f"Y[{i}] is not the argmax of component {i + 1}")
        if picks["Z"][i] != int(np.argmin(col)):
            problems.append(f"Z[{i}] is not the argmin of component {i + 1}")
        # Ties would be broken silently by row order, so surface them instead.
        if (col == col.max()).sum() > 1:
            problems.append(f"component {i + 1}: tied maximum, pick depends on row order")
        if (col == col.min()).sum() > 1:
            problems.append(f"component {i + 1}: tied minimum, pick depends on row order")

    inf_norm = np.abs(sub).max(axis=1)
    outside = np.setdiff1d(np.arange(len(sub)), picks["W"])
    if inf_norm[picks["W"]].max() > inf_norm[outside].min():
        problems.append("W is not the n smallest infinity norms")
    return problems


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def picks_frame(result: fs.MFAResult, anchors: set[str], n: int, n_coord: int = 10) -> pd.DataFrame:
    """One row per pick, with provenance and coordinates."""
    scores = result.scores
    meta = result.matrix.meta
    picks = principled_sample(scores, n)
    anchored_genes = set(meta.loc[meta["gene_name"].isin(anchors), "gene_name"])

    rows = []
    for name, idx in picks.items():
        for slot, j in enumerate(idx):
            j = int(j)
            row = {
                "matrix": result.matrix.name,
                "n": n,
                "set": name,
                # Y/Z are tied to a specific component; W is a global ranking.
                "component": slot + 1 if name in ("Y", "Z") else None,
                "rank": slot + 1 if name == "W" else None,
                "row_index": j,
                "gene_name": meta["gene_name"].iloc[j],
                "tis_id": meta["tis_id"].iloc[j],
                "orf_type": meta["orf_type"].iloc[j],
                "is_anchor": meta["gene_name"].iloc[j] in anchored_genes,
                "value": float(scores[j, slot]) if name in ("Y", "Z") else None,
                "inf_norm": float(np.abs(scores[j, :n]).max()),
            }
            for c in range(min(n_coord, scores.shape[1])):
                row[f"PC{c + 1}"] = float(scores[j, c])
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_picks(frame: pd.DataFrame, pool: pd.DataFrame) -> list[str]:
    """The read-out the exploratory pass exists to produce."""
    lines: list[str] = []
    n_total = len(frame)
    unique = frame["row_index"].nunique()

    lines += [
        f"picks: {n_total} concatenated, {unique} unique rows "
        f"({n_total - unique} duplicate{'s' if n_total - unique != 1 else ''} across Y/Z/W)",
        "",
        "ORF-type composition vs the pool:",
        "",
    ]
    comp = (
        pd.DataFrame(
            {
                "picks": frame.drop_duplicates("row_index")["orf_type"].value_counts(),
                "pool": pool["orf_type"].value_counts(),
            }
        )
        .fillna(0)
        .astype(int)
    )
    comp["pool_pct"] = (comp["pool"] / comp["pool"].sum() * 100).round(1)
    comp["pick_pct"] = (comp["picks"] / comp["picks"].sum() * 100).round(1)
    lines += [comp.to_string(), ""]

    # Rule 3 of the curation guidelines: one isoform per gene, anchors exempt.
    dup_genes = frame["gene_name"].value_counts()
    dup_genes = dup_genes[dup_genes > 1]
    n_anchor_hits = int(frame.drop_duplicates("row_index")["is_anchor"].sum())
    lines += [
        f"gene collisions: {len(dup_genes)} gene(s) picked more than once"
        + (f" ({', '.join(f'{g}×{c}' for g, c in dup_genes.items())})" if len(dup_genes) else ""),
        f"picks landing on an already-anchored gene: {n_anchor_hits}",
        "",
    ]

    for name in ("Y", "Z", "W"):
        sub = frame[frame["set"] == name]
        if sub.empty:
            continue
        lines.append(
            f"{name}: {sub['orf_type'].value_counts().to_dict()}  "
            f"|inf_norm| {sub['inf_norm'].min():.2f}–{sub['inf_norm'].max():.2f}"
        )
    lines.append("")
    return lines


def w_distinctness(result: fs.MFAResult, n: int) -> list[str]:
    """Is W a meaningful set, or arbitrary draws from a dense centre?

    ``W`` takes the n smallest infinity norms. If hundreds of rows share
    essentially the same tiny norm, which n get picked is decided by float noise
    rather than by anything about the isoforms.
    """
    inf_norm = np.abs(result.scores[:, :n]).max(axis=1)
    order = np.sort(inf_norm)
    w_max = order[n - 1]
    n_within_10pct = int((inf_norm <= w_max * 1.1).sum())
    return [
        f"W distinctness (n={n}):",
        f"  infinity norms of the {n} picks: {order[:n].min():.3f}–{w_max:.3f}",
        f"  rows within 10% of the largest W norm: {n_within_10pct} "
        f"(the pool W is effectively drawn from)",
        f"  pool infinity norm: median {np.median(inf_norm):.2f}, "
        f"p05 {np.percentile(inf_norm, 5):.2f}",
        "",
    ]


def plot_picks(result: fs.MFAResult, frame: pd.DataFrame, anchors: set[str], path: Path) -> None:
    """Draw the pool with Y / Z / W marked, on PC1-PC2 and PC3-PC4."""
    scores = result.scores
    meta = result.matrix.meta
    is_anchor = meta["gene_name"].isin(anchors).to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8))
    for ax, (a, b) in zip(axes, ((0, 1), (2, 3))):
        orf = meta["orf_type"].to_numpy()
        for t, c in ORF_COLORS.items():
            m = orf == t
            if m.any():
                ax.scatter(
                    scores[m, a],
                    scores[m, b],
                    c=c,
                    s=3,
                    alpha=0.45,
                    linewidths=0,
                    rasterized=True,
                    label=t if (a, b) == (0, 1) else None,
                )
        ax.scatter(
            scores[is_anchor, a],
            scores[is_anchor, b],
            facecolors="none",
            edgecolors="black",
            s=46,
            linewidths=1.2,
            zorder=4,
            label="cheeseman50 anchors" if (a, b) == (0, 1) else None,
        )
        for name, (color, marker, label) in SET_STYLE.items():
            idx = frame.loc[frame["set"] == name, "row_index"].to_numpy()
            ax.scatter(
                scores[idx, a],
                scores[idx, b],
                c=color,
                marker=marker,
                s=95,
                edgecolors="white",
                linewidths=0.8,
                zorder=6,
                label=label if (a, b) == (0, 1) else None,
            )
        ax.set_xlabel(f"PC{a + 1}")
        ax.set_ylabel(f"PC{b + 1}")
        ax.set_title(f"PC{a + 1} vs PC{b + 1}")
    axes[0].legend(frameon=False, fontsize=8, markerscale=1.4, loc="upper left")
    fig.suptitle(
        f"Principled Sampler v1 (n={frame['n'].iloc[0]}) — {result.matrix.name}", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_anchors() -> set[str]:
    """The 12 hand-curated anchor genes."""
    return set(fs.ANCHOR_GENES)


def main() -> None:
    """Run the sampler and write the exploratory read-out."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default=None, help="all_paired.parquet path or glob")
    ap.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=list(SENSITIVITY_N),
        help=f"component counts to sample at (default {SENSITIVITY_N})",
    )
    args = ap.parse_args()

    anchors = load_anchors()
    ns = sorted(set(args.n) | {DEFAULT_N})
    print(f"anchors: {len(anchors)} genes; n = {ns} (headline n={DEFAULT_N})")

    lines = [
        "# Principled Sampler v1 — exploratory pass",
        "",
        f"Coreset target 50 isoforms; 22 already curated, so 28 needed → n = {DEFAULT_N} "
        f"(3n = {3 * DEFAULT_N}).",
        "",
        "**No constraints enforced.** The curation rules (≥3 per ORF type, one "
        "isoform per gene outside the anchors, exactly 28 picks) are deliberately "
        "ignored here so the method's unmodified behaviour is visible.",
        "",
    ]
    frames = []

    matrix = fs.build_matrix_all_orf(args.parquet)
    print("\nfitting MFA — all-ORF")
    result = fs.fit_mfa(matrix)
    print(fs.summarize(result))

    lines += [
        "## all-ORF",
        "",
        f"`{result.scores.shape[0]:,} × {result.scores.shape[1]}` scores",
        "",
    ]

    for n in ns:
        picks = principled_sample(result.scores, n)
        problems = verify_sample(result.scores, n, picks)
        print(f"  n={n:2d}  verify: {problems or 'OK'}")

        frame = picks_frame(result, anchors, n)
        frames.append(frame)
        lines += [f"### n = {n}", "", "```"]
        lines += summarize_picks(frame, result.matrix.meta)
        lines += w_distinctness(result, n)
        if problems:
            lines += ["VERIFY PROBLEMS:", *(f"  {p}" for p in problems), ""]
        lines += ["```", ""]

        if n == DEFAULT_N:
            lines += [
                f"The {3 * n} picks at the headline n:",
                "",
                "```",
                frame[
                    [
                        "set",
                        "component",
                        "rank",
                        "gene_name",
                        "orf_type",
                        "is_anchor",
                        "value",
                        "inf_norm",
                    ]
                ]
                .round(3)
                .to_string(index=False),
                "```",
                "",
            ]
            plot_picks(result, frame, anchors, HERE / "principled_sample_all_orf.png")

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(HERE / "principled_sample.csv", index=False)

    text = "\n".join(lines)
    (HERE / "principled_sample_report.md").write_text(text)
    print()
    print(text)
    print(f"\nwrote principled_sample.csv + report + figures to {HERE}")


if __name__ == "__main__":
    main()
