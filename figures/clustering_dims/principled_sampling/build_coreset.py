#!/usr/bin/env python
"""Fill the remaining 28 slots of the cheeseman50 LLM-tuning coreset.

`cheeseman50` holds 12 curated genes / 22 isoforms against a target of 50. The
anchors are **static** — never re-derived, re-ranked or filtered. This script
adds the other 28 in two ordered stages:

    22  anchors            (static)
     9  rare-type fill     3 each: internal_oof, 3utr_orf, uoorf
    19  Principled Sampler everything else
    ──
    50

**Stage order is load-bearing.** The exploratory pass showed the unconstrained
sampler gives internal_oof / 3utr_orf / uoorf *zero* picks at every n — they are
0.3-0.9% of the pool and a top-1-per-component rule cannot reach them. Forcing
them first is the only way they appear at all. Equally, 44 pool genes carry both a
rare type and a common one, so the rare strata must claim their genes before the
sampler runs or a gene gets picked twice and the set lands at 27.

``extended``, ``truncated`` and ``uorf`` get no floor: the anchors already carry
12 extended and 9 truncated, and the sampler picked 3 uORFs unprompted at n=10.

Matrix: **all-ORF**, forced rather than preferred — paired-ORF contains only
extended and truncated and cannot supply a rare-type isoform at all. The cost is
the 83 shared-region features, which do carry independent signal (adjusted Rand
0.27 between the two spaces).

Outputs (alongside this script):
  - coreset_selection.csv       the 28 picks, with provenance
  - coreset_50.csv              the final 50 (anchors + picks)
  - coreset_selection.png       the picks drawn on the MFA space
  - coreset_selection_report.md the written read-out

Usage:
    python figures/clustering_dims/principled_sampling/build_coreset.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "feature_space"))

import featurespace as fs  # noqa: E402
from principled_sampler import principled_sample, ranked_candidates  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent

TARGET = 50
# Types the sampler cannot reach on its own, forced to 3 each. uoorf already has
# 1 anchor (SKA3) so it lands at 4 — the rule is a minimum, not a target.
#
# uorf is here despite the anchors having none and the n=10 exploratory run
# picking 3 unprompted: at n=7 with anchors excluded the sampler found only ONE,
# so relying on it to satisfy the floor failed verification. Forced rather than
# assumed.
# The four types the global sampler cannot reach: each is 0.3-4.9% of the pool,
# so a top-1-per-component rule loses every axis to the extensions and
# truncations that dominate it. Missed for lack of COUNT, not lack of variance.
# uoorf already has 1 anchor (SKA3) so it lands at 4 — the rule is a minimum.
RARE_TYPES = ("uorf", "internal_oof", "3utr_orf", "uoorf")

# 3n = 18 ordered candidates, of which 16 are consumed. n=6 rather than 7 so the
# walk reaches W four times instead of twice — Y and Z would otherwise eat 14 of
# the 16 slots and the "typical" set would barely appear.
N_GLOBAL = 6

SET_STYLE = {
    "Y": ("#2563eb", "^", "Y — largest projection"),
    "Z": ("#dc2626", "v", "Z — most negative projection"),
    "W": ("#ca8a04", "s", "W — smallest infinity norm"),
    "rare": ("#16a34a", "D", "rare-type fill"),
}
ORF_COLORS = {
    "extended": "#bfdbfe",
    "truncated": "#fecaca",
    "uorf": "#bbf7d0",
    "uoorf": "#d9f99d",
    "internal_oof": "#e9d5ff",
    "3utr_orf": "#fed7aa",
}


# ---------------------------------------------------------------------------
# Stage 1 — rare-type fill
# ---------------------------------------------------------------------------


def rare_type_fill(scores: np.ndarray, rows: np.ndarray) -> list[dict]:
    """Algorithm 1 at n=1 within one ORF type — its max, min and most typical.

    At n=1 the sampler returns exactly three picks, which is exactly the quota:

        Y   the largest projection on PC1   — one end of the type's spread
        Z   the most negative on PC1        — the other end
        W   the smallest infinity norm      — extreme on nothing, the typical one

    Running the same algorithm here rather than a bespoke "most extreme" rule
    matters for what the set teaches: a pure-magnitude criterion returns three
    outliers of the type and no representative example of it. W is the only
    source of a typical rare-type isoform in the whole coreset.

    Restricted to *rows*, so the picks are extreme among their own type rather
    than against a pool the type could never win against.
    """
    sub = scores[rows]
    picks = principled_sample(sub, 1)
    out = []
    for name in ("Y", "Z", "W"):
        local = int(picks[name][0])
        out.append(
            {
                "row_index": int(rows[local]),
                "set": name,
                "component": 1,
                "value": float(sub[local, 0]),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Stage 2 — Principled Sampler with a rank-extended fallback
# ---------------------------------------------------------------------------


def sampler_fill(
    scores: np.ndarray,
    genes: np.ndarray,
    claimed: set[str],
    n: int,
    k: int,
) -> list[dict]:
    """Walk Algorithm 1's ordered output, taking *k* picks with distinct genes.

    The paper returns an ordered sequence — Y₁..Yₙ, then Z₁..Zₙ, then W₁..Wₙ —
    so consuming it in order and stopping at *k* is the algorithm's own device for
    a target that is not a multiple of 3.

    When a candidate's gene is already claimed, fall through to the next-best
    candidate *on that same component* rather than dropping the slot. The pick
    stays the best available answer to "what is most extreme on this axis",
    which is what the algorithm asks; ``rank_used`` records how far the fallback
    had to reach.
    """
    ranked = ranked_candidates(scores, n)
    slots = (
        [("Y", i, ranked["Y"][i]) for i in range(n)]
        + [("Z", i, ranked["Z"][i]) for i in range(n)]
        + [("W", i, ranked["W"][0]) for i in range(n)]
    )

    picks: list[dict] = []
    w_cursor = 0
    for set_name, i, order in slots:
        if len(picks) == k:
            break
        # Y/Z restart at rank 0 for each component; W is one global ranking, so
        # it resumes where the previous W slot stopped.
        start = w_cursor if set_name == "W" else 0
        for rank, row in enumerate(order[start:], start=start):
            gene = genes[row]
            if gene in claimed:
                continue
            claimed.add(gene)
            picks.append(
                {
                    "row_index": int(row),
                    "set": set_name,
                    "component": i + 1 if set_name in ("Y", "Z") else None,
                    "rank": i + 1 if set_name == "W" else None,
                    "rank_used": rank - start,
                    "value": float(scores[row, i]) if set_name in ("Y", "Z") else None,
                }
            )
            if set_name == "W":
                w_cursor = rank + 1
            break
    if len(picks) < k:
        raise SystemExit(f"sampler exhausted: wanted {k}, got {len(picks)} from 3n={3 * n}")
    return picks


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_coreset(
    result: fs.MFAResult, anchors: set[str], n_coord: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run both stages and return ``(the 28 picks, the final 50)``."""
    scores = result.scores
    meta = result.matrix.meta
    genes = meta["gene_name"].to_numpy()
    orf = meta["orf_type"].to_numpy()

    is_anchor = np.isin(genes, list(anchors))
    n_anchor_iso = int(is_anchor.sum())
    need = TARGET - n_anchor_iso

    # Seeded with the anchors so no pick can land on a gene we already carry, and
    # extended after every pick so no gene is taken twice.
    claimed = set(anchors)
    rows: list[dict] = []

    for orf_type in RARE_TYPES:
        stratum = np.flatnonzero((orf == orf_type) & ~is_anchor)
        for p in rare_type_fill(scores, stratum):
            gene = genes[p["row_index"]]
            if gene in claimed:
                raise SystemExit(f"{orf_type}: {gene} already claimed — needs a fallback")
            claimed.add(gene)
            rows.append({**p, "stage": "rare_fill", "rank": None, "rank_used": 0})

    n_sampler = need - len(rows)
    for p in sampler_fill(scores, genes, claimed, N_GLOBAL, n_sampler):
        rows.append({**p, "stage": "sampler"})

    picks = pd.DataFrame(rows)
    picks.insert(0, "gene_name", genes[picks["row_index"]])
    picks.insert(1, "tis_id", meta["tis_id"].to_numpy()[picks["row_index"]])
    picks.insert(2, "orf_type", orf[picks["row_index"]])
    # Over the components that stage actually read, so the value explains the
    # pick: 10 for the rare fill's axis search, N_GLOBAL for the sampler's
    # infinity-norm ranking. A single window would make W's ranking look
    # non-monotone.
    width = np.where(picks["stage"] == "sampler", N_GLOBAL, n_coord)
    picks["inf_norm"] = [
        float(np.abs(scores[r, :w]).max()) for r, w in zip(picks["row_index"], width)
    ]
    for c in range(n_coord):
        picks[f"PC{c + 1}"] = scores[picks["row_index"], c]

    anchor_rows = meta.loc[is_anchor, ["gene_name", "tis_id", "orf_type"]].copy()
    anchor_rows["source"] = "anchor"
    picked_rows = picks[["gene_name", "tis_id", "orf_type", "stage"]].rename(
        columns={"stage": "source"}
    )
    final = pd.concat([anchor_rows, picked_rows], ignore_index=True)
    return picks, final


def verify(picks: pd.DataFrame, final: pd.DataFrame, anchors: set[str], result) -> list[str]:
    """Check the curation rules and fidelity to the algorithm."""
    problems: list[str] = []
    scores, meta = result.scores, result.matrix.meta
    genes = meta["gene_name"].to_numpy()

    if len(final) != TARGET:
        problems.append(f"final set has {len(final)} rows, expected {TARGET}")
    if picks["gene_name"].nunique() != len(picks):
        problems.append("picks contain a repeated gene (rule 3)")
    if set(picks["gene_name"]) & anchors:
        problems.append(f"picks land on anchor genes: {set(picks['gene_name']) & anchors}")

    counts = final["orf_type"].value_counts()
    for t, c in counts.items():
        if c < 3:
            problems.append(f"orf_type {t} has only {c} isoforms (rule 2 wants >=3)")

    for orf_type, g in picks[picks["stage"] == "rare_fill"].groupby("orf_type"):
        if set(g["set"]) != {"Y", "Z", "W"}:
            problems.append(f"{orf_type}: rare fill is not one each of Y/Z/W")

    # Fidelity: each accepted Y pick must be the best still-available row on its
    # component, i.e. the fallback advanced only as far as it had to.
    claimed_before = set(anchors) | set(picks[picks["stage"] == "rare_fill"]["gene_name"])
    for _, r in picks[(picks["stage"] == "sampler") & (picks["set"] == "Y")].iterrows():
        i = int(r["component"]) - 1
        order = np.argsort(-scores[:, i], kind="stable")
        best = next(row for row in order if genes[row] not in claimed_before)
        if int(r["row_index"]) != int(best):
            problems.append(f"Y[{i + 1}] is not the best available row on its component")
        claimed_before.add(r["gene_name"])
    return problems


def plot_coreset(result: fs.MFAResult, picks: pd.DataFrame, anchors: set[str], path: Path) -> None:
    """Draw the pool with anchors and both stages' picks."""
    scores, meta = result.scores, result.matrix.meta
    is_anchor = meta["gene_name"].isin(anchors).to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 13.6))
    for ax, (a, b) in zip(axes, ((0, 1), (2, 3))):
        first = (a, b) == (0, 1)
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
                    rasterized=True,
                    label=t if first else None,
                )
        ax.scatter(
            scores[is_anchor, a],
            scores[is_anchor, b],
            facecolors="none",
            edgecolors="black",
            s=48,
            linewidths=1.3,
            zorder=4,
            label="anchors (22, static)" if first else None,
        )
        for name, (color, marker, label) in SET_STYLE.items():
            idx = picks.loc[picks["set"] == name, "row_index"].to_numpy()
            if not len(idx):
                continue
            ax.scatter(
                scores[idx, a],
                scores[idx, b],
                c=color,
                marker=marker,
                s=95,
                edgecolors="white",
                linewidths=0.8,
                zorder=6,
                label=label if first else None,
            )
        ax.set_xlabel(f"PC{a + 1}")
        ax.set_ylabel(f"PC{b + 1}")
        ax.set_title(f"PC{a + 1} vs PC{b + 1}")
    axes[0].legend(frameon=False, fontsize=8, markerscale=1.3, loc="upper left")
    fig.suptitle(
        f"cheeseman50 coreset — 22 anchors + {len(picks)} picks — {result.matrix.name}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(
    result: fs.MFAResult,
    picks: pd.DataFrame,
    final: pd.DataFrame,
    anchors: set[str],
    problems: list[str],
    path: Path,
) -> str:
    """Write the markdown read-out and return it."""
    pool = result.matrix.meta
    comp = (
        pd.DataFrame(
            {
                "final_50": final["orf_type"].value_counts(),
                "anchors": final[final["source"] == "anchor"]["orf_type"].value_counts(),
                "rare_fill": picks[picks["stage"] == "rare_fill"]["orf_type"].value_counts(),
                "sampler": picks[picks["stage"] == "sampler"]["orf_type"].value_counts(),
                "pool": pool["orf_type"].value_counts(),
            }
        )
        .fillna(0)
        .astype(int)
    )
    comp["pool_pct"] = (comp["pool"] / comp["pool"].sum() * 100).round(1)

    sampler = picks[picks["stage"] == "sampler"]
    lines = [
        "# cheeseman50 coreset",
        "",
        f"22 static anchors ({len(anchors)} genes) + {len(picks)} picks = {len(final)} isoforms.",
        f"Space: {result.matrix.name}, {result.scores.shape[0]:,} × {result.scores.shape[1]}.",
        "",
        f"**Verification: {'PASS' if not problems else 'FAIL'}**",
        "",
    ]
    if problems:
        lines += ["```", *problems, "```", ""]
    lines += [
        "## ORF-type composition",
        "",
        "```",
        comp.to_string(),
        "```",
        "",
        "## Fallback depth",
        "",
        "How far past its own first choice the sampler had to reach because a gene "
        "was already claimed. Mostly 0 means the constraints barely perturbed the "
        "method.",
        "",
        "```",
        sampler["rank_used"].value_counts().sort_index().rename("slots").to_string(),
        f"\nmax fallback depth: {int(sampler['rank_used'].max())}",
        "```",
        "",
        "## The picks",
        "",
        "```",
        picks[
            [
                "stage",
                "set",
                "component",
                "rank",
                "rank_used",
                "gene_name",
                "orf_type",
                "value",
                "inf_norm",
            ]
        ]
        .round(3)
        .to_string(index=False),
        "```",
        "",
    ]
    text = "\n".join(lines)
    path.write_text(text)
    return text


def load_anchors() -> set[str]:
    """The 12 hand-curated anchor genes."""
    return set(fs.ANCHOR_GENES)


def main() -> None:
    """Build the coreset and write the outputs."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default=None, help="all_paired.parquet path or glob")
    args = ap.parse_args()

    anchors = load_anchors()
    print(f"anchors: {len(anchors)} genes")

    matrix = fs.build_matrices(args.parquet)["all-ORF"]
    result = fs.fit_mfa(matrix)
    print(fs.summarize(result))

    picks, final = build_coreset(result, anchors)
    problems = verify(picks, final, anchors, result)
    print("\nverify:", problems or "OK")

    picks.to_csv(HERE / "coreset_selection.csv", index=False)
    final.to_csv(HERE / "coreset_50.csv", index=False)
    plot_coreset(result, picks, anchors, HERE / "coreset_selection.png")
    print()
    print(
        write_report(result, picks, final, anchors, problems, HERE / "coreset_selection_report.md")
    )
    print(f"\nwrote coreset_selection.csv, coreset_50.csv, figure and report to {HERE}")


if __name__ == "__main__":
    main()
