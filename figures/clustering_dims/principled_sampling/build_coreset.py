#!/usr/bin/env python
"""Fill the remaining 28 slots of the cheeseman50 LLM-tuning coreset.

`cheeseman50` holds 22 curated anchor isoforms (12 genes) against a target of
50. The anchors are **static** — pinned by ``tis_id`` in ``anchors.csv``, never
re-derived, re-ranked or filtered; a missing anchor stops the build. This script
adds the other 28 in two ordered stages:

    22  anchors              (static, anchors.csv)
    12  rare-type fill       3 each: uorf, internal_oof, 3utr_orf, uoorf
    16  per-type sampler     8 extended + 8 truncated, each in its own space
    ──
    50

**Stage order is load-bearing.** The exploratory pass showed the unconstrained
sampler gives internal_oof / 3utr_orf / uoorf *zero* picks at every n — they are
0.3-0.9% of the pool and a top-1-per-component rule cannot reach them — and at
n=7 with anchors excluded it found only one uORF. Forcing all four rare types
first is the only way they appear at all. Equally, 44 pool genes carry both a
rare type and a common one, so the rare strata must claim their genes before the
samplers run or a gene gets picked twice and the set lands short.

chrY isoforms are out of the pool for every pick (``off_pool``): PAR genes sit on
chrY in this catalog while the cell lines are female.

Spaces. The rare types are filled in the **all-ORF** matrix (no shared-region
features: they have no shared region). Extended and truncated are each sampled
in their **own paired-ORF** matrix — every feature, the 83 unique-vs-shared
contrast features included, fitted on that type's rows alone. Those contrasts
drive C3/M1/M2/P1/P2, and the two types together are 93% of the pool, so
sampling them blind to it would choose the bulk of the set on axes the tags do
not evaluate. Two per-type strata rather than one pooled draw, so neither type
can crowd the other out of the extremes. The 16 slots split evenly.

Outputs (alongside this script):
  - coreset_selection.csv        the 28 picks, with the space each was drawn in
  - coreset_50.csv               the final 50 (anchors + picks)
  - coreset_provenance.json      source parquet(s) + sha256, catalog/anchors sha256,
                                 code commit, command — the panel's identity
  - coreset_selection.png        the picks on the all-ORF map  (written, untracked)

The read-out (composition, picks, coordinates) is printed to stdout, not written
beside them: it restates numbers already in the CSVs and the provenance sidecar.

Nothing is written if verification fails.

Usage:
    python figures/clustering_dims/principled_sampling/build_coreset.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "feature_space"))

import featurespace as fs  # noqa: E402
from export_feature_catalog import ROOT, resolve_parquet  # noqa: E402
from principled_sampler import principled_sample, ranked_candidates  # noqa: E402

HERE = Path(__file__).resolve().parent
ANCHORS_CSV = fs.ANCHORS_CSV

TARGET = 50
# The four types no global sampler reaches: each is 0.3-4.9% of the pool, so a
# top-1-per-component rule loses every axis to the extensions and truncations
# that dominate it. Missed for lack of COUNT, not lack of variance. Forced to 3
# each; uoorf already has 1 anchor (SKA3) so it lands at 4 — a minimum, not a
# target.
RARE_TYPES = ("uorf", "internal_oof", "3utr_orf", "uoorf")
# Sampled per type, each in its own paired-ORF matrix (see module docstring).
PAIRED_STRATA = fs.PAIRED_ORF_TYPES


def off_pool(tis_ids: np.ndarray) -> np.ndarray:
    """True for isoforms no pick may take: everything on chrY.

    PAR genes are annotated on chrY in this catalog (GTPBP6 was a pick) while the
    cell lines are female, so their position-based evidence — conservation,
    gnomAD, ClinVar, COSMIC — can be missing for reasons that are not biology.
    """
    return np.char.startswith(np.asarray(tis_ids, dtype=str), "chrY:")


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


def stratum_sizes(n_slots: int, strata: tuple[str, ...] = PAIRED_STRATA) -> dict[str, int]:
    """Split *n_slots* evenly across the paired strata, the remainder to the first."""
    base, extra = divmod(n_slots, len(strata))
    return {s: base + (1 if i < extra else 0) for i, s in enumerate(strata)}


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def load_anchors(path: Path = ANCHORS_CSV) -> pd.DataFrame:
    """The 22 hand-curated anchor isoforms, by stable ``tis_id``."""
    return pd.read_csv(path)


def _pick_rows(result: fs.MFAResult, picks: list[dict], stage: str, n_coord: int) -> pd.DataFrame:
    """One stage's picks as a frame, labelled with the space they came from."""
    meta = result.matrix.meta
    frame = pd.DataFrame(picks)
    idx = frame["row_index"].to_numpy()
    frame.insert(0, "gene_name", meta["gene_name"].to_numpy()[idx])
    frame.insert(1, "tis_id", meta["tis_id"].to_numpy()[idx])
    frame.insert(2, "orf_type", meta["orf_type"].to_numpy()[idx])
    frame["stage"] = stage
    frame["space"] = result.matrix.name
    for col in ("rank", "rank_used"):
        if col not in frame:
            frame[col] = 0 if col == "rank_used" else None
    width = min(n_coord, result.scores.shape[1])
    frame["inf_norm"] = [float(np.abs(result.scores[r, :width]).max()) for r in idx]
    # Per-space coordinates: a paired pick's PCs are not comparable to an
    # all-ORF pick's, which is why `space` travels with them.
    for c in range(width):
        frame[f"PC{c + 1}"] = result.scores[idx, c]
    return frame


def build_coreset(
    results: dict[str, fs.MFAResult], anchors: pd.DataFrame, n_coord: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run both stages and return ``(the 28 picks, the final 50)``.

    *results* maps a matrix name (``"all-ORF"``, ``"paired-<type>"``) to its MFA
    fit.

    Raises:
        SystemExit: An anchor ``tis_id`` is absent from the pool, or a rare-type
            pick lands on a gene already claimed.
    """
    all_orf = results["all-ORF"]
    meta = all_orf.matrix.meta
    tis = meta["tis_id"].to_numpy()
    anchor_ids = set(anchors["tis_id"])
    missing = sorted(anchor_ids - set(tis))
    if missing:
        raise SystemExit(f"{len(missing)} anchor tis_id(s) absent from the pool: {missing}")
    need = TARGET - len(anchor_ids)

    # Seeded with the anchor genes so no pick can land on a gene we already
    # carry, and extended after every pick so no gene is taken twice.
    claimed = set(anchors["gene_name"])
    frames: list[pd.DataFrame] = []

    genes = meta["gene_name"].to_numpy()
    orf = meta["orf_type"].to_numpy()
    is_anchor = np.isin(tis, list(anchor_ids))
    rare: list[dict] = []
    for orf_type in RARE_TYPES:
        # Anchors are pinned by tis_id, but a gene is still taken once: drop every
        # isoform of a claimed gene (the anchor genes, then each rare pick's), or
        # rare_type_fill — which does not consult `claimed` — can pick one and
        # abort the build below.
        taken = np.isin(genes, list(claimed))
        stratum = np.flatnonzero((orf == orf_type) & ~is_anchor & ~taken & ~off_pool(tis))
        for p in rare_type_fill(all_orf.scores, stratum):
            gene = genes[p["row_index"]]
            if gene in claimed:
                raise SystemExit(f"{orf_type}: {gene} already claimed — needs a fallback")
            claimed.add(gene)
            rare.append({**p, "rank": None, "rank_used": 0})
    frames.append(_pick_rows(all_orf, rare, "rare_fill", n_coord))

    for orf_type, k in stratum_sizes(need - len(rare)).items():
        result = results[f"paired-{orf_type}"]
        pool = result.matrix.meta
        pool_ids = pool["tis_id"].to_numpy()
        keep = ~np.isin(pool_ids, list(anchor_ids)) & ~off_pool(pool_ids)
        rows = np.flatnonzero(keep)
        n = math.ceil(k / 3)
        local = sampler_fill(result.scores[rows], pool["gene_name"].to_numpy()[rows], claimed, n, k)
        for p in local:
            p["row_index"] = int(rows[p["row_index"]])
        frames.append(_pick_rows(result, local, "sampler", n_coord))

    picks = pd.concat(frames, ignore_index=True)
    anchor_rows = anchors[["gene_name", "tis_id", "orf_type"]].copy()
    anchor_rows["source"] = "anchor"
    picked_rows = picks[["gene_name", "tis_id", "orf_type", "stage"]].rename(
        columns={"stage": "source"}
    )
    final = pd.concat([anchor_rows, picked_rows], ignore_index=True)
    return picks, final


def verify(
    picks: pd.DataFrame,
    final: pd.DataFrame,
    anchors: pd.DataFrame,
    results: dict[str, fs.MFAResult],
) -> list[str]:
    """Check the curation rules and fidelity to the algorithm."""
    problems: list[str] = []
    anchor_genes = set(anchors["gene_name"])

    if len(final) != TARGET:
        problems.append(f"final set has {len(final)} rows, expected {TARGET}")
    if final["tis_id"].nunique() != len(final):
        problems.append("final set repeats a tis_id")
    if picks["gene_name"].nunique() != len(picks):
        problems.append("picks contain a repeated gene (rule 3)")
    if set(picks["gene_name"]) & anchor_genes:
        problems.append(f"picks land on anchor genes: {set(picks['gene_name']) & anchor_genes}")
    if off_pool(picks["tis_id"].to_numpy()).any():
        problems.append("picks include a chrY isoform")

    counts = final["orf_type"].value_counts()
    for t, c in counts.items():
        if c < 3:
            problems.append(f"orf_type {t} has only {c} isoforms (rule 2 wants >=3)")

    for orf_type, g in picks[picks["stage"] == "rare_fill"].groupby("orf_type"):
        if set(g["set"]) != {"Y", "Z", "W"}:
            problems.append(f"{orf_type}: rare fill is not one each of Y/Z/W")

    sampled = picks[picks["stage"] == "sampler"]
    for orf_type, k in stratum_sizes(len(sampled)).items():
        got = int((sampled["orf_type"] == orf_type).sum())
        if got != k:
            problems.append(f"stratum {orf_type}: {got} picks, expected {k}")

    # Fidelity: each accepted Y pick must be the best still-available row on its
    # component in its own space, i.e. the fallback advanced only as far as it
    # had to. Strata run in order, so each sees the genes the earlier ones took.
    claimed_before = anchor_genes | set(picks[picks["stage"] == "rare_fill"]["gene_name"])
    anchor_ids = set(anchors["tis_id"])
    for orf_type in PAIRED_STRATA:
        result = results[f"paired-{orf_type}"]
        pool = result.matrix.meta
        genes = pool["gene_name"].to_numpy()
        pool_ids = pool["tis_id"].to_numpy()
        eligible = ~np.isin(pool_ids, list(anchor_ids)) & ~off_pool(pool_ids)
        stratum = sampled[sampled["orf_type"] == orf_type]
        for _, r in stratum[stratum["set"] == "Y"].iterrows():
            i = int(r["component"]) - 1
            order = np.argsort(-result.scores[:, i], kind="stable")
            best = next(
                (row for row in order if eligible[row] and genes[row] not in claimed_before), None
            )
            if best is None or int(r["row_index"]) != int(best):
                problems.append(f"{orf_type} Y[{i + 1}] is not the best available row")
            claimed_before.add(r["gene_name"])
        claimed_before |= set(stratum["gene_name"])
    return problems


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def provenance(
    files: list[Path], results: dict[str, fs.MFAResult], final: pd.DataFrame, argv: list[str]
) -> dict:
    """What this panel was built from — the sidecar written next to the CSVs.

    The panel is frozen and handed to an LLM-tuning experiment; without this a
    rebuild against a different parquet or catalog is indistinguishable from it.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from swissisoform.setup._common import code_provenance, rel_to_root, sha256_file

    return {
        "artifact": "cheeseman50 coreset",
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(["build_coreset.py", *argv]),
        "code": code_provenance(),
        "source_parquet": [
            {"path": rel_to_root(p.resolve()), "sha256": sha256_file(p)} for p in files
        ],
        "feature_catalog": {
            "path": rel_to_root(fs.CATALOG_CSV),
            "sha256": sha256_file(fs.CATALOG_CSV),
        },
        "anchors": {"path": rel_to_root(ANCHORS_CSV), "sha256": sha256_file(ANCHORS_CSV)},
        "spaces": {
            name: {"isoforms": int(r.scores.shape[0]), "features": int(r.matrix.X.shape[1])}
            for name, r in results.items()
        },
        "composition": final["source"].value_counts().to_dict(),
        "orf_types": final["orf_type"].value_counts().to_dict(),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def plot_coreset(
    result: fs.MFAResult, picks: pd.DataFrame, anchors: pd.DataFrame, path: Path
) -> None:
    """Draw the pool on the all-ORF map with anchors and both stages' picks.

    Paired-space picks are placed by ``tis_id`` on the all-ORF map: their own
    coordinates live in a different space. matplotlib is imported here, not at
    module level: it is a dev extra, and the CSVs must not depend on it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores, meta = result.scores, result.matrix.meta
    row_of = {t: i for i, t in enumerate(meta["tis_id"])}
    is_anchor = meta["tis_id"].isin(set(anchors["tis_id"])).to_numpy()
    at = picks.assign(all_orf_row=picks["tis_id"].map(row_of))

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
            label=f"anchors ({int(is_anchor.sum())}, static)" if first else None,
        )
        for name, (color, marker, label) in SET_STYLE.items():
            idx = at.loc[at["set"] == name, "all_orf_row"].dropna().astype(int).to_numpy()
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
        f"cheeseman50 coreset — {int(is_anchor.sum())} anchors + {len(picks)} picks — "
        f"drawn on {result.matrix.name}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report_text(
    results: dict[str, fs.MFAResult],
    picks: pd.DataFrame,
    final: pd.DataFrame,
) -> str:
    """The read-out, for stdout.

    Not written to disk: a generated report restates the CSVs and goes stale the
    moment one of them is edited (CLAUDE.md, "No generated *.md reports").
    """
    pool = results["all-ORF"].matrix.meta
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
    spaces = "; ".join(
        f"{name} {r.scores.shape[0]:,} × {r.matrix.X.shape[1]}" for name, r in results.items()
    )
    lines = [
        "# cheeseman50 coreset",
        "",
        f"{int((final['source'] == 'anchor').sum())} static anchors + {len(picks)} picks "
        f"= {len(final)} isoforms.",
        f"Spaces: {spaces}.",
        "",
        "**Verification: PASS**",
        "",
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
                "space",
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
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Build the coreset and write the outputs — only if verification passes."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default=None, help="all_paired.parquet path or glob")
    args = ap.parse_args(argv)

    anchors = load_anchors()
    print(f"anchors: {len(anchors)} isoforms, {anchors['gene_name'].nunique()} genes")

    files = resolve_parquet(args.parquet)
    results = {name: fs.fit_mfa(m) for name, m in fs.build_matrices(args.parquet).items()}
    for result in results.values():
        print(fs.summarize(result))

    picks, final = build_coreset(results, anchors)
    problems = verify(picks, final, anchors, results)
    if problems:
        raise SystemExit("verify failed — nothing written:\n  " + "\n  ".join(problems))
    print("\nverify: OK")

    picks.to_csv(HERE / "coreset_selection.csv", index=False)
    final.to_csv(HERE / "coreset_50.csv", index=False)
    prov = provenance(files, results, final, sys.argv[1:] if argv is None else argv)
    (HERE / "coreset_provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    plot_coreset(results["all-ORF"], picks, anchors, HERE / "coreset_selection.png")
    print()
    print(report_text(results, picks, final))
    print(f"\nwrote coreset_selection.csv, coreset_50.csv, provenance and figure to {HERE}")


if __name__ == "__main__":
    main()
