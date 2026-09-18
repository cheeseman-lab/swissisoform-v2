#!/usr/bin/env python
"""Plot where genome-wide isoforms fall against each CDLMPS scoring cutoff.

Reads the sharded ``all_paired.parquet`` output of a genome-wide run (one row per
alternative isoform) and, for every scored criterion, histograms the raw value the
criterion thresholds with the live cutoff drawn on top. This is the evidence base
for replacing the thresholds still tagged ``CALIBRATE ON GENOME-WIDE RUN —
provisional`` in ``ScoringConfig``: a cutoff that fires on ~100% or ~0% of
isoforms carries no information, which is how S3's old presence check was caught.

Cutoffs are read live from ``ScoringConfig`` so the figures cannot drift from
``config.py``. Note this is deliberately NOT the same as the criterion booleans
stored in the parquet: those were scored with whatever config the run used, so a
threshold changed since then (S3) shows a different pass rate here. Both are
reported side by side in the summary CSV.

``fires_when`` semantics, ORF-type asymmetries and the not-evaluable conditions
for each criterion are documented in ``scoring_cutoffs.csv`` (written by
``export_scoring_cutoffs.py`` in this directory).

Outputs (alongside this script):
  - cutoff_distributions.png       one histogram panel per continuous cutoff
  - criteria_outcomes.png          True / False / not-evaluable per criterion
  - cutoff_distributions_summary.csv   n, quantiles, live pass% vs as-run fired%
  - <key>.png                      per-field panels, with --individual

Usage:
    python figures/scoring_cutoffs/plot_cutoff_distributions.py
    python figures/scoring_cutoffs/plot_cutoff_distributions.py --individual
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pyarrow.parquet as pq

from swissisoform.config import ScoringConfig

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

DEFAULT_SHARDS = str(ROOT / "data" / "output" / "full_catalog_shard_*" / "all_paired.parquet")

# Cell lines the pipeline can carry. Not every shard has every one (RPE1_Async /
# RPE1_Sen are absent from 2 of 117), so column selection is always intersected
# with the file's own schema.
SAMPLES = ("HeLa", "K562", "U2OS", "RPE1_Async", "RPE1_Que", "RPE1_Sen")

# Criterion booleans as scored AT RUN TIME, nested in this struct column.
CRITERIA_STRUCT = "isoform_scoring_criteria"


# ---------------------------------------------------------------------------
# Transforms — how each criterion's scored quantity comes out of the parquet
# ---------------------------------------------------------------------------


def _num(name: str) -> Callable[[pd.DataFrame], pd.Series]:
    """Return a transform reading one numeric column."""
    return lambda df: pd.to_numeric(df[name], errors="coerce")


def _abs_num(name: str) -> Callable[[pd.DataFrame], pd.Series]:
    """Return a transform reading one numeric column as a magnitude."""
    return lambda df: pd.to_numeric(df[name], errors="coerce").abs()


def _n_cell_lines(df: pd.DataFrame) -> pd.Series:
    """Count cell lines with a detection for this TIS (D1's ``len(site.expression)``)."""
    cols = [c for c in (f"expr_{s}_raw_count" for s in SAMPLES) if c in df.columns]
    return df[cols].notna().sum(axis=1).astype(float)


def _max_initiation_efficiency(df: pd.DataFrame) -> pd.Series:
    """Take the best initiation efficiency across cell lines (D2 scores the max)."""
    cols = [c for c in (f"expr_{s}_initiation_efficiency" for s in SAMPLES) if c in df.columns]
    return df[cols].apply(pd.to_numeric, errors="coerce").max(axis=1)


def _n_validated_unique_peptides(df: pd.DataFrame) -> pd.Series:
    """Count isoform-unique PepQuery-validated peptides (D3).

    Mirrors ``evidence/d3_mass_spec``: strict ``is True`` on both flags, so an
    unknown (None) never counts. NaN when PepQuery did not run for the gene —
    that is D3's not-evaluable case, not a zero.
    """
    out: list[float] = []
    for hits, summary in zip(df["isoform_massspec_hits"], df["isoform_massspec_summary"]):
        if not isinstance(summary, dict) or not summary.get("pepquery_run"):
            out.append(float("nan"))
        elif hits is None:
            out.append(0.0)
        else:
            out.append(
                float(
                    sum(
                        1
                        for h in hits
                        if h.get("unique_to_isoform") is True and h.get("validated") is True
                    )
                )
            )
    return pd.Series(out, index=df.index)


def _min_shared_plddt(df: pd.DataFrame) -> pd.Series:
    """Take the weaker of the two shared-region pLDDT means (P2's confidence gate)."""
    pair = df[
        [
            "isoform_structure_plddt_shared_mean_isoform",
            "isoform_structure_plddt_shared_mean_canonical",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    return pair.min(axis=1)


def _sae_top_delta(df: pd.DataFrame) -> pd.Series:
    """Take the strongest shared-feature activation shift (S3)."""
    pair = df[["isoform_sae_top_gained_delta_max", "isoform_sae_top_lost_delta_max"]].apply(
        pd.to_numeric, errors="coerce"
    )
    return pair.abs().max(axis=1)


# ---------------------------------------------------------------------------
# Series registry — one entry per plottable quantity
# ---------------------------------------------------------------------------
#
# ``config_field`` resolves live against ScoringConfig; ``literal`` is only for
# S1, whose ">= 1" is hardcoded in the scorer and has no config field.
# ``direction`` is "ge" everywhere except M1's depletion branch, which is a
# strict "<". ``scale`` is "log" for the fields spanning orders of magnitude.

SERIES: list[dict[str, Any]] = [
    {
        "key": "C1_primate_mean_pident",
        "criterion": "C1_primate_conservation",
        "label": "C1  primate mean AA %identity",
        "columns": ["isoform_conservation_frame_primate_mean_pident"],
        "transform": _num("isoform_conservation_frame_primate_mean_pident"),
        "config_field": "c1_pident_min",
        "direction": "ge",
        "scale": "linear",
    },
    {
        "key": "C2_mammalian_mean_pident",
        "criterion": "C2_mammalian_conservation",
        "label": "C2  mammalian mean AA %identity",
        "columns": ["isoform_conservation_frame_mammalian_mean_pident"],
        "transform": _num("isoform_conservation_frame_mammalian_mean_pident"),
        "config_field": "c2_pident_min",
        "direction": "ge",
        "scale": "linear",
    },
    {
        "key": "C3_phylop_unique_region_mean",
        "criterion": "C3_phylop_coding_selection",
        "label": "C3  PhyloP mean, unique region",
        "columns": ["isoform_conservation_phylop_unique_region_mean"],
        "transform": _num("isoform_conservation_phylop_unique_region_mean"),
        "config_field": "c3_phylop_min",
        "direction": "ge",
        "scale": "linear",
    },
    {
        "key": "D1_n_cell_lines",
        "criterion": "D1_multi_cell_line",
        "label": "D1  cell lines detecting the TIS",
        "columns": [f"expr_{s}_raw_count" for s in SAMPLES],
        "transform": _n_cell_lines,
        "config_field": "min_cell_lines",
        "direction": "ge",
        "scale": "linear",
        "discrete": True,
    },
    {
        "key": "D2_initiation_efficiency",
        "criterion": "D2_initiation_efficiency",
        "label": "D2  best initiation efficiency",
        "columns": [f"expr_{s}_initiation_efficiency" for s in SAMPLES],
        "transform": _max_initiation_efficiency,
        "config_field": "initiation_efficiency_min",
        "direction": "ge",
        "scale": "log",
    },
    {
        "key": "D3_validated_unique_peptides",
        "criterion": "D3_mass_spec",
        "label": "D3  validated isoform-unique peptides",
        "columns": ["isoform_massspec_hits", "isoform_massspec_summary"],
        "transform": _n_validated_unique_peptides,
        "config_field": "massspec_unique_peptides_min",
        "direction": "ge",
        "scale": "linear",
        "discrete": True,
    },
    {
        "key": "M1_constraint_enrichment",
        "criterion": "M1_pathogenic_variant_enrichment",
        "label": "M1a  ESM-C constraint enrichment",
        "columns": ["isoform_plm_vep_constraint_enrichment"],
        "transform": _num("isoform_plm_vep_constraint_enrichment"),
        "config_field": "m1_constraint_enrichment_min",
        "direction": "ge",
        "scale": "log",
    },
    {
        "key": "M1_gnomad_depletion_ratio",
        "criterion": "M1_pathogenic_variant_enrichment",
        "label": "M1b  gnomAD depletion ratio",
        "columns": ["isoform_variant_intersection_gnomad_depletion_ratio"],
        "transform": _num("isoform_variant_intersection_gnomad_depletion_ratio"),
        "config_field": "m1_depletion_ratio_max",
        "direction": "lt",
        "scale": "linear",
    },
    {
        "key": "M2_disease_enrichment_ratio",
        "criterion": "M2_clinical_variant_overlap",
        "label": "M2  disease-variant density ratio",
        "columns": ["isoform_variant_intersection_disease_enrichment_ratio"],
        "transform": _num("isoform_variant_intersection_disease_enrichment_ratio"),
        "config_field": "m2_disease_enrichment_min",
        "direction": "ge",
        "scale": "linear",
    },
    {
        "key": "P1_plddt_diffregion_mean",
        "criterion": "P1_structured_extension",
        "label": "P1  diff-region mean pLDDT",
        "columns": ["isoform_structure_plddt_diffregion_mean"],
        "transform": _num("isoform_structure_plddt_diffregion_mean"),
        "config_field": "p1_plddt_threshold",
        "direction": "ge",
        "scale": "linear",
    },
    {
        "key": "P2_rmsd_shared",
        "criterion": "P2_shared_structural_change",
        "label": "P2  shared-region Ca RMSD (A)",
        "columns": ["isoform_structure_rmsd_shared"],
        "transform": _num("isoform_structure_rmsd_shared"),
        "config_field": "p2_rmsd_shared_min",
        "direction": "ge",
        "scale": "log",
    },
    {
        "key": "P2_shared_region_len",
        "criterion": "P2_shared_structural_change",
        "label": "P2 gate  shared-region length (aa)",
        "columns": ["isoform_structure_shared_region_len"],
        "transform": _num("isoform_structure_shared_region_len"),
        "config_field": "p2_min_shared_len",
        "direction": "ge",
        "scale": "log",
    },
    {
        "key": "P2_min_shared_plddt",
        "criterion": "P2_shared_structural_change",
        "label": "P2 gate  min shared-region pLDDT",
        "columns": [
            "isoform_structure_plddt_shared_mean_isoform",
            "isoform_structure_plddt_shared_mean_canonical",
        ],
        "transform": _min_shared_plddt,
        "config_field": "p2_plddt_min",
        "direction": "ge",
        "scale": "linear",
    },
    {
        "key": "S1_domains_changed",
        "criterion": "S1_domain_change",
        "label": "S1  real domains changed in diff region",
        "columns": ["cmp_interproscan_n_real_domains_changed_in_diff_region"],
        "transform": _num("cmp_interproscan_n_real_domains_changed_in_diff_region"),
        "config_field": None,
        "literal": 1,
        "direction": "ge",
        "scale": "linear",
        "discrete": True,
        "note": "cutoff hardcoded in the scorer, no config field",
    },
    {
        "key": "S2_gravy_delta",
        "criterion": "S2_biophysics",
        "label": "S2a  |GRAVY delta|",
        "columns": ["cmp_biophysics_gravy_delta"],
        "transform": _abs_num("cmp_biophysics_gravy_delta"),
        "config_field": "s2_gravy_delta_min",
        "direction": "ge",
        "scale": "linear",
    },
    {
        "key": "S2_fraction_charged_delta",
        "criterion": "S2_biophysics",
        "label": "S2b  |fraction-charged delta|",
        "columns": ["cmp_biophysics_fraction_charged_delta"],
        "transform": _abs_num("cmp_biophysics_fraction_charged_delta"),
        "config_field": "s2_fraction_charged_delta_min",
        "direction": "ge",
        "scale": "linear",
    },
    {
        "key": "S2_disorder_delta",
        "criterion": "S2_biophysics",
        "label": "S2c  |disorder delta|",
        "columns": ["cmp_biophysics_disorder_delta"],
        "transform": _abs_num("cmp_biophysics_disorder_delta"),
        "config_field": "s2_disorder_delta_min",
        "direction": "ge",
        "scale": "linear",
    },
    {
        "key": "S3_top_shared_feature_delta",
        "criterion": "S3_sae",
        "label": "S3  top shared-feature |delta|",
        "columns": ["isoform_sae_top_gained_delta_max", "isoform_sae_top_lost_delta_max"],
        "transform": _sae_top_delta,
        "config_field": "s3_top_delta_min",
        "direction": "ge",
        "scale": "linear",
    },
]

# Criteria with no numeric axis, or with no data in this run. Logged explicitly
# so a reader never mistakes an absent panel for an empty distribution.
SKIPPED = {
    "L1_localization_change": "categorical (any *_changed is True) — see criteria_outcomes.png",
    "L2_targeting_change": "categorical (any *_changed is True) — see criteria_outcomes.png",
    "P3_secondary_structure": "no sse_* columns in this run — needs a rerun to plot",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def shard_files(pattern: str) -> list[Path]:
    """Return the shard parquet paths matching *pattern*, sorted."""
    files = sorted(Path(p) for p in glob.glob(pattern))
    if not files:
        raise SystemExit(
            f"no parquet files matched {pattern!r}\n"
            "Point --shards at a genome-wide run's all_paired.parquet shards."
        )
    return files


def load_frame(files: list[Path], columns: list[str]) -> pd.DataFrame:
    """Concatenate *columns* across shards, tolerating per-shard schema drift.

    Shards do not all carry the same columns (a cell line absent from every TIS
    in a shard yields no ``expr_{sample}_*`` fields at all), so the request is
    intersected with each file's own schema and missing columns come back as NaN.
    """
    frames = []
    for path in files:
        present = {f.name for f in pq.ParquetFile(path).schema_arrow}
        want = [c for c in columns if c in present]
        frames.append(pd.read_parquet(path, columns=want))
    df = pd.concat(frames, ignore_index=True)
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def as_run_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Expand the run-time criterion struct into a True/False/None frame."""
    if CRITERIA_STRUCT not in df.columns:
        return pd.DataFrame(index=df.index)
    return pd.DataFrame(list(df[CRITERIA_STRUCT]), index=df.index)


# ---------------------------------------------------------------------------
# Stats + plotting
# ---------------------------------------------------------------------------


def series_stats(values: pd.Series, cutoff: float, direction: str) -> dict[str, Any]:
    """Summarize *values* against *cutoff*, ignoring the criterion's own gates."""
    v = values.dropna()
    passing = (v < cutoff) if direction == "lt" else (v >= cutoff)
    q = v.quantile([0.05, 0.25, 0.5, 0.75, 0.95]) if len(v) else {}
    return {
        "n": int(len(v)),
        "n_missing": int(values.isna().sum()),
        "p05": float(q[0.05]) if len(v) else None,
        "p25": float(q[0.25]) if len(v) else None,
        "median": float(q[0.5]) if len(v) else None,
        "p75": float(q[0.75]) if len(v) else None,
        "p95": float(q[0.95]) if len(v) else None,
        "raw_pass_pct": round(100.0 * passing.mean(), 1) if len(v) else None,
    }


def _draw_panel(ax: Any, values: pd.Series, spec: dict[str, Any], cutoff: float) -> None:
    """Draw one histogram with its cutoff line and the passing side shaded."""
    import numpy as np

    v = values.dropna()
    dropped = 0
    if spec["scale"] == "log":
        dropped = int((v <= 0).sum())
        v = v[v > 0]
    if v.empty:
        ax.set_axis_off()
        return

    lo, hi = float(v.quantile(0.005)), float(v.quantile(0.995))
    lo, hi = min(lo, cutoff), max(hi, cutoff)
    if spec["scale"] == "log":
        lo = max(lo, 1e-9)
        hi = max(hi, lo * 10)
        bins = np.logspace(np.log10(lo), np.log10(hi), 41)
    elif spec.get("discrete"):
        hi = max(hi, lo + 1)
        bins = np.arange(np.floor(lo) - 0.5, np.ceil(hi) + 1.5, 1.0)
    else:
        hi = hi if hi > lo else lo + 1.0
        bins = np.linspace(lo, hi, 41)

    n_clipped = int(((v < lo) | (v > hi)).sum())
    ax.hist(v.clip(lo, hi), bins=bins, color="#4C78A8", edgecolor="white", linewidth=0.3)
    if spec["direction"] == "lt":
        ax.axvspan(lo, cutoff, color="#54A24B", alpha=0.12, lw=0)
    else:
        ax.axvspan(cutoff, hi, color="#54A24B", alpha=0.12, lw=0)
    ax.axvline(cutoff, color="#E45756", ls="--", lw=1.6)
    if spec["scale"] == "log":
        from matplotlib.ticker import NullFormatter

        ax.set_xscale("log")
        # Minor decade labels (2x10^1, 3x10^1, ...) collide on narrow panels.
        ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(lo, hi)

    footnotes = []
    if n_clipped:
        footnotes.append(f"{n_clipped} clipped into edge bins")
    if dropped:
        footnotes.append(f"{dropped} non-positive dropped (log axis)")
    if spec.get("note"):
        footnotes.append(spec["note"])
    if footnotes:
        ax.text(
            0.02,
            0.98,
            "\n".join(footnotes),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=6,
            color="#666666",
        )


def _panel_title(spec: dict[str, Any], cutoff: float, stats: dict[str, Any]) -> str:
    """Compose the per-panel title carrying n, median and the live pass rate."""
    rel = "<" if spec["direction"] == "lt" else ">="
    med = "n/a" if stats["median"] is None else f"{stats['median']:.3g}"
    pct = "n/a" if stats["raw_pass_pct"] is None else f"{stats['raw_pass_pct']:.1f}%"
    return (
        f"{spec['label']}\n"
        f"cutoff {rel} {cutoff:g}   n={stats['n']:,}   median {med}   pass {pct}"
    )


def plot_grid(data: dict[str, pd.Series], specs: list[dict[str, Any]], cfg: ScoringConfig,
              stats: dict[str, dict[str, Any]], out_path: Path, subtitle: str) -> None:
    """Write the multi-panel figure, one histogram per cutoff."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = 4
    nrows = -(-len(specs) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.2 * nrows))
    axes = axes.ravel()
    for ax, spec in zip(axes, specs):
        cutoff = _cutoff(spec, cfg)
        _draw_panel(ax, data[spec["key"]], spec, cutoff)
        ax.set_title(_panel_title(spec, cutoff, stats[spec["key"]]), fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.set_ylabel("isoforms", fontsize=7)
    for ax in axes[len(specs) :]:
        ax.set_axis_off()
    fig.suptitle(
        "CDLMPS scoring cutoffs vs the genome-wide isoform distribution", fontsize=13, y=0.997
    )
    fig.text(0.5, 0.978, subtitle, ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.970))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_outcomes(outcomes: pd.DataFrame, out_path: Path, subtitle: str) -> None:
    """Write the True / False / not-evaluable breakdown scored at run time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(outcomes.columns)
    n = len(outcomes)
    true_ = [(outcomes[c] == True).sum() for c in names]  # noqa: E712 - NA-safe
    false_ = [(outcomes[c] == False).sum() for c in names]  # noqa: E712 - NA-safe
    none_ = [outcomes[c].isna().sum() for c in names]

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(names) + 2.2))
    y = range(len(names))
    ax.barh(y, true_, color="#54A24B", label="True (fired)")
    ax.barh(y, false_, left=true_, color="#B0B7BF", label="False")
    ax.barh(
        y,
        none_,
        left=[t + f for t, f in zip(true_, false_)],
        color="#E45756",
        alpha=0.55,
        label="not evaluable",
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"isoforms (n={n:,})", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Criterion outcomes as scored at run time", fontsize=12)
    fig.text(0.5, 0.965, subtitle, ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_individual(data: dict[str, pd.Series], specs: list[dict[str, Any]], cfg: ScoringConfig,
                    stats: dict[str, dict[str, Any]], out_dir: Path) -> list[Path]:
    """Write one standalone PNG per cutoff; returns the paths written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    for spec in specs:
        cutoff = _cutoff(spec, cfg)
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        _draw_panel(ax, data[spec["key"]], spec, cutoff)
        ax.set_title(_panel_title(spec, cutoff, stats[spec["key"]]), fontsize=9)
        ax.set_ylabel("isoforms", fontsize=8)
        fig.tight_layout()
        path = out_dir / f"cutoff_{spec['key']}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _cutoff(spec: dict[str, Any], cfg: ScoringConfig) -> float:
    """Resolve a series' cutoff from the live config, or its hardcoded literal."""
    field = spec.get("config_field")
    return float(getattr(cfg, field)) if field else float(spec["literal"])


def build_summary(specs: list[dict[str, Any]], cfg: ScoringConfig,
                  stats: dict[str, dict[str, Any]], outcomes: pd.DataFrame) -> pd.DataFrame:
    """Assemble the per-series summary table, live pass% beside as-run fired%."""
    rows = []
    n_total = len(outcomes) if len(outcomes) else 0
    for spec in specs:
        st = stats[spec["key"]]
        crit = spec["criterion"]
        row = {
            "series_key": spec["key"],
            "criterion_id": crit,
            "label": spec["label"].replace("\n", " "),
            "config_field": spec.get("config_field") or "(hardcoded)",
            "cutoff": _cutoff(spec, cfg),
            "direction": "<" if spec["direction"] == "lt" else ">=",
            **st,
            "as_run_true": None,
            "as_run_evaluable": None,
            "as_run_fired_pct": None,
            "as_run_evaluable_pct": None,
        }
        if crit in outcomes.columns:
            col = outcomes[crit]
            n_true = int((col == True).sum())  # noqa: E712 - NA-safe
            n_eval = int(col.notna().sum())
            row["as_run_true"] = n_true
            row["as_run_evaluable"] = n_eval
            row["as_run_fired_pct"] = round(100.0 * n_true / n_eval, 1) if n_eval else None
            row["as_run_evaluable_pct"] = round(100.0 * n_eval / n_total, 1) if n_total else None
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    """Load the shards, compute each cutoff's distribution, and write the figures."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--shards",
        default=DEFAULT_SHARDS,
        help="glob for the run's all_paired.parquet shards (default: full_catalog)",
    )
    p.add_argument(
        "--out", type=Path, default=None, help="output directory (default: next to this script)"
    )
    p.add_argument(
        "--individual", action="store_true", help="also write one PNG per cutoff field"
    )
    args = p.parse_args(argv)

    out_dir = args.out or HERE
    out_dir.mkdir(parents=True, exist_ok=True)

    files = shard_files(args.shards)
    columns = sorted({c for spec in SERIES for c in spec["columns"]} | {CRITERIA_STRUCT})
    df = load_frame(files, columns)
    print(f"loaded {len(df):,} isoform rows from {len(files)} shard(s)")

    cfg = ScoringConfig()
    data = {spec["key"]: spec["transform"](df) for spec in SERIES}
    stats = {
        spec["key"]: series_stats(data[spec["key"]], _cutoff(spec, cfg), spec["direction"])
        for spec in SERIES
    }
    outcomes = as_run_outcomes(df)

    subtitle = (
        f"{len(df):,} isoforms   cutoffs from ScoringConfig defaults   "
        "pass% ignores each criterion's own not-evaluable gates"
    )

    grid_path = out_dir / "cutoff_distributions.png"
    plot_grid(data, SERIES, cfg, stats, grid_path, subtitle)
    print(f"wrote {len(SERIES)} cutoff panels -> {grid_path}")

    if not outcomes.empty:
        outcomes_path = out_dir / "criteria_outcomes.png"
        plot_outcomes(
            outcomes,
            outcomes_path,
            f"{len(df):,} isoforms   scored with the config THIS RUN USED, not today's — "
            "a threshold changed since (S3) reads differently here than in the histograms",
        )
        print(f"wrote {outcomes.shape[1]} criterion outcome bars -> {outcomes_path}")

    summary = build_summary(SERIES, cfg, stats, outcomes)
    summary_path = out_dir / "cutoff_distributions_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {len(summary)} rows -> {summary_path}")

    if args.individual:
        written = plot_individual(data, SERIES, cfg, stats, out_dir)
        print(f"wrote {len(written)} individual panels -> {out_dir}")

    for name, why in SKIPPED.items():
        print(f"skipped {name}: {why}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
