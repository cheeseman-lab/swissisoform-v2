"""Diagnostic: long-read read distribution at divergent source-resolution sites.

For picking the divergent-case dominance threshold
(``SourceResolutionConfig.divergence_dominance_frac``) empirically. For one
sample it re-runs the front of the source-resolution cascade (long-read filter →
window-purity) and, for every site whose surviving candidates diverge in-window,
reports how that sample's long-read reads split across the candidate transcripts.

Outputs:
  1. a per-site CSV (``--out``);
  2. a quantile summary of the top-fraction to stdout;
  3. an optional 100%-stacked-bar PNG (``--plot``) — one bar per divergent TIS,
     each bar split into per-candidate read-percentage segments, colored by
     within-site rank, bars sorted by top fraction.

Usage:
    python scripts/export/export_source_divergence_distribution.py --sample HeLa \
        --plot data/output/diagnostics/HeLa_divergence.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from swissisoform.io.gtf import load_exon_skeletons
from swissisoform.io.rnaseq import load_sample_manifest
from swissisoform.runner import (
    GENOME,
    GTF,
    REPLICATE_MANIFEST,
    SAMPLE_MANIFEST,
    _manifest_single_path,
)
from swissisoform.sourceresolve.diagnostics import divergent_site_distribution

ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_inputs(sample: str) -> tuple[Path, Path]:
    """Return ``(filtered_csv, isoquant_table)`` for *sample* from the manifest."""
    manifest = load_sample_manifest(SAMPLE_MANIFEST, REPLICATE_MANIFEST)
    rows = manifest[manifest["sample"] == sample]
    if rows.empty:
        raise SystemExit(f"sample {sample!r} not in manifest {SAMPLE_MANIFEST}")
    row = rows.iloc[0]
    filtered_csv = ROOT / row["filtered_file"]
    isoquant = _manifest_single_path(row, "isoquant_table")
    if isoquant is None:
        raise SystemExit(
            f"sample {sample!r} has no (existing) isoquant_table in the manifest — "
            "source-resolution diagnostics need long-read counts"
        )
    if not filtered_csv.exists():
        raise SystemExit(f"filtered table not found: {filtered_csv} (run the pipeline first)")
    return filtered_csv, isoquant


def _print_summary(dist: pd.DataFrame) -> None:
    """Print the top-fraction quantile summary that informs the threshold."""
    n = len(dist)
    print(f"divergent sites: {n}")
    if n == 0:
        return
    tf = dist["top_fraction"]
    print("\ntop_fraction summary:")
    print(tf.describe().to_string())
    print("\ntop_fraction deciles:")
    deciles = tf.quantile([i / 10 for i in range(11)])
    print(deciles.to_string())


def _plot(dist: pd.DataFrame, out: Path, max_bars: int) -> None:
    """100%-stacked bar per divergent TIS, segments = per-candidate read %."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ordered = dist.sort_values("top_fraction", ascending=False).reset_index(drop=True)
    if max_bars and len(ordered) > max_bars:
        # Subsample EVENLY across the sorted range (not just the head) so the bars
        # span the full dominance spectrum from most-dominant to most-ambiguous.
        import numpy as np

        idx = np.linspace(0, len(ordered) - 1, max_bars).round().astype(int)
        print(
            f"note: plotting {max_bars} of {len(ordered)} divergent sites, "
            "evenly sampled across the dominance range"
        )
        ordered = ordered.iloc[idx].reset_index(drop=True)

    max_rank = max((len(f) for f in ordered["fractions"]), default=0)
    cmap = plt.get_cmap("viridis", max(max_rank, 1))
    n = len(ordered)
    x = range(n)

    # Contiguous bars (width=1, no edge) and a width-capped figure so thousands
    # of sites fit; for small n keep a comfortable per-bar width.
    fig_w = min(max(8.0, n * 0.12), 48.0)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    bottoms = [0.0] * n
    for rank in range(max_rank):
        heights = [
            (fr[rank] * 100.0 if rank < len(fr) else 0.0) for fr in ordered["fractions"]
        ]
        ax.bar(x, heights, bottom=bottoms, color=cmap(rank), width=1.0, linewidth=0)
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    # Horizontal reference lines at 20/40/60/80% — candidate dominance thresholds.
    # The rank-1 (bottom) segment height IS the top transcript's read share, so a
    # line at y=60 crosses the bars exactly where top_fraction drops below 0.60.
    for thr in (20, 40, 50, 60, 80):
        ax.axhline(thr, color="white", linestyle="--", linewidth=1.0, alpha=0.9, zorder=5)
        ax.text(
            n - 0.5, thr, f" {thr}%", va="center", ha="left", fontsize="small",
            color="0.2", clip_on=False, zorder=6,
        )

    ax.set_ylabel("% of long-read reads at the TIS")
    ax.set_xlabel("divergent TIS (sorted by dominance of the top transcript)")
    ax.set_xlim(-0.5, len(ordered) - 0.5)
    ax.set_ylim(0, 100)

    # X ticks as a percentage of the divergent sites (left = most dominant).
    if n > 1:
        fracs = [i / 20 for i in range(21)]  # every 5% of the divergent sites
        ax.set_xticks([f * (n - 1) for f in fracs])
        ax.set_xticklabels([f"{int(f * 100)}%" for f in fracs])
        ax.set_xlabel(
            "divergent TIS, sorted by dominance "
            "(x = % of divergent sites, left = most dominant)"
        )
    else:
        ax.set_xticks([])
    ax.legend(
        handles=[Patch(color=cmap(r), label=f"rank {r + 1}") for r in range(max_rank)],
        title="within-TIS transcript",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize="small",
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote plot: {out}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--sample", default="HeLa", help="Cell line in the sample manifest")
    p.add_argument(
        "--window-upstream", type=int, default=100, help="Window bound: nt 5' of start codon"
    )
    p.add_argument(
        "--window-downstream", type=int, default=100, help="Window bound: nt 3' of start codon"
    )
    p.add_argument(
        "--isoquant-min-count", type=float, default=3.0, help="Long-read presence threshold"
    )
    p.add_argument(
        "--out", type=Path, default=None, help="Per-site CSV (default under data/output)"
    )
    p.add_argument("--plot", type=Path, default=None, help="Stacked-bar PNG output path")
    p.add_argument("--max-bars", type=int, default=200, help="Max bars per plot (0 = no cap)")
    args = p.parse_args(argv)

    filtered_csv, isoquant = _resolve_inputs(args.sample)
    filtered_df = pd.read_csv(filtered_csv)
    skeletons = load_exon_skeletons(str(GTF))

    import pysam

    genome = pysam.FastaFile(str(GENOME))
    try:
        dist = divergent_site_distribution(
            filtered_df,
            exon_skeletons=skeletons,
            genome=genome,
            isoquant_table=isoquant,
            window_upstream=args.window_upstream,
            window_downstream=args.window_downstream,
            isoquant_min_count=args.isoquant_min_count,
        )
    finally:
        genome.close()

    out = args.out or (ROOT / "data" / "output" / "diagnostics" / f"{args.sample}_divergence.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    dist.to_csv(out, index=False)
    print(f"wrote per-site CSV: {out}  ({len(dist)} divergent sites)")
    _print_summary(dist)
    if args.plot:
        _plot(dist, args.plot, args.max_bars)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
