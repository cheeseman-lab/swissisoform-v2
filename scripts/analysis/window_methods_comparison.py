"""Unambiguous-sequence counts across ALL methods AND window radii.

For each HeLa multi-candidate TIS, extract candidate windows out to ±MAXW and
compute the first-divergence radius for (a) all windowed candidates
[sequence-only] and (b) the salmon-expressed subset at each TPM threshold. Then
sweep the window radius W and count, for every method, how many TIS carry an
unambiguous start-codon sequence:

  - sequence-only purity
  - salmon expression-only (TPM >= τ, both reps), for each τ
  - union (sequence-pure OR expression-resolved), for each τ

Universe: HeLa multi-candidate initiation sites (deliverable grain). A TIS with a
single windowed candidate is trivially pure; >=2 candidates must agree within W.
Long-read can be added later as another provider once IsoQuant lands.

Outputs (under --out-dir):
  - HeLa_methods_vs_window.csv             — count + % per method at every W
  - HeLa_methods_vs_window.png             — % and # curves, all methods
  - HeLa_methods_vs_window_radius.parquet  — per-TIS radii (recompute without GTF)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pysam  # noqa: E402

from swissisoform.io.gtf import load_exon_skeletons  # noqa: E402
from swissisoform.sourceseq.expression import (  # noqa: E402
    expressed_in_replicates,
    expressed_transcripts,
    load_isoquant_abundance,
)
from swissisoform.sourceseq.mrna import extract_tis_window  # noqa: E402
from swissisoform.sourceseq.purity import divergence_radius  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("window_methods_comparison")

REPORT_W = [5, 10, 50, 100, 200, 500]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skeleton-table", default="data/output/init_site_skeleton.parquet")
    ap.add_argument("--genome", default="data/reference/Gencode_v49_GRCh38.primary_assembly.genome.fa")
    ap.add_argument("--gtf", default="data/reference/gencode.v49.primary_assembly.annotation.gtf")
    ap.add_argument("--cell-line", default="HeLa")
    ap.add_argument("--salmon-quant", nargs="+",
                    default=["data/reference/salmon/HeLa_rep1/quant.sf",
                             "data/reference/salmon/HeLa_rep2/quant.sf"])
    ap.add_argument("--thresholds", type=float, nargs="+", default=[1.0, 0.1])
    ap.add_argument("--isoquant-table", default=None,
                    help="long-read IsoQuant transcript counts TSV; adds long-read + union lines")
    ap.add_argument("--isoquant-min-counts", type=float, default=3.0)
    ap.add_argument("--max-w", type=int, default=500)
    ap.add_argument("--out-dir", default="data/output/source_transcripts")
    args = ap.parse_args()

    cell, MAXW = args.cell_line, args.max_w
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    thr = sorted(args.thresholds, reverse=True)

    df = pd.read_parquet(args.skeleton_table)
    sel = df[(df[f"present_{cell}"] == True) & (df["n_transcripts"] >= 2)]  # noqa: E712
    N = len(sel)
    logger.info("%s multi-candidate TIS: %d", cell, N)

    genome = pysam.FastaFile(args.genome)
    logger.info("loading exon skeletons ...")
    skeletons = load_exon_skeletons(args.gtf)
    # one expressed set per threshold (salmon TPM >= τ in both reps)
    exp = {t: expressed_in_replicates(args.salmon_quant, min_tpm=t, require="all") for t in thr}
    for t in thr:
        logger.info("salmon TPM>=%.2f present transcripts: %d", t, len(exp[t]))
    # optional long-read (IsoQuant counts >= N)
    exp_lr = None
    if args.isoquant_table:
        exp_lr = expressed_transcripts(load_isoquant_abundance(args.isoquant_table),
                                       args.isoquant_min_counts)
        logger.info("long-read counts>=%.0f present transcripts: %d",
                    args.isoquant_min_counts, len(exp_lr))

    recs = []
    for _, row in sel.iterrows():
        gstart = int(row["gstart"])
        cands = [c for c in str(row["all_transcripts"]).split(",") if c]
        windows = []  # (tid, TisWindow) for windowed candidates
        for tid in cands:
            coords = skeletons.get(tid)
            if coords is None:
                continue
            win = extract_tis_window(coords, genome, gstart, up=MAXW, down=MAXW)
            if win is not None:
                windows.append((tid, win))
        n_win = len(windows)
        rec = {"init_site": row["init_site"], "n_win": n_win,
               "r_seq": divergence_radius([w for _, w in windows], MAXW) if n_win >= 2 else None}
        for t in thr:
            surv = [w for tid, w in windows if tid in exp[t]]
            rec[f"nsurv_{t}"] = len(surv)
            rec[f"rexpr_{t}"] = divergence_radius(surv, MAXW) if len(surv) >= 2 else None
        if exp_lr is not None:
            surv = [w for tid, w in windows if tid in exp_lr]
            rec["nsurv_lr"] = len(surv)
            rec["rexpr_lr"] = divergence_radius(surv, MAXW) if len(surv) >= 2 else None
        recs.append(rec)
    per = pd.DataFrame(recs)
    per.to_parquet(out_dir / f"{cell}_methods_vs_window_radius.parquet", index=False)

    # --- method evaluators (per-TIS booleans at a given W) ---
    def pure(r, w):
        # NaN/None == divergence_radius found no disagreement out to MAXW == pure.
        # (None becomes NaN once stored in the DataFrame, so guard with pd.isna.)
        return pd.isna(r) or (r > w)

    def seq_ok(w):
        # n_win==1 -> trivially pure; n_win==0 -> no window; n_win>=2 -> agree within w
        return ((per.n_win == 1) | ((per.n_win >= 2) & per.r_seq.map(lambda r: pure(r, w))))

    def expr_ok(t, w):
        ns = per[f"nsurv_{t}"]
        return ((ns == 1) | ((ns >= 2) & per[f"rexpr_{t}"].map(lambda r: pure(r, w))))

    def lr_ok(w):
        ns = per["nsurv_lr"]
        return ((ns == 1) | ((ns >= 2) & per["rexpr_lr"].map(lambda r: pure(r, w))))

    methods = {"sequence-only": lambda w: seq_ok(w)}
    for t in thr:
        methods[f"salmon TPM>={t:g}"] = (lambda w, t=t: expr_ok(t, w))
        methods[f"union @TPM>={t:g}"] = (lambda w, t=t: seq_ok(w) | expr_ok(t, w))
    if exp_lr is not None:
        methods[f"long-read >={args.isoquant_min_counts:g}"] = (lambda w: lr_ok(w))
        methods["union @long-read"] = (lambda w: seq_ok(w) | lr_ok(w))
        # 3-way union: sequence-pure OR salmon(@0.1) OR long-read
        if 0.1 in thr:
            methods["union @3-way"] = (lambda w: seq_ok(w) | expr_ok(0.1, w) | lr_ok(w))

    # --- sweep full grid for the figure ---
    ws = list(range(5, MAXW + 1))
    curve = {"W": ws}
    for name, fn in methods.items():
        curve[name] = [int(fn(w).sum()) for w in ws]
    cdf = pd.DataFrame(curve)
    for name in methods:
        cdf[name + " %"] = 100 * cdf[name] / N
    cdf.to_csv(out_dir / f"{cell}_methods_vs_window.csv", index=False)

    # --- report table at the requested W values ---
    logger.info("N = %d multi-candidate TIS\n", N)
    header = "  W  | " + " | ".join(f"{m:>18s}" for m in methods)
    logger.info(header)
    for w in REPORT_W:
        rowv = cdf[cdf.W == w].iloc[0]
        cells = " | ".join(f"{int(rowv[m]):5d} ({rowv[m+' %']:4.1f}%)" for m in methods)
        logger.info(f" {w:3d} | {cells}")

    # --- figure: % (top) and count (bottom) vs W, all methods ---
    colors = {"sequence-only": "#1f77b4"}
    pal = ["#d62728", "#ff7f0e", "#2ca02c", "#17becf", "#9467bd", "#8c564b"]
    for i, m in enumerate([m for m in methods if m != "sequence-only"]):
        colors[m] = pal[i % len(pal)]
    if "union @3-way" in colors:
        colors["union @3-way"] = "#000000"  # headline line: black
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    for m in methods:
        style = "-" if "union" in m else ("--" if "salmon" in m else "-")
        lw = 3.4 if "3-way" in m else (2.6 if "union" in m else 1.8)
        a1.plot(cdf.W, cdf[m + " %"], style, color=colors[m], lw=lw, label=m)
        a2.plot(cdf.W, cdf[m], style, color=colors[m], lw=lw)
    for w in REPORT_W:
        a1.axvline(w, color="#dddddd", lw=0.8, zorder=0)
        a2.axvline(w, color="#dddddd", lw=0.8, zorder=0)
    a1.set_ylabel("unambiguous TIS (%)")
    a1.set_title(f"{cell}: unambiguous-sequence count by method and window radius  (N={N:,} multi-candidate TIS)")
    a1.legend(fontsize=9, ncol=2); a1.grid(alpha=0.3)
    a2.set_ylabel("unambiguous TIS (#)")
    a2.set_xlabel("window radius W (nt each side of start codon)")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{cell}_methods_vs_window.png", dpi=150)
    logger.info("wrote %s", out_dir / f"{cell}_methods_vs_window.png")


if __name__ == "__main__":
    main()
