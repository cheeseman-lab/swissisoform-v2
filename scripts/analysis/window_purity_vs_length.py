"""How does the ±W window length around the start codon affect purity?

For each multi-candidate HeLa TIS, extract the candidate windows out to ±MAXW nt
and compute the **first-divergence radius** once (the smallest nt distance from
the start codon at which the candidates stop agreeing, or None if identical out
to MAXW). A TIS is then "pure at W" iff its divergence radius is None or > W — so
the entire purity-vs-W curve falls out of one pass, swept cheaply over W=5..MAXW.

Two series:
- **sequence-only** — all windowed candidates (the intrinsic sequence question).
- **expression-filtered** — candidates present in HeLa (salmon TPM >= tau in both
  reps); a TIS resolves at W if it has one expressed survivor, or several that
  agree within W.

Universe = TIS with >= 2 windowed candidates (where W actually matters).

Outputs (under --out-dir):
- ``HeLa_purity_vs_window_radius.parquet`` — per-TIS divergence radii (recompute
  the curve without re-walking the GTF).
- ``HeLa_purity_vs_window.csv`` — the swept curve (W, pct/count, both series).
- ``HeLa_purity_vs_window.png`` — the figure (% pure + count vs W).
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
from swissisoform.sourceseq.expression import expressed_in_replicates  # noqa: E402
from swissisoform.sourceseq.mrna import extract_tis_window  # noqa: E402
from swissisoform.sourceseq.purity import divergence_radius  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("window_purity_vs_length")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skeleton-table", default="data/output/init_site_skeleton.parquet")
    ap.add_argument("--genome", default="data/reference/Gencode_v49_GRCh38.primary_assembly.genome.fa")
    ap.add_argument("--gtf", default="data/reference/gencode.v49.primary_assembly.annotation.gtf")
    ap.add_argument("--cell-line", default="HeLa")
    ap.add_argument("--salmon-quant", nargs="+",
                    default=["data/reference/salmon/HeLa_rep1/quant.sf",
                             "data/reference/salmon/HeLa_rep2/quant.sf"])
    ap.add_argument("--min-tpm", type=float, default=0.1)
    ap.add_argument("--max-w", type=int, default=500)
    ap.add_argument("--min-w", type=int, default=5)
    ap.add_argument("--out-dir", default="data/output/source_transcripts")
    args = ap.parse_args()

    cell = args.cell_line
    MAXW = args.max_w
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.skeleton_table)
    sel = df[(df[f"present_{cell}"] == True) & (df["n_transcripts"] >= 2)].copy()  # noqa: E712
    logger.info("%s: %d multi-candidate TIS", cell, len(sel))

    genome = pysam.FastaFile(args.genome)
    logger.info("loading exon skeletons ...")
    skeletons = load_exon_skeletons(args.gtf)
    expressed = expressed_in_replicates(args.salmon_quant, min_tpm=args.min_tpm, require="all")
    logger.info("loaded %d skeletons; %d transcripts present (TPM>=%.2f)",
                len(skeletons), len(expressed), args.min_tpm)

    # --- per TIS: first-divergence radius over all windows and over survivors ---
    recs = []
    for _, row in sel.iterrows():
        gstart = int(row["gstart"])
        cands = [c for c in str(row["all_transcripts"]).split(",") if c]
        windows, surv_windows = [], []
        for tid in cands:
            coords = skeletons.get(tid)
            if coords is None:
                continue
            win = extract_tis_window(coords, genome, gstart, up=MAXW, down=MAXW)
            if win is None:
                continue
            windows.append(win)
            if tid in expressed:
                surv_windows.append(win)
        if len(windows) < 2:
            continue  # W doesn't matter for <2 windowed candidates
        r_seq = divergence_radius(windows, MAXW)
        n_surv = len(surv_windows)
        r_expr = divergence_radius(surv_windows, MAXW) if n_surv >= 2 else None
        recs.append({
            "init_site": row["init_site"], "gene": row["gene"],
            "n_windows": len(windows), "r_seq": r_seq,
            "n_surv": n_surv, "r_expr": r_expr,
        })
    per_tis = pd.DataFrame(recs)
    N = len(per_tis)
    logger.info("testable TIS (>=2 windowed candidates): %d", N)
    per_tis.to_parquet(out_dir / f"{cell}_purity_vs_window_radius.parquet", index=False)

    # --- sweep W ---
    def pure(r, w):  # identical out to MAXW (None/NaN) or first divergence beyond w
        return pd.isna(r) or (r > w)

    ws = list(range(args.min_w, MAXW + 1))
    rseq = per_tis["r_seq"].tolist()
    rexpr = per_tis["r_expr"].tolist()
    nsurv = per_tis["n_surv"].tolist()
    curve = []
    for w in ws:
        seq_pure = sum(pure(r, w) for r in rseq)
        # expression: 1 survivor -> always resolved; >=2 -> agree within w; 0 -> never
        expr_res = sum((ns == 1) or (ns >= 2 and pure(r, w))
                       for ns, r in zip(nsurv, rexpr))
        curve.append({"W": w,
                      "seq_pure_n": seq_pure, "seq_pure_pct": 100 * seq_pure / N,
                      "expr_res_n": expr_res, "expr_res_pct": 100 * expr_res / N})
    cdf = pd.DataFrame(curve)
    cdf.to_csv(out_dir / f"{cell}_purity_vs_window.csv", index=False)

    # --- plot: % (top) and count (bottom) vs W ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax1.plot(cdf["W"], cdf["seq_pure_pct"], label="sequence-only", color="C0")
    ax1.plot(cdf["W"], cdf["expr_res_pct"], label=f"expression-filtered (TPM>={args.min_tpm})", color="C1")
    ax1.set_ylabel("pure / resolved (%)")
    ax1.set_title(f"{cell}: start-codon window purity vs window radius  (N={N} multi-candidate TIS)")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(cdf["W"], cdf["seq_pure_n"], color="C0")
    ax2.plot(cdf["W"], cdf["expr_res_n"], color="C1")
    ax2.set_ylabel("pure / resolved (# TIS)")
    ax2.set_xlabel("window radius W (nt each side of start codon)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{cell}_purity_vs_window.png", dpi=150)
    logger.info("wrote %s", out_dir / f"{cell}_purity_vs_window.png")

    # --- console summary at representative radii ---
    for w in (5, 10, 25, 50, 100, 200, 300, 500):
        r = cdf[cdf["W"] == w].iloc[0]
        logger.info("W=%3d  seq-pure %5d (%4.1f%%)   expr-resolved %5d (%4.1f%%)",
                    w, int(r.seq_pure_n), r.seq_pure_pct, int(r.expr_res_n), r.expr_res_pct)


if __name__ == "__main__":
    main()
