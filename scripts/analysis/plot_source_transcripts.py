"""Render the source-transcript-resolution figures from on-disk outputs.

Reads the parquet/CSV products of ``candidate_mrna_divergence.py``,
``disambiguate_expression.py`` and ``window_purity_vs_length.py`` and writes a
set of PNGs to ``--out-dir`` (default ``data/figures/source_transcripts``). Pure
plotting — no GTF walk, no reference loading — so it runs anywhere in seconds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell-line", default="HeLa")
    ap.add_argument("--src-dir", default="data/output/source_transcripts")
    ap.add_argument("--out-dir", default="data/figures/source_transcripts")
    args = ap.parse_args()

    cell = args.cell_line
    src = Path(args.src_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = []

    # 1) purity vs window radius (% and count) -----------------------------------
    cur = pd.read_csv(src / f"{cell}_purity_vs_window.csv")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    a1.plot(cur.W, cur.seq_pure_pct, label="sequence-only", color="C0")
    a1.plot(cur.W, cur.expr_res_pct, label="expression-filtered (TPM>=0.1)", color="C1")
    cx = cur[cur.expr_res_pct >= cur.seq_pure_pct].iloc[0].W
    a1.axvline(cx, ls="--", color="grey", alpha=0.7)
    a1.annotate(f"crossover W={int(cx)}", (cx, 60), xytext=(cx + 20, 65), color="grey")
    a1.set_ylabel("pure / resolved (%)"); a1.legend(); a1.grid(alpha=0.3)
    a1.set_title(f"{cell}: start-codon window purity vs window radius")
    a2.plot(cur.W, cur.seq_pure_n, color="C0")
    a2.plot(cur.W, cur.expr_res_n, color="C1")
    a2.axvline(cx, ls="--", color="grey", alpha=0.7)
    a2.set_ylabel("pure / resolved (# TIS)")
    a2.set_xlabel("window radius W (nt each side of start codon)"); a2.grid(alpha=0.3)
    fig.tight_layout(); p = out / f"{cell}_purity_vs_window.png"; fig.savefig(p, dpi=150); plt.close(fig)
    saved.append(p)

    # 2) first-divergence radius distribution ------------------------------------
    rad = pd.read_parquet(src / f"{cell}_purity_vs_window_radius.parquet")
    fig, ax = plt.subplots(figsize=(8, 5))
    r = rad.r_seq.dropna()
    ax.hist(r, bins=range(0, 501, 10), color="C0", alpha=0.85)
    med = r.median()
    ax.axvline(med, color="k", ls="--"); ax.annotate(f"median {med:.0f} nt", (med + 8, ax.get_ylim()[1] * 0.9))
    ax.set_xlabel("first-divergence radius (nt from start codon)")
    ax.set_ylabel("# multi-candidate TIS")
    ax.set_title(f"{cell}: where candidate transcripts first diverge "
                 f"({int(rad.r_seq.isna().sum())} identical to ±500)")
    ax.grid(alpha=0.3); fig.tight_layout()
    p = out / f"{cell}_divergence_radius_hist.png"; fig.savefig(p, dpi=150); plt.close(fig)
    saved.append(p)

    # 3) surviving expressed isoforms per TIS + source-abundance confidence -------
    dis = pd.read_parquet(src / f"{cell}_disambiguation_salmon_tpm0.1.parquet")
    n_exp = dis.n_expressed_window.fillna(0).astype(int).clip(upper=4)
    counts = n_exp.value_counts().sort_index()
    labels = {0: "0\n(lost)", 1: "1\n(unique)", 2: "2", 3: "3", 4: "4+"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    a1.bar([labels[k] for k in counts.index], counts.values, color="C1")
    a1.set_ylabel("# TIS"); a1.set_xlabel("expressed candidate isoforms (TPM>=0.1, both reps)")
    a1.set_title(f"{cell}: candidates surviving expression, per TIS"); a1.grid(alpha=0.3, axis="y")
    res = dis[dis.keep_expr.fillna(False)]
    ab = res.source_abundance.clip(lower=0.05)
    a2.hist(ab, bins=[10 ** (x / 4) for x in range(-5, 14)], color="C2", alpha=0.85)
    a2.set_xscale("log")
    a2.axvline(1.0, color="k", ls="--"); a2.annotate("TPM=1", (1.05, a2.get_ylim()[1] * 0.9))
    lo = int((res.source_abundance < 1).sum())
    a2.set_xlabel("source-isoform mean TPM (resolved TIS)"); a2.set_ylabel("# TIS")
    a2.set_title(f"resolved-TIS source confidence  ({lo} of {len(res)} below 1 TPM)")
    a2.grid(alpha=0.3); fig.tight_layout()
    p = out / f"{cell}_resolved_confidence.png"; fig.savefig(p, dpi=150); plt.close(fig)
    saved.append(p)

    print(f"wrote {len(saved)} figures to {out}/:")
    for s in saved:
        print(" ", s.name)


if __name__ == "__main__":
    main()
