"""Window-purity analysis for TIS candidate transcripts.

For each TIS (genomic init site) with multiple candidate transcripts, anchor the
transcripts at the shared start codon, reconstruct each one's spliced mRNA, and
ask how far out the candidates stay identical. Purity at radius W = the fraction
of multi-candidate TIS whose candidates agree within ±W nt. The script reports
purity vs window size, split by ORF type and start codon, and writes a
per-init-site table.

Candidates come from the Ribo-TISH combined table (``present_{sample}``); an
optional salmon expression gate restricts them. Reuses
``swissisoform.sourceseq`` (mRNA reconstruction + the window-purity primitive).

Usage:
  python scripts/analysis/window_purity_analysis.py \
      --sample HeLa --gate supported --max-w 200 --out-dir data/output/window_purity
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pysam

from swissisoform.io.gtf import load_exon_skeletons
from swissisoform.sourceseq.mrna import TisWindow, build_transcript_mrna, start_codon_index
from swissisoform.sourceseq.purity import divergence_radius

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "data" / "reference"
COMBINED = ROOT / "data" / "output" / "filtered" / "all_samples_combined.parquet"
GENOME = REF / "Gencode_v49_GRCh38.primary_assembly.genome.fa"
GTF = REF / "gencode.v49.primary_assembly.annotation.gtf"


def gate_set(kind: str, sample: str) -> set[str] | None:
    """Transcript ids passing an expression gate from salmon quant.sf, or None."""
    if kind == "none":
        return None
    reps = sorted((REF / "salmon").glob(f"{sample}_rep*/quant.sf"))
    if not reps:
        raise SystemExit(f"no salmon quant.sf for {sample} under {REF}/salmon")
    merged = None
    for i, p in enumerate(reps):
        d = pd.read_csv(p, sep="\t")[["Name", "TPM", "NumReads"]]
        d["tid"] = d["Name"].str.split("|").str[0]
        d = d[["tid", "TPM", "NumReads"]].rename(columns={"TPM": f"TPM{i}", "NumReads": f"NR{i}"})
        merged = d if merged is None else merged.merge(d, on="tid")
    tpm = [c for c in merged if c.startswith("TPM")]
    nr = [c for c in merged if c.startswith("NR")]
    if kind == "present":  # TPM >= 1 in every replicate
        return set(merged.loc[(merged[tpm] >= 1).all(axis=1), "tid"])
    if kind == "supported":  # >= 1 estimated read across replicates
        return set(merged.loc[merged[nr].sum(axis=1) >= 1, "tid"])
    raise ValueError(kind)


def ribotish_groups(sample: str) -> pd.DataFrame:
    """Init-site groups (pos, codon, ORF type, candidate Tids) for a sample."""
    df = pd.read_parquet(
        COMBINED, columns=["Tid", "GenomePos", "StartCodon", "RecatTISType", f"present_{sample}"]
    )
    df = df[df[f"present_{sample}"].fillna(False).astype(bool)].copy()

    def parse(gp: str):
        chrom, coords, strand = gp.rsplit(":", 2)
        lo, hi = coords.split("-")
        return chrom, (int(lo) if strand == "+" else int(hi)), strand

    pp = df["GenomePos"].map(parse)
    df["pos"] = pp.map(lambda x: x[1])
    prefix = pp.map(lambda x: f"{x[0]}:{x[1]}:{x[2]}")
    df["init"] = prefix + ":" + df["StartCodon"]
    return (
        df.groupby("init")
        .agg(
            pos=("pos", "first"),
            codon=("StartCodon", "first"),
            orf_type=("RecatTISType", lambda s: s.iloc[0]),
            tids=("Tid", lambda s: sorted(set(s))),
        )
        .reset_index()
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Window-purity analysis for TIS candidates.")
    ap.add_argument("--sample", default="HeLa")
    ap.add_argument("--gate", choices=["none", "supported", "present"], default="supported")
    ap.add_argument("--max-w", type=int, default=200)
    ap.add_argument("--out-dir", default="data/output/window_purity")
    args = ap.parse_args()
    maxw = args.max_w
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    gate = gate_set(args.gate, args.sample)
    g = ribotish_groups(args.sample)
    g["cand"] = g["tids"].map(lambda ts: [t for t in ts if gate is None or t in gate])
    multi = g[g["cand"].map(len) >= 2].reset_index(drop=True)
    print(
        f"{args.sample}: {len(g):,} init-sites | multi-candidate (gate={args.gate}): {len(multi):,}"
    )

    print("loading skeletons + genome ...")
    skel = load_exon_skeletons(str(GTF))
    genome = pysam.FastaFile(str(GENOME))
    mrna_cache: dict[str, str | None] = {}

    def mrna_of(tid: str) -> str | None:
        if tid not in mrna_cache:
            c = skel.get(tid)
            mrna_cache[tid] = build_transcript_mrna(c, genome) if c else None
        return mrna_cache[tid]

    def divergence(tids: list[str], pos: int) -> int | None:
        wins = []
        for tid in tids:
            c = skel.get(tid)
            if c is None:
                continue
            idx = start_codon_index(c, pos)
            m = mrna_of(tid)
            if idx is None or m is None or idx >= len(m):
                continue
            wins.append(
                TisWindow(m[max(0, idx - maxw) : idx], m[idx : idx + maxw], idx, idx - maxw < 0)
            )
        if len(wins) < 2:
            return None
        d = divergence_radius(wins, maxw)
        return maxw + 1 if d is None else d

    rows = []
    for r in multi.itertuples():
        d = divergence(r.cand, r.pos)
        if d is not None:
            rows.append((r.init, r.orf_type, r.codon, len(r.cand), d))
    res = pd.DataFrame(
        rows, columns=["init_site", "orf_type", "start_codon", "n_candidates", "divergence_nt"]
    )
    table = out / f"{args.sample}_window_purity_{args.gate}.parquet"
    res.to_parquet(table, index=False)
    print(f"scored {len(res):,} init-sites -> {table}")

    ws = np.arange(5, maxw + 1, 5)
    r0 = res["divergence_nt"].to_numpy()

    def curve(mask) -> list[float]:
        sub = r0[mask]
        return [100 * (sub > w).mean() for w in ws] if len(sub) else [float("nan")] * len(ws)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    ax1.plot(ws, curve(np.ones(len(r0), bool)), lw=2.6, color="k", label=f"all (n={len(r0):,})")
    for t in ["Annotated", "Truncated", "Extended", "uORF"]:
        m = (res["orf_type"] == t).to_numpy()
        if m.sum() >= 20:
            ax1.plot(ws, curve(m), lw=2, label=f"{t} (n={int(m.sum()):,})")
    ax1.set(xlabel="W (± nt)", ylabel="% pure", ylim=(0, 100), xlim=(0, maxw))
    ax1.set_title(f"by ORF type — {args.sample}, gate={args.gate}")
    ax1.grid(alpha=0.3)
    ax1.legend()
    for c in ["ATG", "CTG", "GTG", "TTG", "AAG", "AGG", "ACG"]:
        m = (res["start_codon"] == c).to_numpy()
        if m.sum() >= 20:
            ax2.plot(ws, curve(m), lw=2, label=f"{c} (n={int(m.sum()):,})")
    ax2.set(xlabel="W (± nt)", xlim=(0, maxw))
    ax2.set_title("by start codon")
    ax2.grid(alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    fig.savefig(out / f"{args.sample}_purity_split_{args.gate}.png", dpi=130)

    w0 = min(100, maxw)
    matrix = (
        res.assign(pure=res["divergence_nt"] > w0)
        .pivot_table(index="orf_type", columns="start_codon", values="pure", aggfunc="mean")
        .mul(100)
        .round(0)
    )
    matrix.to_csv(out / f"{args.sample}_purity_matrix_w{w0}_{args.gate}.csv")

    print(f"\npurity at ±{w0} nt by ORF type:")
    for t in ["Annotated", "Truncated", "Extended", "uORF"]:
        m = (res["orf_type"] == t).to_numpy()
        if m.sum():
            print(f"  {t:<10} n={int(m.sum()):>5}  pure={100 * (r0[m] > w0).mean():.0f}%")
    print(f"\nsaved table + plot + matrix under {out}/")


if __name__ == "__main__":
    main()
