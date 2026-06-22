"""Assemble the union-disambiguated HeLa TIS dataset (window = 100 nt).

Joins everything produced today into one table: each HeLa multi-candidate TIS
with its metadata, Ribo-TISH read counts, per-transcript initiation efficiency
(TIE), and — for the TIS the **union strategy** resolves — the disambiguated
source transcript and its mRNA sequence (the unambiguous ±100 nt start-codon
window, plus the full transcript mRNA where it is unambiguous).

Union rule (see docs/plans/union_strategy_unambiguous_tis.md), at W=100:
  keep a TIS if its candidates are sequence-identical within ±100 nt
  (sequence-pure, Tier 1)  OR  expression collapses it to a single / window-
  agreeing source (expression-resolved, Tier 2/3).

Caveat encoded in the schema: ±100 window is unambiguous for every kept TIS;
the FULL mRNA is unambiguous only when the TIS resolves to a single transcript
(expression-resolved single survivor). For sequence-pure TIS with >1 candidate
the candidates diverge outside the window, so `full_mrna` is a representative
and `full_mrna_unambiguous=False`.

Inputs (all on disk):
  - data/output/source_transcripts/HeLa_tis_window_summary.parquet      (seq purity @100)
  - data/output/source_transcripts/HeLa_tis_candidate_detail.parquet    (per-candidate ±100 windows)
  - data/output/source_transcripts/HeLa_disambiguation_salmon_tpm0.1.parquet (expression resolve + source TPM)
  - data/output/init_site_skeleton.parquet                              (TIS metadata)
  - data/output/filtered/all_samples_combined.parquet                   (read counts)
  - data/reference/gencode.v49.transcripts.fa                           (full mRNA by ENST)
Output:
  - data/output/source_transcripts/HeLa_tis_disambiguated_union_W100.parquet (+ .csv)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pysam

from swissisoform.combine import _init_site_from_genome_pos
from swissisoform.sourceseq.expression import load_salmon_replicates

SRC = Path("data/output/source_transcripts")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--transcriptome", default="data/reference/gencode.v49.transcripts.fa")
    ap.add_argument("--salmon-quant", nargs="+",
                    default=["data/reference/salmon/HeLa_rep1/quant.sf",
                             "data/reference/salmon/HeLa_rep2/quant.sf"])
    ap.add_argument("--out", default=str(SRC / "HeLa_tis_disambiguated_union_W100.parquet"))
    args = ap.parse_args()

    summary = pd.read_parquet(SRC / "HeLa_tis_window_summary.parquet")
    detail = pd.read_parquet(SRC / "HeLa_tis_candidate_detail.parquet")
    disamb = pd.read_parquet(SRC / "HeLa_disambiguation_salmon_tpm0.1.parquet")
    skel = pd.read_parquet("data/output/init_site_skeleton.parquet")
    comb = pd.read_parquet("data/output/filtered/all_samples_combined.parquet")
    tpm = load_salmon_replicates(args.salmon_quant)  # {Tid: mean TPM}

    fa = pysam.FastaFile(args.transcriptome)
    enst_key = {r.split("|")[0]: r for r in fa.references}

    # read counts per (init_site, Tid) from the combined table
    comb = comb.copy()
    comb["init_site"] = _init_site_from_genome_pos(comb["GenomePos"], comb["StartCodon"])
    count_cols = ["HeLa_TISCounts", "HeLa_NormTISCounts", "HeLa_TotalRNASeqCounts", "HeLa_GeneRNASeqCounts"]
    # The TIS footprint count is a property of the initiation SITE, not of which
    # candidate transcript we pick (Ribo-TISH spreads the same footprints over
    # compatible transcripts). Aggregate per init_site (max = the detected
    # signal); this also guarantees counts are always populated.
    counts = comb.groupby("init_site")[count_cols].max()

    # windowed candidates per TIS, with TPM, for representative selection
    win = detail[detail["window_ok"] == True].copy()  # noqa: E712
    win["tpm"] = win["Tid"].map(tpm).fillna(0.0)

    # quick lookups
    seq_keep = dict(zip(summary["init_site"], summary["keep"]))
    seq_reason = dict(zip(summary["init_site"], summary["reason"]))
    seq_div = dict(zip(summary["init_site"], summary["divergence_nt"]))
    dz = disamb.set_index("init_site")
    sk = skel.set_index("init_site")

    rows = []
    for init_site, grp in win.groupby("init_site"):
        if init_site not in sk.index:
            continue
        meta = sk.loc[init_site]
        d = dz.loc[init_site] if init_site in dz.index else None
        is_seq_pure = bool(seq_keep.get(init_site, False))
        expr_keep = bool(d["keep_expr"]) if d is not None and pd.notna(d["keep_expr"]) else False

        # --- union resolution + tier ---
        if is_seq_pure:
            tier = "1_sequence_pure"
            # representative among identical-window candidates: max HeLa TPM
            rep = grp.sort_values("tpm", ascending=False).iloc[0]
            source_tid = rep["Tid"]
            full_unambig = grp["Tid"].nunique() == 1  # only if literally one candidate
        elif expr_keep:
            source_tid = d["source_tid"]
            tier = "2_expression_highTPM" if (d["source_abundance"] or 0) >= 1.0 else "3_expression_lowTPM"
            full_unambig = int(d["n_expressed_window"]) == 1
        else:
            # not resolved by the union at W=100
            rows.append({"init_site": init_site, "gene": meta["gene"], "resolved": False,
                         "resolution_tier": "unresolved",
                         "resolution_reason": d["reason_expr"] if d is not None else "ambiguous_window"})
            continue

        # --- the ±100 window for the chosen source (unambiguous by construction) ---
        wrow = grp[grp["Tid"] == source_tid]
        wrow = wrow.iloc[0] if len(wrow) else grp.iloc[0]
        up, dn = wrow["upstream"] or "", wrow["downstream"] or ""
        mrna_window = up + dn                      # contiguous ±100; A at index len(up)
        full_mrna = fa.fetch(enst_key[source_tid]) if source_tid in enst_key else None

        # --- read counts at this init site (footprint signal is site-level) ---
        cnt = counts.loc[init_site] if init_site in counts.index else None
        tis_counts = float(cnt["HeLa_TISCounts"]) if cnt is not None else None
        norm_tis = float(cnt["HeLa_NormTISCounts"]) if cnt is not None else None
        total_rna = float(cnt["HeLa_TotalRNASeqCounts"]) if cnt is not None else None
        gene_rna = float(cnt["HeLa_GeneRNASeqCounts"]) if cnt is not None else None

        # --- TIE: per-transcript initiation efficiency = RPM footprints / source TPM ---
        src_tpm = float(d["source_abundance"]) if (d is not None and pd.notna(d["source_abundance"])) else float(tpm.get(source_tid, 0.0))
        src_tpm_sum = float(d["source_abundance_sum"]) if (d is not None and pd.notna(d["source_abundance_sum"])) else src_tpm
        tie = (norm_tis / src_tpm_sum) if (norm_tis is not None and src_tpm_sum and src_tpm_sum > 0) else None

        rows.append({
            "init_site": init_site, "gene": meta["gene"], "chrom": meta["chrom"],
            "gstart": int(meta["gstart"]), "strand": meta["strand"],
            "start_codon": meta["start_codon"], "orf_type": meta["orf_type"],
            "n_candidates": int(meta["n_transcripts"]),
            # read counts
            "hela_tis_counts": tis_counts, "hela_norm_tis_counts": norm_tis,
            "hela_total_rnaseq_counts": total_rna, "hela_gene_rnaseq_counts": gene_rna,
            # disambiguation
            "resolved": True, "resolution_tier": tier,
            "resolution_reason": "sequence_pure" if is_seq_pure else d["reason_expr"],
            "window_clean_to_nt": (seq_div.get(init_site) if not is_seq_pure else None),
            "source_transcript": source_tid,
            "source_tpm": src_tpm, "source_tpm_sum": src_tpm_sum,
            "n_expressed_candidates": int(d["n_expressed_window"]) if d is not None and pd.notna(d["n_expressed_window"]) else None,
            # efficiency
            "tie_initiation_efficiency": tie,
            # sequences (window = 100 nt each side)
            "start_codon_pos_in_window": len(up),
            "mrna_window_100": mrna_window,
            "mrna_window_upstream": up, "mrna_window_downstream": dn,
            "full_mrna": full_mrna, "full_mrna_unambiguous": full_unambig,
        })

    out = pd.DataFrame(rows)
    op = Path(args.out)
    out.to_parquet(op, index=False)
    out.drop(columns=["full_mrna"]).to_csv(op.with_suffix(".csv"), index=False)  # csv without the long seqs

    res = out[out["resolved"]]
    print(f"HeLa multi-candidate TIS in table: {len(out):,}")
    print(f"  resolved by union (W=100): {len(res):,} ({100*len(res)/len(out):.1f}%)")
    print("  by tier:")
    print(res["resolution_tier"].value_counts().to_string())
    print(f"  full mRNA unambiguous (single transcript): {int(res['full_mrna_unambiguous'].sum()):,}")
    print(f"  TIE computable: {int(res['tie_initiation_efficiency'].notna().sum()):,}")
    print(f"\nwrote {op}\n  and {op.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
