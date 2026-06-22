"""Three-way union dataset: resolve each TIS by sequence-purity OR salmon OR long-read.

Combines all three disambiguation signals into one table, tiered by evidence
strength and with a precedence-chosen source transcript:

  resolved  = sequence-pure (S)  OR  salmon-resolved (A)  OR  long-read-resolved (B)
  tier 1    = sequence-pure        (correct by construction; identity-agnostic)
  tier 2    = long-read present    (direct molecular evidence)
  tier 3    = salmon-only          (weakest — likely read-dilution artifact)
  source    = long_read > short_read > sequence-representative   (most-trustworthy available)

See docs/analysis/disambiguation_method_comparison.md (three-way union section).
Inputs are the per-method outputs already on disk; output:
  data/output/source_transcripts/HeLa_tis_union3_W100.parquet (+ .csv without seqs)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pysam

from swissisoform.combine import _init_site_from_genome_pos
from swissisoform.sourceseq.expression import load_isoquant_abundance, load_salmon_replicates

SRC = Path("data/output/source_transcripts")
ISOQUANT = "data/reference/longread/isoquant_HeLa/OUT/OUT.transcript_counts.tsv"
SALMON = ["data/reference/salmon/HeLa_rep1/quant.sf", "data/reference/salmon/HeLa_rep2/quant.sf"]
FASTA = "data/reference/gencode.v49.transcripts.fa"


def main() -> None:
    summ = pd.read_parquet(SRC / "HeLa_tis_window_summary.parquet").set_index("init_site")
    sal = pd.read_parquet(SRC / "HeLa_disambiguation_salmon_tpm0.1.parquet").set_index("init_site")
    iso = pd.read_parquet(SRC / "HeLa_disambiguation_isoquant.parquet").set_index("init_site")
    skel = pd.read_parquet("data/output/init_site_skeleton.parquet").set_index("init_site")
    comb = pd.read_parquet("data/output/filtered/all_samples_combined.parquet")
    det = pd.read_parquet(SRC / "HeLa_tis_candidate_detail.parquet")

    tpm = load_salmon_replicates(SALMON)                 # {Tid: mean TPM}
    lrc = load_isoquant_abundance(ISOQUANT)              # {Tid: long-read count}
    fa = pysam.FastaFile(FASTA)
    enst_key = {r.split("|")[0]: r for r in fa.references}

    count_cols = ["HeLa_TISCounts", "HeLa_NormTISCounts", "HeLa_TotalRNASeqCounts", "HeLa_GeneRNASeqCounts"]
    comb = comb.copy()
    comb["init_site"] = _init_site_from_genome_pos(comb["GenomePos"], comb["StartCodon"])
    counts = comb.groupby("init_site")[count_cols].max()

    win = det[det["window_ok"] == True].copy()           # noqa: E712
    win["tpm"] = win["Tid"].map(tpm).fillna(0.0)
    win_by_site = {k: g for k, g in win.groupby("init_site")}

    rows = []
    for init_site, srow in summ.iterrows():
        meta = skel.loc[init_site] if init_site in skel.index else None
        if meta is None:
            continue
        S = bool(srow["keep"])
        arow = sal.loc[init_site] if init_site in sal.index else None
        brow = iso.loc[init_site] if init_site in iso.index else None
        A = bool(arow["keep_expr"]) if arow is not None and pd.notna(arow["keep_expr"]) else False
        B = bool(brow["keep_expr"]) if brow is not None and pd.notna(brow["keep_expr"]) else False
        a_src = arow["source_tid"] if A else None
        b_src = brow["source_tid"] if B else None

        resolved_by = [n for n, f in (("seq", S), ("short_read", A), ("long_read", B)) if f]
        resolved = bool(resolved_by)
        if not resolved:
            rows.append({"init_site": init_site, "gene": meta["gene"], "resolved": False,
                         "resolved_by": "", "n_methods": 0, "agreement_tier": 0,
                         "agreement_label": "unresolved"})
            continue

        tier = 1 if S else (2 if B else 3)
        label = {1: "sequence-pure", 2: "long-read / corroborated", 3: "salmon-only"}[tier]
        # source precedence: long_read > short_read > sequence representative (max-salmon-TPM candidate)
        grp = win_by_site.get(init_site)
        if B and isinstance(b_src, str):
            source, evidence = b_src, "long_read"
        elif A and isinstance(a_src, str):
            source, evidence = a_src, "short_read"
        else:
            rep = grp.sort_values("tpm", ascending=False).iloc[0] if grp is not None and len(grp) else None
            source, evidence = (rep["Tid"] if rep is not None else None), "sequence"

        # window for the chosen source
        up = dn = None
        if grp is not None and source is not None:
            wr = grp[grp["Tid"] == source]
            wr = wr.iloc[0] if len(wr) else grp.iloc[0]
            up, dn = wr["upstream"] or "", wr["downstream"] or ""
        mrna_window = (up + dn) if up is not None else None
        full_mrna = fa.fetch(enst_key[source]) if (source in enst_key) else None

        # full mRNA unambiguous only if resolved to a single transcript
        a_n = int(arow["n_expressed_window"]) if (A and pd.notna(arow["n_expressed_window"])) else None
        b_n = int(brow["n_expressed_window"]) if (B and pd.notna(brow["n_expressed_window"])) else None
        full_unambig = ((evidence == "long_read" and b_n == 1) or
                        (evidence == "short_read" and a_n == 1) or
                        int(meta["n_transcripts"]) == 1)
        source_agreement = (a_src == b_src) if (A and B) else None

        cnt = counts.loc[init_site] if init_site in counts.index else None
        norm_tis = float(cnt["HeLa_NormTISCounts"]) if cnt is not None else None
        src_tpm = float(tpm.get(source, 0.0))
        tie = (norm_tis / src_tpm) if (norm_tis is not None and src_tpm > 0) else None

        rows.append({
            "init_site": init_site, "gene": meta["gene"], "chrom": meta["chrom"],
            "gstart": int(meta["gstart"]), "strand": meta["strand"],
            "start_codon": meta["start_codon"], "orf_type": meta["orf_type"],
            "n_candidates": int(meta["n_transcripts"]),
            "hela_tis_counts": float(cnt["HeLa_TISCounts"]) if cnt is not None else None,
            "hela_norm_tis_counts": norm_tis,
            "resolved": True, "resolved_by": ",".join(resolved_by), "n_methods": len(resolved_by),
            "agreement_tier": tier, "agreement_label": label,
            "source_transcript": source, "source_evidence": evidence,
            "source_salmon_tpm": src_tpm, "source_longread_counts": float(lrc.get(source, 0.0)),
            "source_agreement_salmon_vs_longread": source_agreement,
            "tie_initiation_efficiency": tie,
            "start_codon_pos_in_window": (len(up) if up is not None else None),
            "mrna_window_100": mrna_window, "full_mrna": full_mrna,
            "full_mrna_unambiguous": full_unambig,
        })

    out = pd.DataFrame(rows)
    op = SRC / "HeLa_tis_union3_W100.parquet"
    out.to_parquet(op, index=False)
    out.drop(columns=["full_mrna"]).to_csv(op.with_suffix(".csv"), index=False)

    res = out[out["resolved"]]
    N = len(out)
    print(f"TIS: {N:,} | three-way resolved: {len(res):,} ({100*len(res)/N:.1f}%)")
    print("by agreement tier:")
    print(res.groupby(["agreement_tier", "agreement_label"]).size().to_string())
    print("\nby evidence (source precedence):")
    print(res.source_evidence.value_counts().to_string())
    print(f"\nresolved by >=2 methods: {int((res.n_methods >= 2).sum()):,}")
    print(f"full mRNA unambiguous: {int(res.full_mrna_unambiguous.sum()):,}")
    print(f"TIE computable: {int(res.tie_initiation_efficiency.notna().sum()):,}")
    ab = res[res.source_agreement_salmon_vs_longread.notna()]
    if len(ab):
        print(f"\nsalmon & long-read both resolved: {len(ab):,} | "
              f"agree on source isoform: {int(ab.source_agreement_salmon_vs_longread.sum()):,} "
              f"({100*ab.source_agreement_salmon_vs_longread.mean():.0f}%)")
    print(f"\nwrote {op}")


if __name__ == "__main__":
    main()
