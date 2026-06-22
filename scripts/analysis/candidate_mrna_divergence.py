"""Pull candidate-transcript mRNAs per TIS and test start-codon-window divergence.

Precursor to the long-read disambiguation step of the source-transcript
resolution workstream (``docs/plans/source_transcript_resolution.md``). For each
TIS (one genomic initiation site) detected in a cell line, this:

1. takes the candidate transcripts Ribo-TISH associated with that site
   (``all_transcripts`` in ``init_site_skeleton.parquet``);
2. pulls each candidate's full spliced mRNA **directly from the GENCODE
   transcriptome FASTA** — the exact reference the RNA-seq was mapped against —
   keyed by versioned ENST;
3. locates the TIS start codon *inside* each candidate's mRNA via the exon
   skeleton (``start_codon_index``), since the FASTA gives sequence but not the
   position of our (frequently non-canonical) start;
4. slices the ±W window around the start codon and runs the window-purity test
   (``purity_decision``) to report whether — and at what radius — the candidate
   mRNAs diverge around the start codon.

The transcriptome sequence is cross-checked against an independent
exon+genome reconstruction (``build_transcript_mrna``) so a coordinate-math
error surfaces as a mismatch rather than silently corrupting a window.

Outputs (under ``--out-dir``):
- ``{cell}_tis_window_summary.parquet`` — one row per TIS.
- ``{cell}_tis_candidate_detail.parquet`` — one row per (TIS × candidate).

Run via ``scripts/analysis/run_candidate_mrna_divergence.sbatch`` (the
exon-skeleton load over the 3 GB GTF is the heavy step).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import pysam

from swissisoform.io.gtf import load_exon_skeletons
from swissisoform.sourceseq.mrna import (
    build_transcript_mrna,
    extract_tis_window,
    start_codon_index,
)
from swissisoform.sourceseq.purity import purity_decision

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("candidate_mrna_divergence")


def _fasta_enst_index(fa: pysam.FastaFile) -> dict[str, str]:
    """Map bare versioned ENST -> full GENCODE FASTA header (the faidx key)."""
    return {ref.split("|")[0]: ref for ref in fa.references}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skeleton-table", default="data/output/init_site_skeleton.parquet")
    ap.add_argument("--transcriptome", default="data/reference/gencode.v49.transcripts.fa")
    ap.add_argument("--genome", default="data/reference/Gencode_v49_GRCh38.primary_assembly.genome.fa")
    ap.add_argument("--gtf", default="data/reference/gencode.v49.primary_assembly.annotation.gtf")
    ap.add_argument("--cell-line", default="HeLa")
    ap.add_argument("--min-candidates", type=int, default=2,
                    help="only TIS with >= this many candidate transcripts")
    ap.add_argument("-W", "--window", type=int, default=100, help="window radius (nt)")
    ap.add_argument("--out-dir", default="data/output/source_transcripts")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    W = args.window

    # --- TIS set: multi-candidate sites detected in the chosen cell line ---
    df = pd.read_parquet(args.skeleton_table)
    present_col = f"present_{args.cell_line}"
    if present_col not in df.columns:
        raise SystemExit(f"{present_col} not in {args.skeleton_table}; cols={list(df.columns)}")
    sel = df[(df[present_col] == True) & (df["n_transcripts"] >= args.min_candidates)].copy()  # noqa: E712
    logger.info("%s: %d multi-candidate TIS (>=%d candidates)", args.cell_line, len(sel), args.min_candidates)

    # --- references ---
    fa = pysam.FastaFile(args.transcriptome)
    enst_key = _fasta_enst_index(fa)
    genome = pysam.FastaFile(args.genome)
    logger.info("loading exon skeletons from %s ...", args.gtf)
    skeletons = load_exon_skeletons(args.gtf)
    logger.info("loaded %d exon skeletons", len(skeletons))

    summary_rows: list[dict] = []
    detail_rows: list[dict] = []

    for _, row in sel.iterrows():
        init_site = row["init_site"]
        gstart = int(row["gstart"])
        cands = [c for c in str(row["all_transcripts"]).split(",") if c]

        windows = []
        n_skel = n_window_ok = n_no_window = n_crosscheck_fail = 0
        for tid in cands:
            coords = skeletons.get(tid)
            has_skel = coords is not None
            n_skel += int(has_skel)
            fa_seq = fa.fetch(enst_key[tid]) if tid in enst_key else None

            a_idx = window_ok = clamped = crosscheck_ok = None
            up_seq = down_seq = None
            if has_skel:
                a_idx = start_codon_index(coords, gstart)
                if a_idx is None:
                    n_no_window += 1  # candidate's mRNA does not contain this start exonically
                else:
                    win = extract_tis_window(coords, genome, gstart, up=W, down=W)
                    if win is not None:
                        window_ok = True
                        n_window_ok += 1
                        clamped = win.clamped_5p
                        up_seq, down_seq = win.upstream, win.downstream
                        windows.append(win)
                        # cross-check: reconstructed mRNA == transcriptome FASTA
                        recon = build_transcript_mrna(coords, genome)
                        crosscheck_ok = (fa_seq is not None and recon == fa_seq)
                        n_crosscheck_fail += int(crosscheck_ok is False)
                    else:
                        n_no_window += 1

            detail_rows.append({
                "init_site": init_site, "gene": row["gene"], "Tid": tid,
                "has_skeleton": has_skel,
                "mrna_len_fasta": len(fa_seq) if fa_seq is not None else None,
                "a_index": a_idx, "window_ok": window_ok, "clamped_5p": clamped,
                "crosscheck_ok": crosscheck_ok,
                "upstream": up_seq, "downstream": down_seq,
            })

        pur = purity_decision(windows, W)
        # consensus window only meaningful when pure and at least one window exists
        consensus = None
        if pur.keep and windows:
            consensus = windows[0].upstream + "|" + windows[0].downstream
        n_distinct = len({(w.upstream, w.downstream) for w in windows})

        summary_rows.append({
            "init_site": init_site, "chrom": row["chrom"], "gstart": gstart,
            "strand": row["strand"], "start_codon": row["start_codon"],
            "gene": row["gene"], "orf_type": row["orf_type"],
            "n_candidates": len(cands), "n_skeleton_found": n_skel,
            "n_window_ok": n_window_ok, "n_no_window": n_no_window,
            "n_distinct_windows": n_distinct,
            "keep": pur.keep, "reason": pur.reason, "divergence_nt": pur.divergence_nt,
            "consensus_window": consensus,
            "crosscheck_fail": n_crosscheck_fail,
        })

    summary = pd.DataFrame(summary_rows)
    detail = pd.DataFrame(detail_rows)
    sp = out_dir / f"{args.cell_line}_tis_window_summary.parquet"
    dp = out_dir / f"{args.cell_line}_tis_candidate_detail.parquet"
    summary.to_parquet(sp, index=False)
    detail.to_parquet(dp, index=False)

    # --- report ---
    logger.info("wrote %s (%d TIS) and %s (%d candidate rows)", sp, len(summary), dp, len(detail))
    if not summary.empty:
        logger.info("reason breakdown:\n%s", summary["reason"].value_counts().to_string())
        logger.info("kept (pure/single) %d / %d  (%.1f%%)",
                    int(summary["keep"].sum()), len(summary),
                    100 * summary["keep"].mean())
        tot_cc = int(summary["crosscheck_fail"].sum())
        logger.info("sequence cross-check failures: %d (should be 0)", tot_cc)
        div = summary.loc[~summary["keep"] & summary["divergence_nt"].notna(), "divergence_nt"]
        if not div.empty:
            logger.info("ambiguous-TIS divergence radius (nt): median=%.0f min=%d max=%d",
                        div.median(), int(div.min()), int(div.max()))


if __name__ == "__main__":
    main()
