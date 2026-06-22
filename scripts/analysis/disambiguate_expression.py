"""Disambiguate each TIS's candidate transcripts by expression, then re-test purity.

Second step of the source-transcript funnel: take the per-TIS candidate windows
already extracted by ``candidate_mrna_divergence.py`` and, for each TIS, keep only
the candidates **expressed** in the cell line, then re-run the window-purity test
on the survivors. Answers "does the expression evidence resolve the source
transcript?" — the candidate-reduction + ambiguous→pure transitions are the
disambiguation-power metric in ``docs/plans/source_transcript_resolution.md``.

Provider-agnostic: ``--provider salmon`` (Arm A, per-replicate TPM, presence =
TPM ≥ τ in every rep) or ``--provider isoquant`` (Arm B, counts ≥ τ). Same
downstream logic, so the two arms are directly comparable.

Output: ``{cell}_disambiguation_{provider}.parquet`` — the window summary
augmented with post-expression columns (``n_expressed_window``, ``keep_expr``,
``reason_expr``, ``divergence_nt_expr``).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from swissisoform.sourceseq.expression import (
    expressed_in_replicates,
    expressed_transcripts,
    load_isoquant_abundance,
    load_salmon_replicates,
)
from swissisoform.sourceseq.mrna import TisWindow
from swissisoform.sourceseq.purity import purity_decision

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("disambiguate_expression")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell-line", default="HeLa")
    ap.add_argument("--summary", default="data/output/source_transcripts/{cell}_tis_window_summary.parquet")
    ap.add_argument("--detail", default="data/output/source_transcripts/{cell}_tis_candidate_detail.parquet")
    ap.add_argument("--provider", choices=["salmon", "isoquant"], default="salmon")
    ap.add_argument("--salmon-quant", nargs="+",
                    default=["data/reference/salmon/HeLa_rep1/quant.sf",
                             "data/reference/salmon/HeLa_rep2/quant.sf"])
    ap.add_argument("--isoquant-table", default=None)
    ap.add_argument("--min-tpm", type=float, default=1.0, help="salmon presence threshold")
    ap.add_argument("--min-counts", type=float, default=3.0, help="isoquant presence threshold")
    ap.add_argument("-W", "--window", type=int, default=100)
    ap.add_argument("--out-dir", default="data/output/source_transcripts")
    args = ap.parse_args()

    cell = args.cell_line
    W = args.window
    summary = pd.read_parquet(args.summary.format(cell=cell))
    detail = pd.read_parquet(args.detail.format(cell=cell))

    # --- present-transcript set + abundance map from the chosen provider ---
    # `expressed` is the presence set (drives filtering); `abundance` holds the
    # per-transcript value (salmon mean TPM / isoquant count) used to tag each
    # resolved TIS with its source-isoform confidence.
    if args.provider == "salmon":
        expressed = expressed_in_replicates(args.salmon_quant, min_tpm=args.min_tpm, require="all")
        abundance = load_salmon_replicates(args.salmon_quant)
        logger.info("salmon: %d transcripts present (TPM>=%.2f in all reps)", len(expressed), args.min_tpm)
    else:
        if not args.isoquant_table:
            raise SystemExit("--isoquant-table required for --provider isoquant")
        abundance = load_isoquant_abundance(args.isoquant_table)
        expressed = expressed_transcripts(abundance, args.min_counts)
        logger.info("isoquant: %d transcripts present (counts>=%.0f)", len(expressed), args.min_counts)

    # windowed candidates only (purity operates on candidates that contain the start)
    win = detail[detail["window_ok"] == True].copy()  # noqa: E712
    win["expressed"] = win["Tid"].isin(expressed)

    rows = []
    for init_site, grp in win.groupby("init_site"):
        kept = grp[grp["expressed"]]
        windows = [TisWindow(upstream=r.upstream or "", downstream=r.downstream or "",
                             a_index=int(r.a_index), clamped_5p=bool(r.clamped_5p))
                   for r in kept.itertuples()]
        pur = purity_decision(windows, W)
        # confidence tag: the surviving isoforms' abundance. Representative =
        # the max-abundance survivor (the plan's choice when several agree
        # in-window); source_abundance_sum = total over survivors (the plan's
        # efficiency denominator for indistinguishable agreeing candidates).
        survivors = [(r.Tid, float(abundance.get(r.Tid, 0.0))) for r in kept.itertuples()]
        if survivors:
            rep_tid, rep_val = max(survivors, key=lambda x: x[1])
            sum_val = sum(v for _, v in survivors)
        else:
            rep_tid, rep_val, sum_val = None, None, None
        rows.append({
            "init_site": init_site,
            "n_window": len(grp),                  # windowed candidates before expression
            "n_expressed_window": len(kept),       # after expression filter
            "keep_expr": pur.keep,
            "reason_expr": pur.reason if windows else "no_expressed_candidate",
            "divergence_nt_expr": pur.divergence_nt,
            "source_tid": rep_tid,                 # max-abundance surviving isoform
            "source_abundance": rep_val,           # its value (salmon mean TPM / isoquant count)
            "source_abundance_sum": sum_val,       # summed over agreeing survivors
            "provider": args.provider,
        })
    expr = pd.DataFrame(rows)
    out = summary.merge(expr, on="init_site", how="left")

    out_path = Path(args.out_dir) / f"{cell}_disambiguation_{args.provider}.parquet"
    out.to_parquet(out_path, index=False)

    # --- report: before vs after ---
    n = len(out)
    before_pure = out["keep"].sum()
    after_pure = out["keep_expr"].fillna(False).sum()
    lost = (out["reason_expr"] == "no_expressed_candidate").sum()
    amb_before = ~out["keep"]
    newly = (amb_before & out["keep_expr"].fillna(False)).sum()
    logger.info("wrote %s", out_path)
    logger.info("TIS: %d", n)
    logger.info("window-pure BEFORE expression: %d (%.1f%%)", before_pure, 100 * before_pure / n)
    logger.info("window-pure AFTER  expression: %d (%.1f%%)", after_pure, 100 * after_pure / n)
    logger.info("  newly resolved (ambiguous -> pure): %d", newly)
    logger.info("  TIS with NO expressed windowed candidate (lost): %d", lost)
    red = out.loc[out["n_expressed_window"].notna()]
    logger.info("candidate count: mean %.2f windowed -> %.2f expressed",
                red["n_window"].mean(), red["n_expressed_window"].mean())
    logger.info("reason_expr breakdown:\n%s", out["reason_expr"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
