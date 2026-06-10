"""Thin CLI for the differential-ORF BED12 export.

Logic lives in ``swissisoform.export.bed``. For every alternative TIS in the
cross-cell-line combined catalog, emit a BED12 record covering the
isoform-unique region (strand-aware, intron-aware).

Usage:
    python scripts/export/export_alt_regions_bed.py [--genes ...] [--out PATH]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from swissisoform.assembly import assemble_genes
from swissisoform.export.bed import gene_bed_rows
from swissisoform.pipeline import UpstreamReference

logger = logging.getLogger("alt_bed")

ROOT = Path(__file__).resolve().parent.parent
COMBINED_PARQUET = ROOT / "data" / "output" / "filtered" / "all_samples_combined.parquet"
GTF = ROOT / "data" / "reference" / "gencode.v49.primary_assembly.annotation.gtf"
GENOME = ROOT / "data" / "reference" / "Gencode_v49_GRCh38.primary_assembly.genome.fa"
PROTEIN = ROOT / "data" / "reference" / "gencode.v49.pc_translations.fa"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes", nargs="*", default=None, help="Optional gene-name subset")
    ap.add_argument("--out", default=str(ROOT / "data" / "output" / "alt_orf_regions.bed"))
    args = ap.parse_args()

    logger.info("Loading combined catalog: %s", COMBINED_PARQUET)
    df = pd.read_parquet(COMBINED_PARQUET)
    if args.genes:
        before = len(df)
        df = df[df["Symbol"].isin(args.genes)]
        logger.info("Filtered to %d genes -> %d rows (was %d)", len(args.genes), len(df), before)

    samples = sorted(col[len("present_"):] for col in df.columns if col.startswith("present_"))
    logger.info("Samples: %s", samples)

    logger.info("Loading reference (GTF + exon skeletons)")
    ref = UpstreamReference.load(gtf_path=GTF, genome_fasta=GENOME, protein_fasta=PROTEIN)

    logger.info("Assembling genes from %d rows", len(df))
    genes = assemble_genes(
        df,
        gene_names=args.genes,
        genome_fasta=str(GENOME),
        exon_skeletons=ref.exon_skeletons,
    )
    logger.info("Assembled %d genes", len(genes))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    n_skipped_no_diff = 0
    n_genes_with_rows = 0
    with open(out_path, "w") as fh:
        fh.write("track name=\"SwissIsoform_alt_regions\" "
                 "description=\"Alternative-TIS differential ORF regions (Ribo-TISH)\" "
                 "itemRgb=On useScore=1\n")
        for gene in sorted(genes, key=lambda g: g.gene_name):
            rows = gene_bed_rows(gene, samples or [])
            if rows:
                n_genes_with_rows += 1
                n_total += len(rows)
                for r in rows:
                    fh.write(r + "\n")
            else:
                n_skipped_no_diff += 1
    logger.info("Wrote %d BED12 records across %d genes -> %s",
                n_total, n_genes_with_rows, out_path)
    logger.info("Genes with no alt-region rows: %d", n_skipped_no_diff)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
