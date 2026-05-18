"""Full-pipeline data loading: upstream → combine → assemble.

Loads all cell lines, filters/imputes, combines into a single deduplicated
table, and assembles Gene + TIS domain objects.  Module wiring is NOT done
here — see ``inspect_e2e.py`` for the fully-wired 5-gene reference
implementation.  Copy that pattern when extending to all genes.

Usage:
    python scripts/run_e2e_all.py [--genes GENE1 GENE2 ...]

Without ``--genes``, assembles every gene in the combined dataset.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from swissisoform.assembly import assemble_genes
from swissisoform.combine import combine_filtered_samples
from swissisoform.config import PipelineConfig
from swissisoform.io.rnaseq import load_sample_manifest
from swissisoform.pipeline import UpstreamReference, run_sample

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "reference"
OUT = ROOT / "data" / "output"

GTF = DATA / "gencode.v49.primary_assembly.annotation.gtf"
GENOME = DATA / "Gencode_v49_GRCh38.primary_assembly.genome.fa"
PROTEIN = DATA / "gencode.v49.pc_translations.fa"
SAMPLE_MANIFEST = DATA / "ribotish_sample_manifest.csv"
REPLICATE_MANIFEST = DATA / "ribotish_replicate_manifest.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_e2e_all")


def run_upstream_all(
    reference: UpstreamReference,
    cfg: PipelineConfig,
) -> dict[str, pd.DataFrame]:
    """Run upstream per cell line.  Write CSV, return in-memory dict."""
    manifest = load_sample_manifest(SAMPLE_MANIFEST, REPLICATE_MANIFEST)
    per_sample: dict[str, pd.DataFrame] = {}

    for _, row in manifest.iterrows():
        sample = row["sample"]
        predict = ROOT / row["predict_file"]
        filtered_out = ROOT / row["filtered_file"]
        rnaseq_files = [ROOT / f for f in row["rnaseq_count_files"]]

        t0 = time.perf_counter()
        final_df, _ = run_sample(
            predict,
            rnaseq_files,
            GTF,
            sample=sample,
            config=cfg,
            reference=reference,
        )
        filtered_out.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(filtered_out, index=False)
        per_sample[sample] = final_df
        logger.info(
            "%s: %d filtered rows (%.1fs)",
            sample,
            len(final_df),
            time.perf_counter() - t0,
        )
    return per_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="SwissIsoform v2 — data loading pipeline")
    parser.add_argument(
        "--genes", nargs="*", default=None,
        help="Restrict assembly to these gene names (default: all genes)",
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = PipelineConfig()

    # ── Stage 1: Load shared references ──────────────────────────────────
    logger.info("Loading GTF + genome + protein-product reference tables")
    reference = UpstreamReference.load(gtf_path=GTF, genome_fasta=GENOME, protein_fasta=PROTEIN)

    # ── Stage 2: Upstream per cell line ──────────────────────────────────
    logger.info("Running upstream (filter + impute) per cell line")
    per_sample = run_upstream_all(reference, cfg)

    # ── Stage 3: Combine ─────────────────────────────────────────────────
    combined = combine_filtered_samples(per_sample)
    combined_out = OUT / "filtered" / "all_samples_combined.parquet"
    combined_out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(combined_out, index=False)
    logger.info(
        "Combined upstream: %d unique TIS across %d samples -> %s",
        len(combined),
        len(per_sample),
        combined_out,
    )

    # ── Stage 4: Assembly ────────────────────────────────────────────────
    t0 = time.perf_counter()
    genes = assemble_genes(
        combined,
        gene_names=args.genes,
        genome_fasta=GENOME,
        exon_skeletons=reference.exon_skeletons,
    )
    n_tis = sum(len(g.tis_sites) for g in genes)
    logger.info(
        "Assembled %d genes, %d unique TIS (%.1fs)",
        len(genes),
        n_tis,
        time.perf_counter() - t0,
    )

    # ── Done ─────────────────────────────────────────────────────────────
    # Module wiring, precompute, annotation, comparison, and scoring are
    # handled by inspect_e2e.py (5-gene reference) or a future production
    # script that mirrors its pattern for all genes.
    logger.info(
        "Data loading complete.  %d genes with %d TIS ready for annotation.",
        len(genes),
        n_tis,
    )


if __name__ == "__main__":
    main()

