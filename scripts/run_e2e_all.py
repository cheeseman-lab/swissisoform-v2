"""End-to-end pipeline: combine all cell lines first, annotate once.

Flow:

    for each cell line:
        run_sample (filter + impute)       -> per-sample filtered CSV
    combine_filtered_samples                -> one wide deduped table
    assemble_genes (one pass)               -> Gene objects with
                                                TIS.expression populated
                                                per cell line
    AnnotationPipeline (one pass)           -> biophysics, motifs,
                                                core_identity,
                                                initiation_context
    tis_to_dataframe                        -> single parquet

Every unique TIS is annotated exactly once, regardless of how many
samples called it.  This is the precondition for wiring the expensive
modules (clinical, conservation, massspec) cost-effectively —
deduplication is 6× — 15× fewer module invocations vs the per-sample
loop.

Excludes the expensive modules; those come next with caching.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from swissisoform.assembly import assemble_genes
from swissisoform.combine import combine_filtered_samples
from swissisoform.config import ConservationConfig, PipelineConfig
from swissisoform.io.parquet import tis_to_dataframe
from swissisoform.io.rnaseq import load_sample_manifest
from swissisoform.modules.biophysics import BiophysicsModule
from swissisoform.modules.conservation import ConservationModule
from swissisoform.modules.conservation_frame import ConservationFrameModule
from swissisoform.modules.core_identity import CoreIdentityModule
from swissisoform.modules.initiation_context import InitiationContextModule
from swissisoform.modules.massspec import MassSpecModule
from swissisoform.modules.motifs import MotifsModule
from swissisoform.pipeline import AnnotationPipeline, UpstreamReference, run_sample

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "reference"
OUT = ROOT / "data" / "output"

GTF = DATA / "gencode.v49.primary_assembly.annotation.gtf"
GENOME = DATA / "Gencode_v49_GRCh38.primary_assembly.genome.fa"
PROTEIN = DATA / "gencode.v49.pc_translations.fa"
SAMPLE_MANIFEST = DATA / "ribotish_sample_manifest.csv"
REPLICATE_MANIFEST = DATA / "ribotish_replicate_manifest.csv"

ZOONOMIA = DATA / "zoonomia"
PHYLOP_BW = ZOONOMIA / "cactus241way.phyloP.bw"
PHASTCONS_BW = ZOONOMIA / "hg38.phastCons100way.bw"
HAL_PATH = ZOONOMIA / "241-mammalian-2020v2.hal"
HAL_TREE = ZOONOMIA / "hg38.cactus241way.nh"
HAL2MAF_BIN = ROOT / "scripts" / "bin" / "hal2maf"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_e2e_all")


def hdr(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


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


def spot_check(combined_annot: pd.DataFrame, samples: list[str]) -> None:
    hdr("SPOT CHECK — combined annotated DataFrame")
    print(f"shape: {combined_annot.shape}  (one row per unique TIS)")
    print(f"unique genes: {combined_annot['gene_name'].nunique():,}")

    hdr("TIS presence across samples")
    # Presence flags from combine step were on the upstream combined table,
    # not the annotated one; re-derive from expression columns that survived
    # serialization.
    presence_cols = [c for c in combined_annot.columns if c.endswith("_raw_count")]
    presence_cols.sort()
    if presence_cols:
        presence = combined_annot[presence_cols].notna()
        presence.columns = [c.replace("_raw_count", "") for c in presence.columns]
        print("n_samples a TIS was called in (distribution):")
        dist = presence.sum(axis=1).value_counts().sort_index()
        print(dist.to_string())
        print("\nTIS count by sample (from expression columns):")
        print(presence.sum().to_string())

    hdr("ORF type distribution (one row = one unique TIS)")
    print(combined_annot["orf_type"].value_counts().to_string())

    hdr("Sample rows — 5 random TIS with per-sample expression")
    show = [
        "gene_name",
        "transcript_id",
        "orf_type",
        "start_codon",
        "biophysics_length",
        "biophysics_pI",
    ]
    show = [c for c in show if c in combined_annot.columns]
    expr_show = [c for c in combined_annot.columns if c.endswith("_raw_count")][:3]
    view = combined_annot[show + expr_show].sample(n=min(5, len(combined_annot)), random_state=0)
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(view.to_string(index=False))


def build_conservation_config() -> ConservationConfig:
    """Build ConservationConfig from files in ``data/reference/zoonomia/``.

    Fields are populated only when the underlying file exists — missing
    BigWigs / HAL cause the respective modules to emit ``status="not_run"``
    rather than crash.
    """
    tree_newick = HAL_TREE.read_text() if HAL_TREE.exists() else None
    return ConservationConfig(
        phylop_bigwig=PHYLOP_BW if PHYLOP_BW.exists() else None,
        phastcons_bigwig=PHASTCONS_BW if PHASTCONS_BW.exists() else None,
        hal_path=HAL_PATH if HAL_PATH.exists() else None,
        hal2maf_binary=str(HAL2MAF_BIN) if HAL2MAF_BIN.exists() else "hal2maf",
        hal_tree_newick=tree_newick,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = PipelineConfig(conservation=build_conservation_config())

    print("Loading shared GTF + genome + protein-product reference tables…")
    reference = UpstreamReference.load(gtf_path=GTF, genome_fasta=GENOME, protein_fasta=PROTEIN)

    hdr("STAGE 1 — Upstream per cell line (filter + impute)")
    per_sample = run_upstream_all(reference, cfg)

    hdr("STAGE 2 — Combine")
    combined = combine_filtered_samples(per_sample)
    combined_upstream_out = OUT / "filtered" / "all_samples_combined.parquet"
    combined.to_parquet(combined_upstream_out, index=False)
    logger.info(
        "combined upstream: %d unique TIS across %d samples → %s",
        len(combined),
        len(per_sample),
        combined_upstream_out,
    )

    hdr("STAGE 3 — Assembly (one pass)")
    t0 = time.perf_counter()
    genes = assemble_genes(
        combined,
        genome_fasta=GENOME,
        exon_skeletons=reference.exon_skeletons,
    )
    logger.info(
        "assembled %d genes, %d unique TIS (%.1fs)",
        len(genes),
        sum(len(g.tis_sites) for g in genes),
        time.perf_counter() - t0,
    )

    hdr("STAGE 4 — Annotation (one pass)")
    pipeline = AnnotationPipeline(
        protein_modules=[
            BiophysicsModule(cfg),
            MotifsModule(cfg),
            MassSpecModule(cfg),
        ],
        site_modules=[
            CoreIdentityModule(cfg),
            InitiationContextModule(cfg),
            ConservationModule(cfg),
            ConservationFrameModule(cfg),
        ],
    )
    t0 = time.perf_counter()
    pipeline.run(genes)
    logger.info("annotation complete (%.1fs)", time.perf_counter() - t0)

    hdr("STAGE 5 — Serialize")
    all_sites = [s for g in genes for s in g.tis_sites]
    annot_df = tis_to_dataframe(all_sites)
    out_path = OUT / "annotated" / "all_samples_annotated.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    annot_df.to_parquet(out_path, index=False)
    logger.info("wrote %d rows × %d cols → %s", *annot_df.shape, out_path)

    spot_check(annot_df, sorted(per_sample))


if __name__ == "__main__":
    main()
