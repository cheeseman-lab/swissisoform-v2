"""5-gene end-to-end diagnostic pipeline — full module set.

Reference implementation of the complete annotation workflow on
TP53, EIF4G1, VEGFA, CTNND1, MYC (HeLa only).  Exercises every module
including precompute steps for DeepLoc, PepQuery, SignalP, TargetP,
InterProScan, and cache lookups for PLM VEP / Structure.

Usage:
    python scripts/inspect_e2e.py          # default 5 genes
    python scripts/inspect_e2e.py --genes TP53 MYC  # subset
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from swissisoform.assembly import assemble_genes
from swissisoform.clinical.validate import ConsequenceValidator
from swissisoform.compare.comparator import compare_genes
from swissisoform.config import (
    ClinicalConfig,
    ConservationConfig,
    PipelineConfig,
    ScoringConfig,
)
from swissisoform.io.parquet import paired_tis_dataframe
from swissisoform.io.ribotish import load_ribotish_predictions
from swissisoform.modules.biophysics import BiophysicsModule
from swissisoform.modules.clinical import ClinicalModule
from swissisoform.modules.conservation import ConservationModule
from swissisoform.modules.conservation_frame import ConservationFrameModule
from swissisoform.modules.core_identity import CoreIdentityModule
from swissisoform.modules.initiation_context import InitiationContextModule
from swissisoform.modules.interproscan import InterProScanModule, precompute_interproscan
from swissisoform.modules.localization import LocalizationModule, precompute_deeploc
from swissisoform.modules.massspec import (
    MassSpecModule,
    collect_unique_peptides,
    precompute_pepquery,
)
from swissisoform.modules.motifs import MotifsModule
from swissisoform.modules.plm_vep import PLMVEPModule
from swissisoform.modules.scoring import EvidenceScoringModule
from swissisoform.modules.signalp import SignalPModule, precompute_signalp
from swissisoform.modules.structure import StructureModule
from swissisoform.modules.targetp import TargetPModule, precompute_targetp
from swissisoform.modules.variant_intersection import VariantIntersectionModule
from swissisoform.pipeline import AnnotationPipeline, UpstreamReference, run_sample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("inspect_e2e")

# ── Paths ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "reference"
GTF = DATA / "gencode.v49.primary_assembly.annotation.gtf"
GENOME = DATA / "Gencode_v49_GRCh38.primary_assembly.genome.fa"
PROTEIN = DATA / "gencode.v49.pc_translations.fa"
HELA = DATA / "HeLa_TIS_predict_all.txt"
HELA_RNASEQ = [
    DATA / "rnaseq_counts/GATCAG_8_htseqcount.txt",
    DATA / "rnaseq_counts/CACGGT_9_htseqcount.txt",
]

# Reference DBs (optional — modules degrade gracefully when absent)
GNOMAD_DB = DATA / "gnomad" / "gnomad_v4.1_exome.parquet"
CLINVAR_DB = DATA / "clinvar" / "variant_summary.parquet"
COSMIC_DB = DATA / "cosmic" / "cosmic_variants.parquet"
PHYLOP_BW = DATA / "zoonomia" / "cactus241way.phyloP.bw"
PHASTCONS_BW = DATA / "zoonomia" / "hg38.phastCons100way.bw"
HAL_FILE = DATA / "zoonomia" / "241-mammalian-2020v2.hal"
HAL_SIF = DATA / "zoonomia" / "singularity" / "cactus.sif"
HAL2MAF_BIN = ROOT / "scripts" / "bin" / "hal2maf"
HALSTATS_BIN = ROOT / "scripts" / "bin" / "halStats"

DEFAULT_GENES = ["TP53", "EIF4G1", "VEGFA", "CTNND1", "MYC"]


# ── Configuration ────────────────────────────────────────────────────────


def build_config() -> PipelineConfig:
    """Build PipelineConfig populated with local DB paths when available."""
    cfg = PipelineConfig()
    cfg.clinical = ClinicalConfig(
        gnomad_db=GNOMAD_DB if GNOMAD_DB.exists() else None,
        clinvar_db=CLINVAR_DB if CLINVAR_DB.exists() else None,
        cosmic_db=COSMIC_DB if COSMIC_DB.exists() else None,
    )
    cfg.conservation = ConservationConfig(
        phylop_bigwig=PHYLOP_BW if PHYLOP_BW.exists() else None,
        phastcons_bigwig=PHASTCONS_BW if PHASTCONS_BW.exists() else None,
        hal_path=HAL_FILE if HAL_FILE.exists() else None,
        hal_tree_newick=None,  # pulled from HAL via halStats --tree
        hal2maf_binary=str(HAL2MAF_BIN) if HAL_SIF.exists() else "hal2maf",
        halstats_binary=str(HALSTATS_BIN) if HAL_SIF.exists() else "halStats",
        hal_ref_genome="Homo_sapiens",
    )
    cfg.scoring = ScoringConfig(
        primate_frac_intact_min=0.3,
        mammalian_frac_intact_min=0.2,
        phylop_coding_min=1.0,
        min_cell_lines=1,
        existence_high_threshold=3,
        functional_high_threshold=2,
    )
    return cfg


# ── Precompute helpers ───────────────────────────────────────────────────


def collect_all_proteins(genes) -> list[str]:
    """Return every protein sequence (canonical + isoform) across genes."""
    proteins: list[str] = []
    for g in genes:
        if g.canonical_protein:
            proteins.append(g.canonical_protein)
        for s in g.tis_sites:
            if s.isoform_protein:
                proteins.append(s.isoform_protein)
            if s.canonical_protein:
                proteins.append(s.canonical_protein)
    return proteins


def write_proteins_fasta(proteins: list[str], out_path: Path) -> int:
    """Write deduped, sha1-keyed FASTA for GPU precompute jobs."""
    from swissisoform.plm.embed import protein_hash

    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    n = 0
    with open(out_path, "w") as fh:
        for seq in proteins:
            stripped = seq.rstrip("*").upper()
            if not stripped:
                continue
            h = protein_hash(stripped)
            if h in seen:
                continue
            seen.add(h)
            fh.write(f">{h}\n{stripped}\n")
            n += 1
    return n


def run_precompute(genes, all_proteins: list[str]) -> dict:
    """Run all precompute steps.  Returns dict of prediction lookups."""
    preds = {}

    # DeepLoc (CPU, subprocess into py3.8 env)
    deeploc_input = {}
    for gene in genes:
        deeploc_input[f"canonical_{gene.gene_name}"] = gene.canonical_protein
        for site in gene.tis_sites:
            deeploc_input[site.tis_id] = site.isoform_protein
    preds["deeploc"] = precompute_deeploc(deeploc_input, model="Fast", device="cpu")

    # PepQuery (Java subprocess + cache)
    unique_peps = collect_unique_peptides(genes)
    preds["pepquery"] = precompute_pepquery(
        unique_peps,
        dataset="Deep_29_healthy_human_tissues_PXD010154,GTEx_32_Tissues_Proteome_PXD016999",
        reference_db="swissprot:human",
        cache_dir=ROOT / "data" / "cache" / "pepquery",
    )

    # SignalP, TargetP, InterProScan (subprocess / Nextflow)
    preds["signalp"] = precompute_signalp(all_proteins)
    preds["targetp"] = precompute_targetp(all_proteins)
    preds["interproscan"] = precompute_interproscan(all_proteins)

    # PLM VEP + Structure: cache lookup only (populate via sbatch scripts)
    from swissisoform.plm.embed import precompute_plm_esm2
    from swissisoform.structure.fold import precompute_fold

    preds["plm"] = precompute_plm_esm2(all_proteins, inline=False)
    preds["structure"] = precompute_fold(all_proteins, backend="boltz", inline=False)

    return preds


def build_pipeline(cfg, preds, ref, genes) -> AnnotationPipeline:
    """Construct the full annotation pipeline with all 16 modules."""
    validator = ConsequenceValidator(cds_df=ref.cds_df, genome_fasta=str(GENOME))
    clinical_mod = ClinicalModule(cfg, validator=validator)

    # Prefetch clinical variants per gene
    gene_by_name = {g.gene_name: g for g in genes}
    for gene_name in sorted(gene_by_name):
        canonical_tid = gene_by_name[gene_name].canonical_transcript_id
        variants = clinical_mod.fetch_variants(gene_name, transcript_id=canonical_tid)
        clinical_mod._variant_cache[gene_name] = variants

    return AnnotationPipeline(
        protein_modules=[
            BiophysicsModule(cfg),
            MotifsModule(cfg),
            MassSpecModule(cfg, validated_peptides=preds["pepquery"]),
            clinical_mod,
            LocalizationModule(cfg, predictions=preds["deeploc"]),
            SignalPModule(cfg, predictions=preds["signalp"]),
            TargetPModule(cfg, predictions=preds["targetp"]),
            InterProScanModule(cfg, predictions=preds["interproscan"]),
        ],
        site_modules=[
            CoreIdentityModule(cfg),
            InitiationContextModule(cfg),
            ConservationModule(cfg),
            ConservationFrameModule(cfg),
            VariantIntersectionModule(),
            PLMVEPModule(cfg),
            StructureModule(cfg),
        ],
    )


# ── Spot-check output ────────────────────────────────────────────────────


def print_spot_check(genes) -> None:
    """Print condensed per-gene / per-TIS spot check."""
    for gene in sorted(genes, key=lambda g: g.gene_name):
        can = gene.canonical_annotations
        bio = can.get("biophysics", {})
        clin_sum = can.get("clinical", {}).get("summary", {})
        loc = can.get("localization", {})
        n_tis = len(gene.tis_sites)
        print(
            f"\n{gene.gene_name}  ({gene.canonical_transcript_id}, "
            f"{len(gene.canonical_protein.rstrip('*'))} aa, {n_tis} TIS)"
        )
        print(
            f"  bio pI={bio.get('pI', 0):.2f} gravy={bio.get('gravy', 0):.3f}  "
            f"clin={clin_sum.get('total_variants', 0)} vars  "
            f"loc={loc.get('deeploc_prediction')}"
        )

        for site in sorted(gene.tis_sites, key=lambda s: (s.orf_type.value, s.position)):
            ia = site.isoform_annotations
            isc = ia.get("scoring", {}) or {}
            iplm = ia.get("plm_vep", {}) or {}
            istr = ia.get("structure", {}) or {}
            ilen = len(site.isoform_protein.rstrip("*"))
            kozak = site.kozak_context or "—"
            print(
                f"    TIS {site.tis_id} ({site.orf_type.value}, {ilen} aa, kozak={kozak})"
            )
            print(
                f"      score: "
                f"existence={isc.get('existence_score')}/{isc.get('existence_evaluable')} "
                f"(hi={isc.get('existence_high_confidence')})  "
                f"functional={isc.get('functional_score')}/{isc.get('functional_evaluable')} "
                f"(hi={isc.get('functional_high_confidence')})"
            )
            print(
                f"      plm[{iplm.get('status', '—')}]  "
                f"struct[{istr.get('status', '—')}/{istr.get('backend', '—')}]"
            )


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="SwissIsoform v2 — 5-gene E2E diagnostic")
    parser.add_argument(
        "--genes", nargs="*", default=DEFAULT_GENES,
        help=f"Genes to test (default: {' '.join(DEFAULT_GENES)})",
    )
    args = parser.parse_args()
    test_genes = args.genes

    # Stage 0: Load raw predictions
    t_start = time.perf_counter()
    raw = load_ribotish_predictions(HELA)
    logger.info("Raw Ribo-TISH: %d rows", len(raw))

    # Stage 1: Upstream (filter + impute)
    ref = UpstreamReference.load(gtf_path=GTF, genome_fasta=GENOME, protein_fasta=PROTEIN)
    final, dropped = run_sample(HELA, HELA_RNASEQ, GTF, sample="HeLa", reference=ref)
    logger.info(
        "Upstream: %d kept, %d dropped, %d imputed",
        len(final), len(dropped), final["Imputed"].sum(),
    )

    # Stage 2: Assembly
    genes = assemble_genes(
        final[final["Symbol"].isin(test_genes)],
        gene_names=test_genes,
        genome_fasta=GENOME,
        exon_skeletons=ref.exon_skeletons,
    )
    n_tis = sum(len(g.tis_sites) for g in genes)
    logger.info("Assembled %d genes, %d TIS", len(genes), n_tis)
    for g in sorted(genes, key=lambda x: x.gene_name):
        logger.info("  %s: %d TIS, canonical %s (%d aa)",
                     g.gene_name, len(g.tis_sites), g.canonical_transcript_id,
                     len(g.canonical_protein.rstrip("*")))

    # Stage 3: Precompute
    cfg = build_config()
    all_proteins = collect_all_proteins(genes)

    n_written = write_proteins_fasta(all_proteins, ROOT / "data" / "cache" / "proteins.fa")
    logger.info("proteins.fa: %d unique proteins", n_written)

    preds = run_precompute(genes, all_proteins)
    logger.info(
        "Precompute done: deeploc=%d signalp=%d targetp=%d ips=%d plm=%d struct=%d",
        len(preds["deeploc"]), len(preds["signalp"]), len(preds["targetp"]),
        len(preds["interproscan"]), len(preds["plm"]), len(preds["structure"]),
    )

    # Stage 4: Annotate + compare + score
    pipeline = build_pipeline(cfg, preds, ref, genes)
    pipeline.run(genes)
    compare_genes(genes, scope_a_modules=[BiophysicsModule(cfg)])

    scoring_mod = EvidenceScoringModule(cfg)
    all_sites = [site for gene in genes for site in gene.tis_sites]
    scoring_mod.run(all_sites)
    logger.info("Annotation + comparison + scoring complete")

    # Stage 5: Spot check
    print_spot_check(genes)

    # Stage 6: Output
    out_dir = ROOT / "data" / "output" / "5gene_e2e"
    out_dir.mkdir(parents=True, exist_ok=True)
    paired = paired_tis_dataframe(genes)
    all_path = out_dir / "all_paired.parquet"
    paired.to_parquet(all_path, index=False)
    print(f"\nwrote {all_path} ({len(paired)} rows, {len(paired.columns)} cols)")
    for gene_name, sub in paired.groupby("gene_name"):
        gpath = out_dir / f"{gene_name}_paired.parquet"
        sub.to_parquet(gpath, index=False)
        print(f"  {gene_name}: {len(sub)} rows -> {gpath.name}")

    elapsed = time.perf_counter() - t_start
    logger.info("Total wall time: %.1f min", elapsed / 60)


if __name__ == "__main__":
    main()

