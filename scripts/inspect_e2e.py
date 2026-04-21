"""Diagnostic walkthrough of the 5-gene E2E pipeline — full module set.

Runs:

    run_sample (upstream)
      -> assemble_genes (Gene + TIS objects)
      -> AnnotationPipeline with:
           cheap modules:  biophysics, motifs, core_identity, initiation_context
           expensive:      clinical, conservation, massspec, localization

For expensive modules:
    - clinical:     reads ClinVar + (when available) gnomAD + COSMIC
                    local parquets built by scripts/setup_databases.py.
                    Falls back to live HTTP for any missing DB.
    - conservation: Zoonomia PhyloP / PhastCons BigWig lookups at the TIS
                    and Kozak window.  Tracks are optional — each is looked
                    up independently and returns None when the BigWig is
                    absent.
    - massspec:     inline tryptic digest (no DB).
    - localization: precomputes DeepLoc2 on CPU for unique proteins; if
                    DeepLoc2 is not importable the module no-ops with
                    warnings.

Prints per-gene spot checks for each module's output.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from swissisoform.assembly import assemble_genes
from swissisoform.clinical.validate import ConsequenceValidator
from swissisoform.config import (
    ClinicalConfig,
    ConservationConfig,
    PipelineConfig,
)
from swissisoform.io.parquet import paired_tis_dataframe
from swissisoform.io.ribotish import load_ribotish_predictions
from swissisoform.modules.biophysics import BiophysicsModule
from swissisoform.modules.clinical import ClinicalModule
from swissisoform.modules.conservation import ConservationModule
from swissisoform.modules.core_identity import CoreIdentityModule
from swissisoform.modules.initiation_context import InitiationContextModule
from swissisoform.modules.localization import (
    LocalizationModule,
    precompute_deeploc,
)
from swissisoform.modules.massspec import MassSpecModule
from swissisoform.modules.motifs import MotifsModule
from swissisoform.pipeline import AnnotationPipeline, UpstreamReference, run_sample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

DATA = Path(__file__).parent.parent / "data" / "reference"
GTF = DATA / "gencode.v49.primary_assembly.annotation.gtf"
GENOME = DATA / "Gencode_v49_GRCh38.primary_assembly.genome.fa"
PROTEIN = DATA / "gencode.v49.pc_translations.fa"
HELA = DATA / "HeLa_TIS_predict_all.txt"
HELA_RNASEQ = [
    DATA / "rnaseq_counts/GATCAG_8_htseqcount.txt",
    DATA / "rnaseq_counts/CACGGT_9_htseqcount.txt",
]

# Paths to reference DBs built by scripts/setup_databases.py and
# scripts/download_zoonomia_bigwigs.sh.  Each is optional — the module
# gracefully falls back to live HTTP, no-ops, or empty lookups when a DB
# is missing.
GNOMAD_DB = DATA / "gnomad" / "gnomad_v4.1_exome.parquet"
CLINVAR_DB = DATA / "clinvar" / "variant_summary.parquet"
COSMIC_DB = DATA / "cosmic" / "cosmic_variants.parquet"
PHYLOP_BW = DATA / "zoonomia" / "cactus241way.phyloP.bw"
PHASTCONS_BW = DATA / "zoonomia" / "cactus241way.phastCons.bw"

TEST_GENES = ["TP53", "EIF4G1", "VEGFA", "CTNND1", "MYC"]


def hdr(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def build_config() -> PipelineConfig:
    """Build a PipelineConfig populated with local DB paths when available."""
    cfg = PipelineConfig()
    cfg.clinical = ClinicalConfig(
        gnomad_db=GNOMAD_DB if GNOMAD_DB.exists() else None,
        clinvar_db=CLINVAR_DB if CLINVAR_DB.exists() else None,
        cosmic_db=COSMIC_DB if COSMIC_DB.exists() else None,
    )
    cfg.conservation = ConservationConfig(
        phylop_bigwig=PHYLOP_BW if PHYLOP_BW.exists() else None,
        phastcons_bigwig=PHASTCONS_BW if PHASTCONS_BW.exists() else None,
    )
    return cfg


def precompute_localization(genes) -> dict:
    """Run DeepLoc2 on every unique protein across *genes* (CPU, Fast model).

    Returns a ``{protein_hash: prediction}`` dict that `LocalizationModule`
    consumes.  Returns empty dict if DeepLoc2 isn't importable — the
    module degrades gracefully.
    """
    proteins: dict[str, str] = {}
    for gene in genes:
        proteins[f"canonical_{gene.gene_name}"] = gene.canonical_protein
        for site in gene.tis_sites:
            proteins[site.tis_id] = site.isoform_protein
    print(f"precompute_localization: {len(proteins)} proteins (deduped internally)")
    return precompute_deeploc(proteins, model="Fast", device="cpu")


def main() -> None:
    hdr("STAGE 0 — Raw Ribo-TISH")
    raw = load_ribotish_predictions(HELA)
    print(f"Raw rows: {len(raw):,}")
    for g in TEST_GENES:
        gdf = raw[raw["Symbol"] == g]
        ann = gdf[gdf["TisType"].str.startswith("Annotated")]
        print(f"  {g:10s} raw={len(gdf):>4}  native_annotated={len(ann):>3}")

    hdr("STAGE 1 — Upstream (filter + impute + drop uncanonical)")
    ref = UpstreamReference.load(gtf_path=GTF, genome_fasta=GENOME, protein_fasta=PROTEIN)
    final, dropped = run_sample(HELA, HELA_RNASEQ, GTF, sample="HeLa", reference=ref)
    print(f"final: {len(final):,}  dropped: {len(dropped):,}  imputed: {final['Imputed'].sum():,}")
    for g in TEST_GENES:
        fg = final[final["Symbol"] == g]
        native_ann = fg[fg["TisType"].str.startswith("Annotated") & ~fg["Imputed"]]
        imp_ann = fg[fg["Imputed"]]
        alt = fg[~fg["TisType"].str.startswith("Annotated")]
        types = Counter(alt["TisType"])
        print(
            f"  {g:10s} alt={len(alt):>3}  native_ann={len(native_ann):>2}  "
            f"imputed_ann={len(imp_ann):>2}  "
            f"[{', '.join(f'{k}={v}' for k, v in sorted(types.items()))}]"
        )

    hdr("STAGE 2 — Assembly")
    genes = assemble_genes(
        final[final["Symbol"].isin(TEST_GENES)],
        gene_names=TEST_GENES,
        genome_fasta=GENOME,
    )
    for gene in sorted(genes, key=lambda g: g.gene_name):
        can_lens = sorted({len(s.canonical_protein.rstrip("*")) for s in gene.tis_sites})
        print(
            f"  {gene.gene_name}  gene_canonical="
            f"{len(gene.canonical_protein.rstrip('*'))}aa "
            f"({gene.canonical_transcript_id})  TIS={len(gene.tis_sites)}  "
            f"per-TIS canonical lengths={can_lens}"
        )

    hdr("STAGE 3 — Build annotation pipeline + precompute DeepLoc")
    cfg = build_config()
    print("Reference DBs:")
    print(f"  PhyloP:    {'ok' if PHYLOP_BW.exists() else 'MISSING'}  {PHYLOP_BW}")
    print(f"  PhastCons: {'ok' if PHASTCONS_BW.exists() else 'MISSING'}  {PHASTCONS_BW}")
    print(
        f"  gnomAD:    {'ok' if GNOMAD_DB.exists() else 'MISSING (falls back to API)'}  {GNOMAD_DB}"
    )
    print(
        f"  ClinVar:   {'ok' if CLINVAR_DB.exists() else 'MISSING (falls back to API)'}  {CLINVAR_DB}"
    )
    print(f"  COSMIC:    {'ok' if COSMIC_DB.exists() else 'MISSING (skipped)'}  {COSMIC_DB}")

    # DeepLoc: batch-infer every unique protein once, return hash-keyed dict
    deeploc_lookup = precompute_localization(genes)

    # Conservation: BigWig-backed SiteModule — cheap random access per TIS,
    # no precompute needed.  Module is instantiated with the pipeline below.

    # ConsequenceValidator uses CDS features from the shared upstream
    # reference — loaded once at stage 1, reused here to avoid a second
    # GTF pass.  Must be wired (not cds_df=None) so validator produces
    # authoritative protein_pos via codon-level translation, not HGVSP
    # canonical-frame fallback.
    validator = ConsequenceValidator(cds_df=ref.cds_df, genome_fasta=str(GENOME))

    # Clinical: prefetch variants per gene from the local bulk parquets.
    # ClinicalModule.annotate() looks up self._variant_cache before any
    # HTTP/disk fetch; pre-populating here makes the per-TIS path O(1).
    clinical_mod = ClinicalModule(cfg, validator=validator)
    gene_by_name = {g.gene_name: g for g in genes}
    gene_names = sorted(gene_by_name)
    print(f"clinical: prefetching + validating variants for {len(gene_names)} genes")
    for gene_name in gene_names:
        canonical_tid = gene_by_name[gene_name].canonical_transcript_id
        variants = clinical_mod.fetch_variants(gene_name, transcript_id=canonical_tid)
        clinical_mod._variant_cache[gene_name] = variants
        with_pos = sum(1 for v in variants if v.get("protein_pos") is not None)
        print(
            f"  {gene_name} ({canonical_tid}): {len(variants)} variants "
            f"({with_pos} with protein_pos)"
        )

    pipeline = AnnotationPipeline(
        protein_modules=[
            BiophysicsModule(cfg),
            MotifsModule(cfg),
            MassSpecModule(cfg),
            clinical_mod,
            LocalizationModule(cfg, predictions=deeploc_lookup),
        ],
        site_modules=[
            CoreIdentityModule(cfg),
            InitiationContextModule(cfg),
            ConservationModule(cfg),
        ],
    )

    hdr("STAGE 4 — Annotate")
    pipeline.run(genes)

    hdr("STAGE 5 — Per-gene spot check")
    for gene in sorted(genes, key=lambda g: g.gene_name):
        can = gene.canonical_annotations
        bio = can.get("biophysics", {})
        mot = can.get("motifs", {})
        mass = can.get("massspec", {})
        clin = can.get("clinical", {})
        loc = can.get("localization", {})

        print(
            f"\n{gene.gene_name}  ({gene.canonical_transcript_id}, "
            f"{len(gene.canonical_protein.rstrip('*'))} aa)"
        )
        print(f"  biophysics:    pI={bio.get('pI'):.2f}  gravy={bio.get('gravy'):.3f}")
        print(f"  motifs:        {len(mot.get('hits', []))} hits")
        ms_hits = mass.get("hits", [])
        unique_ms = sum(1 for h in ms_hits if h.get("unique_to_isoform") is True)
        print(f"  massspec:      {len(ms_hits)} tryptic peptides ({unique_ms} marked unique)")
        # conservation is per-TIS (SiteModule) — shown in per-TIS block below
        clin_sum = clin.get("summary", {})
        clin_hits = clin.get("hits", [])
        print(
            f"  clinical:      {clin_sum.get('total_variants', 0)} variants  "
            f"pathogenic={clin_sum.get('pathogenic_count', 0)}  "
            f"by_source={clin_sum.get('by_source', {})}  "
            f"with_protein_pos={len(clin_hits)}"
        )
        print(
            f"  localization:  deeploc={loc.get('deeploc_prediction')}  "
            f"signals={loc.get('deeploc_signals')}"
        )

        # Per-TIS spot check so we can SEE canonical vs each isoform.
        for site in sorted(
            gene.tis_sites,
            key=lambda s: (s.orf_type.value, s.position),
        ):
            ia = site.isoform_annotations
            ibio = ia.get("biophysics", {})
            imot = ia.get("motifs", {})
            imass = ia.get("massspec", {})
            icons = ia.get("conservation", {})
            iclin = ia.get("clinical", {})
            iloc = ia.get("localization", {})
            ilen = len(site.isoform_protein.rstrip("*"))
            kozak = site.kozak_context or "—"
            ms_hits = imass.get("hits", []) or []
            ms_unique = sum(1 for h in ms_hits if h.get("unique_to_isoform") is True)
            cons_sum = icons.get("summary", {}) or {}
            clin_sum = iclin.get("summary", {}) or {}
            print(f"    TIS {site.tis_id} ({site.orf_type.value}, {ilen} aa, kozak={kozak})")
            print(
                f"      bio pI={ibio.get('pI')} gravy={ibio.get('gravy')}  "
                f"motifs={len(imot.get('hits', []) or [])}  "
                f"ms={len(ms_hits)} (uniq={ms_unique})  "
                f"cons phyloP@tis={icons.get('phylop_at_tis')}/"
                f"{cons_sum.get('phylop_status')}  "
                f"clin={clin_sum.get('total_variants', 0)} "
                f"(path={clin_sum.get('pathogenic_count', 0)})  "
                f"loc={iloc.get('deeploc_prediction')}/"
                f"{iloc.get('deeploc_signals')}"
            )

    hdr("STAGE 6 — Paired parquet output")
    out_dir = Path(__file__).parent.parent / "data" / "output" / "5gene_e2e"
    out_dir.mkdir(parents=True, exist_ok=True)
    paired = paired_tis_dataframe(genes)
    all_path = out_dir / "all_paired.parquet"
    paired.to_parquet(all_path, index=False)
    print(f"wrote {all_path} ({len(paired)} rows, {len(paired.columns)} cols)")
    for gene_name, sub in paired.groupby("gene_name"):
        gpath = out_dir / f"{gene_name}_paired.parquet"
        sub.to_parquet(gpath, index=False)
        print(f"  {gene_name}: {len(sub)} rows -> {gpath.name}")


if __name__ == "__main__":
    main()
