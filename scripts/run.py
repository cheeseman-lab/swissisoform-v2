"""SwissIsoform v2 — unified end-to-end pipeline driver.

Single unified entry point for the SwissIsoform v2 pipeline.
Runs the full annotation + comparison + scoring pipeline on an
arbitrary gene subset, isoform subset, or the full catalog.

Usage examples:

    # Smoke test on 5 diagnostic genes (HeLa-only single sample)
    python scripts/run.py --preset 5gene

    # 12 reviewer-picked genes across all 6 cell lines
    python scripts/run.py --genes CBX1 CDC34 ... --run-name 12gene_manual

    # Restrict to specific isoforms in a parquet (join keys:
    # Tid + GenomePos + StartCodon)
    python scripts/run.py --isoforms manual_combined.parquet --run-name 12gene_isoforms

    # Full catalog
    python scripts/run.py --all --run-name full_catalog

    # Skip GPU-dependent modules cleanly when caches are empty
    python scripts/run.py --genes TP53 --skip-modules plm_vep,structure

Output is written to data/output/<run_name>/ as per-gene + combined
paired parquets.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import tomllib
from pathlib import Path

import pandas as pd

from swissisoform.assembly import assemble_genes
from swissisoform.clinical.validate import ConsequenceValidator
from swissisoform.combine import combine_filtered_samples
from swissisoform.compare.comparator import compare_genes
from swissisoform.config import (
    ClinicalConfig,
    ConservationConfig,
    PipelineConfig,
    ScoringConfig,
)
from swissisoform.io.parquet import paired_tis_dataframe
from swissisoform.io.rnaseq import load_sample_manifest
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
logger = logging.getLogger("run")

# ── Paths ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "reference"
OUT = ROOT / "data" / "output"

GTF = DATA / "gencode.v49.primary_assembly.annotation.gtf"
GENOME = DATA / "Gencode_v49_GRCh38.primary_assembly.genome.fa"
PROTEIN = DATA / "gencode.v49.pc_translations.fa"

# Single-sample mode (5-gene diagnostic)
HELA_PREDICT = DATA / "HeLa_TIS_predict_all.txt"
HELA_RNASEQ = [
    DATA / "rnaseq_counts/GATCAG_8_htseqcount.txt",
    DATA / "rnaseq_counts/CACGGT_9_htseqcount.txt",
]

# Multi-sample mode (production)
SAMPLE_MANIFEST = DATA / "ribotish_sample_manifest.csv"
REPLICATE_MANIFEST = DATA / "ribotish_replicate_manifest.csv"
COMBINED_PARQUET = OUT / "filtered" / "all_samples_combined.parquet"

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

ALL_CELL_LINES = ["HeLa", "K562", "U2OS", "RPE1_Async", "RPE1_Que", "RPE1_Sen"]

# Presets — auto-discovered from presets/*.toml at the repo root. Each TOML is a
# self-contained named run: either `genes = [...]` (+ optional cell_lines) or an
# inline `[[isoforms]]` array of {gene, tid, genome_pos, start_codon} picks, plus
# run_name / min_cell_lines. Drop a new .toml in presets/ to add a named run.
PRESETS_DIR = ROOT / "presets"


def load_presets() -> dict[str, dict]:
    """Load every presets/*.toml into {name: spec}; the preset name is the stem."""
    presets: dict[str, dict] = {}
    if PRESETS_DIR.is_dir():
        for f in sorted(PRESETS_DIR.glob("*.toml")):
            with open(f, "rb") as fh:
                presets[f.stem] = tomllib.load(fh)
    return presets


PRESETS = load_presets()

ALL_PROTEIN_MODULES = [
    "biophysics", "motifs", "massspec", "clinical",
    "localization", "signalp", "targetp", "interproscan",
]
ALL_SITE_MODULES = [
    "core_identity", "initiation_context", "conservation", "conservation_frame",
    "variant_intersection", "plm_vep", "structure",
]


# ── Configuration ────────────────────────────────────────────────────────


def build_config(min_cell_lines: int = 3) -> PipelineConfig:
    """Build PipelineConfig populated with local DB paths when available.

    Args:
        min_cell_lines: ScoringConfig.min_cell_lines. Default 3 for
            production multi-cell-line runs; presets override to 1 for
            single-sample diagnostics.
    """
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
        hal_tree_newick=None,
        hal2maf_binary=str(HAL2MAF_BIN) if HAL_SIF.exists() else "hal2maf",
        halstats_binary=str(HALSTATS_BIN) if HAL_SIF.exists() else "halStats",
        hal_ref_genome="Homo_sapiens",
    )
    cfg.scoring = ScoringConfig(
        primate_frac_intact_min=0.3,
        mammalian_frac_intact_min=0.2,
        phylop_coding_min=1.0,
        min_cell_lines=min_cell_lines,
        existence_high_threshold=3,
        functional_high_threshold=2,
    )
    return cfg


# ── Data loading ─────────────────────────────────────────────────────────


def load_single_sample(cell_line: str, reference: UpstreamReference) -> pd.DataFrame:
    """Load + filter one cell line via run_sample.  Currently HeLa-only."""
    if cell_line != "HeLa":
        raise ValueError(
            f"Single-sample mode currently only supports HeLa; got {cell_line!r}. "
            "Use the multi-sample mode (default for non-preset runs)."
        )
    final, dropped = run_sample(
        HELA_PREDICT, HELA_RNASEQ, GTF, sample="HeLa", reference=reference,
    )
    logger.info(
        "Single-sample (HeLa): %d kept, %d dropped, %d imputed",
        len(final), len(dropped), final["Imputed"].sum(),
    )
    return final


def load_combined(
    cell_lines: list[str],
    reference: UpstreamReference,
    cfg: PipelineConfig,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Load the multi-cell-line combined parquet.

    If the cached combined parquet exists and force_rebuild is False, read
    it (and subset to requested cell lines).  Otherwise run upstream per
    sample and rebuild.
    """
    if COMBINED_PARQUET.exists() and not force_rebuild:
        logger.info("Loading cached combined parquet: %s", COMBINED_PARQUET)
        combined = pd.read_parquet(COMBINED_PARQUET)
        if set(cell_lines) != set(ALL_CELL_LINES):
            present_cols = [
                f"present_{cl}" for cl in cell_lines
                if f"present_{cl}" in combined.columns
            ]
            if present_cols:
                mask = combined[present_cols].any(axis=1)
                combined = combined[mask].copy()
                logger.info("Subset to %s: %d rows", ",".join(cell_lines), len(combined))
        return combined

    logger.info("Rebuilding combined parquet from per-sample upstream runs")
    manifest = load_sample_manifest(SAMPLE_MANIFEST, REPLICATE_MANIFEST)
    per_sample: dict[str, pd.DataFrame] = {}
    for _, row in manifest.iterrows():
        sample = row["sample"]
        if sample not in cell_lines:
            continue
        predict = ROOT / row["predict_file"]
        rnaseq_files = [ROOT / f for f in row["rnaseq_count_files"]]
        t0 = time.perf_counter()
        final_df, _ = run_sample(
            predict, rnaseq_files, GTF, sample=sample, config=cfg, reference=reference,
        )
        filtered_out = ROOT / row["filtered_file"]
        filtered_out.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(filtered_out, index=False)
        per_sample[sample] = final_df
        logger.info(
            "%s: %d filtered rows (%.1fs)", sample, len(final_df),
            time.perf_counter() - t0,
        )

    combined = combine_filtered_samples(per_sample)
    COMBINED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(COMBINED_PARQUET, index=False)
    logger.info("Wrote combined parquet: %s (%d rows)", COMBINED_PARQUET, len(combined))
    return combined


def load_isoform_picks(source: Path | str | list[dict]) -> pd.DataFrame:
    """Normalize isoform picks from a parquet/CSV path or inline preset entries.

    Inline entries (a preset TOML's ``[[isoforms]]`` array) are dicts with
    ``gene`` / ``tid`` / ``genome_pos`` / ``start_codon``.  File sources may use
    the SwissIso_* column names.  Both are normalized to the join keys
    ``Tid`` / ``GenomePos`` / ``StartCodon`` (plus ``Symbol``).
    """
    if isinstance(source, (str, Path)):
        p = Path(source)
        isos = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    else:
        isos = pd.DataFrame(
            [
                {
                    "Symbol": e.get("gene"),
                    "Tid": e["tid"],
                    "GenomePos": e["genome_pos"],
                    "StartCodon": e["start_codon"],
                }
                for e in source
            ]
        )

    rename_map = {
        "SwissIso_Tid": "Tid",
        "SwissIso_GenomePos": "GenomePos",
        "SwissIso_StartCodon": "StartCodon",
        "Gene": "Symbol",
    }
    for old, new in rename_map.items():
        if old in isos.columns and new not in isos.columns:
            isos = isos.rename(columns={old: new})

    missing = {"Tid", "GenomePos", "StartCodon"} - set(isos.columns)
    if missing:
        raise ValueError(
            f"Isoform picks missing required columns: {sorted(missing)}. Expected "
            f"Tid, GenomePos, StartCodon (or SwissIso_* / inline tid, genome_pos, "
            f"start_codon)."
        )
    return isos


def restrict_to_isoforms(
    df: pd.DataFrame, isos: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Restrict the combined catalog to specific TIS picks.

    ``isos`` is a normalized picks frame from :func:`load_isoform_picks`.

    Returns (filtered_df, gene_names).  The filter keeps:
      - All Annotated rows for the relevant genes (canonical references)
      - The specific picked TIS rows
    """
    required = ["Tid", "GenomePos", "StartCodon"]

    if "Symbol" in isos.columns and isos["Symbol"].notna().any():
        gene_names = sorted(isos["Symbol"].dropna().unique())
    else:
        merged = df.merge(isos[required], on=required, how="inner")
        gene_names = sorted(merged["Symbol"].unique())

    gene_mask = df["Symbol"].isin(gene_names)
    annotated_mask = gene_mask & (df["RecatTISType"] == "Annotated")

    iso_keys = isos[required].drop_duplicates()
    picked = df.merge(iso_keys, on=required, how="left", indicator=True)
    picked_mask = (picked["_merge"] == "both").to_numpy()

    final_mask = annotated_mask.to_numpy() | picked_mask
    restricted = df[final_mask].copy().reset_index(drop=True)

    n_isos_requested = len(isos)
    n_picked_matched = int(picked_mask.sum())
    n_annotated = int(annotated_mask.sum())
    logger.info(
        "Isoform restriction: %d requested, %d matched, %d canonical Annotated rows "
        "across %d genes (total %d rows)",
        n_isos_requested, n_picked_matched, n_annotated, len(gene_names), len(restricted),
    )
    if n_picked_matched < n_isos_requested:
        logger.warning(
            "%d of %d requested isoforms did not match the combined catalog",
            n_isos_requested - n_picked_matched, n_isos_requested,
        )

    return restricted, gene_names


# ── Precompute ───────────────────────────────────────────────────────────


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


def run_precompute(genes, all_proteins: list[str], skip: set[str]) -> dict:
    """Run precompute steps.  `skip` controls which precomputes fire."""
    preds: dict = {}

    if "localization" not in skip:
        deeploc_input = {}
        for gene in genes:
            deeploc_input[f"canonical_{gene.gene_name}"] = gene.canonical_protein
            for site in gene.tis_sites:
                deeploc_input[site.tis_id] = site.isoform_protein
        preds["deeploc"] = precompute_deeploc(deeploc_input, model="Fast", device="cpu")
    else:
        preds["deeploc"] = {}

    if "massspec" not in skip:
        unique_peps = collect_unique_peptides(genes)
        preds["pepquery"] = precompute_pepquery(
            unique_peps,
            dataset="Deep_29_healthy_human_tissues_PXD010154,GTEx_32_Tissues_Proteome_PXD016999",
            reference_db="swissprot:human",
            cache_dir=ROOT / "data" / "cache" / "pepquery",
        )
    else:
        preds["pepquery"] = {}

    preds["signalp"] = (
        precompute_signalp(all_proteins) if "signalp" not in skip else {}
    )
    preds["targetp"] = (
        precompute_targetp(all_proteins) if "targetp" not in skip else {}
    )
    preds["interproscan"] = (
        precompute_interproscan(all_proteins) if "interproscan" not in skip else {}
    )

    # PLM VEP + Structure: cache lookup only (populate via sbatch GPU scripts).
    from swissisoform.plm.embed import precompute_plm_esm2
    from swissisoform.structure.fold import precompute_fold

    preds["plm"] = (
        precompute_plm_esm2(all_proteins, inline=False) if "plm_vep" not in skip else {}
    )
    preds["structure"] = (
        precompute_fold(all_proteins, backend="boltz", inline=False)
        if "structure" not in skip else {}
    )

    return preds


def build_pipeline(cfg, preds, ref, genes, skip: set[str]) -> AnnotationPipeline:
    """Build the annotation pipeline, omitting modules in `skip`."""
    validator = ConsequenceValidator(cds_df=ref.cds_df, genome_fasta=str(GENOME))

    protein_mods = []
    if "biophysics" not in skip:
        protein_mods.append(BiophysicsModule(cfg))
    if "motifs" not in skip:
        protein_mods.append(MotifsModule(cfg))
    if "massspec" not in skip:
        protein_mods.append(MassSpecModule(cfg, validated_peptides=preds["pepquery"]))
    if "clinical" not in skip:
        clinical_mod = ClinicalModule(cfg, validator=validator)
        for gene in genes:
            try:
                variants = clinical_mod.fetch_variants(
                    gene.gene_name, transcript_id=gene.canonical_transcript_id,
                )
                clinical_mod._variant_cache[gene.gene_name] = variants
            except Exception as e:  # pragma: no cover
                logger.warning("Clinical fetch failed for %s: %s", gene.gene_name, e)
        protein_mods.append(clinical_mod)
    if "localization" not in skip:
        protein_mods.append(LocalizationModule(cfg, predictions=preds["deeploc"]))
    if "signalp" not in skip:
        protein_mods.append(SignalPModule(cfg, predictions=preds["signalp"]))
    if "targetp" not in skip:
        protein_mods.append(TargetPModule(cfg, predictions=preds["targetp"]))
    if "interproscan" not in skip:
        protein_mods.append(InterProScanModule(cfg, predictions=preds["interproscan"]))

    site_mods = []
    if "core_identity" not in skip:
        site_mods.append(CoreIdentityModule(cfg))
    if "initiation_context" not in skip:
        site_mods.append(InitiationContextModule(cfg))
    if "conservation" not in skip:
        site_mods.append(ConservationModule(cfg))
    if "conservation_frame" not in skip:
        site_mods.append(ConservationFrameModule(cfg))
    if "variant_intersection" not in skip:
        site_mods.append(VariantIntersectionModule())
    if "plm_vep" not in skip:
        site_mods.append(PLMVEPModule(cfg))
    if "structure" not in skip:
        site_mods.append(StructureModule(cfg))

    return AnnotationPipeline(protein_modules=protein_mods, site_modules=site_mods)


# ── Spot check ───────────────────────────────────────────────────────────


def print_spot_check(genes, limit: int | None = 5) -> None:
    """Print condensed per-gene / per-TIS spot check.  `limit` caps TIS per gene."""
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

        sites = sorted(gene.tis_sites, key=lambda s: (s.orf_type.value, s.position))
        if limit:
            sites = sites[:limit]
        for site in sites:
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
        if limit and n_tis > limit:
            print(f"    ... and {n_tis - limit} more TIS (use --no-spot-check-limit to see all)")


# ── CLI ──────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SwissIsoform v2 — unified end-to-end pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--preset", choices=sorted(PRESETS),
        help="Named run from presets/<name>.toml (auto-discovered)",
    )
    mode.add_argument("--genes", nargs="+", help="Gene symbols to analyze")
    mode.add_argument("--gene-list", type=Path, help="File with one gene symbol per line")
    mode.add_argument(
        "--isoforms", type=Path,
        help="Parquet/CSV with specific isoforms (join keys: Tid, GenomePos, StartCodon)",
    )
    mode.add_argument("--all", action="store_true", help="Run on the full catalog (every gene)")

    p.add_argument(
        "--cell-lines", default=",".join(ALL_CELL_LINES),
        help=f"Comma-separated cell lines (default: all 6 = {','.join(ALL_CELL_LINES)})",
    )
    p.add_argument(
        "--single-sample", action="store_true",
        help="Single-sample mode (HeLa-only, raw predict file). "
             "Otherwise uses the multi-cell-line combined parquet.",
    )

    p.add_argument(
        "--run-name", default=None,
        help="Output subdirectory under data/output/ (default: auto-derived)",
    )

    p.add_argument(
        "--skip-modules", default="",
        help="Comma-separated module names to skip (e.g. 'plm_vep,structure')",
    )
    p.add_argument(
        "--no-gpu", action="store_true",
        help="Convenience flag: skip PLM VEP + Structure (= --skip-modules plm_vep,structure)",
    )

    p.add_argument(
        "--min-cell-lines", type=int, default=None,
        help="ScoringConfig.min_cell_lines (default: 1 single-sample, 3 multi-sample)",
    )
    p.add_argument(
        "--rebuild-combined", action="store_true",
        help="Force rebuild of the cached all_samples_combined.parquet",
    )
    p.add_argument(
        "--no-spot-check-limit", action="store_true",
        help="Print every TIS in the spot check (default caps at 5 per gene)",
    )
    p.add_argument(
        "--emit-fasta", action="store_true",
        help="Write the proteins FASTA for the selected genes and exit, before "
             "precompute/annotation (seeds the GPU jobs run_plm_embed / run_fold "
             "without a full run)",
    )
    p.add_argument(
        "--fasta-out", type=Path, default=ROOT / "data" / "cache" / "proteins.fa",
        help="Path for the deduped proteins FASTA (default data/cache/proteins.fa); "
             "set per-run so concurrent runs don't clobber each other",
    )

    return p.parse_args(argv)


def resolve_gene_selection(
    args: argparse.Namespace,
    combined: pd.DataFrame | None,
) -> tuple[list[str] | None, pd.DataFrame | None]:
    """Resolve the input mode into (gene_names, restricted_df).

    gene_names: list of HGNC symbols, or None for --all.
    restricted_df: pre-filtered combined df (when --isoforms is used), else None.
    """
    if args.preset:
        spec = PRESETS[args.preset]
        if "isoforms" in spec:
            if combined is None:
                raise RuntimeError(
                    f"preset {args.preset!r} selects isoforms but no combined "
                    "catalog is loaded (isoform presets run multi-sample)"
                )
            restricted, gene_names = restrict_to_isoforms(
                combined, load_isoform_picks(spec["isoforms"]),
            )
            return gene_names, restricted
        return spec["genes"], None
    if args.genes:
        return args.genes, None
    if args.gene_list:
        names = [
            line.strip() for line in args.gene_list.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        return names, None
    if args.isoforms:
        if combined is None:
            raise RuntimeError(
                "--isoforms requires multi-sample mode (combined parquet must exist)"
            )
        restricted, gene_names = restrict_to_isoforms(
            combined, load_isoform_picks(args.isoforms),
        )
        return gene_names, restricted
    if args.all:
        return None, None
    if "5gene" in PRESETS:
        logger.info("No input mode specified; defaulting to --preset 5gene")
        args.preset = "5gene"
        return PRESETS["5gene"]["genes"], None
    raise RuntimeError("No input mode specified and no '5gene' preset available")


def derive_run_name(args: argparse.Namespace, gene_names: list[str] | None) -> str:
    """Auto-derive a run name from the input mode."""
    if args.run_name:
        return args.run_name
    if args.preset:
        return PRESETS[args.preset].get("run_name", args.preset)
    if args.all:
        return "all_samples"
    if args.isoforms:
        return f"{args.isoforms.stem}_e2e"
    if gene_names is not None:
        n = len(gene_names)
        if n <= 3:
            return "_".join(gene_names) + "_e2e"
        return f"{n}gene_e2e"
    return "e2e"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    skip = {m.strip() for m in args.skip_modules.split(",") if m.strip()}
    if args.no_gpu:
        skip |= {"plm_vep", "structure"}
    unknown = skip - set(ALL_PROTEIN_MODULES) - set(ALL_SITE_MODULES)
    if unknown:
        logger.error("Unknown module(s) in --skip-modules: %s", sorted(unknown))
        logger.error("Available: %s", sorted(set(ALL_PROTEIN_MODULES) | set(ALL_SITE_MODULES)))
        return 2
    if skip:
        logger.info("Skipping modules: %s", sorted(skip))

    preset = PRESETS[args.preset] if args.preset else None
    preset_is_isoform = preset is not None and "isoforms" in preset

    cell_lines = [cl.strip() for cl in args.cell_lines.split(",") if cl.strip()]
    if preset is not None and "cell_lines" in preset:
        cell_lines = preset["cell_lines"]
    elif preset_is_isoform:
        cell_lines = ALL_CELL_LINES

    # Isoform presets must run multi-sample (they restrict the combined catalog).
    single_sample = not preset_is_isoform and (
        args.single_sample or (preset is not None and len(cell_lines) == 1)
    )

    if args.min_cell_lines is not None:
        min_cl = args.min_cell_lines
    elif preset is not None and "min_cell_lines" in preset:
        min_cl = preset["min_cell_lines"]
    else:
        min_cl = 1 if single_sample else 3

    t_start = time.perf_counter()

    logger.info("Stage 1: loading shared references")
    ref = UpstreamReference.load(gtf_path=GTF, genome_fasta=GENOME, protein_fasta=PROTEIN)

    if single_sample:
        if len(cell_lines) != 1:
            logger.error("--single-sample requires exactly one cell line; got %s", cell_lines)
            return 2
        logger.info("Stage 2: single-sample mode (%s)", cell_lines[0])
        final = load_single_sample(cell_lines[0], ref)
        combined = None
    else:
        logger.info("Stage 2: multi-sample mode (%s)", ",".join(cell_lines))
        combined = load_combined(
            cell_lines, ref, build_config(),
            force_rebuild=args.rebuild_combined,
        )
        final = combined

    gene_names, restricted_df = resolve_gene_selection(args, combined)
    if restricted_df is not None:
        final = restricted_df
    elif gene_names is not None:
        final = final[final["Symbol"].isin(gene_names)].copy()

    run_name = derive_run_name(args, gene_names)
    n_genes_label = f"{len(gene_names)} genes" if gene_names else "ALL genes"
    logger.info("Stage 3: gene selection → %s (run_name=%s)", n_genes_label, run_name)

    logger.info("Stage 4: assembling gene + TIS domain objects")
    genes = assemble_genes(
        final,
        gene_names=gene_names,
        genome_fasta=GENOME,
        exon_skeletons=ref.exon_skeletons,
    )
    n_tis = sum(len(g.tis_sites) for g in genes)
    logger.info("Assembled %d genes, %d TIS", len(genes), n_tis)
    if not genes:
        logger.error("No genes assembled — check gene names / isoform file / cell-line filter")
        return 1

    cfg = build_config(min_cell_lines=min_cl)
    logger.info("Stage 5: precompute (skip=%s)", sorted(skip) or "none")
    all_proteins = collect_all_proteins(genes)
    n_written = write_proteins_fasta(all_proteins, args.fasta_out)
    logger.info("proteins.fa: %d unique sequences -> %s", n_written, args.fasta_out)
    if args.emit_fasta:
        logger.info(
            "--emit-fasta: wrote %d seqs to %s; stopping before precompute/annotate",
            n_written, args.fasta_out,
        )
        return 0
    preds = run_precompute(genes, all_proteins, skip)
    logger.info(
        "Precompute done: deeploc=%d signalp=%d targetp=%d ips=%d plm=%d struct=%d",
        len(preds["deeploc"]), len(preds["signalp"]), len(preds["targetp"]),
        len(preds["interproscan"]), len(preds["plm"]), len(preds["structure"]),
    )

    logger.info("Stage 6: annotate + compare + score")
    pipeline = build_pipeline(cfg, preds, ref, genes, skip)
    pipeline.run(genes)

    if "biophysics" not in skip:
        compare_genes(genes, scope_a_modules=[BiophysicsModule(cfg)])
    else:
        compare_genes(genes, scope_a_modules=[])

    scoring_mod = EvidenceScoringModule(cfg)
    all_sites = [site for gene in genes for site in gene.tis_sites]
    scoring_mod.run(all_sites)
    logger.info("Annotation + comparison + scoring complete")

    print_spot_check(genes, limit=None if args.no_spot_check_limit else 5)

    out_dir = OUT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    paired = paired_tis_dataframe(genes)
    all_path = out_dir / "all_paired.parquet"
    paired.to_parquet(all_path, index=False)
    print(f"\nwrote {all_path} ({len(paired)} rows, {len(paired.columns)} cols)")
    for gene_name, sub in paired.groupby("gene_name"):
        gpath = out_dir / f"{gene_name}_paired.parquet"
        sub.to_parquet(gpath, index=False)
        if len(genes) <= 50:
            print(f"  {gene_name}: {len(sub)} rows -> {gpath.name}")
    if len(genes) > 50:
        print(f"  (per-gene parquets written for {len(genes)} genes)")

    elapsed = time.perf_counter() - t_start
    logger.info("Total wall time: %.1f min", elapsed / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
