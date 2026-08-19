"""Shared path constants, module lists, preset discovery, and config builder.

Extracted from scripts/run.py so that both scripts/run.py and future
front-ends (runner.py, CLI) can import them without duplicating logic.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from swissisoform.config import (
    ClinicalConfig,
    ConservationConfig,
    PipelineConfig,
    ScoringConfig,
    SourceResolutionConfig,
    StructureConfig,
)

# ── Paths ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "reference"
OUT = ROOT / "data" / "output"

GTF = DATA / "gencode.v49.primary_assembly.annotation.gtf"
GENOME = DATA / "Gencode_v49_GRCh38.primary_assembly.genome.fa"
PROTEIN = DATA / "gencode.v49.pc_translations.fa"

# Reference DBs (optional — modules degrade gracefully when absent)
GNOMAD_DB = DATA / "gnomad" / "gnomad_v4.1_exome.parquet"
CLINVAR_DB = DATA / "clinvar" / "variant_summary.parquet"
COSMIC_DB = DATA / "cosmic" / "cosmic_variants.parquet"
ALPHAMISSENSE_DB = DATA / "alphamissense" / "AlphaMissense_hg38.tsv.gz"
GENEREF_JSON = DATA / "generef" / "generef.json"
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
    "variant_intersection", "plm_vep", "varianteffect", "structure",
]
ALL_GENE_MODULES = ["generef"]


# ── Configuration ────────────────────────────────────────────────────────


def build_config(
    min_cell_lines: int = 3,
    *,
    source_resolution: bool = True,
    divergence_threshold: float | None = 0.5,
    window_upstream: int = 100,
    window_downstream: int = 100,
) -> PipelineConfig:
    """Build PipelineConfig populated with local DB paths when available.

    Args:
        min_cell_lines: ScoringConfig.min_cell_lines. Default 3 for
            production multi-cell-line runs; presets override to 1 for
            single-sample diagnostics.
        source_resolution: When ``False`` the source-resolution cascade is
            disabled (``cfg.source_resolution = None``) — no per-TIS mRNA
            resolution and no downstream collapse.
        divergence_threshold: ``SourceResolutionConfig.divergence_dominance_frac``
            — the fraction of long-read abundance the top transcript must hold at
            a divergent site (default 0.5). ``None`` = most-abundant-wins.
        window_upstream: nt 5' of the start codon in the purity window.
        window_downstream: nt 3' of the start codon in the purity window.
    """
    cfg = PipelineConfig()
    cfg.clinical = ClinicalConfig(
        gnomad_db=GNOMAD_DB if GNOMAD_DB.exists() else None,
        clinvar_db=CLINVAR_DB if CLINVAR_DB.exists() else None,
        cosmic_db=COSMIC_DB if COSMIC_DB.exists() else None,
        alphamissense_db=ALPHAMISSENSE_DB if ALPHAMISSENSE_DB.exists() else None,
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
    # Use ScoringConfig defaults for the high-confidence cutoffs (5/3) and the
    # principled C3/D1 anchors; the 13-gene set validates scoring LOGIC, not
    # calibration, so no per-run override of those thresholds.
    cfg.scoring = ScoringConfig(min_cell_lines=min_cell_lines)
    # Folding backend is config-driven: the GPU fold job, the cache-lookup
    # precompute, and StructureModule all read cfg.structure.backend, so they
    # never disagree on which backend's cache to write/read.
    cfg.structure = StructureConfig()
    # Source-transcript resolution: the cascade runs per sample only when that
    # sample's manifest row supplies an ``isoquant_table`` (HeLa today). Disabled
    # entirely when ``source_resolution=False`` (CLI --skip-source-resolution).
    cfg.source_resolution = (
        SourceResolutionConfig(
            window_upstream=window_upstream,
            window_downstream=window_downstream,
            divergence_dominance_frac=divergence_threshold,
        )
        if source_resolution
        else None
    )
    return cfg
