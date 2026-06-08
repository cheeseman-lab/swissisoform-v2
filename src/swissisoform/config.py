"""Pipeline configuration for SwissIsoform v2.

Defines typed config dataclasses for each pipeline subsystem with sensible defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FilterConfig:
    """Configuration for TIS site filtering (Module 0).

    Attributes:
        transcript_support_levels: Ensembl TSL values to keep.
        min_normalized_counts: Minimum CPM threshold across cell lines.
        tis_enrichment_max_p: Maximum p-value for TIS enrichment test.
        frame_test_max_p: Maximum p-value for reading frame test.
        combined_test_max_q: Maximum q-value for combined Fisher test.
        tis_distance_buffer: Minimum distance (nt) between TIS sites to merge.
        exempt_annotated: If True, Annotated TIS from reference transcripts
            skip count/significance drops (they are reference material, not
            hypotheses).  Must still pass the reference-transcript filter.
    """

    transcript_support_levels: list[str] = field(default_factory=lambda: ["1", "2", "3"])
    min_normalized_counts: float = 0.1
    tis_enrichment_max_p: float = 0.01
    frame_test_max_p: float = 0.01
    combined_test_max_q: float = 0.05
    tis_distance_buffer: int = 30
    exempt_annotated: bool = True


@dataclass
class ConservationConfig:
    """Configuration for conservation analysis (Module 8).

    The active conservation module is BigWig-lookup-based (Zoonomia 241-mammal
    PhyloP / PhastCons tracks — see ``docs/reviews/conservation_module_spec.md``).
    Point ``phylop_bigwig`` and ``phastcons_bigwig`` at the files written by
    ``scripts/setup/download_zoonomia_bigwigs.sh``.

    The homology-search fields (``diamond_db``, ``tblastn_db``) feed the dormant
    ``conservation_homology`` module kept for reference.  They are not wired
    into the main pipeline.

    Attributes:
        phylop_bigwig: Path to 241-mammal PhyloP BigWig file.
        phastcons_bigwig: Path to 241-mammal PhastCons BigWig file.
        hal_path: Path to the Zoonomia 241-mammal Cactus HAL file, used by
            Path 1/2 (primate + mammalian reading-frame integrity). When
            ``None`` or missing, Path 1/2 reports ``status="not_run"`` and
            the BigWig paths continue to serve Path 3 metrics.
        hal_ref_genome: Reference genome name inside the HAL (default
            ``"hg38"``). Must match one of ``halStats --genomes`` output.
        hal2maf_binary: ``hal2maf`` executable name resolved on ``PATH``.
        halstats_binary: ``halStats`` executable name; used once at module
            init to pull the species tree for phylogenetic-depth scoring.
        hal_tree_newick: Optional override. When set, bypasses
            ``halStats --tree`` (useful for tests or offline precompute).
        primate_species: Override for the Path 1 species list; ``None``
            falls back to :data:`conservation_frame.PRIMATE_SPECIES`.
        mammalian_species: Override for the Path 2 species list; ``None``
            falls back to :data:`conservation_frame.MAMMALIAN_SPECIES`.
        diamond_db: Dormant — path to DIAMOND protein database for legacy
            homology search.
        tblastn_db: Dormant — path to tBLASTn nucleotide database for legacy
            homology search.
    """

    phylop_bigwig: Path | None = None
    phastcons_bigwig: Path | None = None
    hal_path: Path | None = None
    hal_ref_genome: str = "hg38"
    # HAL hal2maf subprocess timeout per interval. The default was 120s;
    # under cluster load this can be too tight (the production 5-gene
    # run saw ~60 timeouts vs ~10 historically). Raise to 300s to widen
    # the window. Sites that are still slow probably hit a HAL chunking
    # pathology and need to be skipped anyway.
    hal_timeout: float = 300.0
    hal2maf_binary: str = "hal2maf"
    halstats_binary: str = "halStats"
    hal_tree_newick: str | None = None
    primate_species: list[str] | None = None
    mammalian_species: list[str] | None = None
    diamond_db: Path | None = None
    tblastn_db: Path | None = None


@dataclass
class StructureConfig:
    """Configuration for structure prediction (Module 7).

    Attributes:
        method: Structure prediction method ('chai1', 'esmfold', etc.).
        device: Compute device ('cuda' or 'cpu').
        batch_size: Number of sequences per prediction batch.
    """

    method: str = "chai1"
    device: str = "cuda"
    batch_size: int = 4


@dataclass
class ScoringConfig:
    """Configuration for evidence scoring (Module 10).

    Evidence scoring produces two independent scores per TIS:
    ``existence_score`` (E1–E6 — does this isoform really exist?) and
    ``functional_score`` (F1–F6 — does it change function?). Each
    criterion returns ``True`` / ``False`` / ``None``; the score is the
    count of ``True``. Thresholds below are the cutoffs for turning
    continuous signals into the boolean criterion outputs.

    Attributes:
        min_cell_lines: E4 threshold — minimum cell lines with expression.
        existence_high_threshold: Score cutoff for
            ``existence_high_confidence``.
        functional_high_threshold: Score cutoff for
            ``functional_high_confidence``.
        truncation_max_aa: Maximum truncation length still considered
            potentially functional (downstream consumers only).
        primate_frac_intact_min: E1 threshold — minimum fraction of
            primate species with intact reading frame.
        mammalian_frac_intact_min: E2 threshold — same for mammalian
            radiation.
        phylop_coding_min: E3 threshold — minimum mean PhyloP over the
            unique region to count as under coding-level selection.
        initiation_efficiency_min: E5 threshold — minimum ribosome
            initiation efficiency across any cell line.
        massspec_unique_peptides_min: E6 threshold — minimum number of
            peptides uniquely assigned to the isoform.
    """

    min_cell_lines: int = 3
    existence_high_threshold: int = 5
    functional_high_threshold: int = 3
    truncation_max_aa: int = 200
    primate_frac_intact_min: float = 0.5
    mammalian_frac_intact_min: float = 0.3
    phylop_coding_min: float = 1.0
    initiation_efficiency_min: float = 0.01
    massspec_unique_peptides_min: int = 1
    # F1 threshold for mean pLDDT over the differential region. Scale
    # matches whatever the structure backend emits: Boltz-2 emits 0–1
    # (so use 0.70); AlphaFold-style backends emit 0–100 (use 70.0).
    f1_plddt_threshold: float = 0.70
    # F5: minimum predicted-damaging variants in the isoform-unique region to
    # call the criterion True. Damaging = AlphaMissense likely_pathogenic OR
    # ESM-2 ΔLLR ≤ f5_llr_damaging_threshold (per-variant evidence from
    # VariantEffectModule). Avoids scoring on isolated singletons.
    f5_min_pathogenic_in_unique: int = 1
    # F5: ESM-2 masked-marginal ΔLLR = logP(alt) − logP(wt) cutoff below which
    # a substitution counts as damaging. −7.5 matches the threshold used for
    # the analogous LLR in Brandes et al. (2023).
    f5_llr_damaging_threshold: float = -7.5


@dataclass
class ClinicalConfig:
    """Configuration for clinical variant analysis (Module 9).

    Attributes:
        gnomad_api_url: gnomAD GraphQL API endpoint (fallback when
            ``gnomad_db`` is None).
        gnomad_db: Path to local gnomAD bulk parquet built by
            ``scripts/setup/setup_databases.py gnomad``.  When set, the
            fetcher reads from disk instead of calling the API.
        clinvar_email: Email for NCBI E-utilities authentication
            (fallback when ``clinvar_db`` is None).
        clinvar_api_key: API key for NCBI E-utilities (fallback only).
        clinvar_db: Path to local ClinVar parquet built by
            ``scripts/setup/setup_databases.py clinvar``.
        cosmic_db: Path to local COSMIC parquet.
        alphamissense_db: Path to the tabix-indexed AlphaMissense hg38 table
            (``AlphaMissense_hg38.tsv.gz`` + ``.tbi``) built by
            ``scripts/setup/setup_databases.py alphamissense``. Drives the
            per-variant calibrated missense pathogenicity in
            ``VariantEffectModule``.
        fetch_timeout: HTTP timeout (fallback paths only).
        max_retries: Max retry attempts (fallback paths only).
        retry_delay: Base delay for exponential backoff (fallback only).
    """

    gnomad_api_url: str = "https://gnomad.broadinstitute.org/api"
    gnomad_db: Path | None = None
    clinvar_email: str = ""
    clinvar_api_key: str = ""
    clinvar_db: Path | None = None
    cosmic_db: Path | None = None
    alphamissense_db: Path | None = None
    fetch_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class PipelineConfig:
    """Top-level configuration for the SwissIsoform v2 pipeline.

    Attributes:
        cell_lines: Cell line names matching Ribo-TISH input directories.
        data_dir: Base directory for reference data files.
        genome_fasta: Path to genome FASTA file.
        gtf_path: Path to gene annotation GTF file.
        protein_fasta: Path to canonical protein FASTA file.
        output_dir: Directory for pipeline outputs.
        filtering: Filtering configuration.
        conservation: Conservation analysis configuration.
        structure: Structure prediction configuration.
        scoring: Evidence scoring configuration.
        clinical: Clinical variant analysis configuration.
    """

    cell_lines: list[str] = field(
        default_factory=lambda: ["HeLa", "K562", "U2OS", "RPE1_Async", "RPE1_Que", "RPE1_Sen"]
    )
    data_dir: Path = Path("data/reference")
    genome_fasta: Path | None = None
    gtf_path: Path | None = None
    protein_fasta: Path | None = None
    output_dir: Path = Path("results")
    filtering: FilterConfig = field(default_factory=FilterConfig)
    conservation: ConservationConfig | None = None
    structure: StructureConfig | None = None
    scoring: ScoringConfig | None = None
    clinical: ClinicalConfig | None = None
