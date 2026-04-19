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

    Attributes:
        diamond_db: Path to DIAMOND protein database for ortholog search.
        tblastn_db: Path to tBLASTn nucleotide database.
        phylop_bigwig: Path to PhyloP conservation scores BigWig file.
    """

    diamond_db: Path | None = None
    tblastn_db: Path | None = None
    phylop_bigwig: Path | None = None


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

    Attributes:
        min_cell_lines: Minimum cell lines with expression for scoring.
        existence_high_threshold: Score threshold for high-confidence existence.
        functional_high_threshold: Score threshold for high-confidence function.
        truncation_max_aa: Maximum amino acid truncation to consider functional.
    """

    min_cell_lines: int = 3
    existence_high_threshold: int = 5
    functional_high_threshold: int = 3
    truncation_max_aa: int = 200


@dataclass
class ClinicalConfig:
    """Configuration for clinical variant analysis (Module 9).

    Attributes:
        gnomad_api_url: gnomAD GraphQL API endpoint (fallback when
            ``gnomad_db`` is None).
        gnomad_db: Path to local gnomAD bulk parquet built by
            ``scripts/setup_databases.py gnomad``.  When set, the
            fetcher reads from disk instead of calling the API.
        clinvar_email: Email for NCBI E-utilities authentication
            (fallback when ``clinvar_db`` is None).
        clinvar_api_key: API key for NCBI E-utilities (fallback only).
        clinvar_db: Path to local ClinVar parquet built by
            ``scripts/setup_databases.py clinvar``.
        cosmic_db: Path to local COSMIC parquet.
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
        default_factory=lambda: [
            "HeLa", "K562", "U2OS", "RPE1_Async", "RPE1_Que", "RPE1_Sen"
        ]
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
