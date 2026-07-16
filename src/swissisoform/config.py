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
class SourceResolutionConfig:
    """Configuration for per-sample source-transcript resolution.

    Filtering-cascade stage (``pipeline.run_sample``) that pins each TIS to one
    source mRNA using the sample's own long-read (IsoQuant) RNA-seq. A single
    cascade: long-read presence filter → window-purity → abundance-based
    labeling. Off unless ``PipelineConfig.source_resolution`` is set; skipped
    for samples without an IsoQuant input (HeLa only today).

    The purity window is defined by two independent bounds — ``window_upstream``
    nt 5' of the start-codon A and ``window_downstream`` nt 3' of it — so the
    local-context test can be made asymmetric. Both default to 100.

    Attributes:
        window_upstream: nt 5' of the start codon in the purity window.
        window_downstream: nt 3' of the start codon in the purity window.
        isoquant_min_count: IsoQuant presence threshold (long-read counts) for
            the long-read filter step.
        divergence_dominance_frac: For divergent (ambiguous-window) sites, the
            top survivor must hold at least this fraction of the total long-read
            abundance across divergent candidates to be called the source;
            otherwise the site is ``unresolved``. Defaults to 0.5 (≥50%).
            ``None`` disables the check (most-abundant-wins).
        drop_unresolved: Reserved — superseded by the assembly-boundary collapse
            (``sourceresolve.collapse_to_source``), which is what actually
            subsets to resolved TIS for the next stage. Kept ``False``; the
            per-sample filtered table stays tag-only (full rows) for audit.
    """

    window_upstream: int = 100
    window_downstream: int = 100
    isoquant_min_count: float = 3.0
    divergence_dominance_frac: float | None = 0.5
    drop_unresolved: bool = False


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
        backend: Folding backend ('esmfold2' default, 'boltz', or 'chai').
        device: Compute device ('cuda' or 'cpu').
        batch_size: Number of sequences per prediction batch.
    """

    backend: str = "esmfold2"
    device: str = "cuda"
    batch_size: int = 4


@dataclass
class ScoringConfig:
    """Configuration for evidence scoring (Module 10).

    Evidence scoring produces two independent scores per TIS:
    ``existence_score`` (Conservation + Detection — does this isoform really
    exist?) and ``functional_score`` (Localization + Mutation + Predicted
    structure + Structural characteristics — does it change function?). Each
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
        primate_frac_intact_min: Context display only — fraction of primate
            species with intact reading frame. No longer the E1 score basis.
        mammalian_frac_intact_min: Context display only — same for mammalian
            radiation. No longer the E2 score basis.
        e1_pident_min: E1 threshold — minimum primate mean AA percent-identity
            over the unique region.
        e2_pident_min: E2 threshold — minimum mammalian mean AA percent-identity.
        e3_phylop_min: E3 threshold — minimum absolute mean PhyloP over the
            unique region (strong purifying selection).
        phylop_coding_min: Legacy E3 alias retained for context display.
        initiation_efficiency_min: E5 threshold — minimum ribosome
            initiation efficiency across any cell line.
        massspec_unique_peptides_min: E6 threshold — minimum number of
            peptides uniquely assigned to the isoform.
        pepquery_fix_mods: PepQuery2 ``-fixMod`` UNIMOD ids (default
            Carbamidomethyl C).
        pepquery_var_mods: PepQuery2 ``-varMod`` UNIMOD ids (default
            Oxidation M + Acetyl peptide N-term).
        pepquery_max_var: PepQuery2 ``-maxVar`` — max variable mods per peptide.
        f5_constraint_enrichment_min: F5 threshold — minimum ESM-C constraint
            enrichment (unique vs shared) to call germline constraint.
        f5_depletion_ratio_max: F5 threshold — gnomAD depletion ratio below
            which germline variation is judged to avoid the unique region.
        f6_disease_enrichment_min: F6 threshold — disease-variant density
            enrichment (unique vs shared) zero-point.
        s2_gravy_delta_min: S2 distinctness — |gravy_unique − gravy_shared| cutoff.
        s2_fraction_charged_delta_min: S2 distinctness — |fraction_charged
            region-vs-core| cutoff.
        s2_disorder_delta_min: S2 distinctness — |disorder region-vs-core| cutoff.
        s3_sae_min_unique_features: S3 threshold — minimum count of interpretable
            SAE features firing in the isoform-unique region.
        f7_rmsd_shared_min: F7 threshold — minimum shared-region Cα RMSD (Å) to
            call a significant structural change in the retained region.
        f7_min_shared_len: F7 guard — minimum shared-region length (aa) below
            which the RMSD is too noisy to score (criterion → None).
        f7_plddt_min: F7 confidence gate — minimum of the two per-side mean
            shared-region pLDDTs required before the RMSD is trusted (a floppy
            low-confidence region must not read as a structural change).
    """

    min_cell_lines: int = 3
    existence_high_threshold: int = 5
    functional_high_threshold: int = 3
    truncation_max_aa: int = 200
    # Context display only — primate/mammalian frame-intact fractions are no
    # longer the E1/E2 score basis (mean_pident is). Kept for reason strings.
    primate_frac_intact_min: float = 0.5
    mammalian_frac_intact_min: float = 0.3
    # E3 principled anchor: absolute mean PhyloP over the unique region, strong
    # purifying selection. Replaces the unique-vs-shared framing.
    e3_phylop_min: float = 2.0
    phylop_coding_min: float = 1.0
    initiation_efficiency_min: float = 0.01
    massspec_unique_peptides_min: int = 1
    # PepQuery2 modification ids (`-printPTM`): keep Carbamidomethyl(C)=1 fixed
    # and Oxidation(M)=2 variable (PepQuery's own defaults), and add id 5 =
    # "Acetylation of peptide N-term" (+42.0106) so the initiator / NME-acetyl
    # proteoform is searchable. We use the *peptide* N-term acetyl (5), not the
    # *protein* N-term acetyl (33): in peptide-input mode PepQuery has no protein
    # context to anchor a protein-N-term mod, so 33 would never fire — and since
    # we only submit N-terminal diagnostic peptides, the peptide N-term is the
    # protein N-term anyway.
    pepquery_fix_mods: str = "1"
    pepquery_var_mods: str = "2,5"
    pepquery_max_var: int = 3
    # F6 principled anchor: disease-variant density enrichment zero-point.
    f6_disease_enrichment_min: float = 1.0
    # E1/E2 score on AA percent-identity over the unique region.
    e1_pident_min: float = 0.80  # CALIBRATE ON GENOME-WIDE RUN — provisional
    e2_pident_min: float = 0.50  # CALIBRATE ON GENOME-WIDE RUN — provisional
    # F5 germline tolerance/constraint thresholds.
    f5_constraint_enrichment_min: float = 2.0  # CALIBRATE ON GENOME-WIDE RUN — provisional
    f5_depletion_ratio_max: float = 0.80  # CALIBRATE ON GENOME-WIDE RUN — provisional
    # P1 threshold for mean pLDDT over the differential region. Scale
    # matches whatever the structure backend emits: Boltz-2 emits 0–1
    # (so use 0.70); AlphaFold-style backends emit 0–100 (use 70.0).
    # (Field name kept ``f1_*`` for back-compat; P1 is the renamed F1.)
    f1_plddt_threshold: float = 0.70
    # S2 biophysical-distinctness cutoffs (region-vs-core; any one satisfied →
    # distinct). Migrated from the old F1 distinctness half, now applied to the
    # ``<feat>_unique`` vs ``<feat>_shared`` Scope-A keys rather than the
    # whole-protein ``_delta``.
    s2_gravy_delta_min: float = 0.3  # PROVISIONAL — set in threshold discussion
    s2_fraction_charged_delta_min: float = 0.05  # PROVISIONAL — set in threshold discussion
    s2_disorder_delta_min: float = 0.05  # PROVISIONAL — set in threshold discussion
    # S3 SAE unique-region feature count threshold.
    s3_sae_min_unique_features: int = 1  # PROVISIONAL — set in threshold discussion
    # F7 shared-region structural change. RMSD scale is Å; pLDDT gate is on the
    # 0–1 ESMFold2 scale (mirror of f1_plddt_threshold). The real RMSD cutoff is
    # to be picked from the genome-wide distribution (near-zero spike + tail).
    f7_rmsd_shared_min: float = 2.0  # CALIBRATE ON GENOME-WIDE RUN — provisional
    f7_min_shared_len: int = 20  # CALIBRATE ON GENOME-WIDE RUN — provisional
    f7_plddt_min: float = 0.70  # CALIBRATE ON GENOME-WIDE RUN — provisional
    # Dormant per-variant damaging cutoffs (VariantEffectModule still reads
    # f5_llr_damaging_threshold for ΔLLR labeling; F5 scoring no longer uses
    # these — see f5_constraint_enrichment_min / f5_depletion_ratio_max).
    f5_min_pathogenic_in_unique: int = 1
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
    source_resolution: SourceResolutionConfig | None = None
    conservation: ConservationConfig | None = None
    structure: StructureConfig | None = None
    scoring: ScoringConfig | None = None
    clinical: ClinicalConfig | None = None
