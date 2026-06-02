"""Domain model for SwissIsoform v2 pipeline.

Defines the core data structures: ORFType enum, DifferentialRegion,
TranslationInitiationSite, Gene, CellLineExpression, and VariantAnnotation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

# Confidence tiers for DifferentialRegion.
# - "exact": canonical diff (annotated → empty; uORF/altORF/internal/3utr → entire isoform)
# - "tail_verified": sequence relationship confirmed by ≥95% tail match
# - "length_fallback": truncation with unmatched sequences; diff derived from length delta
# - "whole_isoform_fallback": sequences unrelated; entire isoform marked differential
DiffRegionConfidence = Literal[
    "exact", "tail_verified", "length_fallback", "whole_isoform_fallback"
]


class ORFType(Enum):
    """Classification of open reading frame types relative to canonical CDS."""

    ANNOTATED = "annotated"
    EXTENDED = "extended"
    TRUNCATED = "truncated"
    UORF = "uorf"
    UOORF = "uoorf"
    INTERNAL_OUT_OF_FRAME = "internal_oof"
    THREE_UTR_ORF = "3utr_orf"
    ALT_ORF = "alt_orf"


def orf_type_from_ribotish(tis_type: str) -> ORFType:
    """Map a Ribo-TISH TisType string to an ORFType enum value.

    Ribo-TISH produces 16 compound type strings like "Extended:CDSFrameOverlap"
    or "5'UTR:Known". This function normalizes them to the 8-value ORFType enum.

    Args:
        tis_type: Raw TisType string from Ribo-TISH predict_all.txt.

    Returns:
        Corresponding ORFType enum member.
    """
    if tis_type.startswith("Annotated"):
        return ORFType.ANNOTATED
    if tis_type.startswith("Truncated"):
        return ORFType.TRUNCATED
    if tis_type.startswith("Extended"):
        return ORFType.EXTENDED
    if tis_type.startswith("Internal"):
        return ORFType.INTERNAL_OUT_OF_FRAME
    if tis_type.startswith("5'UTR"):
        if "CDSFrameOverlap" in tis_type:
            return ORFType.UOORF
        return ORFType.UORF
    if tis_type.startswith("3'UTR"):
        return ORFType.THREE_UTR_ORF
    if tis_type.startswith("Novel"):
        if "CDSFrameOverlap" in tis_type:
            return ORFType.UOORF
        return ORFType.ALT_ORF
    return ORFType.ALT_ORF


@dataclass
class TranscriptCoordinates:
    """Exon skeleton for a single transcript (Layer 1 of ORF exon infrastructure).

    Built once per transcript at GTF loading time and shared across all ORFs
    on that transcript. Downstream Layer 2 walkers consume this to produce
    per-ORF genomic intervals.

    All coordinates are 0-based half-open plus-strand reference coordinates,
    regardless of transcript strand. Exons are stored in ascending genomic
    order; transcript (mRNA) order is ascending for ``+`` strand and
    descending for ``-`` strand.

    Attributes:
        transcript_id: Ensembl transcript ID (e.g. ``ENST00000269305.9``).
        chrom: Chromosome name as it appears in the GTF.
        strand: ``'+'`` or ``'-'``.
        exons: Full exon structure (not just CDS) as ``[(start, end), ...]``,
            ascending genomic order. Includes 5'UTR + CDS + 3'UTR exons so
            extensions that initiate in the 5'UTR can be walked.
        cds_start: Plus-strand 0-based genomic position of the canonical ATG's
            first nucleotide. For ``+`` strand this is the lower coordinate;
            for ``-`` strand it is the higher coordinate (the A of ATG sits
            on the minus strand). ``None`` if the transcript has no annotated
            start codon.
        cds_end: Plus-strand 0-based genomic position delimiting the end of
            the canonical CDS. For ``+`` strand this is the exclusive upper
            bound of the stop codon; for ``-`` strand it is the lower bound
            (exclusive on the minus-strand walk direction). ``None`` if
            unavailable.
    """

    transcript_id: str
    chrom: str
    strand: str
    exons: list[tuple[int, int]]
    cds_start: int | None = None
    cds_end: int | None = None


@dataclass
class CellLineExpression:
    """Expression measurements for a TIS in a single cell line.

    Attributes:
        raw_count: Raw read count at the TIS position.
        cpm: Counts per million normalization.
        p_value: Statistical significance of TIS enrichment.
        initiation_efficiency: Ratio of TIS reads to total gene reads, if available.
    """

    raw_count: int
    cpm: float
    p_value: float
    initiation_efficiency: float | None = None


@dataclass
class DifferentialRegion:
    """Coordinates of the isoform-unique region in both protein spaces.

    For extensions: isoform_start=0, isoform_end=delta_aa (the N-terminal prefix).
    For truncations: canonical_start=0, canonical_end=abs(delta_aa) (the lost region).
    For uORFs/altORFs: isoform_start=0, isoform_end=len(isoform_protein).

    Attributes:
        isoform_start: Start position in isoform protein coords (0-indexed), or None.
        isoform_end: End position in isoform protein coords (exclusive), or None.
        canonical_start: Start position in canonical protein coords (0-indexed), or None.
        canonical_end: End position in canonical protein coords (exclusive), or None.
        sequence: The actual differential amino acid sequence.
        confidence: How the region was derived. Downstream positional analyses
            should treat ``length_fallback`` and ``whole_isoform_fallback`` as
            low-confidence and optionally filter or down-weight them.
    """

    isoform_start: int | None = None
    isoform_end: int | None = None
    canonical_start: int | None = None
    canonical_end: int | None = None
    sequence: str = ""
    confidence: DiffRegionConfidence = "exact"


@dataclass
class TranslationInitiationSite:
    """The atomic unit of the SwissIsoform pipeline.

    Each TIS represents a single translation initiation event detected by
    Ribo-TISH. Per-protein annotation modules write to isoform_annotations,
    and the comparison layer writes to comparison.

    Attributes:
        tis_id: Unique identifier (typically chrom:pos:strand:codon).
        gene_name: HGNC gene symbol.
        transcript_id: Ensembl transcript ID.
        chrom: Chromosome name.
        position: Genomic position of the start codon.
        strand: Genomic strand ('+' or '-').
        start_codon: Start codon sequence (e.g. 'ATG', 'CTG').
        orf_type: Classified ORF type relative to canonical CDS.
        gene_id: Ensembl gene ID.
        transcript_start: Transcript-relative start position.
        aa_len: Predicted amino acid length of the ORF.
        tis_pvalue: TIS enrichment p-value from Ribo-TISH.
        ribo_pvalue: Ribosome frame test p-value.
        fisher_qvalue: Combined Fisher's method q-value.
        expression: Per-cell-line expression data.
        canonical_protein: Full canonical protein sequence (populated from Gene).
        isoform_protein: Predicted isoform protein sequence (from Ribo-TISH AASeq).
        diff_region: Coordinates and sequence of the isoform-unique region.
        kozak_context: Kozak sequence context around start codon.
        isoform_annotations: Per-isoform module outputs keyed by MODULE_NAME.
        comparison: Comparison results keyed by MODULE_NAME.
    """

    # Identity
    tis_id: str
    gene_name: str
    transcript_id: str
    chrom: str
    position: int
    strand: str
    start_codon: str
    orf_type: ORFType

    # Ribo-TISH IDs
    gene_id: str = ""
    transcript_start: int = 0
    aa_len: int = 0

    # Statistics
    tis_pvalue: float | None = None
    ribo_pvalue: float | None = None
    fisher_qvalue: float | None = None

    # Expression
    expression: dict[str, CellLineExpression] = field(default_factory=dict)

    # Per-cell-line expression of this transcript's *canonical* (Annotated) start,
    # so downstream comparison can put canonical-start vs alt-start initiation
    # efficiency side by side (E4/E5). Keyed by cell line; empty when the Tid has
    # no Annotated row or in single-sample mode.
    canonical_expression: dict[str, CellLineExpression] = field(default_factory=dict)

    # Proteins
    canonical_protein: str = ""
    isoform_protein: str = ""

    # Differential region
    diff_region: DifferentialRegion | None = None

    # Context
    kozak_context: str | None = None

    # Genomic exon intervals (plus-strand, 0-based half-open) covering
    # the isoform ORF's nucleotide sequence, skipping introns. Populated
    # by the assembly layer from the transcript skeleton. Empty list when
    # the skeleton isn't available. Symmetric with ``isoform_protein``.
    orf_exons: list[tuple[int, int]] = field(default_factory=list)

    # Canonical ORF exons for the TIS's *own transcript* — symmetric with
    # ``canonical_protein`` (which is the per-Tid canonical). Keeping a
    # per-TIS copy lets SiteModules (conservation, future clinical-genomic)
    # compute unique/shared region metrics without needing the parent Gene.
    canonical_orf_exons: list[tuple[int, int]] = field(default_factory=list)

    # Isoform-level annotations — per-protein modules write here
    isoform_annotations: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Differential-region annotations — Scope-A baseline. Populated by the
    # Comparator when it re-runs ProteinModules on ``diff_region.sequence``
    # so that enrichment ratios (unique vs. shared) can be derived.
    diff_annotations: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Comparison results — comparator writes here. Per-module dict with
    # scalar deltas, enrichment ratios, and positional hits restricted to
    # the differential region.
    comparison: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class Gene:
    """A gene with its canonical transcript and associated TIS sites.

    Canonical annotations are computed once per gene and shared across all
    TIS sites. Gene-level reference annotations (generef) live in
    gene_annotations and are not diffed against isoform annotations.

    Attributes:
        gene_name: HGNC gene symbol.
        gene_id: Ensembl gene ID.
        canonical_transcript_id: Ensembl ID of the canonical transcript.
        canonical_protein: Full canonical protein sequence.
        tis_sites: All TIS sites belonging to this gene.
        canonical_annotations: Per-protein module outputs for the canonical protein.
        gene_annotations: Gene-level reference annotations (generef, etc.).
    """

    gene_name: str
    gene_id: str
    canonical_transcript_id: str
    canonical_protein: str
    tis_sites: list[TranslationInitiationSite] = field(default_factory=list)
    canonical_annotations: dict[str, dict[str, Any]] = field(default_factory=dict)
    gene_annotations: dict[str, Any] = field(default_factory=dict)

    # Genomic exon intervals (plus-strand, 0-based half-open) covering
    # the canonical ORF, symmetric with ``canonical_protein``. Populated
    # by the assembly layer from the transcript skeleton; empty list when
    # the skeleton isn't available.
    canonical_orf_exons: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class VariantAnnotation:
    """A genetic variant overlapping a TIS region.

    Attributes:
        tis_id: ID of the associated TranslationInitiationSite.
        source: Data source (e.g. 'gnomAD', 'ClinVar', 'COSMIC').
        variant_id: Source-specific variant identifier.
        chrom: Chromosome.
        position: Genomic position of the variant.
        ref: Reference allele.
        alt: Alternate allele.
        protein_pos: Position in the protein sequence (0-indexed).
        metadata: Additional source-specific fields.
    """

    tis_id: str
    source: str
    variant_id: str
    chrom: str
    position: int
    ref: str
    alt: str
    protein_pos: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
