"""Domain model for SwissIsoform v2 pipeline.

Defines the core data structures: ORFType enum, TranslationInitiationSite,
Gene, CellLineExpression, and VariantAnnotation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ORFType(Enum):
    """Classification of open reading frame types relative to canonical CDS."""

    ANNOTATED = "Annotated"
    EXTENDED = "Extended"
    TRUNCATED = "Truncated"
    UORF = "uORF"
    UOORF = "uoORF"
    INTERNAL_OUT_OF_FRAME = "internal_oof"
    THREE_UTR_ORF = "3utr_orf"
    ALT_ORF = "altORF"


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
class TranslationInitiationSite:
    """The atomic unit of the SwissIsoform pipeline.

    Each TIS represents a single translation initiation event detected by
    Ribo-TISH. Modules enrich this object by writing to annotations[MODULE_NAME].

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
        canonical_protein: Full canonical protein sequence.
        isoform_protein: Predicted isoform protein sequence.
        differential_sequence: Sequence unique to the isoform.
        shared_sequence: Sequence shared with canonical.
        kozak_context: Kozak sequence context around start codon.
        annotations: Module outputs keyed by MODULE_NAME.
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

    # Proteins
    canonical_protein: str = ""
    isoform_protein: str = ""
    differential_sequence: str = ""
    shared_sequence: str = ""

    # Context
    kozak_context: str | None = None

    # Annotations — modules write here
    annotations: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class Gene:
    """A gene with its canonical transcript and associated TIS sites.

    Attributes:
        gene_name: HGNC gene symbol.
        gene_id: Ensembl gene ID.
        canonical_transcript_id: Ensembl ID of the canonical transcript.
        canonical_protein: Full canonical protein sequence.
        tis_sites: All TIS sites belonging to this gene.
        gene_annotations: Gene-level annotation outputs.
    """

    gene_name: str
    gene_id: str
    canonical_transcript_id: str
    canonical_protein: str
    tis_sites: list[TranslationInitiationSite] = field(default_factory=list)
    gene_annotations: dict[str, Any] = field(default_factory=dict)


@dataclass
class VariantAnnotation:
    """A genetic variant overlapping a TIS differential region.

    Attributes:
        tis_id: ID of the associated TranslationInitiationSite.
        source: Data source (e.g. 'gnomAD', 'ClinVar', 'COSMIC').
        variant_id: Source-specific variant identifier.
        chrom: Chromosome.
        position: Genomic position of the variant.
        ref: Reference allele.
        alt: Alternate allele.
        in_differential_region: Whether the variant falls in the isoform-unique region.
        metadata: Additional source-specific fields.
    """

    tis_id: str
    source: str
    variant_id: str
    chrom: str
    position: int
    ref: str
    alt: str
    in_differential_region: bool
    metadata: dict[str, Any] = field(default_factory=dict)
