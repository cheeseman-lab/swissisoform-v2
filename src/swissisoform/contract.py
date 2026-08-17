"""Canonical-vs-alternative contract: ORF types and differential-region rules.

Single source of truth for the two definitions that everything downstream
depends on:

1. :class:`ORFType` — the classification of an open reading frame relative to
   the canonical CDS, plus :func:`orf_type_from_ribotish` mapping Ribo-TISH
   compound type strings onto it.
2. :func:`diff_region_rule` — which coordinate space the isoform-unique
   "differential region" lives in for each ORF type, and whether the entire
   isoform is unique (no shared region).

Differential-region spaces by ORF type:
    - Extensions → isoform space, partial (``isoform[0:delta_aa]``).
    - Truncations → canonical space, partial (``canonical[0:abs(delta_aa)]``).
    - uORF / altORF / uoORF / internal-OOF / 3'UTR ORF → isoform space, the
      entire isoform (no shared region).
    - Annotated → none (no differential region).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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


# ORF types whose unique region was never canonical coding sequence: an extension's
# came from 5'UTR or intron, and a separate ORF shares no reading frame with the CDS
# at all. Any metric that contrasts the unique region against the canonical-shared
# one is baseline-free for these — the two sides are not comparable quantities — so
# a criterion built on such a contrast reports "not evaluable" rather than a verdict.
# ANNOTATED and TRUNCATED are excluded: both live in canonical coding sequence.
NO_CANONICAL_BASELINE_ORFS = frozenset(
    {
        ORFType.EXTENDED,
        ORFType.UORF,
        ORFType.UOORF,
        ORFType.INTERNAL_OUT_OF_FRAME,
        ORFType.THREE_UTR_ORF,
        ORFType.ALT_ORF,
    }
)


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


@dataclass(frozen=True)
class DiffRegionRule:
    """Where an ORF type's isoform-unique region lives.

    Attributes:
        space: Coordinate space of the differential region —
            ``"isoform"``, ``"canonical"``, or ``"none"``.
        whole_isoform: ``True`` when the entire isoform is unique (no shared
            region with the canonical protein).
    """

    space: str  # "isoform" | "canonical" | "none"
    whole_isoform: bool


_DIFF_REGION_RULES: dict[ORFType, DiffRegionRule] = {
    ORFType.ANNOTATED: DiffRegionRule(space="none", whole_isoform=False),
    ORFType.EXTENDED: DiffRegionRule(space="isoform", whole_isoform=False),
    ORFType.TRUNCATED: DiffRegionRule(space="canonical", whole_isoform=False),
    ORFType.UORF: DiffRegionRule(space="isoform", whole_isoform=True),
    ORFType.UOORF: DiffRegionRule(space="isoform", whole_isoform=True),
    ORFType.INTERNAL_OUT_OF_FRAME: DiffRegionRule(space="isoform", whole_isoform=True),
    ORFType.THREE_UTR_ORF: DiffRegionRule(space="isoform", whole_isoform=True),
    ORFType.ALT_ORF: DiffRegionRule(space="isoform", whole_isoform=True),
}


def diff_region_rule(orf_type: ORFType) -> DiffRegionRule:
    """Return the differential-region rule for an ORF type.

    Args:
        orf_type: The classified ORF type.

    Returns:
        The :class:`DiffRegionRule` describing the coordinate space and whether
        the entire isoform is unique.
    """
    return _DIFF_REGION_RULES[orf_type]
