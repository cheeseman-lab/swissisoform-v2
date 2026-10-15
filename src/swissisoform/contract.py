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

import math
import re
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

# ORF types read as a separate protein: they share no reading frame with the CDS,
# so the whole isoform is unique and there is no shared region.
SEPARATE_ORF_TYPES: tuple[str, ...] = (
    ORFType.UORF.value,
    ORFType.UOORF.value,
    ORFType.INTERNAL_OUT_OF_FRAME.value,
    ORFType.THREE_UTR_ORF.value,
    ORFType.ALT_ORF.value,
)

# What the unique region is called per ORF type — the nouns the category prompts
# use, the tag labels carry (``Tag.label_for``) and the site renders. An extension
# adds it, a truncation lost it (canonical-space), a separate ORF is wholly unique.
REGION_NOUN: dict[str, str] = {
    ORFType.EXTENDED.value: "extension",
    ORFType.TRUNCATED.value: "lost region",
    **dict.fromkeys(SEPARATE_ORF_TYPES, "ORF"),
}
# What a reader calls the region when the ORF type is unknown.
DEFAULT_REGION_NOUN = "differential region"
# Not after a hyphen: "isoform-unique region" is a compound, not the region's name.
_UNIQUE_REGION = re.compile(r"(?<![\w-])[Uu]nique region\b")


def region_noun(
    orf_type: str | None, *, cap: bool = False, default: str | None = None
) -> str | None:
    """The unique region's name for one ORF type (``default`` for an unknown type).

    ``cap`` capitalises the first letter, for a heading or a legend entry.
    """
    noun = REGION_NOUN.get((orf_type or "").lower()) or default
    return noun[0].upper() + noun[1:] if cap and noun else noun


def stratum_label(text: str, orf_type: str | None) -> str:
    """Rewrite a generic "unique region" phrase the way it reads for one ORF type.

    "Unique region more basic" → "Extension more basic" / "Lost region more
    basic" / "ORF more basic", keeping the capital when the phrase had one. On a
    truncation a trailing " gained" reads " lost" ("Constrained residues gained"
    → "Constrained residues lost"). An unknown ORF type returns ``text`` as is.
    """
    noun = region_noun(orf_type)
    if noun is None:
        return text
    noun_cap = region_noun(orf_type, cap=True)
    out = _UNIQUE_REGION.sub(lambda m: noun_cap if m.group(0)[0] == "U" else noun, text)
    if (orf_type or "").lower() == ORFType.TRUNCATED.value and out.endswith(" gained"):
        out = out[: -len(" gained")] + " lost"
    return out


# Codons that can initiate translation: ATG plus the near-cognate starts — its
# single-substitution neighbours, all observed driving real alt-TIS events (the
# Ribo-TISH calls in this pipeline use CTG, GTG, TTG, ACG and ATC). Membership is
# what decides a start-codon variant: a substitution that leaves the codon in this
# set keeps the isoform initiating, one that drops it out ablates the start and
# with it the whole proteoform. The amino acid is irrelevant to that question —
# a third-base change can abolish a near-cognate start while translating to the
# same residue.
NEAR_COGNATE_STARTS = frozenset(
    {"ATG", "CTG", "GTG", "TTG", "ACG", "AGG", "AAG", "ATA", "ATC", "ATT"}
)

#: Notes explaining a codon-0 call whose term alone does not say what happened.
#: A start that got *stronger* and a start that merely moved sideways both come
#: out of the amino-acid branches looking like ordinary missense, which is true
#: but not the interesting part.
START_STRENGTHENED_NOTE = "near-cognate start replaced by ATG — initiation strengthened"
START_STILL_NEAR_COGNATE_NOTE = "start remains a near-cognate; initiation is not ablated"
START_LOST_ATG_NOTE = "annotated ATG start destroyed"
START_LOST_NEAR_COGNATE_NOTE = "near-cognate start left the initiating set"
START_NOT_AN_INITIATOR_NOTE = "annotated start is not an initiating codon"


def start_codon_effect(ref_codon: str, alt_codon: str) -> tuple[str | None, str]:
    """Decide what a substitution does to an ORF's start codon.

    **The direction of the change carries the meaning, and a membership test throws
    it away.** ``NEAR_COGNATE_STARTS`` is exactly ATG plus its nine single-base
    neighbours, so asking only "is the mutated codon still in the set?" can never
    fire for an ATG start: every SNV of ATG lands on another member. That labelled
    the classic pathogenic ``p.Met1?`` variants missense and kept them out of the
    loss-of-function gate entirely.

    Losing an ATG and gaining one are not the same event, so the rule is asymmetric:

    * ATG → anything else: the annotated initiator is gone. A near-cognate in its
      place initiates far less efficiently, so this is ``start_lost`` even when the
      replacement is itself a near-cognate.
    * near-cognate → outside the set: ``start_lost``, as before. A third-base change
      can do this while translating to the same residue, which is why the codon and
      not the amino acid decides.
    * near-cognate → ATG: the start got *stronger*. Calling that start-loss would be
      backwards, so the amino-acid classification stands and carries a note.
    * near-cognate → another near-cognate: a lateral move between weak starts. Note,
      no override.

    Args:
        ref_codon: The ORF's reference start trinucleotide.
        alt_codon: The same position after the substitution.

    Returns:
        ``(consequence_override, note)``. The override is ``None`` when the caller
        should keep its amino-acid-derived term; the note may be set either way, and
        is ``""`` only when the start is untouched.
    """
    ref = ref_codon[:3].upper()
    alt = alt_codon[:3].upper()

    if ref == alt:
        # A multi-base substitution can touch codon 0's span while leaving the
        # trinucleotide itself intact, changing only the codon after it.
        return None, ""

    if ref == "ATG":
        return "start_lost", START_LOST_ATG_NOTE

    if ref not in NEAR_COGNATE_STARTS:
        # Not reachable for a well-formed ORF — the annotated start is by
        # construction an initiator. Said out loud rather than silently treated as
        # a near-cognate, because it means the annotation and the sequence disagree.
        return None, START_NOT_AN_INITIATOR_NOTE

    if alt not in NEAR_COGNATE_STARTS:
        return "start_lost", START_LOST_NEAR_COGNATE_NOTE

    if alt == "ATG":
        return None, START_STRENGTHENED_NOTE

    return None, START_STILL_NEAR_COGNATE_NOTE


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


def ratio(subject: float | None, reference: float | None) -> float | None:
    """``subject / reference``, or None where that does not read as a ratio.

    Undefined for a missing or non-finite side, a reference ≤ 0 (a negative
    denominator inverts it) or a negative subject. The pipeline's region ratios
    and the site's comparison tables both use this, so they cannot disagree.
    """
    if subject is None or reference is None:
        return None
    if not (math.isfinite(subject) and math.isfinite(reference)):
        return None
    if reference <= 0 or subject < 0:
        return None
    return subject / reference


def density_ratio(
    n_unique: float, unique_len: float, n_shared: float, shared_len: float
) -> float | None:
    """Per-length density ratio ``(n_unique / unique_len) / (n_shared / shared_len)``.

    None when either length is ≤ 0 or the shared density is 0 (see :func:`ratio`).
    """
    if not unique_len or not shared_len or unique_len <= 0 or shared_len <= 0:
        return None
    return ratio(n_unique / unique_len, n_shared / shared_len)
