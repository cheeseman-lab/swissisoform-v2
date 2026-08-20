"""Which protein a variant is numbered against, and where it sits on the figure.

The coordinate walk itself lives in :mod:`swissisoform.coords` and is shared with
the pipeline's position mapper — one traversal, two access patterns. What is here is
the part with no pipeline counterpart: choosing *which* of an isoform's two reading
frames numbers a variant, and converting that residue into the gene figure's axis.
"""

from __future__ import annotations

from swissisoform.coords import coding_offset, span_overlaps_intervals
from swissisoform.variantquery.index import OrfRecord

__all__ = [
    "FRAME_CANONICAL",
    "FRAME_ISOFORM",
    "REGION_OTHER",
    "REGION_SHARED",
    "REGION_UNIQUE",
    "canonical_x",
    "coding_offset",
    "plotly_x",
    "region_for",
    "resolve_residue",
]

#: Which protein a residue is numbered against.
FRAME_ISOFORM = "isoform"
FRAME_CANONICAL = "canonical"

#: Which part of the isoform/canonical pair the position falls in.
REGION_UNIQUE = "unique"
REGION_SHARED = "shared"
REGION_OTHER = "other"


def region_for(record: OrfRecord, start: int, end: int) -> str:
    """Classify a 1-based inclusive span as isoform-unique, shared, or other.

    Reads the **precomputed** ``unique_``/``shared_genomic_intervals`` from the
    parquet rather than re-deriving them, so this can never disagree with what
    the pipeline scored. ``unique`` wins a tie: a span straddling the boundary is
    the more interesting claim, and is what the differential-region cards show.
    """
    if span_overlaps_intervals(record.unique_intervals, start, end):
        return REGION_UNIQUE
    if span_overlaps_intervals(record.shared_intervals, start, end):
        return REGION_SHARED
    return REGION_OTHER


def resolve_residue(record: OrfRecord, start: int, end: int) -> tuple[int | None, str, int | None]:
    """Residue number, frame, and the genomic base it was read from.

    Tries the isoform ORF first, then the per-transcript canonical ORF. That
    ordering is what puts a **truncation's lost N-terminus** in canonical frame:
    those bases are absent from the isoform protein, so only a canonical-frame
    residue number exists for them — matching how the gene figure already places
    such variants.

    For multi-base REFs the span is walked in ascending genomic order and the
    first base that lands inside the ORF is used, so a deletion starting in an
    intron and running into an exon still reports a residue.

    Returns:
        ``(residue, frame, genomic_pos)`` with a **0-based** residue (p.R248 is
        ``247``), or ``(None, "", None)`` when no base of the span is inside
        either ORF.
    """
    for exons, frame in (
        (record.orf_exons, FRAME_ISOFORM),
        (record.canonical_orf_exons, FRAME_CANONICAL),
    ):
        for pos in range(start, end + 1):
            offset = coding_offset(exons, record.strand, pos)
            if offset is not None:
                return offset // 3, frame, pos
    return None, "", None


def canonical_x(
    residue: int, frame: str, canonical_len: int | None, isoform_len: int | None
) -> int | None:
    """Residue → the gene figure's x coordinate, given the two protein lengths.

    The combined gene figure draws everything in canonical-residue space with
    isoforms right-aligned on the shared region, so an isoform-frame residue is
    shifted by ``canonical_len - isoform_len``; a canonical-frame residue is already
    in that space.

    The figure adapter writes the same conversion as ``residue + 1 + offset``
    followed by a global ``-1`` that anchors canonical residue 1 at x = 0; those
    cancel, so ``residue + offset`` is the same coordinate.
    """
    if frame == FRAME_CANONICAL:
        return residue
    if canonical_len is None or isoform_len is None:
        return None
    return residue + (canonical_len - isoform_len)


def plotly_x(record: OrfRecord, residue: int, frame: str) -> int | None:
    """:func:`canonical_x` for a scan hit, reading both lengths off its ORF."""
    return canonical_x(residue, frame, record.canonical_len, record.isoform_len)
