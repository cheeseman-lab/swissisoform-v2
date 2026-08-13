"""Genomic position → coding offset → protein residue, per ORF.

Pure arithmetic mirror of ``ConsequenceValidator.build_position_map_from_orf``
(``clinical/validate.py:221``), which walks the exons in mRNA order and assigns
consecutive coding offsets. That method materialises a full ``{gpos: offset}``
dict and needs biopython in its module; here the same walk is done as O(exons)
arithmetic with no dependencies, so it can run inside the website image.

The two are cross-checked against each other in the test suite.
"""

from __future__ import annotations

from swissisoform.variantquery.index import Interval, OrfRecord

#: Which protein a residue is numbered against.
FRAME_ISOFORM = "isoform"
FRAME_CANONICAL = "canonical"

#: Which part of the isoform/canonical pair the position falls in.
REGION_UNIQUE = "unique"
REGION_SHARED = "shared"
REGION_OTHER = "other"


def coding_offset(exons: tuple[Interval, ...], strand: str, pos: int) -> int | None:
    """0-based coding offset of 1-based ``pos`` within an ORF, or None if outside.

    Exons are 0-based half-open plus-strand ascending; on the minus strand they
    are walked in reverse (and each exon high→low) so offset 0 is always the
    ORF's first translated base.
    """
    if not exons:
        return None
    ordered = exons if strand != "-" else tuple(reversed(exons))
    offset = 0
    for start, end in ordered:
        if start < pos <= end:
            # start+1 is the first base of the exon in plus-strand terms.
            return offset + (pos - start - 1 if strand != "-" else end - pos)
        offset += end - start
    return None


def _overlaps(intervals: tuple[Interval, ...], start: int, end: int) -> bool:
    """True when any 0-based half-open interval overlaps 1-based ``[start, end]``."""
    return any(s < end and e >= start for s, e in intervals)


def region_for(record: OrfRecord, start: int, end: int) -> str:
    """Classify a 1-based inclusive span as isoform-unique, shared, or other.

    Reads the **precomputed** ``unique_``/``shared_genomic_intervals`` from the
    parquet rather than re-deriving them, so this can never disagree with what
    the pipeline scored. ``unique`` wins a tie: a span straddling the boundary is
    the more interesting claim, and is what the differential-region cards show.
    """
    if _overlaps(record.unique_intervals, start, end):
        return REGION_UNIQUE
    if _overlaps(record.shared_intervals, start, end):
        return REGION_SHARED
    return REGION_OTHER


def resolve_residue(record: OrfRecord, start: int, end: int) -> tuple[int | None, str, int | None]:
    """Residue number, frame, and the genomic base it was read from.

    Tries the isoform ORF first, then the per-transcript canonical ORF. That
    ordering is what puts a **truncation's lost N-terminus** in canonical frame:
    those bases are absent from the isoform protein, so only a canonical-frame
    residue number exists for them — matching how the gene figure already places
    such variants (``app.py:715-753``).

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


def plotly_x(record: OrfRecord, residue: int, frame: str) -> int | None:
    """Residue → the gene figure's x coordinate.

    The combined gene figure draws everything in canonical-residue space with
    isoforms right-aligned on the shared region, so an isoform-frame residue is
    shifted by ``canonical_len - isoform_len``; a canonical-frame residue is
    already in that space.

    ``app.py:604-673`` writes this as ``residue + 1 + offset`` followed by a
    global ``-1`` that anchors canonical residue 1 at x = 0; those cancel, so
    plain ``residue + offset`` is the same coordinate.
    """
    if frame == FRAME_CANONICAL:
        return residue
    if record.canonical_len is None or record.isoform_len is None:
        return None
    return residue + (record.canonical_len - record.isoform_len)
