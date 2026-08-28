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

    For multi-base REFs every base of the span is mapped and the **lowest coding
    offset** wins — the span's first base in *translation* order. Taking the first
    base that maps while walking ascending genomic order would be the same thing on
    the plus strand and a codon late on the minus, where mRNA runs against genomic
    order and the lowest genomic coordinate is the span's *last* translated base.

    A deletion starting in an intron and running into an exon still reports a
    residue: unmapped bases are skipped rather than ending the walk.

    This matters only where nothing better is available. When the index carries a
    CDS, ``scan`` prefers the classifier's ``protein_pos`` (also a minimum over the
    span) and uses this function for the frame alone; an index built with
    ``--no-cds`` has no such override, and this number is what gets reported.

    Returns:
        ``(residue, frame, genomic_pos)`` with a **0-based** residue (p.R248 is
        ``247``), or ``(None, "", None)`` when no base of the span is inside
        either ORF.
    """
    for exons, frame in (
        (record.orf_exons, FRAME_ISOFORM),
        (record.canonical_orf_exons, FRAME_CANONICAL),
    ):
        mapped = [
            (offset, pos)
            for pos in range(start, end + 1)
            if (offset := coding_offset(exons, record.strand, pos)) is not None
        ]
        if mapped:
            offset, pos = min(mapped)
            return offset // 3, frame, pos
    return None, "", None


def x_offset_residues(x_offset_nt: int) -> int | float:
    """``canonical_x_offset_nt`` → the figure's x shift in residues.

    Exact ``int`` when the ORF reads in the canonical frame — which is every
    extension and truncation, so residue arithmetic downstream is unchanged. A
    ``float`` otherwise: a uORF three-quarters of a codon out of phase has no
    canonical residue to be rounded onto, and inventing one would put its bar and
    its markers half a residue apart.
    """
    shift, remainder = divmod(x_offset_nt, 3)
    return shift if remainder == 0 else x_offset_nt / 3


def canonical_x(
    residue: int,
    frame: str,
    canonical_len: int | None,
    isoform_len: int | None,
    x_offset_nt: int | None = None,
    canonical_per_tid_length: int | None = None,
) -> int | float | None:
    """Residue → the gene figure's x coordinate.

    The combined gene figure draws everything in canonical-residue space, so an
    isoform-frame residue is shifted by the mRNA distance between the two start
    codons: ``x_offset_nt / 3``.

    A canonical-frame residue is *not* already in that space. ``resolve_residue``
    numbers it against ``canonical_orf_exons``, the **per-transcript** canonical,
    while the figure draws one bar per gene — the gene-level representative, of
    length ``canonical_len``. Those are different proteins for 1,670 of 6,462 ORFs,
    so the residue is shifted by ``canonical_len - canonical_per_tid_length``: the
    same term ``derive_x_offsets`` folds into the isoform offset, on the assumption
    the two canonicals share a C-terminus. Without it a variant in a truncation's
    lost N-terminus — the only way canonical frame is reached, and the case this
    whole path exists for — lands up to 3,230 residues from where it belongs.

    That offset comes from the index (``build_orf_index.derive_x_offsets``) and
    is the same number the site's figure adapter uses to place the bar itself — one
    stored value, so a marker cannot land off the bar it belongs to.

    Falls back to ``canonical_len - isoform_len`` when the offset is absent (an
    index built before the column existed). That shortcut is exact wherever the two
    proteins share a C-terminus and undefined otherwise, which is the whole reason
    the offset exists.

    The result is an ``int`` unless the ORF reads out of the canonical frame, in
    which case the offset is not a whole number of residues (see
    :func:`x_offset_residues`).

    The figure adapter writes the same conversion as ``residue + 1 + offset``
    followed by a global ``-1`` that anchors canonical residue 1 at x = 0; those
    cancel, so ``residue + offset`` is the same coordinate.
    """
    if frame == FRAME_CANONICAL:
        if canonical_len is None or canonical_per_tid_length is None:
            return residue
        return residue + (canonical_len - canonical_per_tid_length)
    if x_offset_nt is not None:
        return residue + x_offset_residues(x_offset_nt)
    if canonical_len is None or isoform_len is None:
        return None
    return residue + (canonical_len - isoform_len)


def plotly_x(record: OrfRecord, residue: int, frame: str) -> int | float | None:
    """:func:`canonical_x` for a scan hit, reading the offset off its ORF."""
    return canonical_x(
        residue,
        frame,
        record.canonical_len,
        record.isoform_len,
        record.canonical_x_offset_nt,
        record.canonical_per_tid_length,
    )
