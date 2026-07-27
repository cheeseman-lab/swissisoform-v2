"""Genomic interval utilities for ORF exon infrastructure.

Layer-2 walker and set-algebra helpers that translate between transcript
skeletons (:class:`TranscriptCoordinates`) and per-ORF genomic intervals. All
intervals are 0-based half-open plus-strand reference coordinates.
"""

from __future__ import annotations

from swissisoform.models import TranscriptCoordinates


def orf_exons_from_skeleton(
    coords: TranscriptCoordinates,
    orf_start_genomic: int,
    aa_len: int,
) -> list[tuple[int, int]]:
    """Walk an exon skeleton to produce genomic intervals covering an ORF.

    Starting at ``orf_start_genomic``, consumes ``aa_len * 3`` nucleotides
    forward in transcript order (across introns), returning the resulting
    plus-strand half-open intervals. The stop codon is not included.

    ``orf_start_genomic`` uses the same convention as ``TIS.position``:

    - ``+`` strand: 0-based plus-strand position of the A of ATG (first nt
      consumed). The walker takes ``[orf_start_genomic, orf_start_genomic+1)``
      as the first base and proceeds ascending.
    - ``-`` strand: 0-based plus-strand *exclusive* end of the ORF range.
      The A of ATG on the minus strand sits at plus-strand
      ``orf_start_genomic - 1``. The walker takes
      ``[orf_start_genomic - 1, orf_start_genomic)`` as the first base and
      proceeds descending.

    Args:
        coords: Transcript skeleton with exons in ascending genomic order.
        orf_start_genomic: Genomic start of the ORF (see convention above).
        aa_len: Length of the ORF in amino acids. The walker consumes
            ``aa_len * 3`` nucleotides.

    Returns:
        List of ``(start, end)`` plus-strand half-open intervals covering the
        ORF. Empty list if ``aa_len <= 0`` or the ORF start falls outside
        every exon. Intervals are returned in ascending genomic order
        regardless of strand (so downstream set operations can treat them
        uniformly).
    """
    nt_needed = aa_len * 3
    if nt_needed <= 0 or not coords.exons:
        return []

    exons = sorted(coords.exons)
    result: list[tuple[int, int]] = []

    if coords.strand == "+":
        for ex_start, ex_end in exons:
            if ex_end <= orf_start_genomic:
                continue
            take_start = max(ex_start, orf_start_genomic)
            if take_start >= ex_end:
                continue
            take_end = min(ex_end, take_start + nt_needed)
            result.append((take_start, take_end))
            nt_needed -= take_end - take_start
            if nt_needed <= 0:
                break
    else:  # minus strand
        for ex_start, ex_end in reversed(exons):
            if ex_start >= orf_start_genomic:
                continue
            take_end = min(ex_end, orf_start_genomic)
            if take_end <= ex_start:
                continue
            take_start = max(ex_start, take_end - nt_needed)
            result.append((take_start, take_end))
            nt_needed -= take_end - take_start
            if nt_needed <= 0:
                break
        result.reverse()

    return result


def _normalize(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping/adjacent intervals; drop empties."""
    cleaned = [(s, e) for s, e in intervals if e > s]
    if not cleaned:
        return []
    cleaned.sort()
    merged: list[tuple[int, int]] = [cleaned[0]]
    for start, end in cleaned[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def interval_length(intervals: list[tuple[int, int]]) -> int:
    """Total length of half-open intervals."""
    return sum(end - start for start, end in intervals)


def interval_difference(
    a: list[tuple[int, int]],
    b: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    r"""Return ``a \ b`` as a sorted, merged list of half-open intervals."""
    a_norm = _normalize(a)
    b_norm = _normalize(b)
    if not a_norm:
        return []
    if not b_norm:
        return a_norm

    result: list[tuple[int, int]] = []
    for start, end in a_norm:
        cursor = start
        for bs, be in b_norm:
            if be <= cursor:
                continue
            if bs >= end:
                break
            if bs > cursor:
                result.append((cursor, bs))
            cursor = max(cursor, be)
            if cursor >= end:
                break
        if cursor < end:
            result.append((cursor, end))
    return result


def interval_intersection(
    a: list[tuple[int, int]],
    b: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Return ``a ∩ b`` as a sorted, merged list of half-open intervals."""
    a_norm = _normalize(a)
    b_norm = _normalize(b)
    result: list[tuple[int, int]] = []
    i = j = 0
    while i < len(a_norm) and j < len(b_norm):
        a_start, a_end = a_norm[i]
        b_start, b_end = b_norm[j]
        start = max(a_start, b_start)
        end = min(a_end, b_end)
        if start < end:
            result.append((start, end))
        if a_end <= b_end:
            i += 1
        else:
            j += 1
    return result
