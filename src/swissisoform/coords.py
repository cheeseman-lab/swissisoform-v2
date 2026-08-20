"""Genomic primitives shared across the pipeline and the VCF scan.

Layer-2 walker and set-algebra helpers that translate between transcript
skeletons (:class:`TranscriptCoordinates`) and per-ORF genomic intervals, plus the
conventions everything downstream depends on: how a genomic position maps into an
ORF's coding sequence, what "inside an interval" means, and how a chromosome is
named.

All intervals are 0-based half-open plus-strand reference coordinates, ascending
regardless of strand; genomic positions are 1-based, so membership is
``start < pos <= end``.

**Deliberately stdlib-only.** ``website/prepare_deploy.sh`` vendors this module into
the Railway image, so it must import nothing the container does not install — hence
the ``TYPE_CHECKING`` guard below, which keeps the only annotation that needs
:mod:`swissisoform.models` out of the runtime import graph.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swissisoform.models import TranscriptCoordinates

#: Complement of an uppercase DNA base. Anything else passes through unchanged,
#: which is what lets a sequence carrying ``N`` round-trip without special-casing.
COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}


def revcomp(seq: str) -> str:
    """Reverse-complement a DNA string; characters outside ACGTN pass through.

    Uppercases first, because genome sequence can be soft-masked and a lowercase base
    would otherwise be reversed but not complemented — silently wrong in a way that
    only shows up on repeat-masked regions.

    Not for alignment strings: gap characters need their own handling, which
    ``conservation_frame.module._revcomp_maf`` does.
    """
    return "".join(COMPLEMENT.get(base, base) for base in reversed(seq.upper()))


def normalize_chrom(raw: str) -> str:
    """Normalise a chromosome name to the UCSC-style one the pipeline uses.

    Source databases and VCFs write bare names (``17``) even on GRCh38, while our
    coordinates are ``chr17``. Ensembl and 1000G call the mitochondrion ``MT``; UCSC,
    and therefore our GTF and genome, call it ``M``.
    """
    name = raw.strip()
    if not name.startswith("chr"):
        name = f"chr{name}"
    if name in ("chrMT", "chrmt", "chrMt"):
        return "chrM"
    return name


def iter_coding_positions(
    exons: Sequence[tuple[int, int]], strand: str
) -> Iterator[tuple[int, int]]:
    """Yield ``(1-based genomic position, 0-based coding offset)`` in mRNA order.

    The single traversal every coordinate mapping in the codebase is built on. Exons
    arrive in ascending genomic order whatever the strand, so on the minus strand
    they are walked in reverse and each one high→low — which makes offset 0 the ORF's
    first *translated* base in both directions.
    """
    ordered = list(exons)
    if strand == "-":
        ordered.reverse()
    coding_pos = 0
    for start, end in ordered:
        positions = range(end, start, -1) if strand == "-" else range(start + 1, end + 1)
        for gpos in positions:
            yield gpos, coding_pos
            coding_pos += 1


def coding_offset(exons: Sequence[tuple[int, int]], strand: str, pos: int) -> int | None:
    """0-based coding offset of 1-based ``pos`` within an ORF, or None if outside.

    The same answer as walking :func:`iter_coding_positions` and looking ``pos`` up,
    computed in O(exons) without building the map — which is what a scan wants, since
    it usually asks about one position per ORF. Build the map instead when many
    positions share an ORF.
    """
    ordered = list(exons)
    if strand == "-":
        ordered.reverse()
    offset = 0
    for start, end in ordered:
        if start < pos <= end:
            return offset + (end - pos if strand == "-" else pos - start - 1)
        offset += end - start
    return None


def position_in_intervals(pos: int, intervals: Iterable[tuple[int, int]]) -> bool:
    """True when 1-based ``pos`` falls in any 0-based half-open interval."""
    return any(start < pos <= end for start, end in intervals)


def span_overlaps_intervals(
    intervals: Iterable[tuple[int, int]], start: int, end: int
) -> bool:
    """True when the 1-based inclusive span ``[start, end]`` meets any interval.

    The span form of :func:`position_in_intervals`: an exon ``[s, e)`` covers 1-based
    ``s+1 .. e``, so it overlaps when ``s < end and e >= start``. A multi-base variant
    needs this — testing only its first base would miss a deletion that reaches into
    an exon from outside.
    """
    return any(s < end and e >= start for s, e in intervals)


def unique_shared_intervals(
    is_truncation: bool,
    orf_exons: Sequence[tuple[int, int]] | None,
    canonical_orf_exons: Sequence[tuple[int, int]] | None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    r"""Genomic ``(unique, shared)`` interval sets for one ORF pair.

    ORF-type-aware, and that asymmetry is the whole point: extensions, uORFs and
    altORFs contribute *new* isoform sequence (``isoform \ canonical``), while a
    truncation *loses* canonical sequence (``canonical \ isoform``). The shared set is
    the intersection either way. Empty when either skeleton is missing.

    Takes a bool rather than an ``ORFType`` so this module stays importable in the
    website image; callers pass ``orf_type == ORFType.TRUNCATED``.

    Pure set algebra: a missing skeleton is *not* special-cased here, because the two
    callers mean different things by it. The parquet writer reports empty sets ("we
    cannot say"), while the scoring pass treats an ORF with no canonical counterpart
    as entirely unique — so each guards before calling.
    """
    a, b = (canonical_orf_exons, orf_exons) if is_truncation else (orf_exons, canonical_orf_exons)
    unique = interval_difference(list(a or []), list(b or []))
    shared = interval_intersection(list(orf_exons or []), list(canonical_orf_exons or []))
    return unique, shared


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
