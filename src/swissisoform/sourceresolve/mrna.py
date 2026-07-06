"""Spliced mRNA reconstruction and ±W start-codon window extraction.

Phase 1 of the source-transcript resolution workstream (data-independent).
Reconstructs a transcript's full spliced mRNA (5'UTR + CDS + 3'UTR) from its
exon skeleton + a genome, and extracts the sequence window around a TIS start
codon used by the window-purity test (:mod:`swissisoform.sourceresolve.purity`).

Coordinate conventions match the rest of the package: all genomic positions
are 0-based half-open plus-strand reference coordinates. ``genome`` is any
object exposing ``.fetch(chrom, start, end) -> str`` (e.g. ``pysam.FastaFile``),
so callers pass a real indexed FASTA in production and a tiny in-memory stub in
tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from swissisoform.models import TranscriptCoordinates

# Reuse the same complement table the assembly layer uses for revcomp.
_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def _revcomp(seq: str) -> str:
    """Reverse-complement an upper-cased DNA sequence."""
    return seq.upper().translate(_COMPLEMENT)[::-1]


def _cdna_index(coords: TranscriptCoordinates, gpos: int) -> int | None:
    """0-based mRNA (cDNA) index of plus-strand genomic position ``gpos``.

    Walks exons in mRNA order — ascending genomic for ``+`` strand, descending
    for ``-`` strand — accumulating exonic length until the exon containing
    ``gpos`` is reached.

    Args:
        coords: Transcript skeleton (exons in any order; sorted internally).
        gpos: 0-based plus-strand genomic position.

    Returns:
        The mRNA index of the base at ``gpos``, or ``None`` if ``gpos`` is
        intronic or outside every exon.
    """
    exons = sorted(coords.exons)
    acc = 0
    if coords.strand == "+":
        for start, end in exons:
            if start <= gpos < end:
                return acc + (gpos - start)
            acc += end - start
    else:
        for start, end in reversed(exons):
            if start <= gpos < end:
                return acc + (end - 1 - gpos)
            acc += end - start
    return None


def build_transcript_mrna(coords: TranscriptCoordinates, genome: Any) -> str:
    """Reconstruct the full spliced mRNA (5'→3'): 5'UTR + CDS + 3'UTR.

    Concatenates exon sequences in ascending genomic order and
    reverse-complements the result for minus-strand transcripts to recover
    mRNA orientation.

    Args:
        coords: Transcript skeleton with the full exon structure.
        genome: Object with ``.fetch(chrom, start, end) -> str`` (0-based
            half-open), e.g. ``pysam.FastaFile``.

    Returns:
        The spliced mRNA sequence (upper-cased), or ``""`` if no exons.
    """
    exons = sorted(coords.exons)
    if not exons:
        return ""
    plus = "".join(genome.fetch(coords.chrom, start, end).upper() for start, end in exons)
    return _revcomp(plus) if coords.strand == "-" else plus


def start_codon_index(coords: TranscriptCoordinates, start_genomic: int) -> int | None:
    """mRNA index of the A of the start codon for a TIS at ``start_genomic``.

    ``start_genomic`` uses the ``TIS.position`` convention shared with
    :func:`swissisoform.coords.orf_exons_from_skeleton`:

    - ``+`` strand: 0-based plus-strand position of the A of the start codon.
    - ``-`` strand: 0-based plus-strand *exclusive* end of the ORF range; the
      A sits at plus-strand ``start_genomic - 1``.

    Returns:
        mRNA index of the A, or ``None`` if it is not exonic.
    """
    a_pos = start_genomic if coords.strand == "+" else start_genomic - 1
    return _cdna_index(coords, a_pos)


@dataclass
class TisWindow:
    """mRNA sequence around a TIS start codon, anchored at the A (index 0).

    Attributes:
        upstream: Bases immediately 5' of the A, in 5'→3' order (length ≤ the
            requested ``up``; shorter when the transcript 5' end is reached).
            ``upstream[-1]`` is the base immediately 5' of the A.
        downstream: The A plus bases 3' of it, in 5'→3' order (length ≤ the
            requested ``down``). ``downstream[0]`` is the A; ``downstream[:3]``
            is the start codon.
        a_index: mRNA index of the A in the full transcript.
        clamped_5p: True if the upstream window hit the transcript 5' end
            (i.e. the 5'UTR is shorter than ``up``).
    """

    upstream: str
    downstream: str
    a_index: int
    clamped_5p: bool


def extract_tis_window(
    coords: TranscriptCoordinates,
    genome: Any,
    start_genomic: int,
    up: int = 100,
    down: int = 100,
) -> TisWindow | None:
    """Extract the mRNA window around a TIS start codon.

    Builds the spliced mRNA, locates the start-codon A, and slices ``up`` bases
    5' of it and ``down`` bases starting at it (the A + 3' context). The window
    is clamped at the transcript 5' end (a short 5'UTR yields a short
    ``upstream``).

    Args:
        coords: Transcript skeleton.
        genome: Object with ``.fetch(chrom, start, end) -> str``.
        start_genomic: TIS start (see :func:`start_codon_index` convention).
        up: Bases to take 5' of the A (default 100).
        down: Bases to take starting at the A (default 100).

    Returns:
        A :class:`TisWindow`, or ``None`` if the A is intronic / outside the
        exons or falls beyond the reconstructed mRNA.
    """
    a_idx = start_codon_index(coords, start_genomic)
    if a_idx is None:
        return None
    mrna = build_transcript_mrna(coords, genome)
    if a_idx >= len(mrna):
        return None
    lo = a_idx - up
    upstream = mrna[max(0, lo) : a_idx]
    downstream = mrna[a_idx : a_idx + down]
    return TisWindow(
        upstream=upstream,
        downstream=downstream,
        a_index=a_idx,
        clamped_5p=lo < 0,
    )
