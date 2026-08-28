"""Tests for the ORF-exon Layer-2 walker and interval set-algebra."""

from __future__ import annotations

from swissisoform.coords import (
    first_coding_base,
    interval_difference,
    interval_intersection,
    interval_length,
    orf_exons_from_skeleton,
    start_offset_nt,
)
from swissisoform.models import TranscriptCoordinates


def _coords(
    strand: str,
    exons: list[tuple[int, int]],
    cds_start: int | None = None,
) -> TranscriptCoordinates:
    return TranscriptCoordinates(
        transcript_id="ENST_TEST",
        chrom="chr1",
        strand=strand,
        exons=exons,
        cds_start=cds_start,
    )


class TestWalkerPlusStrand:
    def test_single_exon_contained(self):
        coords = _coords("+", [(1000, 2000)])
        out = orf_exons_from_skeleton(coords, orf_start_genomic=1100, aa_len=10)
        assert out == [(1100, 1130)]

    def test_spans_multiple_exons(self):
        # Walk 30 nt from 990 across intron into next exon at 1100
        coords = _coords("+", [(900, 1000), (1100, 2000)])
        out = orf_exons_from_skeleton(coords, orf_start_genomic=990, aa_len=10)
        # 10 nt from [990, 1000) + 20 nt from [1100, 1120)
        assert out == [(990, 1000), (1100, 1120)]
        assert interval_length(out) == 30

    def test_zero_len(self):
        coords = _coords("+", [(1000, 2000)])
        assert orf_exons_from_skeleton(coords, 1000, 0) == []

    def test_start_outside_exons(self):
        coords = _coords("+", [(1000, 1100)])
        # ORF start past the last exon — nothing to walk.
        assert orf_exons_from_skeleton(coords, 2000, 10) == []


class TestWalkerMinusStrand:
    def test_single_exon_contained(self):
        # Minus strand: orf_start_genomic is the exclusive upper bound
        coords = _coords("-", [(1000, 2000)])
        out = orf_exons_from_skeleton(coords, orf_start_genomic=1900, aa_len=10)
        # 30 nt consuming downward: [1870, 1900)
        assert out == [(1870, 1900)]

    def test_spans_multiple_exons(self):
        # Exons (ascending): [900, 1000), [1100, 2000)
        # mRNA order on minus strand: [1100, 2000) then [900, 1000)
        # ORF end at 1110 (exclusive); consume 30 nt downward
        coords = _coords("-", [(900, 1000), (1100, 2000)])
        out = orf_exons_from_skeleton(coords, orf_start_genomic=1110, aa_len=10)
        # Take 10 nt from [1100, 1110), then 20 nt from [980, 1000)
        assert out == [(980, 1000), (1100, 1110)]
        assert interval_length(out) == 30


class TestIntervalAlgebra:
    def test_difference_disjoint(self):
        assert interval_difference([(0, 10)], [(20, 30)]) == [(0, 10)]

    def test_difference_overlapping(self):
        # [0,10) \ [3,7) = [0,3) + [7,10)
        assert interval_difference([(0, 10)], [(3, 7)]) == [(0, 3), (7, 10)]

    def test_difference_full_cover(self):
        assert interval_difference([(0, 10)], [(0, 10)]) == []

    def test_intersection_partial(self):
        assert interval_intersection([(0, 10)], [(5, 20)]) == [(5, 10)]

    def test_intersection_disjoint(self):
        assert interval_intersection([(0, 10)], [(20, 30)]) == []

    def test_intersection_multiple(self):
        a = [(0, 10), (20, 30)]
        b = [(5, 25)]
        assert interval_intersection(a, b) == [(5, 10), (20, 25)]


class TestStartOffsetNt:
    """The figure's isoform→canonical x shift, in mRNA nucleotides.

    ``start_offset_nt / 3`` is the shift; unlike ``canonical_len - isoform_len`` it
    stays defined when the two proteins share no C-terminus (uORFs, altORFs), which
    is the case the old shortcut placed at the canonical N-terminus instead.
    """

    # Two exons, one intron: mRNA is [100,200) + [300,400).
    TX = [(100, 200), (300, 400)]

    def test_first_base_is_strand_aware(self):
        # The A of the start codon: lowest coordinate on +, highest on -.
        assert first_coding_base([(100, 200)], "+") == 101
        assert first_coding_base([(100, 200)], "-") == 200
        assert first_coding_base([], "+") is None

    def test_upstream_orf_is_negative(self):
        # uORF at 110 vs canonical at 150, both in the first exon.
        assert start_offset_nt(self.TX, "+", [(110, 140)], [(150, 200), (300, 340)]) == -40

    def test_downstream_orf_is_positive_and_skips_the_intron(self):
        # 150 is mRNA offset 50; 310 is 100 + 10 = 110. The 100 nt of intron
        # between them must not be counted.
        assert start_offset_nt(self.TX, "+", [(310, 340)], [(150, 200), (300, 340)]) == 60

    def test_minus_strand_walks_the_other_way(self):
        # mRNA order is [300,400) high→low, then [100,200). The ORF starting at
        # 400 is 50 nt upstream of the canonical starting at 350.
        assert start_offset_nt(self.TX, "-", [(370, 400)], [(150, 200), (300, 350)]) == -50

    def test_out_of_frame_offset_is_not_a_multiple_of_three(self):
        # A uORF need not read in the canonical frame. Canonical starts at mRNA
        # offset 50; this ORF starts at 12, so it is 38 nt upstream — not a whole
        # number of codons. That remainder is what makes the x coordinate
        # fractional rather than something to round away.
        offset = start_offset_nt(self.TX, "+", [(112, 142)], [(150, 200)])
        assert offset == -38
        assert offset % 3 != 0

    def test_generalises_right_alignment(self):
        # Where both proteins share a C-terminus, the answer IS the old shortcut:
        # a 10-residue N-terminal extension starts 30 nt upstream.
        canonical = [(160, 200), (300, 400)]
        extension = [(130, 200), (300, 400)]
        assert start_offset_nt(self.TX, "+", extension, canonical) == -30

    def test_missing_orf_returns_none(self):
        assert start_offset_nt(self.TX, "+", [], [(150, 200)]) is None
        assert start_offset_nt(self.TX, "+", [(110, 140)], None) is None

    def test_start_outside_the_transcript_returns_none(self):
        # A skeleton/ORF mismatch. None so callers fall back rather than guess.
        assert start_offset_nt(self.TX, "+", [(1000, 1030)], [(150, 200)]) is None
