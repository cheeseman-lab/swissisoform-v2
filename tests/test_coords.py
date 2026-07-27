"""Tests for the ORF-exon Layer-2 walker and interval set-algebra."""

from __future__ import annotations

from swissisoform.coords import (
    interval_difference,
    interval_intersection,
    interval_length,
    orf_exons_from_skeleton,
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
