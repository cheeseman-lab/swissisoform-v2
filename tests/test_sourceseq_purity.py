"""Tests for the window-purity decision."""

from __future__ import annotations

from swissisoform.models import TranscriptCoordinates
from swissisoform.sourceseq.mrna import TisWindow, extract_tis_window
from swissisoform.sourceseq.purity import divergence_radius, purity_decision
from tests.test_sourceseq_mrna import FakeGenome


def _w(upstream: str, downstream: str) -> TisWindow:
    return TisWindow(
        upstream=upstream,
        downstream=downstream,
        a_index=len(upstream),
        clamped_5p=False,
    )


class TestPurityDecision:
    def test_identical_windows_kept(self):
        wins = [_w("AAAAA", "ATGCC"), _w("AAAAA", "ATGCC")]
        res = purity_decision(wins, w=5)
        assert res.keep is True
        assert res.reason == "pure_window"
        assert res.divergence_nt is None

    def test_downstream_divergence(self):
        # downstream differs at the 3rd codon base (radius 2)
        wins = [_w("AAAAA", "ATGCC"), _w("AAAAA", "ATTCC")]
        res = purity_decision(wins, w=5)
        assert res.keep is False
        assert res.reason == "ambiguous_window"
        assert res.divergence_nt == 2

    def test_upstream_divergence(self):
        # upstream differs 3 bases 5' of the A
        wins = [_w("AAAAA", "ATGCC"), _w("AACAA", "ATGCC")]
        res = purity_decision(wins, w=5)
        assert res.keep is False
        assert res.divergence_nt == 3

    def test_divergence_beyond_window_is_ignored(self):
        # differ only at downstream radius 6, but W=5 → still pure
        wins = [_w("AAAAA", "ATGCCCA"), _w("AAAAA", "ATGCCCT")]
        res = purity_decision(wins, w=5)
        assert res.keep is True
        assert res.divergence_nt is None

    def test_different_5p_extent_is_divergence(self):
        # one candidate has a shorter 5'UTR within the window → divergence at r=4
        wins = [_w("AAA", "ATGCC"), _w("GGAAA", "ATGCC")]
        res = purity_decision(wins, w=5)
        assert res.keep is False
        assert res.divergence_nt == 4

    def test_equal_short_5p_is_not_divergence(self):
        # both end the 5'UTR at the same radius and agree → pure
        wins = [_w("AAA", "ATGCC"), _w("AAA", "ATGCC")]
        res = purity_decision(wins, w=5)
        assert res.keep is True
        assert res.reason == "pure_window"

    def test_single_candidate(self):
        res = purity_decision([_w("AAAAA", "ATGCC")], w=5)
        assert res.keep is True
        assert res.reason == "single_candidate"

    def test_no_candidates(self):
        res = purity_decision([], w=5)
        assert res.keep is False
        assert res.reason == "no_candidates"


class TestDivergenceRadius:
    def test_returns_none_when_identical(self):
        assert divergence_radius([_w("AAAAA", "ATGCC"), _w("AAAAA", "ATGCC")], 5) is None


def _coords(strand, exons):
    return TranscriptCoordinates(transcript_id="T", chrom="chr1", strand=strand, exons=exons)


class TestPurityIntegration:
    """Two transcripts sharing the start codon but with different 5'UTR exons."""

    def _genome(self):
        # [10,20)= T1 5'UTR "A"*10 ; [30,40)= T2 5'UTR "G"*10 ;
        # [50,80)= shared CDS exon starting with ATG (A at genomic 50).
        chrom = "N" * 10 + "A" * 10 + "N" * 10 + "G" * 10 + "N" * 10 + "ATG" + "C" * 27
        return FakeGenome({"chr1": chrom})

    def test_divergent_5utr_discarded(self):
        genome = self._genome()
        t1 = _coords("+", [(10, 20), (50, 80)])
        t2 = _coords("+", [(30, 40), (50, 80)])
        w1 = extract_tis_window(t1, genome, start_genomic=50, up=5, down=5)
        w2 = extract_tis_window(t2, genome, start_genomic=50, up=5, down=5)
        assert w1.upstream == "AAAAA" and w2.upstream == "GGGGG"
        assert w1.downstream == w2.downstream == "ATGCC"
        res = purity_decision([w1, w2], w=5)
        assert res.keep is False
        assert res.divergence_nt == 1  # upstream differs immediately 5' of the A

    def test_shared_5utr_kept(self):
        genome = self._genome()
        t1 = _coords("+", [(10, 20), (50, 80)])
        t1b = _coords("+", [(10, 20), (50, 80)])
        w1 = extract_tis_window(t1, genome, start_genomic=50, up=5, down=5)
        w1b = extract_tis_window(t1b, genome, start_genomic=50, up=5, down=5)
        res = purity_decision([w1, w1b], w=5)
        assert res.keep is True
        assert res.reason == "pure_window"
