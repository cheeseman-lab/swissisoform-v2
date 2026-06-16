"""Tests for spliced mRNA reconstruction + ±W start-codon window extraction."""

from __future__ import annotations

from swissisoform.models import TranscriptCoordinates
from swissisoform.sourceseq.mrna import (
    build_transcript_mrna,
    extract_tis_window,
    start_codon_index,
)

_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def _rc(seq: str) -> str:
    return seq.upper().translate(_COMPLEMENT)[::-1]


class FakeGenome:
    """Minimal ``pysam.FastaFile``-like stub over in-memory sequences."""

    def __init__(self, seqs: dict[str, str]) -> None:
        """Store ``{chrom: sequence}`` for ``fetch()``."""
        self.seqs = seqs

    def fetch(self, chrom: str, start: int, end: int) -> str:
        return self.seqs[chrom][start:end]


# A designed 30-nt mRNA with the start codon ATG at index 10.
MRNA = "ACGTACGTAC" + "ATG" + "C" * 17
assert len(MRNA) == 30 and MRNA[10:13] == "ATG"


def _coords(strand, exons, chrom="chr1"):
    return TranscriptCoordinates(transcript_id="T", chrom=chrom, strand=strand, exons=exons)


class TestBuildMrnaPlus:
    def test_single_exon(self):
        genome = FakeGenome({"chr1": "N" * 10 + MRNA + "N" * 20})
        coords = _coords("+", [(10, 40)])
        assert build_transcript_mrna(coords, genome) == MRNA

    def test_multi_exon_splices_out_intron(self):
        # exon1 [10,25)="A"*15, intron [25,35), exon2 [35,50)="CCCCCATGCCCCCCC"
        chrom = "N" * 10 + "A" * 15 + "N" * 10 + "CCCCCATGCCCCCCC" + "N" * 10
        genome = FakeGenome({"chr1": chrom})
        coords = _coords("+", [(10, 25), (35, 50)])
        mrna = build_transcript_mrna(coords, genome)
        assert mrna == "A" * 15 + "CCCCCATGCCCCCCC"
        assert mrna[20:23] == "ATG"

    def test_empty_exons(self):
        assert build_transcript_mrna(_coords("+", []), FakeGenome({"chr1": "ACGT"})) == ""


class TestStartCodonIndex:
    def test_plus_single_exon(self):
        coords = _coords("+", [(10, 40)])
        # A of ATG at genomic 20 (10 + index 10)
        assert start_codon_index(coords, 20) == 10

    def test_plus_multi_exon_across_junction(self):
        coords = _coords("+", [(10, 25), (35, 50)])
        # mRNA index 20 → genomic 40 (15 exon1 bases + 5 into exon2)
        assert start_codon_index(coords, 40) == 20

    def test_minus_single_exon(self):
        coords = _coords("-", [(10, 40)])
        # minus: start_genomic is exclusive end; A at plus pos start_genomic-1.
        # For mRNA index 10 we need plus pos 29 → start_genomic 30.
        assert start_codon_index(coords, 30) == 10

    def test_intronic_returns_none(self):
        coords = _coords("+", [(10, 25), (35, 50)])
        assert start_codon_index(coords, 30) is None  # 30 is in the intron


class TestExtractWindowPlus:
    def test_basic_window(self):
        genome = FakeGenome({"chr1": "N" * 10 + MRNA + "N" * 20})
        coords = _coords("+", [(10, 40)])
        win = extract_tis_window(coords, genome, start_genomic=20, up=5, down=5)
        assert win is not None
        assert win.a_index == 10
        assert win.downstream == "ATGCC"  # A + start codon + 3' context
        assert win.upstream == MRNA[5:10] == "CGTAC"
        assert win.clamped_5p is False

    def test_5p_clamp(self):
        genome = FakeGenome({"chr1": "N" * 10 + MRNA + "N" * 20})
        coords = _coords("+", [(10, 40)])
        win = extract_tis_window(coords, genome, start_genomic=20, up=20, down=5)
        assert win.upstream == MRNA[0:10]  # only 10 bases of 5'UTR available
        assert win.clamped_5p is True

    def test_upstream_crosses_splice_junction(self):
        chrom = "N" * 10 + "A" * 15 + "N" * 10 + "CCCCCATGCCCCCCC" + "N" * 10
        genome = FakeGenome({"chr1": chrom})
        coords = _coords("+", [(10, 25), (35, 50)])
        win = extract_tis_window(coords, genome, start_genomic=40, up=8, down=3)
        # 8 upstream bases walk across the intron: 3 from exon1 ("A") + 5 "C"
        assert win.upstream == "AAA" + "CCCCC"
        assert win.downstream == "ATG"

    def test_intronic_start_returns_none(self):
        genome = FakeGenome({"chr1": "N" * 60})
        coords = _coords("+", [(10, 25), (35, 50)])
        assert extract_tis_window(coords, genome, start_genomic=30) is None


class TestExtractWindowMinus:
    def test_basic_window(self):
        # minus-strand transcript whose mRNA equals MRNA → plus exon = rc(MRNA)
        genome = FakeGenome({"chr1": "N" * 10 + _rc(MRNA) + "N" * 20})
        coords = _coords("-", [(10, 40)])
        assert build_transcript_mrna(coords, genome) == MRNA
        win = extract_tis_window(coords, genome, start_genomic=30, up=5, down=5)
        assert win is not None
        assert win.a_index == 10
        assert win.downstream == "ATGCC"
        assert win.upstream == "CGTAC"
