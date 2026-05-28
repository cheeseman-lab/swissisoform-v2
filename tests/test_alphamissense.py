"""Tests for the AlphaMissense tabix lookup (synthetic local table)."""

from __future__ import annotations

import pysam
import pytest

from swissisoform.clinical.alphamissense import AlphaMissenseLookup

# CHROM POS REF ALT genome uniprot transcript protein_variant am_path am_class
_ROWS = [
    "chr17\t48071438\tG\tC\thg38\tP83916\tENST00000225603.9\tN185K\t0.6784\tlikely_pathogenic",
    "chr17\t48071438\tG\tT\thg38\tP83916\tENST00000225603.9\tN185K\t0.6784\tlikely_pathogenic",
    # Same position, second transcript with a different score (preference test).
    "chr17\t48071438\tG\tC\thg38\tP83916\tENST00000999999.1\tN12K\t0.1100\tlikely_benign",
    "chr1\t69103\tT\tC\thg38\tQ8NH21\tENST00000335137.4\tF5L\t0.9110\tlikely_pathogenic",
]


@pytest.fixture()
def am_db(tmp_path):
    """Build a tiny bgzipped + tabix-indexed AlphaMissense table."""
    tsv = tmp_path / "AlphaMissense_hg38.tsv"
    cols = [
        "#CHROM", "POS", "REF", "ALT", "genome", "uniprot_id",
        "transcript_id", "protein_variant", "am_pathogenicity", "am_class",
    ]
    tsv.write_text("\t".join(cols) + "\n" + "\n".join(_ROWS) + "\n")
    gz = str(tsv) + ".gz"
    pysam.tabix_compress(str(tsv), gz, force=True)
    pysam.tabix_index(gz, seq_col=0, start_col=1, end_col=1, meta_char="#", force=True)
    return gz


class TestAlphaMissenseLookup:
    def test_available(self, am_db):
        assert AlphaMissenseLookup(am_db).available is True

    def test_missing_db_unavailable(self, tmp_path):
        lk = AlphaMissenseLookup(tmp_path / "nope.tsv.gz")
        assert lk.available is False
        assert lk.lookup("chr1", 100, "A", "T") is None

    def test_snv_hit(self, am_db):
        rec = AlphaMissenseLookup(am_db).lookup("chr1", 69103, "T", "C")
        assert rec is not None
        assert rec["am_pathogenicity"] == pytest.approx(0.911)
        assert rec["am_class"] == "likely_pathogenic"
        assert rec["protein_variant"] == "F5L"

    def test_allele_must_match(self, am_db):
        # Position exists but no T>G row → None.
        assert AlphaMissenseLookup(am_db).lookup("chr1", 69103, "T", "G") is None

    def test_indel_returns_none(self, am_db):
        assert AlphaMissenseLookup(am_db).lookup("chr17", 48071439, "TTCTTG", "T") is None

    def test_transcript_preference(self, am_db):
        lk = AlphaMissenseLookup(am_db)
        # Without transcript hint → first matching allele row (the canonical one).
        rec_any = lk.lookup("chr17", 48071438, "G", "C")
        assert rec_any["transcript_id"] == "ENST00000225603.9"
        # With a hint matching the second transcript → its (versionless) row.
        rec_tx = lk.lookup("chr17", 48071438, "G", "C", transcript_id="ENST00000999999")
        assert rec_tx["transcript_id"] == "ENST00000999999.1"
        assert rec_tx["am_class"] == "likely_benign"

    def test_absent_chrom(self, am_db):
        assert AlphaMissenseLookup(am_db).lookup("chrZ", 100, "A", "T") is None
