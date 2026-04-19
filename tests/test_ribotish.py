"""Tests for Ribo-TISH TSV reader and GTF annotation loader."""

from __future__ import annotations

import pandas as pd
import pytest

from swissisoform.io.ribotish import (
    load_ribotish_predictions,
    parse_genome_pos,
    recategorize_tis_type,
)

# ---------------------------------------------------------------------------
# parse_genome_pos
# ---------------------------------------------------------------------------


class TestParseGenomePos:
    """Tests for parse_genome_pos helper."""

    def test_forward_strand(self):
        chrom, start, end, strand = parse_genome_pos("chr1:100-200:+")
        assert chrom == "chr1"
        assert start == 100
        assert end == 200
        assert strand == "+"

    def test_reverse_strand(self):
        chrom, start, end, strand = parse_genome_pos("chr1:944693-959240:-")
        assert chrom == "chr1"
        assert start == 944693
        assert end == 959240
        assert strand == "-"

    def test_chrX(self):
        chrom, start, end, strand = parse_genome_pos("chrX:5000-6000:+")
        assert chrom == "chrX"

    def test_invalid_format_raises(self):
        with pytest.raises((ValueError, IndexError)):
            parse_genome_pos("bad_string")


# ---------------------------------------------------------------------------
# load_ribotish_predictions
# ---------------------------------------------------------------------------

RIBOTISH_HEADER = (
    "Gid\tTid\tSymbol\tGeneType\tGenomePos\tStartCodon\tStart\tStop\t"
    "TisType\tTISGroup\tTISCounts\tTISPvalue\tRiboPvalue\tRiboPStatus\t"
    "FisherPvalue\tTISQvalue\tFrameQvalue\tFisherQvalue\tAALen\tSeq\tAASeq"
)


def _make_row(
    genome_pos: str = "chr1:100-200:+",
    tis_type: str = "Annotated",
    symbol: str = "GENE1",
) -> str:
    """Build a single TSV row with sensible defaults."""
    return (
        f"ENSG00000000001.1\tENST00000000001.1\t{symbol}\tprotein_coding\t"
        f"{genome_pos}\tATG\t10\t100\t{tis_type}\t1\t50\t0.001\t0.0001\tT\t"
        f"0.0005\t0.01\t0.01\t0.005\t30\tATGCCC\tMP"
    )


class TestLoadRibotishPredictions:
    """Tests for load_ribotish_predictions."""

    def test_basic_load(self, tmp_path):
        tsv = tmp_path / "test.txt"
        tsv.write_text(RIBOTISH_HEADER + "\n" + _make_row() + "\n")
        df = load_ribotish_predictions(tsv)
        assert len(df) == 1
        assert "Chromosome" in df.columns
        assert "Strand" in df.columns
        assert "Locus" in df.columns

    def test_locus_forward_strand(self, tmp_path):
        tsv = tmp_path / "test.txt"
        tsv.write_text(RIBOTISH_HEADER + "\n" + _make_row("chr1:100-200:+") + "\n")
        df = load_ribotish_predictions(tsv)
        assert df.iloc[0]["Locus"] == 100

    def test_locus_reverse_strand(self, tmp_path):
        tsv = tmp_path / "test.txt"
        tsv.write_text(RIBOTISH_HEADER + "\n" + _make_row("chr1:944693-959240:-") + "\n")
        df = load_ribotish_predictions(tsv)
        assert df.iloc[0]["Locus"] == 959240

    def test_multiple_rows(self, tmp_path):
        tsv = tmp_path / "test.txt"
        rows = "\n".join(
            [
                RIBOTISH_HEADER,
                _make_row("chr1:100-200:+"),
                _make_row("chr2:300-400:-"),
                _make_row("chrX:500-600:+"),
            ]
        )
        tsv.write_text(rows + "\n")
        df = load_ribotish_predictions(tsv)
        assert len(df) == 3
        assert list(df["Chromosome"]) == ["chr1", "chr2", "chrX"]
        assert list(df["Strand"]) == ["+", "-", "+"]
        assert list(df["Locus"]) == [100, 400, 500]

    def test_empty_file(self, tmp_path):
        tsv = tmp_path / "test.txt"
        tsv.write_text(RIBOTISH_HEADER + "\n")
        df = load_ribotish_predictions(tsv)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# recategorize_tis_type
# ---------------------------------------------------------------------------


class TestRecategorizeTisType:
    """Tests for recategorize_tis_type."""

    @pytest.fixture()
    def tis_types_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "TisType": [
                    "Annotated",
                    "Truncated",
                    "Truncated:Known",
                    "Extended",
                    "Extended:CDSFrameOverlap",
                    "5'UTR",
                    "5'UTR:CDSFrameOverlap",
                    "5'UTR:Known",
                    "Novel",
                    "Novel:CDSFrameOverlap",
                    "Novel:Known",
                    "Internal",
                    "Internal:CDSFrameOverlap",
                    "3'UTR",
                    "3'UTR:CDSFrameOverlap",
                ],
            }
        )

    def test_annotated(self, tis_types_df):
        result = recategorize_tis_type(tis_types_df)
        assert result.loc[0, "RecatTISType"] == "Annotated"

    def test_truncated_variants(self, tis_types_df):
        result = recategorize_tis_type(tis_types_df)
        assert result.loc[1, "RecatTISType"] == "Truncated"
        assert result.loc[2, "RecatTISType"] == "Truncated"

    def test_extended_variants(self, tis_types_df):
        result = recategorize_tis_type(tis_types_df)
        assert result.loc[3, "RecatTISType"] == "Extended"
        assert result.loc[4, "RecatTISType"] == "Extended"

    def test_uorf_variants(self, tis_types_df):
        result = recategorize_tis_type(tis_types_df)
        assert result.loc[5, "RecatTISType"] == "uORF"
        assert result.loc[6, "RecatTISType"] == "uORF"
        assert result.loc[7, "RecatTISType"] == "uORF"

    def test_other(self, tis_types_df):
        result = recategorize_tis_type(tis_types_df)
        # Novel, Novel:CDSFrameOverlap, Novel:Known, Internal, etc.
        for idx in [8, 9, 10, 11, 12, 13, 14]:
            assert result.loc[idx, "RecatTISType"] == "Other"

    def test_custom_columns(self, tis_types_df):
        renamed = tis_types_df.rename(columns={"TisType": "OrigType"})
        result = recategorize_tis_type(
            renamed, original_column="OrigType", output_column="SimpleType"
        )
        assert "SimpleType" in result.columns

    def test_does_not_mutate_input(self, tis_types_df):
        original_cols = set(tis_types_df.columns)
        recategorize_tis_type(tis_types_df)
        assert set(tis_types_df.columns) == original_cols
