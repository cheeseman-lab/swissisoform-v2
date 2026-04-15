"""Tests for GTF annotation loader."""

from __future__ import annotations

from swissisoform.io.gtf import load_transcript_annotations

# Build GTF lines individually to stay under line-length limit
_HEADER = "##description: test gtf\n##provider: GENCODE\n"
_GENE_LINE = (
    'chr1\tHAVANA\tgene\t100\t500\t.\t+\t.\t'
    'gene_id "ENSG00000000001.1"; gene_type "protein_coding";'
)
_TXN1_ATTRS = (
    'gene_id "ENSG00000000001.1"; gene_type "protein_coding"; '
    'transcript_id "ENST00000000001.1"; transcript_type "protein_coding"; '
    'transcript_support_level "1"; tag "MANE_Select";'
)
_TXN1_LINE = f"chr1\tHAVANA\ttranscript\t100\t500\t.\t+\t.\t{_TXN1_ATTRS}"
_EXON_LINE = (
    'chr1\tHAVANA\texon\t100\t200\t.\t+\t.\t'
    'gene_id "ENSG00000000001.1"; transcript_id "ENST00000000001.1";'
)
_TXN2_ATTRS = (
    'gene_id "ENSG00000000001.1"; gene_type "protein_coding"; '
    'transcript_id "ENST00000000002.1"; '
    'transcript_type "nonsense_mediated_decay"; '
    'transcript_support_level "3";'
)
_TXN2_LINE = f"chr1\tHAVANA\ttranscript\t100\t500\t.\t+\t.\t{_TXN2_ATTRS}"

SAMPLE_GTF = "\n".join([
    _HEADER.rstrip(),
    _GENE_LINE,
    _TXN1_LINE,
    _EXON_LINE,
    _TXN2_LINE,
    "",
])


class TestLoadTranscriptAnnotations:
    """Tests for load_transcript_annotations."""

    def test_basic_load(self, tmp_path):
        gtf = tmp_path / "test.gtf"
        gtf.write_text(SAMPLE_GTF)
        df = load_transcript_annotations(str(gtf))
        # Should only pick up the 2 transcript lines
        assert len(df) == 2

    def test_columns_present(self, tmp_path):
        gtf = tmp_path / "test.gtf"
        gtf.write_text(SAMPLE_GTF)
        df = load_transcript_annotations(str(gtf))
        expected = {
            "chromosome", "source", "feature_type",
            "start", "end", "strand",
            "gene_id", "gene_type", "transcript_id",
            "transcript_type", "transcript_support_level",
            "MANE_Select",
        }
        assert expected.issubset(set(df.columns))

    def test_mane_select_boolean(self, tmp_path):
        gtf = tmp_path / "test.gtf"
        gtf.write_text(SAMPLE_GTF)
        df = load_transcript_annotations(str(gtf))
        assert bool(df.iloc[0]["MANE_Select"]) is True
        assert bool(df.iloc[1]["MANE_Select"]) is False

    def test_gene_id_parsed(self, tmp_path):
        gtf = tmp_path / "test.gtf"
        gtf.write_text(SAMPLE_GTF)
        df = load_transcript_annotations(str(gtf))
        assert df.iloc[0]["gene_id"] == "ENSG00000000001.1"

    def test_transcript_support_level(self, tmp_path):
        gtf = tmp_path / "test.gtf"
        gtf.write_text(SAMPLE_GTF)
        df = load_transcript_annotations(str(gtf))
        assert df.iloc[0]["transcript_support_level"] == "1"
        assert df.iloc[1]["transcript_support_level"] == "3"

    def test_feature_type_filter(self, tmp_path):
        gtf = tmp_path / "test.gtf"
        gtf.write_text(SAMPLE_GTF)
        df = load_transcript_annotations(str(gtf), feature_type="gene")
        assert len(df) == 1
        assert df.iloc[0]["feature_type"] == "gene"

    def test_empty_gtf(self, tmp_path):
        gtf = tmp_path / "empty.gtf"
        gtf.write_text("##description: empty\n")
        df = load_transcript_annotations(str(gtf))
        assert len(df) == 0

    def test_start_end_are_int(self, tmp_path):
        gtf = tmp_path / "test.gtf"
        gtf.write_text(SAMPLE_GTF)
        df = load_transcript_annotations(str(gtf))
        assert df["start"].dtype in ("int64", "int32")
        assert df["end"].dtype in ("int64", "int32")
