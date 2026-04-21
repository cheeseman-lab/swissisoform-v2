"""Tests for GTF annotation loader."""

from __future__ import annotations

from swissisoform.io.gtf import load_exon_skeletons, load_transcript_annotations

# Build GTF lines individually to stay under line-length limit
_HEADER = "##description: test gtf\n##provider: GENCODE\n"
_GENE_LINE = (
    "chr1\tHAVANA\tgene\t100\t500\t.\t+\t.\t"
    'gene_id "ENSG00000000001.1"; gene_type "protein_coding";'
)
_TXN1_ATTRS = (
    'gene_id "ENSG00000000001.1"; gene_type "protein_coding"; '
    'transcript_id "ENST00000000001.1"; transcript_type "protein_coding"; '
    'transcript_support_level "1"; tag "MANE_Select";'
)
_TXN1_LINE = f"chr1\tHAVANA\ttranscript\t100\t500\t.\t+\t.\t{_TXN1_ATTRS}"
_EXON_LINE = (
    "chr1\tHAVANA\texon\t100\t200\t.\t+\t.\t"
    'gene_id "ENSG00000000001.1"; transcript_id "ENST00000000001.1";'
)
_TXN2_ATTRS = (
    'gene_id "ENSG00000000001.1"; gene_type "protein_coding"; '
    'transcript_id "ENST00000000002.1"; '
    'transcript_type "nonsense_mediated_decay"; '
    'transcript_support_level "3";'
)
_TXN2_LINE = f"chr1\tHAVANA\ttranscript\t100\t500\t.\t+\t.\t{_TXN2_ATTRS}"

SAMPLE_GTF = "\n".join(
    [
        _HEADER.rstrip(),
        _GENE_LINE,
        _TXN1_LINE,
        _EXON_LINE,
        _TXN2_LINE,
        "",
    ]
)


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
            "chromosome",
            "source",
            "feature_type",
            "start",
            "end",
            "strand",
            "gene_id",
            "gene_type",
            "transcript_id",
            "transcript_type",
            "transcript_support_level",
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


# ---------------------------------------------------------------------------
# load_exon_skeletons
# ---------------------------------------------------------------------------


def _line(feat: str, start: int, end: int, strand: str, tid: str) -> str:
    attrs = (
        f'gene_id "ENSG00000000001.1"; transcript_id "{tid}"; '
        f'gene_type "protein_coding"; transcript_type "protein_coding";'
    )
    return f"chr1\tHAVANA\t{feat}\t{start}\t{end}\t.\t{strand}\t.\t{attrs}"


_SKELETON_GTF_PLUS = "\n".join(
    [
        _line("exon", 101, 200, "+", "ENST_PLUS.1"),
        _line("exon", 301, 400, "+", "ENST_PLUS.1"),
        # Start codon at 0-based 120 (GTF 1-based inclusive 121..123)
        _line("start_codon", 121, 123, "+", "ENST_PLUS.1"),
        _line("CDS", 121, 200, "+", "ENST_PLUS.1"),
        _line("CDS", 301, 350, "+", "ENST_PLUS.1"),
        _line("stop_codon", 351, 353, "+", "ENST_PLUS.1"),
        "",
    ]
)


_SKELETON_GTF_MINUS = "\n".join(
    [
        _line("exon", 101, 200, "-", "ENST_MINUS.1"),
        _line("exon", 301, 400, "-", "ENST_MINUS.1"),
        # Minus-strand start_codon: A of ATG on mRNA is the higher plus-strand
        # coord of the codon feature.  GTF 1-based inclusive 398..400 → 0-based
        # half-open [397, 400), so cds_start (plus-strand exclusive upper) = 400.
        _line("start_codon", 398, 400, "-", "ENST_MINUS.1"),
        _line("CDS", 301, 397, "-", "ENST_MINUS.1"),
        _line("CDS", 101, 200, "-", "ENST_MINUS.1"),
        "",
    ]
)


class TestLoadExonSkeletons:
    def test_plus_strand(self, tmp_path):
        gtf = tmp_path / "sk.gtf"
        gtf.write_text(_SKELETON_GTF_PLUS)
        sk = load_exon_skeletons(str(gtf))
        assert "ENST_PLUS.1" in sk
        coords = sk["ENST_PLUS.1"]
        assert coords.strand == "+"
        assert coords.chrom == "chr1"
        # 1-based inclusive 101..200 → 0-based half-open [100, 200)
        assert coords.exons == [(100, 200), (300, 400)]
        assert coords.cds_start == 120
        assert coords.cds_end == 353

    def test_minus_strand(self, tmp_path):
        gtf = tmp_path / "sk.gtf"
        gtf.write_text(_SKELETON_GTF_MINUS)
        sk = load_exon_skeletons(str(gtf))
        coords = sk["ENST_MINUS.1"]
        assert coords.strand == "-"
        # cds_start on minus = higher plus-strand coord of start_codon (exclusive end)
        assert coords.cds_start == 400

    def test_empty_gtf(self, tmp_path):
        gtf = tmp_path / "empty.gtf"
        gtf.write_text("##empty\n")
        assert load_exon_skeletons(str(gtf)) == {}
