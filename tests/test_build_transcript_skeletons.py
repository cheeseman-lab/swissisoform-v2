"""Tests for the transcript-skeleton parquet builder."""

from pathlib import Path

import pandas as pd

from scripts.site import build_transcript_skeletons as bts


def _make_synthetic_gtf(tmp_path: Path) -> Path:
    """Tiny GTF with one + strand transcript (3 exons) and one CDS span."""
    attrs = 'transcript_id "ENST_A.1"; gene_name "GENE_A";'
    rows = [
        ("transcript", 101, 900),
        ("exon", 101, 300),
        ("exon", 401, 600),
        ("exon", 701, 900),
        ("CDS", 201, 600),
        ("CDS", 701, 800),
    ]
    lines = [f"chr1\tHAVANA\t{f}\t{s}\t{e}\t.\t+\t.\t{attrs}" for f, s, e in rows]
    gtf = tmp_path / "synthetic.gtf"
    gtf.write_text("\n".join(lines) + "\n")
    return gtf


def test_skeleton_exons_are_plus_strand_zero_based_half_open(tmp_path: Path) -> None:
    gtf = _make_synthetic_gtf(tmp_path)
    skeletons = bts.build_skeletons(gtf, transcript_ids={"ENST_A.1"})
    assert "ENST_A.1" in skeletons
    sk = skeletons["ENST_A.1"]
    # GTF is 1-based inclusive; parquet stores 0-based half-open.
    assert sk["exons"] == [(100, 300), (400, 600), (700, 900)]
    assert sk["chrom"] == "chr1"
    assert sk["strand"] == "+"


def test_skeleton_cds_bounds_are_min_and_max_across_cds_segments(tmp_path: Path) -> None:
    gtf = _make_synthetic_gtf(tmp_path)
    skeletons = bts.build_skeletons(gtf, transcript_ids={"ENST_A.1"})
    sk = skeletons["ENST_A.1"]
    assert sk["cds_start"] == 200  # 0-based half-open: GTF 201 → 200
    assert sk["cds_end"] == 800  # GTF 800 (inclusive) → 800 (exclusive)


def test_write_parquet_schema(tmp_path: Path) -> None:
    gtf = _make_synthetic_gtf(tmp_path)
    out = tmp_path / "skeletons.parquet"
    bts.write_skeletons_parquet(gtf, transcript_ids={"ENST_A.1"}, out_path=out)
    df = pd.read_parquet(out)
    assert list(df.columns) == [
        "transcript_id",
        "gene_name",
        "chrom",
        "strand",
        "exons",
        "cds_start",
        "cds_end",
        "length_nt",
        "length_aa",
    ]
    assert len(df) == 1
    row = df.iloc[0]
    assert row["transcript_id"] == "ENST_A.1"
    assert row["gene_name"] == "GENE_A"
    assert row["length_nt"] == 600  # (300-100) + (600-400) + (900-700)


def test_unknown_transcript_id_is_silently_skipped(tmp_path: Path) -> None:
    gtf = _make_synthetic_gtf(tmp_path)
    skeletons = bts.build_skeletons(gtf, transcript_ids={"ENST_MISSING.1"})
    assert skeletons == {}


def _make_minus_strand_gtf(tmp_path: Path) -> Path:
    """Tiny GTF with one minus-strand transcript, 2 exons, one CDS."""
    attrs = 'transcript_id "ENST_B.1"; gene_name "GENE_B";'
    rows = [
        ("exon", 1001, 1200),
        ("exon", 1401, 1600),
        ("CDS", 1051, 1200),
        ("CDS", 1401, 1500),
    ]
    lines = [f"chr2\tHAVANA\t{f}\t{s}\t{e}\t.\t-\t.\t{attrs}" for f, s, e in rows]
    gtf = tmp_path / "synthetic_minus.gtf"
    gtf.write_text("\n".join(lines) + "\n")
    return gtf


def test_minus_strand_exons_are_plus_strand_sorted(tmp_path: Path) -> None:
    gtf = _make_minus_strand_gtf(tmp_path)
    sk = bts.build_skeletons(gtf, transcript_ids={"ENST_B.1"})["ENST_B.1"]
    assert sk["strand"] == "-"
    # Even on minus strand, exons are stored in plus-strand-sorted half-open form.
    assert sk["exons"] == [(1000, 1200), (1400, 1600)]


def test_multi_transcript_gtf_only_returns_requested(tmp_path: Path) -> None:
    """Two transcripts present in GTF; only one is requested."""
    gtf = tmp_path / "two.gtf"
    gtf.write_text(
        "\n".join(
            [
                'chr1\tHAVANA\texon\t101\t200\t.\t+\t.\ttranscript_id "WANTED.1"; gene_name "GA";',
                'chr1\tHAVANA\texon\t301\t400\t.\t+\t.\ttranscript_id "WANTED.1"; gene_name "GA";',
                'chr1\tHAVANA\texon\t101\t200\t.\t+\t.\ttranscript_id "DROPPED.1"; gene_name "GB";',
                'chr1\tHAVANA\texon\t501\t600\t.\t+\t.\ttranscript_id "DROPPED.1"; gene_name "GB";',
            ]
        )
        + "\n"
    )
    sk = bts.build_skeletons(gtf, transcript_ids={"WANTED.1"})
    assert set(sk) == {"WANTED.1"}
    assert sk["WANTED.1"]["exons"] == [(100, 200), (300, 400)]


def test_length_aa_uses_spliced_cds(tmp_path: Path) -> None:
    """length_aa is the spliced CDS / 3, not the genomic span / 3.

    Synthetic GTF: exons 101-300, 401-600, 701-900; CDS 201-600 + 701-800.
    Spliced CDS = (300-200) + (600-400) + (800-700) = 400 nt → 133 aa.
    Genomic span = 800 - 200 = 600 → would be 200 aa (the bug).
    """
    gtf = _make_synthetic_gtf(tmp_path)
    out = tmp_path / "skeletons.parquet"
    bts.write_skeletons_parquet(gtf, transcript_ids={"ENST_A.1"}, out_path=out)
    df = pd.read_parquet(out)
    assert df.iloc[0]["length_aa"] == 133
