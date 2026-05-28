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
