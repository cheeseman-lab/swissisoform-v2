"""Tests for the website Plotly figure builders."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "website" / "src"))

from swissisoform_site.plots.transcript import build_transcript_figure  # noqa: E402


class _Skel:
    transcript_id = "ENSTTEST"
    chrom = "chr1"
    strand = "+"
    cds_start = 1000
    cds_end = 4000
    exons = [(0, 500), (1000, 2000), (3000, 4000), (4000, 10000)]


class _Iso:
    focal_tis_id = "chr1:1000:+:ATG:ENSTTEST"
    tis_id = "chr1:1000:+:ATG:ENSTTEST"
    all_tis_on_transcript = [
        {"tis_id": "chr1:1000:+:ATG:ENSTTEST", "genomic_pos": 1000, "orf_type": "truncated"}
    ]
    cell_line_bars = {"chr1:1000:+:ATG:ENSTTEST": {"HeLa": 1.5}}


def test_transcript_bars_have_explicit_width():
    fig = build_transcript_figure(_Iso(), _Skel(), overlays={})
    bars = [t for t in fig["data"] if t.get("type") == "bar"]
    assert bars, "expected at least one bar trace"
    for b in bars:
        assert b.get("width"), "bar trace must set an explicit width (else sub-pixel)"
        assert b["width"][0] if isinstance(b["width"], list) else b["width"]


def test_transcript_zoom_spans_at_least_2kb():
    fig = build_transcript_figure(_Iso(), _Skel(), overlays={})
    xr = fig["layout"]["xaxis"]["range"]
    assert xr[1] - xr[0] >= 2000, f"zoom window too tight: {xr}"
