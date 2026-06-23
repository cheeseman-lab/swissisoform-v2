"""Tests for the IGV-style per-cell-line initiation figure builder."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("swissisoform_site")  # optional website package; skip if not installed

from swissisoform_site.plots import transcript as tplot


def _skeleton():
    return SimpleNamespace(chrom="chr1", strand="+", cds_start=200, cds_end=4000)


def _iso_two_tis():
    return SimpleNamespace(
        tis_id="chr1:250:+:ATG:ENST_A.1",
        focal_tis_id="chr1:250:+:ATG:ENST_A.1",
        all_tis_on_transcript=[
            {"tis_id": "chr1:200:+:ATG:ENST_A.1", "genomic_pos": 200, "orf_type": "Annotated"},
            {"tis_id": "chr1:250:+:ATG:ENST_A.1", "genomic_pos": 250, "orf_type": "Truncated"},
        ],
        cell_line_bars={
            "chr1:200:+:ATG:ENST_A.1": {"HeLa": 0.3, "K562": -0.1},
            "chr1:250:+:ATG:ENST_A.1": {"HeLa": -0.8, "K562": 0.0},
        },
    )


def test_figure_is_a_plotly_dict():
    fig = tplot.build_transcript_figure(_iso_two_tis(), _skeleton(), overlays={})
    assert "data" in fig and "layout" in fig
    assert isinstance(fig["data"], list)


def test_one_marker_trace_per_tis():
    fig = tplot.build_transcript_figure(_iso_two_tis(), _skeleton(), overlays={})
    markers = [t for t in fig["data"] if t.get("mode") == "markers"]
    assert len(markers) == 2  # one trace per TIS


def test_cell_lines_are_the_y_lanes():
    fig = tplot.build_transcript_figure(_iso_two_tis(), _skeleton(), overlays={})
    ticktext = fig["layout"]["yaxis"]["ticktext"]
    assert "HeLa" in ticktext and "K562" in ticktext


def test_x_axis_is_genomic_position():
    fig = tplot.build_transcript_figure(_iso_two_tis(), _skeleton(), overlays={})
    xr = fig["layout"]["xaxis"]["range"]
    # ROI spans the TIS cluster (200, 250) with padding.
    assert xr[0] < 200 and xr[1] > 250


def test_focal_tis_uses_distinguishing_color():
    fig = tplot.build_transcript_figure(_iso_two_tis(), _skeleton(), overlays={})
    focal = [t for t in fig["data"] if "(focal)" in t.get("name", "")]
    assert len(focal) == 1
    assert focal[0]["marker"]["color"] == tplot._FOCAL_COLOR


def test_marker_size_scales_with_initiation_efficiency():
    fig = tplot.build_transcript_figure(_iso_two_tis(), _skeleton(), overlays={})
    sizes = [s for t in fig["data"] if t.get("mode") == "markers" for s in t["marker"]["size"]]
    assert sizes and max(sizes) > min(sizes)  # not all the same


def test_canonical_start_and_diff_region_drawn():
    fig = tplot.build_transcript_figure(_iso_two_tis(), _skeleton(), overlays={})
    shapes = fig["layout"]["shapes"]
    # a dotted vertical line (canonical start) and a filled rect (diff region)
    assert any(s.get("type") == "line" for s in shapes)
    assert any(s.get("type") == "rect" for s in shapes)
    texts = [a["text"] for a in fig["layout"]["annotations"]]
    assert "canonical start" in texts and "differential region" in texts


def test_no_cell_line_data_returns_empty_figure_with_caption():
    iso = SimpleNamespace(
        tis_id="chr1:250:+:ATG:ENST_A.1",
        focal_tis_id="chr1:250:+:ATG:ENST_A.1",
        all_tis_on_transcript=[
            {"tis_id": "chr1:250:+:ATG:ENST_A.1", "genomic_pos": 250, "orf_type": "Truncated"},
        ],
        cell_line_bars={},
    )
    fig = tplot.build_transcript_figure(iso, _skeleton(), overlays={})
    assert fig["data"] == []
    assert "initiation" in fig["layout"]["annotations"][0]["text"].lower()


def test_figure_uses_system_font_stack():
    fig = tplot.build_transcript_figure(_iso_two_tis(), _skeleton(), overlays={})
    family = fig["layout"]["font"]["family"]
    assert "sans-serif" in family or "Helvetica" in family or "Segoe" in family
