"""Tests for the per-cell-line TIS-usage figure builder."""

from __future__ import annotations

from types import SimpleNamespace

from swissisoform_site.plots import transcript as tplot


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
    fig = tplot.build_transcript_figure(_iso_two_tis(), None, overlays={})
    assert "data" in fig and "layout" in fig
    assert isinstance(fig["data"], list)


def test_one_grouped_bar_trace_per_tis():
    fig = tplot.build_transcript_figure(_iso_two_tis(), None, overlays={})
    bars = [t for t in fig["data"] if t.get("type") == "bar"]
    assert len(bars) == 2  # one bar group per TIS
    assert fig["layout"]["barmode"] == "group"


def test_cell_lines_are_the_categorical_x_axis():
    fig = tplot.build_transcript_figure(_iso_two_tis(), None, overlays={})
    bars = [t for t in fig["data"] if t.get("type") == "bar"]
    assert fig["layout"]["xaxis"]["type"] == "category"
    # x values are cell-line names, in canonical display order.
    assert bars[0]["x"] == ["HeLa", "K562"]


def test_focal_tis_uses_distinguishing_color():
    fig = tplot.build_transcript_figure(_iso_two_tis(), None, overlays={})
    focal = [t for t in fig["data"] if "(focal)" in t.get("name", "")]
    assert len(focal) == 1
    assert focal[0]["marker"]["color"] == tplot._FOCAL_COLOR


def test_tis_label_includes_orf_type_and_codon():
    fig = tplot.build_transcript_figure(_iso_two_tis(), None, overlays={})
    names = [t["name"] for t in fig["data"] if t.get("type") == "bar"]
    assert any("Truncated" in n and "ATG" in n for n in names)


def test_no_cell_line_data_returns_empty_figure_with_caption():
    iso = SimpleNamespace(
        tis_id="chr1:250:+:ATG:ENST_A.1",
        focal_tis_id="chr1:250:+:ATG:ENST_A.1",
        all_tis_on_transcript=[
            {"tis_id": "chr1:250:+:ATG:ENST_A.1", "genomic_pos": 250, "orf_type": "Truncated"},
        ],
        cell_line_bars={},
    )
    fig = tplot.build_transcript_figure(iso, None, overlays={})
    assert fig["data"] == []
    assert "initiation" in fig["layout"]["annotations"][0]["text"].lower()


def test_y_axis_is_log2_initiation_efficiency():
    fig = tplot.build_transcript_figure(_iso_two_tis(), None, overlays={})
    assert "initiation efficiency" in fig["layout"]["yaxis"]["title"]["text"].lower()


def test_figure_uses_system_font_stack():
    fig = tplot.build_transcript_figure(_iso_two_tis(), None, overlays={})
    family = fig["layout"]["font"]["family"]
    assert "sans-serif" in family or "Helvetica" in family or "Segoe" in family
