"""Tests for the transcript-graph figure builder."""

from __future__ import annotations

from types import SimpleNamespace

from swissisoform_site.plots import transcript as tplot


def _synthetic_skeleton():
    return SimpleNamespace(
        transcript_id="ENST_A.1",
        chrom="chr1",
        strand="+",
        exons=[(100, 300), (400, 600), (700, 900)],
        cds_start=200,
        cds_end=800,
    )


def _synthetic_isoform_with_two_tis():
    return SimpleNamespace(
        tis_id="chr1:250:+:ATG:ENST_A.1",
        transcript_id="ENST_A.1",
        all_tis_on_transcript=[
            {"tis_id": "chr1:200:+:ATG:ENST_A.1", "genomic_pos": 200, "orf_type": "Annotated"},
            {"tis_id": "chr1:250:+:ATG:ENST_A.1", "genomic_pos": 250, "orf_type": "Truncated"},
        ],
        focal_tis_id="chr1:250:+:ATG:ENST_A.1",
        cell_line_bars={
            "chr1:200:+:ATG:ENST_A.1": {"HeLa": 0.3, "K562": -0.1},
            "chr1:250:+:ATG:ENST_A.1": {"HeLa": -0.8, "K562": 0.0},
        },
    )


def test_figure_is_a_plotly_dict():
    fig = tplot.build_transcript_figure(
        _synthetic_isoform_with_two_tis(), _synthetic_skeleton(), overlays={"variants": False}
    )
    assert "data" in fig
    assert "layout" in fig
    assert isinstance(fig["data"], list)


def test_figure_has_one_marker_trace_per_tis_plus_bar_traces():
    fig = tplot.build_transcript_figure(
        _synthetic_isoform_with_two_tis(), _synthetic_skeleton(), overlays={}
    )
    trace_types = [t.get("type") for t in fig["data"]]
    assert trace_types.count("scatter") >= 2  # transcript shape + TIS markers
    assert trace_types.count("bar") >= 1


def test_focal_tis_renders_with_distinguishing_color():
    fig = tplot.build_transcript_figure(
        _synthetic_isoform_with_two_tis(), _synthetic_skeleton(), overlays={}
    )
    tis_marker_traces = [t for t in fig["data"] if t.get("name", "").startswith("TIS:")]
    focal = [t for t in tis_marker_traces if "focal" in t.get("name", "").lower()]
    assert len(focal) == 1


def test_cell_line_bars_match_sample_count():
    iso = _synthetic_isoform_with_two_tis()
    fig = tplot.build_transcript_figure(iso, _synthetic_skeleton(), overlays={})
    bar_traces = [t for t in fig["data"] if t.get("type") == "bar"]
    sample_names = {t.get("name") for t in bar_traces}
    assert {"HeLa", "K562"}.issubset(sample_names)


def test_missing_skeleton_returns_empty_figure_with_caption():
    iso = _synthetic_isoform_with_two_tis()
    fig = tplot.build_transcript_figure(iso, None, overlays={})
    assert fig["data"] == []
    assert "skeleton" in fig["layout"]["annotations"][0]["text"].lower()


def test_xaxis_window_includes_focal_tis_and_canonical_start():
    """The visible window must frame the alternative TIS *and* the canonical
    start codon together, clamped within the skeleton span — so the reader sees
    where the TIS sits relative to the canonical ORF (the point of the rework).
    """
    iso = _synthetic_isoform_with_two_tis()
    skel = _synthetic_skeleton()  # cds_start=200 (canonical start, + strand)
    fig = tplot.build_transcript_figure(iso, skel, overlays={})
    xrange = fig["layout"]["xaxis"]["range"]
    # Clamped within the skeleton span [100, 900].
    assert xrange[0] >= 100
    assert xrange[1] <= 900
    # Contains both TIS positions (200, 250).
    assert xrange[0] <= 200
    assert xrange[1] >= 250
    # Contains the canonical start codon (cds_start = 200).
    assert xrange[0] <= 200 <= xrange[1]


def test_cell_line_bars_have_explicit_width():
    """Bars must set an explicit width — Plotly's 1-nt default is sub-pixel on a
    genomic axis and renders the expression panel blank.
    """
    iso = _synthetic_isoform_with_two_tis()
    fig = tplot.build_transcript_figure(iso, _synthetic_skeleton(), overlays={})
    bars = [t for t in fig["data"] if t.get("type") == "bar"]
    assert bars
    for b in bars:
        w = b.get("width")
        assert w, "bar trace must set an explicit width"
        # width is a per-point list; every entry must be a positive number.
        assert all(x and x > 0 for x in (w if isinstance(w, list) else [w]))


def test_figure_uses_system_font_stack():
    iso = _synthetic_isoform_with_two_tis()
    fig = tplot.build_transcript_figure(iso, _synthetic_skeleton(), overlays={})
    font_family = fig["layout"]["font"]["family"]
    assert "sans-serif" in font_family or "Helvetica" in font_family or "Segoe" in font_family


def test_no_cell_line_bars_renders_single_panel_with_markers():
    """If cell_line_bars is empty, the figure should still render the transcript
    track + TIS markers cleanly (no empty bar panel).
    """
    iso = SimpleNamespace(
        tis_id="chr1:250:+:ATG:ENST_A.1",
        transcript_id="ENST_A.1",
        all_tis_on_transcript=[
            {"tis_id": "chr1:250:+:ATG:ENST_A.1", "genomic_pos": 250, "orf_type": "Truncated"},
        ],
        focal_tis_id="chr1:250:+:ATG:ENST_A.1",
        cell_line_bars={},
    )
    fig = tplot.build_transcript_figure(iso, _synthetic_skeleton(), overlays={})
    # No bar traces — only transcript / exon / TIS markers.
    assert not [t for t in fig["data"] if t.get("type") == "bar"]
    # Still has at least the transcript line + a TIS marker.
    tis_markers = [t for t in fig["data"] if t.get("name", "").startswith("TIS:")]
    assert len(tis_markers) == 1
