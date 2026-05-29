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


from swissisoform_site.plots.protein import build_protein_figure  # noqa: E402


class _ProtIso:
    orf_type = "extended"
    canonical_len = 185
    isoform_len = 239
    diff_start = 0
    diff_end = 54
    diff_space = "isoform"
    differential_sequence = "M" * 54
    domains = [{"name": "PF00001", "start": 60, "end": 120}]
    motifs = [{"name": "NLS", "start": 70, "end": 78}]
    # mixed significance, all in-unique
    variants_in_unique = [
        {
            "isoform_protein_pos": 10,
            "clinical_significance": "Pathogenic",
            "hgvsp": "p.M1?",
            "source": "ClinVar",
        },
        {
            "isoform_protein_pos": 25,
            "clinical_significance": "Uncertain_significance",
            "hgvsp": "p.A5T",
            "source": "gnomAD",
        },
    ]


def test_protein_has_legend():
    fig = build_protein_figure(_ProtIso(), overlays={})
    assert any(t.get("showlegend") for t in fig["data"]), "expected a legend entry"


def test_protein_draws_two_length_bars():
    fig = build_protein_figure(_ProtIso(), overlays={})
    bar_lines = [t for t in fig["data"] if t.get("name") in ("Canonical", "Isoform")]
    assert len(bar_lines) >= 2, "expected separate canonical + isoform length tracks"


def test_protein_shows_non_pathogenic_variants():
    fig = build_protein_figure(_ProtIso(), overlays={})
    # 2 variants supplied (1 pathogenic, 1 VUS) — both must appear as marker x-positions
    xs = [
        x
        for t in fig["data"]
        if t.get("mode", "").startswith("markers")
        for x in (t.get("x") or [])
    ]
    assert 10 in xs and 25 in xs, "both pathogenic and VUS variants must render"
