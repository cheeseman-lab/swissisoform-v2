"""Tests for the protein-lollipop figure builder."""

from __future__ import annotations

from types import SimpleNamespace

from swissisoform_site.plots import protein as pplot


def _iso(orf_type="truncated", diff_space="canonical", iso_len=405, can_len=434):
    return SimpleNamespace(
        tis_id="chr3:3129127:+:ATG:ENST00000434583.5",
        orf_type=orf_type,
        diff_space=diff_space,
        differential_sequence="MLRCLYHWHRPVLNRRWSRLCLPKQYLFT",
        isoform_length_aa=iso_len,
        canonical_length_aa=can_len,
        variants_in_unique=[
            {
                "variant_id": "ClinVar:1",
                "protein_pos": 10,
                "hgvsp": "p.Leu13fs",
                "clinical_significance": "Pathogenic",
            },
            {
                "variant_id": "ClinVar:2",
                "protein_pos": 24,
                "hgvsp": "p.Gln25Ter",
                "clinical_significance": "Pathogenic",
            },
        ],
        domains=[
            {"name": "PCMP-domain", "start": 100, "end": 380},
        ],
        motifs=[
            {"name": "NLS", "start": 35, "end": 41},
        ],
    )


def test_figure_is_plotly_dict_with_traces():
    fig = pplot.build_protein_figure(_iso(), overlays={})
    assert isinstance(fig["data"], list)
    assert fig["data"]  # non-empty


def test_diff_region_span_uses_canonical_length_for_truncation():
    """For diff_space=canonical, the shaded diff spans 1..len(diff_seq) on the canonical axis."""
    fig = pplot.build_protein_figure(_iso(), overlays={})
    span_traces = [t for t in fig["data"] if t.get("name") == "diff region"]
    assert len(span_traces) == 1
    assert max(span_traces[0]["x"]) == 29  # len("MLRCLYHWHRPVLNRRWSRLCLPKQYLFT")


def test_variants_drawn_as_lollipops_at_protein_positions():
    fig = pplot.build_protein_figure(_iso(), overlays={"variants": True})
    variant_traces = [t for t in fig["data"] if t.get("name", "").startswith("variant")]
    xs = sorted(x for t in variant_traces for x in t["x"])
    assert xs == [10, 24]


def test_domains_drawn_as_rectangles_in_separate_track():
    fig = pplot.build_protein_figure(_iso(), overlays={"domains": True})
    rect_traces = [t for t in fig["data"] if t.get("name", "").startswith("domain:")]
    assert len(rect_traces) == 1
    assert min(rect_traces[0]["x"]) == 100
    assert max(rect_traces[0]["x"]) == 380


def test_motifs_off_when_overlay_disabled():
    fig = pplot.build_protein_figure(_iso(), overlays={"motifs": False})
    motif_traces = [t for t in fig["data"] if t.get("name", "").startswith("motif:")]
    assert motif_traces == []


def test_extension_diff_space_is_isoform():
    """For extensions, diff space is isoform — shaded span starts at 1 on isoform axis."""
    fig = pplot.build_protein_figure(_iso(orf_type="extended", diff_space="isoform"), overlays={})
    span = [t for t in fig["data"] if t.get("name") == "diff region"][0]
    assert min(span["x"]) == 1
