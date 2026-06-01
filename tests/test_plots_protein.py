"""Tests for the protein-track figure builder.

The V2 protein view draws two left-aligned length bars (``Canonical`` y=1.0,
``Isoform`` y=0.5), a shaded ``Differential region``, clinical-variant
lollipops coloured by significance (head trace named by the significance
class, stem trace sharing its ``legendgroup``), domain boxes (``Domain
(InterPro)``), and motif spans (``Motif``), with a horizontal legend.
"""

from __future__ import annotations

from types import SimpleNamespace

from swissisoform_site.plots import protein as pplot

# Lollipop heads/stems live at this y (see protein._ISO_Y + 0.55).
_LOLLIPOP_Y = 1.05


def _iso(orf_type="truncated", diff_space="canonical", iso_len=405, can_len=434):
    return SimpleNamespace(
        tis_id="chr3:3129127:+:ATG:ENST00000434583.5",
        orf_type=orf_type,
        diff_space=diff_space,
        diff_start=0,
        diff_end=29,  # len(differential_sequence)
        differential_sequence="MLRCLYHWHRPVLNRRWSRLCLPKQYLFT",
        canonical_len=can_len,
        isoform_len=iso_len,
        variants_in_unique=[
            {
                "variant_id": "ClinVar:1",
                "isoform_protein_pos": 10,
                "protein_pos": 10,
                "hgvsp": "p.Leu13fs",
                "clinical_significance": "Pathogenic",
                "source": "ClinVar",
                "consequence": "frameshift_variant",
            },
            {
                "variant_id": "ClinVar:2",
                "isoform_protein_pos": 24,
                "protein_pos": 24,
                "hgvsp": "p.Gln25Ter",
                "clinical_significance": "Pathogenic",
                "source": "ClinVar",
                "consequence": "stop_gained",
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


def test_draws_two_named_length_bars():
    """Canonical + isoform are separate, named length tracks (the length delta)."""
    fig = pplot.build_protein_figure(_iso(), overlays={})
    names = {t.get("name") for t in fig["data"]}
    assert "Canonical" in names
    assert "Isoform" in names


def test_has_a_legend():
    fig = pplot.build_protein_figure(_iso(), overlays={})
    assert fig["layout"]["showlegend"] is True
    assert any(t.get("showlegend") for t in fig["data"])


def test_diff_region_span_uses_canonical_length_for_truncation():
    """For a truncation the differential region spans 1..len(diff_seq)."""
    fig = pplot.build_protein_figure(_iso(), overlays={})
    span_traces = [t for t in fig["data"] if t.get("name") == "Differential region"]
    assert len(span_traces) == 1
    assert max(span_traces[0]["x"]) == 29  # len("MLRCLYHWHRPVLNRRWSRLCLPKQYLFT")


def test_extension_diff_space_is_isoform():
    """For extensions the differential region starts at residue 1 on the isoform."""
    fig = pplot.build_protein_figure(_iso(orf_type="extended", diff_space="isoform"), overlays={})
    span = [t for t in fig["data"] if t.get("name") == "Differential region"][0]
    assert min(span["x"]) == 1


def _variant_marks(fig):
    """Marker traces sitting in the variant tracks above the bars (y > 1.0)."""
    return [
        t
        for t in fig["data"]
        if t.get("mode") == "markers" and (t.get("y") or []) and all(y > 1.0 for y in t["y"])
    ]


def test_variants_render_above_bars_at_protein_positions():
    fig = pplot.build_protein_figure(_iso(), overlays={"variants": True})
    xs = sorted(x for t in _variant_marks(fig) for x in t["x"])
    assert xs == [10, 24]


def test_variants_grouped_into_one_track_per_consequence():
    """Two consequence types (frameshift, stop_gained) → two variant tracks."""
    fig = pplot.build_protein_figure(_iso(), overlays={"variants": True})
    marks = _variant_marks(fig)
    assert len(marks) == 2  # one track per consequence type
    # each track sits at its own y row
    ys = {t["y"][0] for t in marks}
    assert len(ys) == 2


def test_pathogenic_variants_are_red():
    fig = pplot.build_protein_figure(_iso(), overlays={"variants": True})
    colors = {c for t in _variant_marks(fig) for c in t["marker"]["color"]}
    assert "#d62728" in colors


def test_non_pathogenic_variants_also_render():
    """VUS / benign variants render too (not only pathogenic)."""
    iso = _iso()
    iso.variants_in_unique = iso.variants_in_unique + [
        {
            "variant_id": "gnomAD:1",
            "isoform_protein_pos": 15,
            "clinical_significance": "Uncertain_significance",
            "source": "gnomAD",
            "consequence": "missense_variant",
        }
    ]
    fig = pplot.build_protein_figure(iso, overlays={"variants": True})
    xs = sorted(x for t in _variant_marks(fig) for x in t["x"])
    assert 15 in xs
    assert len(_variant_marks(fig)) == 3  # frameshift, stop_gained, missense


def test_variants_overlay_off_hides_variant_tracks():
    fig = pplot.build_protein_figure(_iso(), overlays={"variants": False})
    assert _variant_marks(fig) == []


def test_domains_drawn_as_rectangles_in_separate_track():
    fig = pplot.build_protein_figure(_iso(), overlays={"domains": True})
    rect_traces = [t for t in fig["data"] if t.get("name") == "Domain (InterPro)"]
    assert len(rect_traces) == 1
    # Truncation fixture (diff_end=29): isoform-coord domain shifts +29 onto the
    # canonical display axis so it aligns with the shifted isoform bar.
    assert min(rect_traces[0]["x"]) == 129
    assert max(rect_traces[0]["x"]) == 409


def test_motifs_drawn_as_spans():
    fig = pplot.build_protein_figure(_iso(), overlays={"motifs": True})
    motif_traces = [t for t in fig["data"] if t.get("name") == "Motif"]
    assert len(motif_traces) == 1
    # Span covers start..end (+29 offset onto the canonical axis for truncations).
    assert motif_traces[0]["x"] == [64, 70]


def test_motifs_off_when_overlay_disabled():
    fig = pplot.build_protein_figure(_iso(), overlays={"motifs": False})
    assert [t for t in fig["data"] if t.get("name") == "Motif"] == []


def test_no_protein_length_returns_empty_figure_with_caption():
    iso = _iso(iso_len=0, can_len=0)
    fig = pplot.build_protein_figure(iso, overlays={})
    assert fig["data"] == []
    annotations = fig.get("layout", {}).get("annotations", [])
    assert any("length" in a.get("text", "").lower() for a in annotations)


def test_protein_figure_uses_system_font_stack():
    fig = pplot.build_protein_figure(_iso(), overlays={})
    font_family = fig["layout"]["font"]["family"]
    assert "sans-serif" in font_family or "Helvetica" in font_family or "Segoe" in font_family
