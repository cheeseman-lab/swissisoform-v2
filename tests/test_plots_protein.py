"""Tests for the protein-track figure builder.

The V2 protein view draws two left-aligned length bars (``Canonical`` y=1.0,
``Isoform`` y=0.5), a shaded ``Differential region``, clinical-variant
lollipops coloured by significance (head trace named by the significance
class, stem trace sharing its ``legendgroup``), domain boxes (``Domain
(InterPro)``), and motif spans (``Motif``), with a horizontal legend.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("swissisoform_site")  # optional website package; skip if not installed

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
        variants=[
            {
                "variant_id": "ClinVar:1",
                "isoform_protein_pos": 10,
                "protein_pos": 10,
                "hgvsp": "p.Leu13fs",
                "clinical_significance": "Pathogenic",
                "source": "ClinVar",
                "consequence": "frameshift_variant",
                "in_unique": True,
            },
            {
                "variant_id": "ClinVar:2",
                "isoform_protein_pos": 24,
                "protein_pos": 24,
                "hgvsp": "p.Gln25Ter",
                "clinical_significance": "Pathogenic",
                "source": "ClinVar",
                "consequence": "stop_gained",
                "in_unique": True,
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
    iso.variants = iso.variants + [
        {
            "variant_id": "gnomAD:1",
            "isoform_protein_pos": 15,
            "clinical_significance": "Uncertain_significance",
            "source": "gnomAD",
            "consequence": "missense_variant",
            "in_unique": True,
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
    xs = motif_traces[0]["x"]
    # Span covers start..end (+29 offset onto the canonical axis for truncations).
    # The bar is densified into collinear points by ``_bar_samples`` so Plotly fires
    # hover across its interior (it only fires at vertices), so assert the extent
    # rather than a literal 2-point segment.
    assert (min(xs), max(xs)) == (64, 70)
    assert xs == sorted(xs)
    # Densified points stay inside the span, and hover text covers every vertex.
    assert len(motif_traces[0]["hovertext"]) == len(xs)
    assert len(motif_traces[0]["y"]) == len(xs)


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


# --------------------------------------------------------------------------- #
# Combined gene view — build_gene_protein_figure
# --------------------------------------------------------------------------- #


def _gene_view():
    """One canonical + one extension + one truncation on the canonical frame."""
    return SimpleNamespace(
        canonical_len=185,
        bars=[
            {"label": "extended · CTG", "x0": -40, "x1": 185, "orf_type": "extended",
             "is_trunc": False, "diff_x0": -40, "diff_x1": 0, "diff_on_canonical": False,
             "slug": "chr1-100-+-CTG-ENST1"},
            {"label": "truncated · AAG", "x0": 44, "x1": 186, "orf_type": "truncated",
             "is_trunc": True, "diff_x0": 1, "diff_x1": 43, "diff_on_canonical": True,
             "slug": "chr1-200-+-AAG-ENST1"},
        ],
        variants=[
            {"variant_id": "ClinVar:1", "pos": 60, "consequence": "missense_variant",
             "significance": "Pathogenic", "hgvsp": "p.X", "source": "ClinVar", "in_unique": False},
        ],
        domains=[{"name": "Chromo", "interpro_id": "IPR000953", "x0": 20, "x1": 80,
                  "isoforms": ["extended · CTG", "truncated · AAG"]}],
        domain_segments=[{"x0": 20, "x1": 80, "depth": 1}],
        disorder=[{"x0": 90, "x1": 110}],
        coiled_coil=[{"x0": 120, "x1": 140}],
        motifs=[{"name": "NLS", "x0": 5, "x1": 12}],
        cell_lines=[{"sample": "HeLa",
                     "marks": [{"residue": -40, "log2_ie": 0.5, "label": "extended · CTG"}]}],
        x_left=-44.0,
    )


def test_gene_protein_figure_one_canonical_plus_isoform_bars():
    fig = pplot.build_gene_protein_figure(_gene_view())
    assert isinstance(fig, dict) and fig["data"]
    names = [t.get("name") for t in fig["data"]]
    # Exactly one canonical bar + one bar per isoform.
    assert names.count("Canonical") == 1
    assert "extended · CTG" in names
    assert "truncated · AAG" in names
    # Deduped domain band + a variant trace present.
    assert "Domain (InterPro)" in names
    # Residue-frame axis, not genomic.
    assert fig["layout"]["xaxis"]["title"]["text"] == "Protein residue (0 = canonical start)"


def test_gene_protein_figure_empty_without_bars():
    fig = pplot.build_gene_protein_figure(SimpleNamespace(canonical_len=0, bars=[]))
    assert fig["data"] == []


def test_gene_protein_figure_collapse_domains_depth_lane():
    """Collapsed mode renders one lane of depth-shaded segments with overlap counts."""
    view = _gene_view()
    view.domains = view.domains + [
        {"name": "Chromo-shadow", "interpro_id": "IPR008251", "x0": 30, "x1": 120,
         "isoforms": ["extended · CTG"]},
    ]
    # Two segments: residues 20–120 covered by 2 domains, plus a 1-deep tail.
    view.domain_segments = [{"x0": 20, "x1": 79, "depth": 1},
                            {"x0": 80, "x1": 80, "depth": 2},
                            {"x0": 81, "x1": 120, "depth": 1}]
    collapsed = pplot.build_gene_protein_figure(view, collapse_domains=True)
    dboxes = [t for t in collapsed["data"] if t.get("name") == "Domain (InterPro)"]
    # One box per depth segment, hover shows the overlap count.
    assert len(dboxes) == 3
    assert any("2 overlapping domains" in (t.get("text") or "") for t in dboxes)
    # Left padding: axis starts left of the most-negative bar x0.
    assert collapsed["layout"]["xaxis"]["range"][0] < min(b["x0"] for b in view.bars)


def test_gene_protein_figure_isoform_bars_carry_click_slug():
    """Isoform bars carry the isoform slug as customdata; the canonical bar doesn't."""
    fig = pplot.build_gene_protein_figure(_gene_view())
    by_name = {t.get("name"): t for t in fig["data"]}
    assert "customdata" not in by_name["Canonical"]
    assert by_name["extended · CTG"]["customdata"][0] == "chr1-100-+-CTG-ENST1"
    assert by_name["truncated · AAG"]["customdata"][0] == "chr1-200-+-AAG-ENST1"
