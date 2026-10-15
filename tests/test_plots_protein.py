"""Tests for the combined gene protein-residue figure builder.

``build_gene_protein_figure`` draws one plain canonical bar (anchored at x=0)
plus one row per isoform aligned on the shared C-terminus, each carrying its own
difference (extension / lost region / ORF), with deduplicated
variant / domain / disorder / coiled-coil / motif tracks and per-cell-line
initiation lanes. The coordinate invariants it relies on are computed upstream
by ``_make_gene_protein_view`` (app.py) and are covered here too.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("swissisoform_site")  # optional website package; skip if not installed

from swissisoform_site.app import _make_gene_protein_view
from swissisoform_site.plots import protein as pplot

# --------------------------------------------------------------------------- #
# Combined gene view — build_gene_protein_figure
# --------------------------------------------------------------------------- #


def _gene_view():
    """One canonical + one extension + one truncation on the canonical frame."""
    return SimpleNamespace(
        canonical_len=185,
        bars=[
            {"label": "extended · CTG · 225 aa", "x0": -40, "x1": 185, "orf_type": "extended",
             "is_trunc": False, "region": "extension", "diff_x0": -40, "diff_x1": 0,
             "diff_is_lost": False, "slug": "chr1-100-+-CTG-ENST1"},
            {"label": "truncated · AAG · 142 aa", "x0": 44, "x1": 185, "orf_type": "truncated",
             "is_trunc": True, "region": "lost region", "diff_x0": 0, "diff_x1": 43,
             "diff_is_lost": True, "slug": "chr1-200-+-AAG-ENST1"},
        ],
        variants=[
            {"variant_id": "ClinVar:1", "pos": 60, "consequence": "missense_variant",
             "significance": "Pathogenic", "protein_change": "p.X", "source": "ClinVar",
             "in_unique": False},
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


def _row_traces(fig, y):
    return [t for t in fig["data"] if t.get("mode") == "lines" and set(t["y"]) == {y}]


def test_gene_protein_figure_one_canonical_plus_isoform_bars():
    fig = pplot.build_gene_protein_figure(_gene_view())
    assert isinstance(fig, dict) and fig["data"]
    names = [t.get("name") for t in fig["data"]]
    assert names.count("Canonical") == 1
    # One legend entry per kind of segment, never one per isoform.
    legend = [t["name"] for t in fig["data"] if t.get("showlegend")]
    assert len(legend) == len(set(legend))
    assert {"Canonical", "Isoform", "Extension", "Lost region"} <= set(legend)
    # Every isoform row is labelled, uniquely.
    labels = [a["text"] for a in fig["layout"]["annotations"] if a.get("name")]
    assert labels == ["extended · CTG · 225 aa", "truncated · AAG · 142 aa"]
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
    """Every segment of an isoform row — and its label — carries that isoform's slug."""
    view = _gene_view()
    fig = pplot.build_gene_protein_figure(view)
    canonical = next(t for t in fig["data"] if t.get("name") == "Canonical")
    assert "customdata" not in canonical
    for bar in view.bars:
        segs = _row_traces(fig, bar["_y"])
        assert len(segs) == 2  # its difference + its shared body
        assert all(t["customdata"][0] == bar["slug"] for t in segs)
    names = {a["name"] for a in fig["layout"]["annotations"] if a.get("captureevents")}
    assert names == {b["slug"] for b in view.bars}


def test_canonical_bar_is_plain_and_lost_region_sits_on_its_own_row():
    """The canonical row is only the canonical bar; a truncation's lost region is
    drawn on the truncation's row, not shaded onto the canonical."""
    view = _gene_view()
    fig = pplot.build_gene_protein_figure(view)
    assert [t["name"] for t in _row_traces(fig, 0.0)] == ["Canonical"]
    trunc = next(b for b in view.bars if b["diff_is_lost"])
    lost = [t for t in _row_traces(fig, trunc["_y"]) if t["name"] == "Lost region"]
    assert len(lost) == 1
    assert min(lost[0]["x"]) == 0  # lines up under the canonical bar's left edge
    assert max(lost[0]["x"]) == trunc["x0"]  # meets the shared body flush


def test_gene_figure_has_no_deepest_truncation_guide():
    fig = pplot.build_gene_protein_figure(_gene_view())
    assert not [s for s in fig["layout"]["shapes"] if s.get("line", {}).get("dash") == "dot"]


def test_gene_variants_make_no_unique_region_claim():
    """Whether a variant is isoform-unique depends on the isoform, so the gene view's
    hover never says "unique" or "shared"."""
    fig = pplot.build_gene_protein_figure(_gene_view())
    hovers = [h for t in fig["data"] if t.get("mode") == "markers" for h in t.get("hovertext") or []]
    assert hovers and not any("unique" in h or "shared region" in h for h in hovers)


# --------------------------------------------------------------------------- #
# Coordinate invariants — computed by _make_gene_protein_view (app.py), NOT the
# renderer. The _gene_view() fixture above hardcodes post-computed coordinates,
# so it can't catch a regression in the x=0 anchoring / C-terminus alignment
# math. These drive _make_gene_protein_view on a synthetic gene instead.
# --------------------------------------------------------------------------- #

_CAN_LEN = 100
_EXT_LEN = 130  # +30-aa N-terminal extension
_TRUNC_LEN = 70  # -30-aa N-terminal truncation


def _synth_isoform(orf_type, diff_space, iso_len, start_codon, tis_id):
    """Minimal isoform carrying only what _make_gene_protein_view reads."""
    return SimpleNamespace(
        orf_type=orf_type,
        diff_space=diff_space,
        diff_end=30,  # differential-sequence length
        isoform_len=iso_len,
        canonical_len=_CAN_LEN,
        start_codon=start_codon,
        tis_id=tis_id,
        raw={},
        variants_all=[],
    )


def _synthetic_gene():
    """One canonical + one extension + one truncation, C-termini shared."""
    return SimpleNamespace(
        canonical_len=_CAN_LEN,
        isoforms=[
            _synth_isoform("extended", "isoform", _EXT_LEN, "CTG", "chr1:100:+:CTG:ENST1"),
            _synth_isoform("truncated", "canonical", _TRUNC_LEN, "AAG", "chr1:200:+:AAG:ENST2"),
        ],
    )


def _bar(view, orf_type):
    return next(b for b in view.bars if b["orf_type"] == orf_type)


def test_canonical_bar_anchored_at_x0():
    """The canonical bar starts at x=0 (canonical start) and ends at can_len-1."""
    fig = pplot.build_gene_protein_figure(_make_gene_protein_view(_synthetic_gene()))
    canonical = next(t for t in fig["data"] if t.get("name") == "Canonical")
    assert min(canonical["x"]) == 0
    assert max(canonical["x"]) == _CAN_LEN - 1


def test_extension_bar_runs_negative():
    """An extension's added N-terminus sits left of the canonical start (x<0)."""
    view = _make_gene_protein_view(_synthetic_gene())
    ext = _bar(view, "extended")
    assert ext["x0"] < 0
    assert ext["x0"] == -(_EXT_LEN - _CAN_LEN)  # -30
    # The differential (added) region lies left of the x=0 anchor.
    assert ext["diff_x1"] <= 0


def test_cterminus_alignment_extension_and_truncation():
    """Both extension and truncation share the canonical C-terminus (x1 == can_len-1)."""
    view = _make_gene_protein_view(_synthetic_gene())
    ext = _bar(view, "extended")
    trunc = _bar(view, "truncated")
    assert ext["x1"] == _CAN_LEN - 1
    assert trunc["x1"] == _CAN_LEN - 1
    assert ext["x1"] == trunc["x1"]  # same right edge → shared C-terminus


def test_truncation_lost_region_flagged():
    """The truncation starts right of x=0 and flags its N-terminus as lost."""
    view = _make_gene_protein_view(_synthetic_gene())
    trunc = _bar(view, "truncated")
    ext = _bar(view, "extended")
    assert trunc["x0"] > 0
    assert trunc["diff_is_lost"] is True
    assert trunc["diff_x0"] == 0  # canonical residue 1, at x=0 after the shift
    assert trunc["region"] == "lost region"
    assert ext["diff_is_lost"] is False
    assert ext["region"] == "extension"


def test_truncation_lost_region_stops_before_shared_core():
    """The lost region ends at x0-1 (last lost residue), not x0 (first retained).

    x0 is the first RETAINED (shared-core) residue where the isoform body begins,
    so the lost region must end before x0 — otherwise it claims one residue of
    the shared core.
    """
    view = _make_gene_protein_view(_synthetic_gene())
    trunc = _bar(view, "truncated")
    assert trunc["diff_x1"] == trunc["x0"] - 1
    assert trunc["diff_x1"] < trunc["x0"]  # no overlap into the shared body


def test_separate_orf_hover_counts_every_residue():
    """A 50-aa uORF hovers as 50 aa, matching its row label (PR #56 review)."""
    import json

    view = _gene_view()
    view.bars = [
        {"label": "uORF · ATG · 50 aa", "x0": -120, "x1": -71, "orf_type": "uORF",
         "is_trunc": False, "region": "ORF", "diff_x0": -120, "diff_x1": -71,
         "diff_is_lost": False, "slug": "chr1-50-+-ATG-ENST2"},
    ]
    fig = pplot.build_gene_protein_figure(view)
    dumped = json.dumps(fig["data"])
    assert "ORF: 50 aa" in dumped and "ORF: 49 aa" not in dumped
    assert "ORF" in [t["name"] for t in fig["data"] if t.get("showlegend")]


def test_variant_dots_carry_their_outbound_link():
    """Gene-level variants link out (issue #31) rather than claim an isoform."""
    view = _gene_view()
    url = "https://www.ncbi.nlm.nih.gov/clinvar/variation/1/"
    view.variants = [
        {**view.variants[0], "url": url},
        {**view.variants[0], "variant_id": "x", "pos": 70, "url": None},
    ]
    fig = pplot.build_gene_protein_figure(view)
    dots = [t for t in fig["data"] if t.get("mode") == "markers" and t.get("customdata")]
    assert len(dots) == 1
    assert dots[0]["customdata"] == [{"href": url}, None]
    assert "click to open the database page" in dots[0]["hovertext"][0]
    assert "click to open" not in dots[0]["hovertext"][1]
