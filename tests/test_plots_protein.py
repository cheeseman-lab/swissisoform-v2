"""Tests for the combined gene protein-residue figure builder.

``build_gene_protein_figure`` draws one canonical length bar (anchored at x=0)
plus one bar per isoform aligned on the shared C-terminus, with deduplicated
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
            {"label": "extended · CTG", "x0": -40, "x1": 185, "orf_type": "extended",
             "is_trunc": False, "diff_x0": -40, "diff_x1": 0, "diff_on_canonical": False,
             "slug": "chr1-100-+-CTG-ENST1"},
            {"label": "truncated · AAG", "x0": 44, "x1": 186, "orf_type": "truncated",
             "is_trunc": True, "diff_x0": 1, "diff_x1": 43, "diff_on_canonical": True,
             "slug": "chr1-200-+-AAG-ENST1"},
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


def test_truncation_lost_region_on_canonical():
    """The truncation starts right of x=0 and shades its lost N-terminus on the canonical."""
    view = _make_gene_protein_view(_synthetic_gene())
    trunc = _bar(view, "truncated")
    ext = _bar(view, "extended")
    assert trunc["x0"] > 0
    assert trunc["diff_on_canonical"] is True
    assert ext["diff_on_canonical"] is False


def test_truncation_lost_region_stops_before_shared_core():
    """The lost region ends at x0-1 (last lost residue), not x0 (first retained).

    x0 is the first RETAINED (shared-core) residue where the isoform body begins,
    so the red lost-region overlay must not extend to x0 — otherwise it bleeds one
    residue into the shared core on the canonical bar.
    """
    view = _make_gene_protein_view(_synthetic_gene())
    trunc = _bar(view, "truncated")
    assert trunc["diff_x1"] == trunc["x0"] - 1
    assert trunc["diff_x1"] < trunc["x0"]  # no overlap into the shared body
