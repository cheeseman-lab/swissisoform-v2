"""The gene figure's variant-row contract with ``gene_overlay.js``.

The browser adds uploaded-VCF markers itself, including rows for consequences the
server drew none of. It can only do that the way the server would if the figure
says how: which rows exist, which traces and labels belong to each, and the
numbers behind the y-range and height. These tests pin that contract.
"""

from __future__ import annotations

from types import SimpleNamespace

from swissisoform_site.plots.protein import UPLOADED_MARKER, build_gene_protein_figure


def _view(variants):
    return SimpleNamespace(
        canonical_len=100,
        bars=[
            {
                "label": "extended · CTG",
                "x0": -20,
                "x1": 99,
                "is_trunc": False,
                "diff_x0": -20,
                "diff_x1": -1,
                "diff_on_canonical": False,
                "orf_type": "extended",
                "slug": "s",
            }
        ],
        variants=variants,
        domains=[],
        domain_segments=[],
        disorder=[],
        coiled_coil=[],
        motifs=[],
        cell_lines=[],
        x_left=-20,
    )


def _variant(consequence, pos=10):
    return {
        "variant_id": f"v-{consequence}-{pos}",
        "pos": pos,
        "consequence": consequence,
        "significance": None,
        "protein_change": "",
        "source": "clinvar",
        "in_unique": False,
    }


def test_meta_lists_the_rows_drawn_in_severity_order() -> None:
    fig = build_gene_protein_figure(
        _view([_variant("synonymous_variant"), _variant("stop_gained"), _variant("zzz_term")])
    )
    meta = fig["layout"]["meta"]["variant_rows"]
    assert meta["present"] == ["stop_gained", "synonymous_variant", "zzz_term"]
    assert meta["order"][0] == "frameshift_variant"
    assert meta["uploaded_marker"] == UPLOADED_MARKER


def test_every_row_trace_and_label_is_tagged_at_its_row_y() -> None:
    fig = build_gene_protein_figure(_view([_variant("missense_variant"), _variant("stop_gained")]))
    meta = fig["layout"]["meta"]["variant_rows"]
    for i, term in enumerate(meta["present"]):
        y = meta["base"] + i * meta["step"]
        (trace,) = [t for t in fig["data"] if (t.get("meta") or {}).get("row") == term]
        assert set(trace["y"]) == {y}
        labels = fig["layout"]["annotations"]
        (label,) = [a for a in labels if a.get("name") == f"variant-row:{term}"]
        assert label["y"] == y


def test_height_and_range_follow_the_formula_the_meta_states() -> None:
    fig = build_gene_protein_figure(_view([_variant("missense_variant"), _variant("stop_gained")]))
    layout = fig["layout"]
    meta = layout["meta"]["variant_rows"]
    y_lo, y_hi = layout["yaxis"]["range"]
    top = meta["base"] + (len(meta["present"]) - 1) * meta["step"] + meta["top_pad"]
    assert y_hi == max(top, meta["canon_y"] + meta["min_above_canon"]) + meta["y_hi_pad"]
    assert layout["height"] == int(meta["height_base"] + meta["height_per_y"] * (y_hi - y_lo))


def test_a_gene_without_variants_still_says_how_to_add_rows() -> None:
    meta = build_gene_protein_figure(_view([]))["layout"]["meta"]["variant_rows"]
    assert meta["present"] == []
    assert meta["base"] > meta["canon_y"]


def test_the_server_never_draws_uploaded_markers() -> None:
    """Uploaded markers are the browser's job now; a stray one would be drawn twice."""
    uploaded = dict(_variant("missense_variant"), source="uploaded")
    fig = build_gene_protein_figure(_view([uploaded]))
    assert not any(t.get("name") == "uploaded VCF" for t in fig["data"])
    assert not any((t.get("marker") or {}).get("symbol") == "diamond" for t in fig["data"])
