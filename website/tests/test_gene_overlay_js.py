"""``gene_overlay.js``: uploaded-VCF markers added to the server's gene figure.

The server stopped drawing these when the scan moved into the browser, so these
tests are the contract the server-side figure tests used to be: markers on the
right row at the hit's x, one per variant, distinct from ClinVar's circles, one
legend entry — and, for a consequence the gene has no row for, the same geometry
the server would have produced had that row existed.

Fixture-free: the figure comes from a hand-built view and the hits are written
here. Needs ``node`` (see ``jsrun``).
"""

from __future__ import annotations

from types import SimpleNamespace

from jsrun import apply_uploaded_hits, needs_node

from swissisoform_site.plots.protein import UPLOADED_MARKER, build_gene_protein_figure

pytestmark = needs_node


def _view(variants, *, truncation=False):
    bar = {
        "label": "truncated · ATG" if truncation else "extended · CTG",
        "x0": 20 if truncation else -20,
        "x1": 99,
        "is_trunc": truncation,
        "diff_x0": 0 if truncation else -20,
        "diff_x1": 19 if truncation else -1,
        "diff_on_canonical": truncation,
        "orf_type": "truncated" if truncation else "extended",
        "slug": "s",
    }
    return SimpleNamespace(
        canonical_len=100,
        bars=[bar],
        variants=variants,
        domains=[],
        domain_segments=[],
        disorder=[],
        coiled_coil=[],
        motifs=[],
        cell_lines=[],
        x_left=-20,
    )


def _clinvar(consequence, pos=10):
    return {
        "variant_id": f"cv-{consequence}-{pos}",
        "pos": pos,
        "consequence": consequence,
        "significance": None,
        "protein_change": "",
        "source": "clinvar",
        "in_unique": False,
    }


def _hit(consequence="missense_variant", x=30, **overrides):
    hit = {
        "line_no": 7,
        "chrom": "chr1",
        "pos": 1234567,
        "ref": "C",
        "alt": "T",
        "gene": "G",
        "tis_id": "chr1:1:+:ATG:ENST1",
        "frame": "isoform",
        "residue": 29,
        "region": "shared",
        "consequence": consequence,
        "aa_ref": "P",
        "aa_alt": "L",
        "consequence_note": "",
        "x": x,
        "protein_change": "p.Pro30Leu · isoform frame",
    }
    hit.update(overrides)
    return hit


def _uploaded(fig):
    return [t for t in fig["data"] if t.get("name") == "uploaded VCF"]


def _row_y(fig, term):
    meta = fig["layout"]["meta"]["variant_rows"]
    (label,) = [a for a in fig["layout"]["annotations"] if a.get("name") == f"variant-row:{term}"]
    return label["y"], meta


def test_no_hits_leaves_the_figure_alone() -> None:
    fig = build_gene_protein_figure(_view([_clinvar("missense_variant")]))
    assert apply_uploaded_hits(fig, []) == fig


def test_a_mark_sits_at_its_x_on_its_consequences_row() -> None:
    fig = build_gene_protein_figure(_view([_clinvar("missense_variant")]))
    out = apply_uploaded_hits(fig, [_hit(x=42)])
    (trace,) = _uploaded(out)
    y, _ = _row_y(out, "missense_variant")
    assert trace["x"] == [42]
    assert trace["y"] == [y]


def test_marks_are_styled_from_the_server_and_coloured_by_consequence() -> None:
    fig = build_gene_protein_figure(_view([]))
    (trace,) = _uploaded(apply_uploaded_hits(fig, [_hit("synonymous_variant")]))
    marker = trace["marker"]
    for key, value in UPLOADED_MARKER.items():
        assert marker[key] == value
    assert marker["color"] == "#94a3b8"  # synonymous grey, not missense amber


def test_exactly_one_legend_entry_however_many_rows() -> None:
    fig = build_gene_protein_figure(_view([]))
    hits = [_hit("missense_variant"), _hit("stop_gained", x=5), _hit("inframe_deletion", x=9)]
    traces = _uploaded(apply_uploaded_hits(fig, hits))
    assert len(traces) == 3
    assert sum(1 for t in traces if t["showlegend"]) == 1


def test_one_variant_in_several_isoforms_is_one_mark() -> None:
    """Shared-region hits map to the same x in every isoform; stacked, only one hover
    would be reachable, so they merge and the count goes in the tooltip."""
    fig = build_gene_protein_figure(_view([]))
    hits = [_hit(tis_id=f"chr1:{i}:+:ATG:ENST1") for i in range(3)]
    hits.append(_hit(alt="G", tis_id="chr1:9:+:ATG:ENST1"))  # a different allele, same x
    (trace,) = _uploaded(apply_uploaded_hits(fig, hits))
    assert trace["x"] == [30, 30]
    assert "in 3 isoforms" in trace["hovertext"][0]
    assert "in 3 isoforms" not in trace["hovertext"][1]


def test_a_hit_in_any_unique_region_marks_the_merge_unique() -> None:
    fig = build_gene_protein_figure(_view([]))
    hits = [_hit(tis_id="a"), _hit(tis_id="b", region="unique")]
    (trace,) = _uploaded(apply_uploaded_hits(fig, hits))
    assert trace["hovertext"][0].endswith("unique region")


def test_hover_names_the_source_the_change_and_the_frame() -> None:
    fig = build_gene_protein_figure(_view([]))
    (trace,) = _uploaded(apply_uploaded_hits(fig, [_hit()]))
    hover = trace["hovertext"][0]
    assert hover.startswith("<b>From your VCF</b>")
    assert "chr1:1,234,567 C&gt;T · line 7" in hover
    assert "p.Pro30Leu · isoform frame" in hover


def test_vcf_text_is_escaped_in_the_hover() -> None:
    """Hover text is HTML to Plotly, and every field here came from an upload."""
    fig = build_gene_protein_figure(_view([]))
    (trace,) = _uploaded(apply_uploaded_hits(fig, [_hit(chrom="<img src=x>")]))
    assert "<img" not in trace["hovertext"][0]
    assert "&lt;img src=x&gt;" in trace["hovertext"][0]


def test_an_unplaced_hit_draws_nothing() -> None:
    fig = build_gene_protein_figure(_view([_clinvar("missense_variant")]))
    assert apply_uploaded_hits(fig, [_hit(x=None)]) == fig


def test_a_new_row_lands_where_the_server_would_have_put_it() -> None:
    """start_lost sorts between stop_gained and missense, so existing rows must move.

    The reference is the server's own figure for the same gene with a start_lost row
    already present: rows, labels, range, height and the truncation guide must match.
    """
    known = [_clinvar("stop_gained"), _clinvar("missense_variant")]
    fig = build_gene_protein_figure(_view(known, truncation=True))
    out = apply_uploaded_hits(fig, [_hit("start_lost", x=0)])
    reference = build_gene_protein_figure(
        _view(known + [_clinvar("start_lost", pos=1)], truncation=True)
    )

    for term in ("stop_gained", "start_lost", "missense_variant"):
        assert _row_y(out, term)[0] == _row_y(reference, term)[0], term
    for term in ("stop_gained", "missense_variant"):
        (trace,) = [t for t in out["data"] if (t.get("meta") or {}).get("row") == term]
        assert set(trace["y"]) == {_row_y(reference, term)[0]}
    assert out["layout"]["yaxis"]["range"] == reference["layout"]["yaxis"]["range"]
    assert out["layout"]["height"] == reference["layout"]["height"]

    def guide(f):
        (shape,) = [s for s in f["layout"]["shapes"] if s.get("name") == "truncation-guide"]
        return shape["y1"]

    assert guide(out) == guide(reference)


def test_the_first_row_on_a_gene_with_none_lands_where_the_server_would_put_it() -> None:
    """No rows yet and no truncation guide: the headroom floor, not the rows, set y_hi."""
    fig = build_gene_protein_figure(_view([]))
    out = apply_uploaded_hits(fig, [_hit("missense_variant", x=0)])
    reference = build_gene_protein_figure(_view([_clinvar("missense_variant", pos=1)]))

    assert _row_y(out, "missense_variant")[0] == _row_y(reference, "missense_variant")[0]
    assert out["layout"]["yaxis"]["range"] == reference["layout"]["yaxis"]["range"]
    assert out["layout"]["height"] == reference["layout"]["height"]


def test_an_unknown_consequence_gets_a_row_after_the_known_ones() -> None:
    fig = build_gene_protein_figure(_view([_clinvar("synonymous_variant")]))
    out = apply_uploaded_hits(fig, [_hit("reference_mismatch")])
    assert _row_y(out, "reference_mismatch")[0] > _row_y(out, "synonymous_variant")[0]
