"""Build the protein-lollipop Plotly figure for the V2 per-isoform page.

The figure has two stacked panels:
- Top (residues 1..N): protein body + diff-region shading + variant lollipops
- Bottom (same X axis): domain rectangles + motif marks

For truncations (diff_space=canonical), the diff region spans residues 1..len(diff_seq)
on the canonical-length axis — the LOST region. For extensions (diff_space=isoform), the
diff region spans 1..len(diff_seq) on the isoform-length axis — the ADDED region.
"""

from __future__ import annotations

from typing import Any

_DIFF_FILL = "rgba(214, 39, 40, 0.18)"  # red wash
_SHARED_FILL = "rgba(31, 119, 180, 0.10)"  # blue wash
_LOLLIPOP_COLOR_PATHOGENIC = "#d62728"
_LOLLIPOP_COLOR_OTHER = "#1f77b4"

_FONT_FAMILY = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
)
_HOVER_FONT_FAMILY = "system-ui, sans-serif"


def build_protein_figure(isoform: Any, overlays: dict[str, bool]) -> dict[str, Any]:
    """Return a Plotly figure dict for the protein view."""
    overlays = overlays or {}
    diff_seq = getattr(isoform, "differential_sequence", "") or ""
    diff_len = len(diff_seq)
    diff_space = getattr(isoform, "diff_space", "isoform")
    iso_len = int(getattr(isoform, "isoform_length_aa", 0) or 0)
    can_len = int(getattr(isoform, "canonical_length_aa", 0) or 0)

    # Pick the axis length based on diff_space: truncations show the canonical axis
    # (so the lost region is visible); extensions show the isoform axis.
    axis_len = can_len if diff_space == "canonical" else iso_len
    if axis_len == 0:
        return {
            "data": [],
            "layout": {
                "font": {"family": _FONT_FAMILY, "size": 13, "color": "#1f2937"},
                "annotations": [
                    {
                        "text": "No protein length available.",
                        "showarrow": False,
                        "x": 0.5,
                        "y": 0.5,
                        "xref": "paper",
                        "yref": "paper",
                        "font": {"size": 14, "color": "#666"},
                    }
                ],
            },
        }

    traces: list[dict[str, Any]] = []

    # Diff region shaded span
    traces.append(_shaded_span(1, diff_len, _DIFF_FILL, "diff region"))
    # Shared region shaded span (everything past the diff)
    if axis_len > diff_len:
        traces.append(_shaded_span(diff_len + 1, axis_len, _SHARED_FILL, "shared region"))

    # Protein body line
    traces.append(
        {
            "type": "scatter",
            "mode": "lines",
            "x": [1, axis_len],
            "y": [0.5, 0.5],
            "line": {"color": "#444", "width": 6},
            "hoverinfo": "skip",
            "showlegend": False,
        }
    )

    # Variant lollipops — each variant gets a stem trace + a head trace; both named
    # with the "variant " prefix so they toggle together via Plotly.restyle by name.
    if overlays.get("variants", True):
        for v in getattr(isoform, "variants_in_unique", []) or []:
            pos = v.get("protein_pos")
            if pos is None:
                continue
            sig = (v.get("clinical_significance") or "").lower()
            is_path = "pathogenic" in sig
            color = _LOLLIPOP_COLOR_PATHOGENIC if is_path else _LOLLIPOP_COLOR_OTHER
            vid = v.get("variant_id", "?")
            tooltip = f"{vid}<br>{v.get('hgvsp', '')}<br>{v.get('clinical_significance', '')}"
            # Stem: 2-point line trace, no markers — toggles with the head.
            traces.append(
                {
                    "type": "scatter",
                    "name": f"variant {vid} stem",
                    "mode": "lines",
                    "x": [pos, pos],
                    "y": [0.5, 1.2],
                    "line": {"color": color, "width": 1.5},
                    "hoverinfo": "skip",
                    "showlegend": False,
                }
            )
            # Head: single-point marker, carries the hover tooltip.
            traces.append(
                {
                    "type": "scatter",
                    "name": f"variant {vid}",
                    "mode": "markers",
                    "x": [pos],
                    "y": [1.2],
                    "marker": {"size": 10, "color": color, "symbol": "circle"},
                    "hovertext": [tooltip],
                    "hoverinfo": "text",
                    "showlegend": False,
                }
            )

    # Domain rectangles (in a separate track at y=-0.5)
    if overlays.get("domains", True):
        for d in getattr(isoform, "domains", []) or []:
            traces.append(
                {
                    "type": "scatter",
                    "name": f"domain: {d.get('name', '?')}",
                    "mode": "lines",
                    "fill": "toself",
                    "x": [d["start"], d["end"], d["end"], d["start"], d["start"]],
                    "y": [-0.7, -0.7, -0.3, -0.3, -0.7],
                    "line": {"color": "rgba(0,0,0,0)"},
                    "fillcolor": "rgba(44, 160, 44, 0.45)",
                    "hovertext": d.get("name", "?"),
                    "hoverinfo": "text",
                    "showlegend": False,
                }
            )

    # Motif marks at y=-1.0
    if overlays.get("motifs", True):
        for m in getattr(isoform, "motifs", []) or []:
            traces.append(
                {
                    "type": "scatter",
                    "name": f"motif: {m.get('name', '?')}",
                    "mode": "markers",
                    "x": [m["start"]],
                    "y": [-1.0],
                    "marker": {"size": 8, "color": "#9467bd", "symbol": "diamond"},
                    "hovertext": [m.get("name", "?")],
                    "hoverinfo": "text",
                    "showlegend": False,
                }
            )

    title = (
        f"Protein view — {getattr(isoform, 'orf_type', '').title()}; "
        f"diff: 1..{diff_len} "
        f"({'lost from canonical' if diff_space == 'canonical' else 'added to isoform'})"
    )
    return {
        "data": traces,
        "layout": {
            "title": {"text": title, "font": {"size": 15, "color": "#111827"}},
            "font": {"family": _FONT_FAMILY, "size": 13, "color": "#1f2937"},
            "hoverlabel": {"font": {"family": _HOVER_FONT_FAMILY, "size": 12}},
            "xaxis": {
                "title": {
                    "text": "Protein residue",
                    "font": {"size": 12, "color": "#4b5563"},
                },
                "tickfont": {"size": 11, "color": "#6b7280"},
                "range": [0, axis_len + 5],
            },
            "yaxis": {
                "range": [-1.5, 1.5],
                "showticklabels": False,
                "zeroline": False,
            },
            "height": 320,
            "margin": {"l": 70, "r": 30, "t": 60, "b": 50},
        },
    }


def _shaded_span(x0: int, x1: int, fill: str, name: str) -> dict[str, Any]:
    return {
        "type": "scatter",
        "name": name,
        "mode": "lines",
        "fill": "toself",
        "x": [x0, x1, x1, x0, x0],
        "y": [-1.5, -1.5, 1.5, 1.5, -1.5],
        "line": {"color": "rgba(0,0,0,0)"},
        "fillcolor": fill,
        "hoverinfo": "skip",
        "showlegend": False,
    }
