"""Build the protein-track Plotly figure for the V2 per-isoform page.

Two left-aligned length bars on separate y-lanes — canonical (y=1.0) and
isoform (y=0.5) — make the length delta legible. The differential region is
shaded on the relevant bar (extensions → isoform residues green; truncations →
canonical residues red). Clinical variants render as lollipops coloured by
significance; motifs as spans; domains as boxes. A horizontal legend names
every visual class.
"""

from __future__ import annotations

from typing import Any

_CANON_COLOR = "#64748b"  # slate
_ISO_COLOR = "#1f77b4"  # blue
_EXT_FILL = "#2ca02c"  # green (added region)
_TRUNC_FILL = "#d62728"  # red (lost region)
_DOMAIN_FILL = "rgba(44, 160, 44, 0.45)"
_MOTIF_COLOR = "#9467bd"

# Variant significance colour classes (label, colour, predicate on lowered sig).
_VARIANT_CLASSES = [
    ("Pathogenic", "#d62728", lambda s: s.startswith(("pathogenic", "likely_path", "likely path"))),
    ("Benign", "#94a3b8", lambda s: "benign" in s),
    ("Uncertain", "#f59e0b", lambda s: "uncertain" in s or s in ("vus", "")),
    ("Other", "#6b7280", lambda _s: True),
]

_FONT_FAMILY = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
)
_HOVER_FONT_FAMILY = "system-ui, sans-serif"

_CANON_Y = 1.0
_ISO_Y = 0.5


def _classify_variant(sig: str) -> tuple[str, str]:
    s = (sig or "").lower()
    for label, color, pred in _VARIANT_CLASSES:
        if pred(s):
            return label, color
    return "Other", "#6b7280"


def build_protein_figure(isoform: Any, overlays: dict[str, bool]) -> dict[str, Any]:
    """Return a Plotly figure dict for the protein view."""
    overlays = overlays or {}
    can_len = int(getattr(isoform, "canonical_len", 0) or 0)
    iso_len = int(getattr(isoform, "isoform_len", 0) or 0)
    diff_start = int(getattr(isoform, "diff_start", 0) or 0)
    diff_end = int(getattr(isoform, "diff_end", 0) or 0)
    diff_space = getattr(isoform, "diff_space", "isoform")
    orf_type = getattr(isoform, "orf_type", "") or ""

    axis_len = max(can_len, iso_len)
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

    # ── Length bars (canonical + isoform) ──
    traces.append(_length_bar(1, can_len or 1, _CANON_Y, _CANON_COLOR, "Canonical"))
    traces.append(_length_bar(1, iso_len or 1, _ISO_Y, _ISO_COLOR, "Isoform"))

    # ── Differential region shading on the relevant bar ──
    is_trunc = diff_space == "canonical" or orf_type == "truncated"
    diff_lo = diff_start + 1
    diff_hi = diff_end
    if diff_hi >= diff_lo:
        if is_trunc:
            diff_y, diff_fill = _CANON_Y, _TRUNC_FILL
        else:
            diff_y, diff_fill = _ISO_Y, _EXT_FILL
        traces.append(_diff_overlay(diff_lo, diff_hi, diff_y, diff_fill))

    # ── Variant lollipops, coloured by significance, one legend entry per class ──
    if overlays.get("variants", True):
        seen_classes: set[str] = set()
        for v in getattr(isoform, "variants_in_unique", []) or []:
            pos = v.get("isoform_protein_pos")
            if pos is None:
                pos = v.get("protein_pos")
            if pos is None:
                continue
            label, color = _classify_variant(v.get("clinical_significance") or "")
            is_path = label == "Pathogenic"
            vid = v.get("variant_id") or v.get("id") or "?"
            tooltip = (
                f"{vid}<br>{v.get('hgvsp', '')}<br>"
                f"{v.get('clinical_significance', '')}<br>{v.get('source', '')}"
            )
            show = label not in seen_classes
            seen_classes.add(label)
            traces.append(
                {
                    "type": "scatter",
                    "mode": "lines",
                    "x": [pos, pos],
                    "y": [_ISO_Y, _ISO_Y + 0.55],
                    "line": {"color": color, "width": 1.5},
                    "legendgroup": label,
                    "hoverinfo": "skip",
                    "showlegend": False,
                }
            )
            traces.append(
                {
                    "type": "scatter",
                    "name": label,
                    "mode": "markers",
                    "x": [pos],
                    "y": [_ISO_Y + 0.55],
                    "marker": {
                        "size": 12 if is_path else 9,
                        "color": color,
                        "symbol": "circle",
                        "line": {"width": 1, "color": "#111"},
                    },
                    "legendgroup": label,
                    "hovertext": [tooltip],
                    "hoverinfo": "text",
                    "showlegend": show,
                }
            )

    # ── Domains (boxes), one legend entry ──
    if overlays.get("domains", True):
        domain_legend_shown = False
        for d in getattr(isoform, "domains", []) or []:
            traces.append(
                {
                    "type": "scatter",
                    "name": "Domain (InterPro)",
                    "mode": "lines",
                    "fill": "toself",
                    "x": [d["start"], d["end"], d["end"], d["start"], d["start"]],
                    "y": [-0.7, -0.7, -0.3, -0.3, -0.7],
                    "line": {"color": "rgba(0,0,0,0)"},
                    "fillcolor": _DOMAIN_FILL,
                    "legendgroup": "domain",
                    "hovertext": d.get("name", "?"),
                    "hoverinfo": "text",
                    "showlegend": not domain_legend_shown,
                }
            )
            domain_legend_shown = True

    # ── Motifs (spans), one legend entry ──
    if overlays.get("motifs", True):
        motif_legend_shown = False
        for m in getattr(isoform, "motifs", []) or []:
            end = m.get("end", m.get("start"))
            label = m.get("name", "?")
            hover = f"{label}" + (f"<br>{m['match']}" if m.get("match") else "")
            traces.append(
                {
                    "type": "scatter",
                    "name": "Motif",
                    "mode": "lines",
                    "x": [m["start"], end],
                    "y": [-1.0, -1.0],
                    "line": {"color": _MOTIF_COLOR, "width": 6},
                    "legendgroup": "motif",
                    "hovertext": [hover, hover],
                    "hoverinfo": "text",
                    "showlegend": not motif_legend_shown,
                }
            )
            motif_legend_shown = True

    # ── Per-cell-line initiation lanes (TIS mapped onto the residue axis) ──
    tracks = getattr(isoform, "cell_line_tracks", None) or []
    canon_residue = getattr(isoform, "canon_residue", None)
    shapes: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    y_bottom = -1.5
    if tracks:
        lane_base, lane_gap = -1.7, 0.45
        all_ie = [m["log2_ie"] for t in tracks for m in t["marks"]]
        vmin, vmax = (min(all_ie), max(all_ie)) if all_ie else (-1.0, 0.0)

        def _ie_size(v: float) -> float:
            return 12.0 if vmax <= vmin else 7.0 + (v - vmin) / (vmax - vmin) * 15.0

        for i, t in enumerate(tracks):
            ly = lane_base - i * lane_gap
            xs, ys, sizes, colors, hover = [], [], [], [], []
            for m in t["marks"]:
                xs.append(m["residue"])
                ys.append(ly)
                sizes.append(_ie_size(m["log2_ie"]))
                colors.append(_TRUNC_FILL if m["focal"] else _ISO_COLOR)
                try:
                    ie = f"{2 ** m['log2_ie']:.3g}"
                except (OverflowError, ValueError):
                    ie = "n/a"
                hover.append(
                    f"<b>{t['sample']}</b><br>{m['label']}<br>residue ~{m['residue']:.0f} · IE {ie}"
                )
            traces.append(
                {
                    "type": "scatter",
                    "mode": "markers",
                    "x": xs,
                    "y": ys,
                    "marker": {
                        "size": sizes,
                        "color": colors,
                        "line": {"width": 1, "color": "#fff"},
                    },
                    "hovertext": hover,
                    "hoverinfo": "text",
                    "showlegend": False,
                }
            )
            annotations.append(
                {
                    "x": 0,
                    "y": ly,
                    "xref": "x",
                    "yref": "y",
                    "xanchor": "right",
                    "xshift": -6,
                    "text": t["sample"].replace("_", " "),
                    "showarrow": False,
                    "font": {"size": 10, "color": "#475569"},
                }
            )
        y_bottom = lane_base - (len(tracks) - 1) * lane_gap - 0.3
        annotations.append(
            {
                "x": axis_len + 5,
                "y": lane_base + 0.35,
                "xref": "x",
                "yref": "y",
                "xanchor": "right",
                "text": "initiation efficiency per cell line (dot size)",
                "showarrow": False,
                "font": {"size": 9, "color": "#94a3b8"},
            }
        )
        if canon_residue is not None:
            shapes.append(
                {
                    "type": "line",
                    "x0": canon_residue,
                    "x1": canon_residue,
                    "y0": y_bottom,
                    "y1": 1.3,
                    "line": {"color": "#111827", "width": 1, "dash": "dot"},
                }
            )

    title = (
        f"Protein — {orf_type.title()}; "
        f"canonical {can_len} aa vs isoform {iso_len} aa; "
        f"diff {diff_lo}..{diff_hi} "
        f"({'lost from canonical' if is_trunc else 'added to isoform'})"
    )
    return {
        "data": traces,
        "layout": {
            "title": {"text": title, "font": {"size": 15, "color": "#111827"}},
            "font": {"family": _FONT_FAMILY, "size": 13, "color": "#1f2937"},
            "hoverlabel": {"font": {"family": _HOVER_FONT_FAMILY, "size": 12}},
            "showlegend": True,
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "left",
                "x": 0,
                "font": {"size": 11},
            },
            "shapes": shapes,
            "annotations": annotations,
            "xaxis": {
                "title": {
                    "text": "Protein residue",
                    "font": {"size": 12, "color": "#4b5563"},
                },
                "tickfont": {"size": 11, "color": "#6b7280"},
                "range": [0, axis_len + 5],
            },
            "yaxis": {
                "range": [y_bottom - 0.2, 1.7],
                "showticklabels": False,
                "zeroline": False,
            },
            "height": 320 + (40 * len(tracks) if tracks else 0),
            "margin": {"l": 95, "r": 30, "t": 70, "b": 50},
        },
    }


def _length_bar(x0: int, x1: int, y: float, color: str, name: str) -> dict[str, Any]:
    return {
        "type": "scatter",
        "name": name,
        "mode": "lines",
        "x": [x0, x1],
        "y": [y, y],
        "line": {"color": color, "width": 12},
        "hovertext": [f"{name}: {x0}", f"{name}: {x1}"],
        "hoverinfo": "text",
        "showlegend": True,
    }


def _diff_overlay(x0: int, x1: int, y: float, color: str) -> dict[str, Any]:
    return {
        "type": "scatter",
        "name": "Differential region",
        "mode": "lines",
        "x": [x0, x1],
        "y": [y, y],
        "line": {"color": color, "width": 12},
        "hovertext": [f"Differential region {x0}", f"Differential region {x1}"],
        "hoverinfo": "text",
        "showlegend": True,
    }
