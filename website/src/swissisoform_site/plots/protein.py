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

_FONT_FAMILY = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
)
_HOVER_FONT_FAMILY = "system-ui, sans-serif"

_CANON_Y = 1.0
_ISO_Y = 0.5

# Variant tracks, one row per consequence type, ordered most→least severe.
_CONSEQ_ORDER = [
    "frameshift_variant",
    "stop_gained",
    "stop_lost",
    "start_lost",
    "splice_donor_variant",
    "splice_acceptor_variant",
    "missense_variant",
    "inframe_deletion",
    "inframe_insertion",
    "synonymous_variant",
]
_CONSEQ_COLOR = {
    "frameshift_variant": "#7f1d1d",
    "stop_gained": "#b91c1c",
    "stop_lost": "#b91c1c",
    "start_lost": "#b91c1c",
    "splice_donor_variant": "#c2410c",
    "splice_acceptor_variant": "#c2410c",
    "missense_variant": "#d97706",
    "inframe_deletion": "#7c3aed",
    "inframe_insertion": "#7c3aed",
    "synonymous_variant": "#94a3b8",
}
_CONSEQ_SHORT = {
    "frameshift_variant": "frameshift",
    "stop_gained": "stop-gain",
    "stop_lost": "stop-lost",
    "start_lost": "start-lost",
    "splice_donor_variant": "splice",
    "splice_acceptor_variant": "splice",
    "missense_variant": "missense",
    "inframe_deletion": "inframe-del",
    "inframe_insertion": "inframe-ins",
    "synonymous_variant": "synonymous",
}


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
    shapes: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    is_trunc = diff_space == "canonical" or orf_type == "truncated"

    # Left axis bound — siblings longer than the focal map left of residue 1.
    cl_tracks = getattr(isoform, "cell_line_tracks", None) or []
    _all_res = [m["residue"] for t in cl_tracks for m in t["marks"]]
    x_left = min(0.0, min(_all_res)) - 4 if _all_res else 0.0

    # ── Length bars, aligned on the SHARED region (not the N-terminus) ──
    # The differential (unique) region occupies residues 1..diff_end on the
    # protein that carries it; the other protein is shifted right by diff_end so
    # the shared sequence lines up. ``feat_offset`` moves isoform-coordinate
    # features (domains/motifs) onto the displayed axis for truncations.
    if is_trunc:
        canon_x0, canon_x1 = 1, (can_len or 1)
        iso_x0, iso_x1 = diff_end + 1, diff_end + (iso_len or 1)
        feat_offset = diff_end
    else:
        iso_x0, iso_x1 = 1, (iso_len or 1)
        canon_x0, canon_x1 = diff_end + 1, diff_end + (can_len or 1)
        feat_offset = 0
    axis_len = max(axis_len, canon_x1, iso_x1)
    traces.append(_length_bar(canon_x0, canon_x1, _CANON_Y, _CANON_COLOR, "Canonical"))
    traces.append(_length_bar(iso_x0, iso_x1, _ISO_Y, _ISO_COLOR, "Isoform"))

    # ── Differential region shading on the bar that carries it (residues 1..diff_end) ──
    diff_lo = diff_start + 1
    diff_hi = diff_end
    if diff_hi >= diff_lo:
        diff_y, diff_fill = (_CANON_Y, _TRUNC_FILL) if is_trunc else (_ISO_Y, _EXT_FILL)
        traces.append(_diff_overlay(diff_lo, diff_hi, diff_y, diff_fill))

    # ── Variant tracks ABOVE the bars: one row per consequence type ──
    mut_base = _CANON_Y + 0.35  # first variant row, above the canonical bar
    mut_top = _CANON_Y
    if overlays.get("variants", True):
        by_conseq: dict[str, list] = {}
        for v in getattr(isoform, "variants", []) or []:
            pos = v.get("isoform_protein_pos")
            if pos is None:
                pos = v.get("protein_pos")
            if pos is None:
                continue
            by_conseq.setdefault(v.get("consequence") or "other", []).append((pos, v))
        conseqs = sorted(
            by_conseq,
            key=lambda c: (_CONSEQ_ORDER.index(c) if c in _CONSEQ_ORDER else 99, c),
        )
        for i, c in enumerate(conseqs):
            ty = mut_base + i * 0.32
            xs, cols, sizes, opac, hover = [], [], [], [], []
            for pos, v in by_conseq[c]:
                sig = (v.get("clinical_significance") or "").lower()
                is_path = "pathogenic" in sig
                in_unique = bool(v.get("in_unique"))
                xs.append(pos)
                cols.append("#d62728" if is_path else _CONSEQ_COLOR.get(c, "#94a3b8"))
                # Differential-region variants are the focus: full size/opacity.
                # Shared canonical-core variants ride along, recessive.
                sizes.append((9 if is_path else 6) if in_unique else (7 if is_path else 4))
                opac.append(1.0 if in_unique else 0.4)
                region = "unique" if in_unique else "shared"
                hover.append(
                    f"{v.get('variant_id', '?')}<br>{v.get('hgvsp') or ''}<br>"
                    f"{c} · {region} region<br>"
                    f"{v.get('clinical_significance') or '—'} · {v.get('source') or ''}"
                )
            traces.append(
                {
                    "type": "scatter",
                    "mode": "markers",
                    "x": xs,
                    "y": [ty] * len(xs),
                    "marker": {
                        "size": sizes,
                        "color": cols,
                        "opacity": opac,
                        "symbol": "circle",
                    },
                    "hovertext": hover,
                    "hoverinfo": "text",
                    "showlegend": False,
                }
            )
            annotations.append(
                {
                    "x": x_left,
                    "y": ty,
                    "xref": "x",
                    "yref": "y",
                    "xanchor": "right",
                    "xshift": -6,
                    "text": _CONSEQ_SHORT.get(c, c)[:14],
                    "showarrow": False,
                    "font": {"size": 9, "color": "#475569"},
                }
            )
        if conseqs:
            mut_top = mut_base + (len(conseqs) - 1) * 0.32 + 0.3

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
                    "x": [
                        d["start"] + feat_offset,
                        d["end"] + feat_offset,
                        d["end"] + feat_offset,
                        d["start"] + feat_offset,
                        d["start"] + feat_offset,
                    ],
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
                    "x": [m["start"] + feat_offset, end + feat_offset],
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
    tracks = cl_tracks
    canon_residue = getattr(isoform, "canon_residue", None)
    y_bottom = -1.5
    if tracks:
        lane_base, lane_gap = -1.7, 0.45
        all_ie = [m["log2_ie"] for t in tracks for m in t["marks"]]
        vmin, vmax = (min(all_ie), max(all_ie)) if all_ie else (-1.0, 0.0)

        def _ie_size(v: float) -> float:
            return 12.0 if vmax <= vmin else 7.0 + (v - vmin) / (vmax - vmin) * 15.0

        for i, t in enumerate(tracks):
            ly = lane_base - i * lane_gap
            xs, ys, sizes, colors, symbols, opac, hover = [], [], [], [], [], [], []
            for m in t["marks"]:
                xs.append(m["residue"])
                ys.append(ly)
                sizes.append(_ie_size(m["log2_ie"]))
                # The focal start (this isoform's TIS) is a solid red dot anchored
                # to the red guide line; the other alternative starts on this
                # transcript ride along as hollow grey dots so they read as
                # context, not as this isoform's start.
                colors.append(_TRUNC_FILL if m["focal"] else "#94a3b8")
                symbols.append("circle" if m["focal"] else "circle-open")
                opac.append(1.0 if m["focal"] else 0.6)
                try:
                    ie = f"{2 ** m['log2_ie']:.3g}"
                except (OverflowError, ValueError):
                    ie = "n/a"
                tag = " (this isoform)" if m["focal"] else ""
                hover.append(
                    f"<b>{t['sample']}</b><br>{m['label']}{tag}<br>"
                    f"residue ~{m['residue']:.0f} · IE {ie}"
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
                        "symbol": symbols,
                        "opacity": opac,
                        "line": {"width": 1, "color": "#fff"},
                    },
                    "hovertext": hover,
                    "hoverinfo": "text",
                    "showlegend": False,
                }
            )
            annotations.append(
                {
                    "x": x_left,
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
    # ── Vertical guides: focal start (red) + canonical start (grey) ──
    guide_top = max(mut_top, _CANON_Y + 0.2)
    shapes.append(
        {
            "type": "line",
            "x0": iso_x0,
            "x1": iso_x0,
            "y0": y_bottom,
            "y1": guide_top,
            "line": {"color": _TRUNC_FILL, "width": 1, "dash": "dot"},
        }
    )
    if canon_residue is not None and abs(canon_residue - iso_x0) > 0.5:
        shapes.append(
            {
                "type": "line",
                "x0": canon_residue,
                "x1": canon_residue,
                "y0": y_bottom,
                "y1": guide_top,
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
                "range": [x_left, axis_len + 5],
            },
            "yaxis": {
                "range": [y_bottom - 0.2, mut_top + 0.3],
                "showticklabels": False,
                "zeroline": False,
            },
            "height": 300
            + int(60 * max(0.0, mut_top - _CANON_Y))
            + (40 * len(tracks) if tracks else 0),
            "margin": {"l": 110, "r": 30, "t": 70, "b": 50},
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
