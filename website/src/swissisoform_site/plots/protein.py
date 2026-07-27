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
_DISORDER_FILL = "rgba(148, 163, 184, 0.55)"  # slate — intrinsically disordered
_COIL_COLOR = "#ea580c"  # orange — coiled-coil
_MOTIF_COLOR = "#9467bd"

_FONT_FAMILY = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
)
_HOVER_FONT_FAMILY = "system-ui, sans-serif"

_CANON_Y = 1.0
_ISO_Y = 0.5

# Rough number of 9px label characters that span the full residue axis, used to
# fit domain names to their box width. Approximate (plot width is responsive);
# tuned to favour showing full names on reasonably wide boxes.
_AXIS_CHARS = 140

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


def build_protein_figure(
    isoform: Any, overlays: dict[str, bool], collapse_domains: bool = False
) -> dict[str, Any]:
    """Return a Plotly figure dict for the protein view.

    Args:
        collapse_domains: When True, render the compact merged-region domain
            summary (``isoform.domains_merged``) instead of the full per-entry
            stack — the collapsed state of the page's domain toggle.
    """
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

    # ── Feature tracks below the bars, laid out top-down with a descending
    #    y-cursor so each lane clears the one above it: InterPro domains (packed
    #    into rows), disorder, coiled-coil, motifs, then per-cell-line
    #    initiation. Feature x-coords use ``feat_offset`` (isoform → display). ──
    row_h, row_gap = 0.30, 0.06
    y_cur = -0.15

    def _left_label(text: str, y: float) -> None:
        annotations.append(
            {
                "x": x_left,
                "y": y,
                "xref": "x",
                "yref": "y",
                "xanchor": "right",
                "xshift": -6,
                "text": text,
                "showarrow": False,
                "font": {"size": 10, "color": "#475569"},
            }
        )

    show_domains = overlays.get("domains", True)
    if show_domains and collapse_domains:
        # Collapsed = one green bar on a single lane whose opacity tracks how
        # many domain signatures overlap at each position — the original density
        # look, no per-entry rows or inline labels.
        segs = list(getattr(isoform, "domain_segments", []) or [])
        if segs:
            max_depth = max(s["depth"] for s in segs) or 1
            top, bot = y_cur, y_cur - row_h
            legend_shown = False
            for s in segs:
                alpha = 0.20 + 0.65 * (s["depth"] / max_depth)
                fill = f"rgba(44, 160, 44, {alpha:.2f})"
                hover = (
                    f"<b>{s['depth']} overlapping domain signature"
                    f"{'s' if s['depth'] != 1 else ''}</b>"
                    f"<br>residues {int(s['start'])}–{int(s['end'])}"
                )
                traces.append(
                    _feature_box(
                        s["start"] + feat_offset, s["end"] + feat_offset, top, bot,
                        fill, "Domain (InterPro)", "domain", hover, not legend_shown,
                    )
                )
                legend_shown = True
            _left_label("domains (InterPro)", (top + bot) / 2)
            y_cur = bot - 0.18
    elif show_domains:
        # Expanded — greedy row-packing so non-overlapping domains share a row
        # and overlapping ones stack instead of smearing into one block.
        domains = list(getattr(isoform, "domains", []) or [])
        if domains:
            row_ends: list[float] = []
            for d in domains:
                d["_x0"] = d["start"] + feat_offset
                d["_x1"] = d["end"] + feat_offset
                ri = next((i for i, rend in enumerate(row_ends) if d["_x0"] > rend + 1), None)
                if ri is None:
                    ri = len(row_ends)
                    row_ends.append(d["_x1"])
                else:
                    row_ends[ri] = d["_x1"]
                d["_row"] = ri
            n_rows = len(row_ends)
            legend_shown = False
            for d in domains:
                top = y_cur - d["_row"] * (row_h + row_gap)
                bot = top - row_h
                hover = f"<b>{d['name']}</b>"
                if d.get("interpro_id"):
                    hover += f"<br>{d['interpro_id']}"
                hover += f"<br>residues {int(d['start'])}–{int(d['end'])}"
                if d.get("dbs"):
                    hover += f"<br>{d.get('n_sig', len(d['dbs']))} signatures: {', '.join(d['dbs'])}"
                traces.append(
                    _feature_box(
                        d["_x0"], d["_x1"], top, bot, _DOMAIN_FILL,
                        "Domain (InterPro)", "domain", hover, not legend_shown,
                    )
                )
                legend_shown = True
                # Label only named InterPro entries; raw member-DB accessions
                # ("G3DSA:…", "cd18654", "PTHR…") stay unlabeled, named on hover.
                # Size the label to the box's own width — full name when it fits,
                # ellipsize only when the box is genuinely too narrow.
                name = d.get("name") or ""
                if d.get("interpro_id"):
                    chars_fit = int((d["_x1"] - d["_x0"]) / max(axis_len, 1) * _AXIS_CHARS)
                    if chars_fit >= 3:
                        label = (
                            name if len(name) <= chars_fit else name[: max(1, chars_fit - 1)] + "…"
                        )
                        annotations.append(
                            {
                                "x": d["_x0"],
                                "y": top - row_h / 2,
                                "xanchor": "left",
                                "xshift": 3,
                                "text": label,
                                "showarrow": False,
                                "font": {"size": 9, "color": "#14532d"},
                            }
                        )
            band_bottom = y_cur - (n_rows - 1) * (row_h + row_gap) - row_h
            _left_label("domains (InterPro)", (y_cur + band_bottom) / 2)
            y_cur = band_bottom - 0.18

    # Disordered regions (MobiDB-lite) — their own slate lane.
    disorder = list(getattr(isoform, "disorder", []) or []) if overlays.get("disorder", True) else []
    if disorder:
        top, bot = y_cur, y_cur - row_h * 0.7
        legend_shown = False
        for seg in disorder:
            traces.append(
                _feature_box(
                    seg["start"] + feat_offset, seg["end"] + feat_offset, top, bot,
                    _DISORDER_FILL, "Disordered (MobiDB)", "disorder",
                    f"<b>Disordered region</b><br>residues {int(seg['start'])}–{int(seg['end'])}",
                    not legend_shown,
                )
            )
            legend_shown = True
        _left_label("disorder (MobiDB-lite)", (top + bot) / 2)
        y_cur = bot - 0.18

    # Coiled-coil regions (COILS) — own orange lane.
    coils = list(getattr(isoform, "coiled_coil", []) or []) if overlays.get("coiled_coil", True) else []
    if coils:
        cy = y_cur - row_h * 0.35
        legend_shown = False
        for seg in coils:
            cxs = _bar_samples(seg["start"] + feat_offset, seg["end"] + feat_offset)
            chov = f"<b>Coiled-coil</b><br>residues {int(seg['start'])}–{int(seg['end'])}"
            traces.append(
                {
                    "type": "scatter",
                    "name": "Coiled-coil",
                    "mode": "lines",
                    "x": cxs,
                    "y": [cy] * len(cxs),
                    "line": {"color": _COIL_COLOR, "width": 8},
                    "legendgroup": "coil",
                    "hovertext": [chov] * len(cxs),
                    "hoverinfo": "text",
                    "showlegend": not legend_shown,
                }
            )
            legend_shown = True
        _left_label("coiled-coil (COILS)", cy)
        y_cur = cy - row_h * 0.35 - 0.18

    # ── Motifs (spans), one legend entry ──
    motifs = list(getattr(isoform, "motifs", []) or []) if overlays.get("motifs", True) else []
    if motifs:
        motif_y = y_cur - row_h * 0.35
        motif_legend_shown = False
        for m in motifs:
            end = m.get("end", m.get("start"))
            label = m.get("name", "?")
            hover = f"{label}" + (f"<br>{m['match']}" if m.get("match") else "")
            mxs = _bar_samples(m["start"] + feat_offset, end + feat_offset)
            traces.append(
                {
                    "type": "scatter",
                    "name": "Motif",
                    "mode": "lines",
                    "x": mxs,
                    "y": [motif_y] * len(mxs),
                    "line": {"color": _MOTIF_COLOR, "width": 6},
                    "legendgroup": "motif",
                    "hovertext": [hover] * len(mxs),
                    "hoverinfo": "text",
                    "showlegend": not motif_legend_shown,
                }
            )
            motif_legend_shown = True
        _left_label("motifs (ELM)", motif_y)
        y_cur = motif_y - row_h * 0.35 - 0.18

    # ── Per-cell-line initiation lanes (TIS mapped onto the residue axis) ──
    tracks = cl_tracks
    canon_residue = getattr(isoform, "canon_residue", None)
    y_bottom = y_cur - 0.1
    if tracks:
        lane_base, lane_gap = y_cur - 0.2, 0.45
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

    return {
        "data": traces,
        "layout": {
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
            # Height tracks the full vertical span, which now grows with the
            # number of stacked feature rows (domains/disorder/coil) + lanes.
            "height": int(150 + 150 * ((mut_top + 0.3) - (y_bottom - 0.2))),
            "margin": {"l": 110, "r": 30, "t": 40, "b": 50},
        },
    }


def build_gene_protein_figure(view: Any, collapse_domains: bool = False) -> dict[str, Any]:
    """Combined gene view: one canonical bar + one bar per isoform, residue axis.

    Reproduces the per-isoform ``build_protein_figure`` layout for a whole gene:
    a single canonical bar (residues ``1..canonical_len``) with every isoform
    aligned on the shared region beneath it, deduplicated variant rows on top,
    and a deduplicated domain band + disorder / coiled-coil / motif tracks +
    per-cell-line initiation lanes below. All coordinates are in the canonical
    residue frame; the adapter has already mapped each isoform's features into
    it (see ``app._make_gene_protein_view``).

    Args:
        view: duck-typed adapter with ``canonical_len``, ``bars`` (per isoform:
            ``label``/``x0``/``x1``/``is_trunc``/``diff_x0``/``diff_x1``/
            ``diff_on_canonical``), ``variants`` (``variant_id``/``pos``/
            ``consequence``/``significance``/``hgvsp``/``source``/``in_unique``),
            ``domains`` (``name``/``interpro_id``/``x0``/``x1``/``isoforms``),
            ``disorder``/``coiled_coil`` (``x0``/``x1``), ``motifs`` (``name``/
            ``x0``/``x1``), ``cell_lines`` (``sample`` + ``marks`` of
            ``residue``/``log2_ie``/``label``), and ``x_left``.
        collapse_domains: when True, merge all domains onto a single lane
            (compact) instead of the greedy per-entry row stack.

    Returns:
        Plotly figure dict.
    """
    can_len = int(getattr(view, "canonical_len", 0) or 0)
    bars = list(getattr(view, "bars", []) or [])
    if can_len <= 0 or not bars:
        return {
            "data": [],
            "layout": {
                "font": {"family": _FONT_FAMILY, "size": 13, "color": "#1f2937"},
                "annotations": [
                    {
                        "text": "No protein length available for this gene.",
                        "showarrow": False, "x": 0.5, "y": 0.5,
                        "xref": "paper", "yref": "paper",
                        "font": {"size": 14, "color": "#666"},
                    }
                ],
            },
        }

    traces: list[dict[str, Any]] = []
    shapes: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []

    x_left = float(getattr(view, "x_left", 0.0) or 0.0)
    # Canonical is drawn at [0, can_len-1] (residue 1 → x=0), so the axis tops out
    # at can_len-1 (or a longer uORF/altORF bar).
    axis_len = max([can_len - 1] + [int(b["x1"]) for b in bars])
    # Pad the left edge so the leftmost initiation dots / extension starts (drawn
    # at x_left) aren't clipped, and anchor the row labels at the padded edge.
    x_left -= max(6, int(0.03 * (axis_len - x_left + 1)))

    def _left_label(text: str, y: float, size: int = 10) -> None:
        annotations.append(
            {
                "x": x_left, "y": y, "xref": "x", "yref": "y",
                "xanchor": "right", "xshift": -6, "text": text,
                "showarrow": False, "font": {"size": size, "color": "#475569"},
            }
        )

    # ── Bars: canonical on top (y=0), isoforms stacked below ──
    bar_gap = 0.34
    canon_y = 0.0
    traces.append(_length_bar(0, can_len - 1, canon_y, _CANON_COLOR, "Canonical"))
    _left_label("canonical", canon_y)
    for i, b in enumerate(bars):
        by = -bar_gap * (i + 1)
        b["_y"] = by
        bar_trace = _length_bar(int(b["x0"]), int(b["x1"]), by, _ISO_COLOR, b["label"])
        # Carry the isoform slug so a click on the bar navigates to its page.
        if b.get("slug"):
            bar_trace["customdata"] = [b["slug"]] * len(bar_trace["x"])
        traces.append(bar_trace)
        _left_label(b["label"], by, size=9)
    lowest_bar = -bar_gap * len(bars)

    # ── Differential-region shading (extension → isoform bar, green; truncation
    #    → canonical bar, red; uORF/altORF → whole isoform bar, green) ──
    legend_ext = legend_trunc = False
    for b in bars:
        d0, d1 = b.get("diff_x0"), b.get("diff_x1")
        if d0 is None or d1 is None or d1 < d0:
            continue
        if b.get("diff_on_canonical"):
            ov = _diff_overlay(int(d0), int(d1), canon_y, _TRUNC_FILL)
            ov["showlegend"] = not legend_trunc
            ov["name"] = "Lost region (truncation)"
            legend_trunc = True
        else:
            ov = _diff_overlay(int(d0), int(d1), b["_y"], _EXT_FILL)
            ov["showlegend"] = not legend_ext
            ov["name"] = "Differential region"
            legend_ext = True
            # Extension shading sits on the isoform's own row → clickable too.
            if b.get("slug"):
                ov["customdata"] = [b["slug"]] * len(ov["x"])
        traces.append(ov)

    # ── Variant rows ABOVE the canonical bar, one row per consequence ──
    mut_base = canon_y + 0.3
    mut_top = canon_y
    variants = list(getattr(view, "variants", []) or [])
    by_conseq: dict[str, list] = {}
    for v in variants:
        if v.get("pos") is None:
            continue
        by_conseq.setdefault(v.get("consequence") or "other", []).append(v)
    conseqs = sorted(
        by_conseq, key=lambda c: (_CONSEQ_ORDER.index(c) if c in _CONSEQ_ORDER else 99, c)
    )
    for i, c in enumerate(conseqs):
        ty = mut_base + i * 0.26
        xs, cols, sizes, opac, hover = [], [], [], [], []
        for v in by_conseq[c]:
            is_path = "pathogenic" in (v.get("significance") or "").lower()
            in_unique = bool(v.get("in_unique"))
            xs.append(v["pos"])
            cols.append("#d62728" if is_path else _CONSEQ_COLOR.get(c, "#94a3b8"))
            sizes.append((9 if is_path else 6) if in_unique else (7 if is_path else 4))
            opac.append(1.0 if in_unique else 0.4)
            hover.append(
                f"{v.get('variant_id', '?')}<br>{v.get('hgvsp') or ''}<br>"
                f"{c} · {'unique' if in_unique else 'shared'} region<br>"
                f"{v.get('significance') or '—'} · {v.get('source') or ''}"
            )
        traces.append(
            {
                "type": "scatter", "mode": "markers", "x": xs, "y": [ty] * len(xs),
                "marker": {"size": sizes, "color": cols, "opacity": opac, "symbol": "circle"},
                "hovertext": hover, "hoverinfo": "text", "showlegend": False,
            }
        )
        _left_label(_CONSEQ_SHORT.get(c, c)[:14], ty, size=9)
    if conseqs:
        mut_top = mut_base + (len(conseqs) - 1) * 0.26 + 0.22

    # ── Feature tracks BELOW the bars: domains, disorder, coil, motifs ──
    row_h, row_gap = 0.28, 0.05
    y_cur = lowest_bar - 0.24

    domains = list(getattr(view, "domains", []) or [])
    segments = list(getattr(view, "domain_segments", []) or [])
    if domains and collapse_domains and segments:
        # Compact: one lane whose shading tracks how many distinct domains
        # overlap each stretch (hover shows the count), so a many-domain gene
        # stays short but the overlap density is still legible.
        max_depth = max(s["depth"] for s in segments) or 1
        top, bot = y_cur, y_cur - row_h
        legend_shown = False
        for s in segments:
            depth = int(s["depth"])
            alpha = 0.20 + 0.65 * (depth / max_depth)
            fill = f"rgba(44, 160, 44, {alpha:.2f})"
            hover = (
                f"<b>{depth} overlapping domain{'' if depth == 1 else 's'}</b>"
                f"<br>frame residues {int(s['x0'])}–{int(s['x1'])}"
            )
            traces.append(
                _feature_box(s["x0"], s["x1"], top, bot, fill,
                             "Domain (InterPro)", "domain", hover, not legend_shown)
            )
            legend_shown = True
        _left_label("domains (InterPro)", (top + bot) / 2)
        y_cur = bot - 0.13
    elif domains:
        row_ends: list[float] = []
        for d in domains:
            ri = next((i for i, rend in enumerate(row_ends) if d["x0"] > rend + 1), None)
            if ri is None:
                ri = len(row_ends)
                row_ends.append(d["x1"])
            else:
                row_ends[ri] = d["x1"]
            d["_row"] = ri
        n_rows = len(row_ends)
        legend_shown = False
        for d in domains:
            top = y_cur - d["_row"] * (row_h + row_gap)
            bot = top - row_h
            hover = f"<b>{d['name']}</b>"
            if d.get("interpro_id"):
                hover += f"<br>{d['interpro_id']}"
            hover += f"<br>frame residues {int(d['x0'])}–{int(d['x1'])}"
            if d.get("isoforms"):
                n_iso = len(d["isoforms"])
                hover += f"<br>in {n_iso} isoform" + ("" if n_iso == 1 else "s")
            traces.append(
                _feature_box(d["x0"], d["x1"], top, bot, _DOMAIN_FILL,
                             "Domain (InterPro)", "domain", hover, not legend_shown)
            )
            legend_shown = True
            name = d.get("name") or ""
            chars_fit = int((d["x1"] - d["x0"]) / max(axis_len, 1) * _AXIS_CHARS)
            if d.get("interpro_id") and chars_fit >= 3:
                label = name if len(name) <= chars_fit else name[: max(1, chars_fit - 1)] + "…"
                annotations.append(
                    {
                        "x": d["x0"], "y": top - row_h / 2, "xanchor": "left", "xshift": 3,
                        "text": label, "showarrow": False, "font": {"size": 9, "color": "#14532d"},
                    }
                )
        band_bottom = y_cur - (n_rows - 1) * (row_h + row_gap) - row_h
        _left_label("domains (InterPro)", (y_cur + band_bottom) / 2)
        y_cur = band_bottom - 0.13

    disorder = list(getattr(view, "disorder", []) or [])
    if disorder:
        top, bot = y_cur, y_cur - row_h * 0.7
        legend_shown = False
        for seg in disorder:
            hover = f"<b>Disordered region</b><br>frame residues {int(seg['x0'])}–{int(seg['x1'])}"
            traces.append(
                _feature_box(seg["x0"], seg["x1"], top, bot, _DISORDER_FILL,
                             "Disordered (MobiDB)", "disorder", hover, not legend_shown)
            )
            legend_shown = True
        _left_label("disorder (MobiDB-lite)", (top + bot) / 2)
        y_cur = bot - 0.13

    coils = list(getattr(view, "coiled_coil", []) or [])
    if coils:
        cy = y_cur - row_h * 0.35
        legend_shown = False
        for seg in coils:
            cxs = _bar_samples(int(seg["x0"]), int(seg["x1"]))
            chov = f"<b>Coiled-coil</b><br>frame residues {int(seg['x0'])}–{int(seg['x1'])}"
            traces.append(
                {
                    "type": "scatter", "name": "Coiled-coil", "mode": "lines",
                    "x": cxs, "y": [cy] * len(cxs),
                    "line": {"color": _COIL_COLOR, "width": 8}, "legendgroup": "coil",
                    "hovertext": [chov] * len(cxs),
                    "hoverinfo": "text", "showlegend": not legend_shown,
                }
            )
            legend_shown = True
        _left_label("coiled-coil (COILS)", cy)
        y_cur = cy - row_h * 0.35 - 0.13

    motifs = list(getattr(view, "motifs", []) or [])
    if motifs:
        motif_y = y_cur - row_h * 0.35
        legend_shown = False
        for m in motifs:
            mxs = _bar_samples(int(m["x0"]), int(m["x1"]))
            traces.append(
                {
                    "type": "scatter", "name": "Motif", "mode": "lines",
                    "x": mxs, "y": [motif_y] * len(mxs),
                    "line": {"color": _MOTIF_COLOR, "width": 6}, "legendgroup": "motif",
                    "hovertext": [f"{m.get('name', '?')}"] * len(mxs),
                    "hoverinfo": "text", "showlegend": not legend_shown,
                }
            )
            legend_shown = True
        _left_label("motifs (ELM)", motif_y)
        y_cur = motif_y - row_h * 0.35 - 0.13

    # ── Per-cell-line initiation lanes (each isoform's start, dot sized by IE) ──
    y_bottom = y_cur - 0.1
    cell_lines = list(getattr(view, "cell_lines", []) or [])
    if cell_lines:
        lane_base, lane_gap = y_cur - 0.18, 0.32
        all_ie = [m["log2_ie"] for t in cell_lines for m in t["marks"]]
        vmin, vmax = (min(all_ie), max(all_ie)) if all_ie else (-1.0, 0.0)

        def _ie_size(v: float) -> float:
            return 12.0 if vmax <= vmin else 7.0 + (v - vmin) / (vmax - vmin) * 15.0

        for i, t in enumerate(cell_lines):
            ly = lane_base - i * lane_gap
            xs, sizes, hover = [], [], []
            for m in t["marks"]:
                xs.append(m["residue"])
                sizes.append(_ie_size(m["log2_ie"]))
                try:
                    ie = f"{2 ** m['log2_ie']:.3g}"
                except (OverflowError, ValueError):
                    ie = "n/a"
                hover.append(f"<b>{t['sample']}</b><br>{m.get('label', '')}<br>IE {ie}")
            traces.append(
                {
                    "type": "scatter", "mode": "markers", "x": xs, "y": [ly] * len(xs),
                    "marker": {"size": sizes, "color": _ISO_COLOR, "opacity": 0.85,
                               "line": {"width": 1, "color": "#fff"}},
                    "hovertext": hover, "hoverinfo": "text", "showlegend": False,
                }
            )
            _left_label(t["sample"].replace("_", " "), ly)
        y_bottom = lane_base - (len(cell_lines) - 1) * lane_gap - 0.3

    # A single black dotted guide at the SHORTEST truncation's differential-region
    # end — the deepest N-terminal cut, where its lost region meets the retained
    # body. (Extensions' differential ends at x=0, marked by the axis zeroline.)
    guide_top = max(mut_top, canon_y + 0.2)
    trunc_ends = [
        b["diff_x1"] for b in bars if b.get("diff_on_canonical") and b.get("diff_x1") is not None
    ]
    if trunc_ends:
        x_guide = max(trunc_ends)  # largest x0 = fewest residues retained = shortest protein
        shapes.append(
            {
                "type": "line", "x0": x_guide, "x1": x_guide, "y0": y_bottom, "y1": guide_top,
                "line": {"color": "#111827", "width": 1, "dash": "dot"},
            }
        )

    y_hi = max(mut_top, canon_y + 0.2) + 0.3
    y_lo = y_bottom - 0.2
    return {
        "data": traces,
        "layout": {
            "font": {"family": _FONT_FAMILY, "size": 13, "color": "#1f2937"},
            "hoverlabel": {"font": {"family": _HOVER_FONT_FAMILY, "size": 12}},
            "showlegend": True,
            "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02,
                       "xanchor": "left", "x": 0, "font": {"size": 11}},
            "shapes": shapes,
            "annotations": annotations,
            "xaxis": {
                "title": {"text": "Protein residue (0 = canonical start)",
                          "font": {"size": 12, "color": "#4b5563"}},
                "tickfont": {"size": 11, "color": "#6b7280"},
                "range": [x_left, axis_len + 5],
                "zeroline": True,  # solid line at x=0 marks the canonical start
            },
            "yaxis": {"range": [y_lo, y_hi], "showticklabels": False, "zeroline": False},
            "height": int(90 + 118 * (y_hi - y_lo)),
            "margin": {"l": 120, "r": 30, "t": 40, "b": 50},
        },
    }


def _feature_box(
    x0: float, x1: float, top: float, bot: float, fill: str, name: str, group: str,
    hover: str, show_legend: bool,
) -> dict[str, Any]:
    """A filled rectangle trace (domain / disorder box) with a hoverable interior."""
    return {
        "type": "scatter",
        "name": name,
        "mode": "lines",
        "fill": "toself",
        "x": [x0, x1, x1, x0, x0],
        "y": [bot, bot, top, top, bot],
        "line": {"color": "rgba(0,0,0,0)"},
        "fillcolor": fill,
        "legendgroup": group,
        # ``hoveron: fills`` makes the whole box interior hoverable (a filled
        # shape otherwise only reacts at its corner vertices), so short boxes
        # still surface their name/accession on hover.
        "text": hover,
        "hoveron": "fills",
        "hoverinfo": "text",
        "showlegend": show_legend,
    }


def _bar_samples(x0: int, x1: int, cap: int = 400) -> list[float]:
    """Evenly sample display coords across ``[x0, x1]`` (endpoints included).

    Plotly fires line-trace hover only at data vertices, so a 2-point bar has no
    hover in its interior. Densifying into collinear points keeps the visual
    identical (a straight line) while making the whole bar hoverable.
    """
    lo, hi = (x0, x1) if x0 <= x1 else (x1, x0)
    n = int(hi - lo)
    if n <= 1:
        return [float(lo), float(hi)]
    step = max(1, n // cap)
    xs = [float(v) for v in range(int(lo), int(hi) + 1, step)]
    if xs[-1] != float(hi):
        xs.append(float(hi))
    return xs


def _length_bar(x0: int, x1: int, y: float, color: str, name: str) -> dict[str, Any]:
    xs = _bar_samples(x0, x1)
    length = int(abs(x1 - x0)) + 1
    base = min(x0, x1)
    hover = [f"<b>{name}</b><br>residue {int(x - base) + 1} / {length} aa" for x in xs]
    return {
        "type": "scatter",
        "name": name,
        "mode": "lines",
        "x": xs,
        "y": [y] * len(xs),
        "line": {"color": color, "width": 12},
        "hovertext": hover,
        "hoverinfo": "text",
        "showlegend": True,
    }


def _diff_overlay(x0: int, x1: int, y: float, color: str) -> dict[str, Any]:
    xs = _bar_samples(x0, x1)
    length = int(abs(x1 - x0)) + 1
    base = min(x0, x1)
    hover = [
        f"<b>Differential region</b><br>residue {int(x - base) + 1} / {length} aa" for x in xs
    ]
    return {
        "type": "scatter",
        "name": "Differential region",
        "mode": "lines",
        "x": xs,
        "y": [y] * len(xs),
        "line": {"color": color, "width": 12},
        "hovertext": hover,
        "hoverinfo": "text",
        "showlegend": True,
    }
