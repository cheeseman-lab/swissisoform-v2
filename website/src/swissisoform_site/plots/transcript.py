"""Build the IGV-style per-cell-line initiation figure for the V2 isoform page.

An IGV-like region-of-interest view of the translation start cluster: genomic
position on x (zoomed to the TIS/canonical-start neighbourhood), one horizontal
lane per cell line on y, each TIS drawn as a dot at its genomic position sized by
initiation efficiency. A reference band marks the canonical start codon and the
differential region. We have per-TIS initiation values (not per-base coverage),
so these are peaks, not read pileups.

Inputs are duck-typed; production callers pass ``swissisoform_site.data.Isoform``
adapters + ``TranscriptSkeleton``. Tests use ``types.SimpleNamespace``.
"""

from __future__ import annotations

from typing import Any

_FOCAL_COLOR = "#d62728"
_OTHER_COLOR = "#1f77b4"
_DIFF_FILL = "rgba(214, 39, 40, 0.10)"
_CANON_LINE = "#111827"

_FONT_FAMILY = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
)
_HOVER_FONT_FAMILY = "system-ui, sans-serif"

_SAMPLE_ORDER = ["HeLa", "K562", "U2OS", "RPE1_Async", "RPE1_Que", "RPE1_Sen"]


def _base_layout(title: str) -> dict[str, Any]:
    return {
        "title": {"text": title, "font": {"size": 15, "color": "#111827"}},
        "font": {"family": _FONT_FAMILY, "size": 13, "color": "#1f2937"},
        "hoverlabel": {"font": {"family": _HOVER_FONT_FAMILY, "size": 12}},
        "margin": {"l": 90, "r": 30, "t": 60, "b": 50},
        "showlegend": True,
    }


def _empty_figure(caption: str) -> dict[str, Any]:
    layout = _base_layout("")
    layout.update(
        {
            "annotations": [
                {
                    "text": caption,
                    "showarrow": False,
                    "x": 0.5,
                    "y": 0.5,
                    "xref": "paper",
                    "yref": "paper",
                    "font": {"size": 14, "color": "#666"},
                }
            ],
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "height": 180,
        }
    )
    return {"data": [], "layout": layout}


def _codon(tis_id: str) -> str:
    parts = tis_id.split(":")
    return parts[3] if len(parts) >= 4 else "?"


def _size(v: float, vmin: float, vmax: float) -> float:
    """Map a log2 initiation-efficiency value to a marker diameter (8–30 px)."""
    if vmax <= vmin:
        return 18.0
    frac = (v - vmin) / (vmax - vmin)
    return 8.0 + frac * 22.0


def build_transcript_figure(
    isoform: Any, skeleton: Any | None, overlays: dict[str, bool]
) -> dict[str, Any]:
    """Return a Plotly figure dict for the IGV-style initiation view.

    Args:
        isoform: Object with ``focal_tis_id``, ``all_tis_on_transcript``
            (list of {tis_id, genomic_pos, orf_type}) and ``cell_line_bars``
            ({tis_id: {sample: log2 initiation efficiency}}).
        skeleton: Object with ``cds_start``/``cds_end``/``strand`` — used for the
            canonical-start reference marker. None → still renders (no reference).
        overlays: Unused; reserved.
    """
    cell_line_bars = getattr(isoform, "cell_line_bars", {}) or {}
    all_tis = list(getattr(isoform, "all_tis_on_transcript", []) or [])
    focal_tis_id = getattr(isoform, "focal_tis_id", None) or getattr(isoform, "tis_id", None)

    samples = [s for s in _SAMPLE_ORDER if any(s in v for v in cell_line_bars.values())]
    samples += sorted({s for v in cell_line_bars.values() for s in v} - set(samples))

    pos_by_tis = {
        t["tis_id"]: t["genomic_pos"] for t in all_tis if t.get("genomic_pos") is not None
    }
    meta = {t["tis_id"]: t for t in all_tis}

    if not samples or not cell_line_bars or not pos_by_tis:
        return _empty_figure("No per-cell-line initiation data for this isoform.")

    # Canonical start codon (genomic), strand-aware.
    cds_start = getattr(skeleton, "cds_start", None)
    cds_end = getattr(skeleton, "cds_end", None)
    strand = getattr(skeleton, "strand", "+")
    canon_start = cds_start if strand == "+" else cds_end

    # Region of interest: span the TIS cluster + canonical start, with padding.
    anchors = list(pos_by_tis.values())
    if canon_start is not None:
        anchors.append(canon_start)
    lo, hi = min(anchors), max(anchors)
    pad = max(60, int((hi - lo) * 0.18))
    x0, x1 = lo - pad, hi + pad

    all_vals = [v for d in cell_line_bars.values() for v in d.values()]
    vmin, vmax = (min(all_vals), max(all_vals)) if all_vals else (-1.0, 0.0)

    lane_y = {s: i for i, s in enumerate(samples)}

    # TIS dots, one trace per TIS (legend = TIS), placed in each cell-line lane.
    tis_ids = sorted(
        pos_by_tis.keys(),
        key=lambda tid: (tid != focal_tis_id, pos_by_tis[tid]),
    )
    traces: list[dict[str, Any]] = []
    for tid in tis_ids:
        is_focal = tid == focal_tis_id
        orf_type = meta.get(tid, {}).get("orf_type", "")
        xs: list[float] = []
        ys: list[float] = []
        sizes: list[float] = []
        hover: list[str] = []
        for s in samples:
            v = cell_line_bars.get(tid, {}).get(s)
            if v is None:
                continue
            xs.append(pos_by_tis[tid])
            ys.append(lane_y[s])
            sizes.append(_size(float(v), vmin, vmax))
            try:
                ie = f"{2 ** float(v):.3g}"
            except (OverflowError, ValueError):
                ie = "n/a"
            hover.append(
                f"<b>{s}</b><br>{orf_type} · {_codon(tid)}<br>"
                f"@ {pos_by_tis[tid]:,}<br>log2(IE): {float(v):.2f} · IE: {ie}"
            )
        if not xs:
            continue
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "name": f"{orf_type} · {_codon(tid)}{' (focal)' if is_focal else ''}",
                "x": xs,
                "y": ys,
                "marker": {
                    "size": sizes,
                    "color": _FOCAL_COLOR if is_focal else _OTHER_COLOR,
                    "line": {"width": 1, "color": "#fff"},
                    "opacity": 0.9,
                },
                "hovertext": hover,
                "hoverinfo": "text",
            }
        )

    shapes: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    y_top = len(samples) - 0.5

    # Differential region: between the focal TIS and the canonical start.
    focal_pos = pos_by_tis.get(focal_tis_id)
    if focal_pos is not None and canon_start is not None:
        d0, d1 = sorted((focal_pos, canon_start))
        shapes.append(
            {
                "type": "rect",
                "x0": d0,
                "x1": d1,
                "y0": -0.5,
                "y1": y_top,
                "fillcolor": _DIFF_FILL,
                "line": {"width": 0},
                "layer": "below",
            }
        )
        annotations.append(
            {
                "x": (d0 + d1) / 2,
                "y": y_top,
                "yshift": 12,
                "text": "differential region",
                "showarrow": False,
                "font": {"size": 10, "color": "#b91c1c"},
            }
        )

    # Canonical start codon reference line.
    if canon_start is not None:
        shapes.append(
            {
                "type": "line",
                "x0": canon_start,
                "x1": canon_start,
                "y0": -0.5,
                "y1": y_top,
                "line": {"color": _CANON_LINE, "width": 1.5, "dash": "dot"},
            }
        )
        annotations.append(
            {
                "x": canon_start,
                "y": y_top,
                "yshift": 12,
                "text": "canonical start",
                "showarrow": False,
                "font": {"size": 10, "color": _CANON_LINE},
            }
        )

    layout = _base_layout("Initiation across cell lines (region of interest)")
    layout.update(
        {
            "shapes": shapes,
            "annotations": annotations,
            "xaxis": {
                "title": {
                    "text": f"Genomic position — {getattr(skeleton, 'chrom', '')} ({strand})",
                    "font": {"size": 12, "color": "#4b5563"},
                },
                "tickfont": {"size": 11, "color": "#6b7280"},
                "range": [x0, x1],
            },
            "yaxis": {
                "tickvals": list(range(len(samples))),
                "ticktext": [s.replace("_", " ") for s in samples],
                "tickfont": {"size": 12, "color": "#374151"},
                "range": [-0.7, y_top + 0.6],
                "zeroline": False,
            },
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "x": 0,
                "font": {"size": 11},
            },
            "height": 120 + 42 * len(samples),
            "plot_bgcolor": "#fafafa",
        }
    )
    return {"data": traces, "layout": layout}
