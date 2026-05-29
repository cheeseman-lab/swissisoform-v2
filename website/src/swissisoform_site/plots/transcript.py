"""Build the transcript-coordinate Plotly figure for the V2 per-isoform page.

The figure has up to two panels (shared X axis):
- Top: transcript track (5'UTR + CDS + 3'UTR rectangles + TIS markers)
- Bottom (only if cell-line data exists): per-TIS bars across cell lines
  (log2 initiation efficiency)

Inputs are duck-typed records; production callers pass real dataclasses
(``swissisoform_site.data.Isoform`` + ``TranscriptSkeleton``). Tests use
``types.SimpleNamespace`` for fixture brevity.
"""

from __future__ import annotations

from typing import Any

_FOCAL_COLOR = "#d62728"
_OTHER_COLOR = "#7f7f7f"
_TRANSCRIPT_LINE_COLOR = "#222"
_CDS_FILL = "rgba(31, 119, 180, 0.45)"

_FONT_FAMILY = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
)
_HOVER_FONT_FAMILY = "system-ui, sans-serif"


def _base_layout(title: str) -> dict[str, Any]:
    """Common layout pieces (fonts, hover, margins) shared across modes."""
    return {
        "title": {"text": title, "font": {"size": 15, "color": "#111827"}},
        "font": {"family": _FONT_FAMILY, "size": 13, "color": "#1f2937"},
        "hoverlabel": {"font": {"family": _HOVER_FONT_FAMILY, "size": 12}},
        "margin": {"l": 70, "r": 30, "t": 60, "b": 50},
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
            "height": 200,
        }
    )
    return {"data": [], "layout": layout}


def build_transcript_figure(
    isoform: Any, skeleton: Any | None, overlays: dict[str, bool]
) -> dict[str, Any]:
    """Return a Plotly figure dict for the transcript view.

    Args:
        isoform: Object with ``focal_tis_id``, ``all_tis_on_transcript``
            (list of {tis_id, genomic_pos, orf_type}), and ``cell_line_bars``
            ({tis_id: {sample: log2_value}}).
        skeleton: Object with ``exons`` (list of (start, end) tuples),
            ``cds_start``, ``cds_end``, ``strand``. None → empty figure.
        overlays: Currently unused; reserved for later variant/domain overlays.

    Returns:
        A Plotly figure dict (``{"data": [...], "layout": {...}}``).
    """
    if skeleton is None:
        return _empty_figure("Transcript skeleton not available for this isoform.")

    exons = list(getattr(skeleton, "exons", []) or [])
    if not exons:
        return _empty_figure("Transcript skeleton has no exons.")

    traces: list[dict[str, Any]] = []

    transcript_start = min(s for s, _ in exons)
    transcript_end = max(e for _, e in exons)

    # ── Transcript skeleton (intron line + exon rectangles) ──
    traces.append(
        {
            "type": "scatter",
            "name": "transcript",
            "mode": "lines",
            "x": [transcript_start, transcript_end],
            "y": [1, 1],
            "line": {"color": _TRANSCRIPT_LINE_COLOR, "width": 1},
            "hoverinfo": "skip",
            "showlegend": False,
            "yaxis": "y2",
        }
    )

    cds_start = getattr(skeleton, "cds_start", None)
    cds_end = getattr(skeleton, "cds_end", None)
    for s, e in exons:
        in_cds_start = max(s, cds_start) if cds_start is not None else s
        in_cds_end = min(e, cds_end) if cds_end is not None else e
        if in_cds_start < in_cds_end:
            traces.append(
                _exon_rect(in_cds_start, in_cds_end, height=0.7, fill=_CDS_FILL, name="CDS")
            )
        if s < (cds_start or s):
            traces.append(_exon_rect(s, cds_start, height=0.45, fill="#cccccc", name="5'UTR"))
        if (cds_end or e) < e:
            traces.append(_exon_rect(cds_end, e, height=0.45, fill="#cccccc", name="3'UTR"))

    # ── TIS markers ──
    focal_tis_id = getattr(isoform, "focal_tis_id", None) or getattr(isoform, "tis_id", None)
    all_tis = list(getattr(isoform, "all_tis_on_transcript", []) or [])
    for entry in all_tis:
        tis_id = entry["tis_id"]
        pos = entry["genomic_pos"]
        is_focal = tis_id == focal_tis_id
        orf_type = entry.get("orf_type", "")
        hover = (
            f"<b>{tis_id}</b><br>"
            f"ORF type: {orf_type}<br>"
            f"Genomic position: {pos:,}<br>"
            f"{'<i>(focal)</i>' if is_focal else ''}"
        )
        traces.append(
            {
                "type": "scatter",
                "name": f"TIS: {tis_id}{' (focal)' if is_focal else ''}",
                "mode": "markers",
                "x": [pos],
                "y": [1.4],
                "marker": {
                    "size": 18 if is_focal else 14,
                    "color": _FOCAL_COLOR if is_focal else _OTHER_COLOR,
                    "symbol": "triangle-down",
                    "line": {"width": 1.5, "color": "#111"},
                },
                "hovertext": [hover],
                "hoverinfo": "text",
                "yaxis": "y2",
                "showlegend": False,
            }
        )

    # ── Auto-zoom xaxis to the TIS cluster ──
    tis_positions = [t["genomic_pos"] for t in all_tis]
    if tis_positions:
        tis_min = min(tis_positions)
        tis_max = max(tis_positions)
        tis_span = max(tis_max - tis_min, 0)
        pad = max(500, int(tis_span * 0.6))
        x0 = tis_min - pad
        x1 = tis_max + pad
        if x1 - x0 < 1000:
            focal_pos = next(
                (t["genomic_pos"] for t in all_tis if t["tis_id"] == focal_tis_id),
                tis_min,
            )
            x0 = focal_pos - 500
            x1 = focal_pos + 500
        x0 = max(x0, transcript_start)
        x1 = min(x1, transcript_end)
        x_range = [x0, x1]
    else:
        x_range = [transcript_start, transcript_end]

    # ── Cell-line bars (bottom panel) ──
    cell_line_bars = getattr(isoform, "cell_line_bars", {}) or {}
    samples = sorted({s for v in cell_line_bars.values() for s in v})

    bar_traces: list[dict[str, Any]] = []
    for sample in samples:
        xs: list[float] = []
        ys: list[float] = []
        hover: list[str] = []
        for entry in all_tis:
            pos = entry["genomic_pos"]
            val = cell_line_bars.get(entry["tis_id"], {}).get(sample)
            if val is None:
                continue
            xs.append(pos)
            ys.append(float(val))
            try:
                raw_ie = 2 ** float(val)
                ie_text = f"{raw_ie:.3g}"
            except (OverflowError, ValueError):
                ie_text = "n/a"
            hover.append(
                f"<b>{sample}</b><br>"
                f"TIS: {entry['tis_id']}<br>"
                f"log2(IE): {float(val):.3f}<br>"
                f"IE: {ie_text}"
            )
        if not xs:
            continue
        bar_traces.append(
            {
                "type": "bar",
                "name": sample,
                "x": xs,
                "y": ys,
                "hovertext": hover,
                "hoverinfo": "text",
                "yaxis": "y",
            }
        )

    has_bars = bool(bar_traces)
    traces.extend(bar_traces)

    title = f"Transcript: {getattr(skeleton, 'transcript_id', '')}"
    layout = _base_layout(title)

    if has_bars:
        layout.update(
            {
                "xaxis": {
                    "title": {
                        "text": "Genomic position (nt)",
                        "font": {"size": 12, "color": "#4b5563"},
                    },
                    "tickfont": {"size": 11, "color": "#6b7280"},
                    "range": x_range,
                },
                "yaxis": {
                    "title": {
                        "text": "log2 initiation efficiency",
                        "font": {"size": 12, "color": "#4b5563"},
                    },
                    "tickfont": {"size": 11, "color": "#6b7280"},
                    "domain": [0.0, 0.65],
                    "zeroline": True,
                    "zerolinecolor": "#999",
                },
                "yaxis2": {
                    "domain": [0.7, 1.0],
                    "showticklabels": False,
                    "range": [0, 2],
                },
                "barmode": "group",
                "bargap": 0.25,
                "bargroupgap": 0.08,
                "height": 380,
            }
        )
    else:
        # Single-panel layout: just the transcript track + TIS markers.
        layout.update(
            {
                "xaxis": {
                    "title": {
                        "text": "Genomic position (nt)",
                        "font": {"size": 12, "color": "#4b5563"},
                    },
                    "tickfont": {"size": 11, "color": "#6b7280"},
                    "range": x_range,
                },
                "yaxis": {
                    "visible": False,
                    "range": [0, 2],
                },
                # Hide yaxis2 entirely so the transcript track owns the y-space.
                "yaxis2": {
                    "overlaying": "y",
                    "showticklabels": False,
                    "range": [0, 2],
                    "visible": False,
                },
                "height": 220,
            }
        )

    return {"data": traces, "layout": layout}


def _exon_rect(x0: int, x1: int, height: float, fill: str, name: str) -> dict[str, Any]:
    return {
        "type": "scatter",
        "name": name,
        "mode": "lines",
        "fill": "toself",
        "x": [x0, x1, x1, x0, x0],
        "y": [1 - height / 2, 1 - height / 2, 1 + height / 2, 1 + height / 2, 1 - height / 2],
        "line": {"color": "rgba(0,0,0,0)"},
        "fillcolor": fill,
        "yaxis": "y2",
        "hoverinfo": "skip",
        "showlegend": False,
    }
