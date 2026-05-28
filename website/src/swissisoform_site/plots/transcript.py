"""Build the transcript-coordinate Plotly figure for the V2 per-isoform page.

The figure has two panels (shared X axis):
- Top: transcript track (5'UTR + CDS + 3'UTR rectangles + TIS markers)
- Bottom: per-TIS bars across cell lines (log2 init. efficiency)

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


def _empty_figure(caption: str) -> dict[str, Any]:
    return {
        "data": [],
        "layout": {
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
        },
    }


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

    traces: list[dict[str, Any]] = []

    # ── Transcript track (top panel) ──
    exons = list(getattr(skeleton, "exons", []) or [])
    if not exons:
        return _empty_figure("Transcript skeleton has no exons.")

    # One scatter line for the transcript body (intron line + exon boxes)
    transcript_start = min(s for s, _ in exons)
    transcript_end = max(e for _, e in exons)
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

    # Exon rectangles (drawn as scatter fill polygons for portability)
    cds_start = getattr(skeleton, "cds_start", None)
    cds_end = getattr(skeleton, "cds_end", None)
    for s, e in exons:
        in_cds_start = max(s, cds_start) if cds_start is not None else s
        in_cds_end = min(e, cds_end) if cds_end is not None else e
        if in_cds_start < in_cds_end:
            traces.append(
                _exon_rect(in_cds_start, in_cds_end, height=0.5, fill=_CDS_FILL, name="CDS")
            )
        if s < (cds_start or s):
            traces.append(_exon_rect(s, cds_start, height=0.3, fill="#cccccc", name="5'UTR"))
        if (cds_end or e) < e:
            traces.append(_exon_rect(cds_end, e, height=0.3, fill="#cccccc", name="3'UTR"))

    # ── TIS markers ──
    focal_tis_id = getattr(isoform, "focal_tis_id", None) or getattr(isoform, "tis_id", None)
    for entry in getattr(isoform, "all_tis_on_transcript", []) or []:
        tis_id = entry["tis_id"]
        pos = entry["genomic_pos"]
        is_focal = tis_id == focal_tis_id
        traces.append(
            {
                "type": "scatter",
                "name": f"TIS: {tis_id}{' (focal)' if is_focal else ''}",
                "mode": "markers",
                "x": [pos],
                "y": [1.4],
                "marker": {
                    "size": 14 if is_focal else 9,
                    "color": _FOCAL_COLOR if is_focal else _OTHER_COLOR,
                    "symbol": "triangle-down",
                    "line": {"width": 1, "color": "black"},
                },
                "hovertext": [f"{tis_id}<br>{entry.get('orf_type', '')}<br>pos={pos}"],
                "hoverinfo": "text",
                "yaxis": "y2",
                "showlegend": False,
            }
        )

    # ── Cell-line bars (bottom panel) ──
    cell_line_bars = getattr(isoform, "cell_line_bars", {}) or {}
    samples = sorted({s for v in cell_line_bars.values() for s in v})
    for sample in samples:
        xs: list[float] = []
        ys: list[float] = []
        for entry in getattr(isoform, "all_tis_on_transcript", []) or []:
            pos = entry["genomic_pos"]
            val = cell_line_bars.get(entry["tis_id"], {}).get(sample)
            if val is None:
                continue
            xs.append(pos)
            ys.append(float(val))
        if not xs:
            continue
        traces.append(
            {
                "type": "bar",
                "name": sample,
                "x": xs,
                "y": ys,
                "yaxis": "y",
            }
        )

    return {
        "data": traces,
        "layout": {
            "title": {"text": f"Transcript: {getattr(skeleton, 'transcript_id', '')}"},
            "xaxis": {"title": "Genomic position (nt)"},
            "yaxis": {
                "title": "log2 init. efficiency",
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
            "height": 380,
            "margin": {"l": 60, "r": 20, "t": 50, "b": 50},
            "showlegend": True,
        },
    }


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
