"""Build the per-cell-line TIS-usage Plotly figure for the V2 isoform page.

The differential region's *position* relative to the canonical ORF is shown in
the protein track; splicing/exon structure is deliberately not the point here.
What this panel answers is: **how strongly is each alternative TIS initiated in
each cell line?** A categorical grouped-bar chart — cell lines on x, one bar
group per TIS on the transcript, log2 initiation efficiency on y, the focal TIS
emphasised.

Inputs are duck-typed records; production callers pass real dataclasses
(``swissisoform_site.data.Isoform``). Tests use ``types.SimpleNamespace``.
"""

from __future__ import annotations

from typing import Any

_FOCAL_COLOR = "#d62728"
_OTHER_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

_FONT_FAMILY = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
)
_HOVER_FONT_FAMILY = "system-ui, sans-serif"

# Display order for the cell-line categorical axis.
_SAMPLE_ORDER = ["HeLa", "K562", "U2OS", "RPE1_Async", "RPE1_Que", "RPE1_Sen"]


def _base_layout(title: str) -> dict[str, Any]:
    return {
        "title": {"text": title, "font": {"size": 15, "color": "#111827"}},
        "font": {"family": _FONT_FAMILY, "size": 13, "color": "#1f2937"},
        "hoverlabel": {"font": {"family": _HOVER_FONT_FAMILY, "size": 12}},
        "margin": {"l": 70, "r": 30, "t": 60, "b": 60},
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


def _tis_label(tis_id: str, orf_type: str, is_focal: bool) -> str:
    """Short legend label for a TIS, e.g. ``truncated · ATG (focal)``."""
    parts = tis_id.split(":")
    codon = parts[3] if len(parts) >= 4 else "?"
    label = f"{orf_type or 'TIS'} · {codon}"
    return f"{label} (focal)" if is_focal else label


def build_transcript_figure(
    isoform: Any, skeleton: Any | None, overlays: dict[str, bool]
) -> dict[str, Any]:
    """Return a Plotly figure dict for the per-cell-line TIS-usage view.

    Args:
        isoform: Object with ``focal_tis_id``, ``all_tis_on_transcript``
            (list of {tis_id, genomic_pos, orf_type}) and ``cell_line_bars``
            ({tis_id: {sample: log2 initiation efficiency}}).
        skeleton: Unused (kept for call-site compatibility — splicing is shown
            elsewhere).
        overlays: Unused; reserved.

    Returns:
        A Plotly figure dict (``{"data": [...], "layout": {...}}``).
    """
    cell_line_bars = getattr(isoform, "cell_line_bars", {}) or {}
    all_tis = list(getattr(isoform, "all_tis_on_transcript", []) or [])
    focal_tis_id = getattr(isoform, "focal_tis_id", None) or getattr(isoform, "tis_id", None)

    samples = [s for s in _SAMPLE_ORDER if any(s in v for v in cell_line_bars.values())]
    # include any samples not in the canonical order (defensive)
    extra = sorted({s for v in cell_line_bars.values() for s in v} - set(samples))
    samples += extra

    if not samples or not cell_line_bars:
        return _empty_figure("No per-cell-line initiation data for this isoform.")

    meta = {t["tis_id"]: t for t in all_tis}
    # Order TIS: focal first, then by genomic position.
    tis_ids = sorted(
        cell_line_bars.keys(),
        key=lambda tid: (tid != focal_tis_id, meta.get(tid, {}).get("genomic_pos", 0)),
    )

    traces: list[dict[str, Any]] = []
    other_i = 0
    for tid in tis_ids:
        bars = cell_line_bars[tid]
        is_focal = tid == focal_tis_id
        orf_type = meta.get(tid, {}).get("orf_type", "")
        if is_focal:
            color = _FOCAL_COLOR
        else:
            color = _OTHER_COLORS[other_i % len(_OTHER_COLORS)]
            other_i += 1
        ys: list[float | None] = []
        hover: list[str] = []
        for s in samples:
            val = bars.get(s)
            ys.append(float(val) if val is not None else None)
            if val is not None:
                try:
                    ie = f"{2 ** float(val):.3g}"
                except (OverflowError, ValueError):
                    ie = "n/a"
                hover.append(
                    f"<b>{s}</b><br>{orf_type} {tid}<br>log2(IE): {float(val):.2f}<br>IE: {ie}"
                )
            else:
                hover.append(f"<b>{s}</b><br>not initiated")
        traces.append(
            {
                "type": "bar",
                "name": _tis_label(tid, orf_type, is_focal),
                "x": samples,
                "y": ys,
                "marker": {"color": color, "line": {"width": 1, "color": "#fff"}},
                "hovertext": hover,
                "hoverinfo": "text",
            }
        )

    layout = _base_layout("Initiation efficiency across cell lines")
    layout.update(
        {
            "barmode": "group",
            "bargap": 0.3,
            "bargroupgap": 0.1,
            "xaxis": {
                "type": "category",
                "tickfont": {"size": 12, "color": "#374151"},
            },
            "yaxis": {
                "title": {
                    "text": "log2 initiation efficiency",
                    "font": {"size": 12, "color": "#4b5563"},
                },
                "tickfont": {"size": 11, "color": "#6b7280"},
                "zeroline": True,
                "zerolinecolor": "#cbd5e1",
            },
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "x": 0,
                "font": {"size": 11},
            },
            "height": 340,
        }
    )
    return {"data": traces, "layout": layout}
