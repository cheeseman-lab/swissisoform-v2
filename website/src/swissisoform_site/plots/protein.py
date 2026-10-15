"""Build the gene-page protein-track Plotly figure.

One canonical bar, drawn identically on every gene, with one row per isoform
beneath it on the canonical residue axis. Each isoform row carries its own
difference: an extension's gained N-terminus (green), a truncation's lost
N-terminus (red, the canonical residues it lacks) or a separate ORF's whole body
(green), named by the stratum's noun (``contract.region_noun``). Clinical
variants render as dots coloured by significance; motifs as spans; domains as
boxes.
"""

from __future__ import annotations

from typing import Any

from swissisoform.clinical.significance import is_pathogenic
from swissisoform.contract import DEFAULT_REGION_NOUN, region_noun

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


# Rough number of 9px label characters that span the full residue axis, used to
# fit domain names to their box width. Approximate (plot width is responsive);
# tuned to favour showing full names on reasonably wide boxes.
_AXIS_CHARS = 140

# Variant tracks, one row per consequence type, ordered most→least severe.
# CONSEQ_COLOR / CONSEQ_SHORT are public: the scan results table renders the
# same vocabulary, and two copies would drift the moment a term is added.
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
CONSEQ_COLOR = {
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
CONSEQ_SHORT = {
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


def build_gene_protein_figure(view: Any, collapse_domains: bool = False) -> dict[str, Any]:
    """Combined gene view: one canonical bar + one bar per isoform, residue axis.

    Lays out a single canonical bar (residues ``1..canonical_len``) with every isoform
    aligned on the shared region beneath it, deduplicated variant rows on top,
    and a deduplicated domain band + disorder / coiled-coil / motif tracks +
    per-cell-line initiation lanes below. All coordinates are in the canonical
    residue frame; the adapter has already mapped each isoform's features into
    it (see ``app._make_gene_protein_view``).

    Args:
        view: duck-typed adapter with ``canonical_len``, ``bars`` (per isoform:
            ``label``/``x0``/``x1``/``is_trunc``/``region``/``diff_x0``/``diff_x1``/
            ``diff_is_lost``/``slug``), ``variants`` (``variant_id``/``pos``/
            ``consequence``/``significance``/``protein_change``/``source``),
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
                "x": x_left,
                "y": y,
                "xref": "x",
                "yref": "y",
                "xanchor": "right",
                "xshift": -6,
                "text": text,
                "showarrow": False,
                "font": {"size": size, "color": "#475569"},
            }
        )

    # ── Bars: canonical on top (y=0), isoforms stacked below ──
    # The canonical bar is the same plain reference on every gene. Each isoform
    # row carries its OWN difference — an extension's gained N-terminus (green),
    # a truncation's lost N-terminus (red: the canonical residues it lacks), a
    # separate ORF's whole body (green) — so rows differ only where isoforms do.
    bar_gap = 0.34
    canon_y = 0.0
    canon = _segment(
        0,
        can_len - 1,
        canon_y,
        _CANON_COLOR,
        "Canonical",
        [f"<b>Canonical</b> · {can_len} aa"],
        residue_hover=True,
    )
    canon["showlegend"] = True
    traces.append(canon)
    _left_label("canonical", canon_y)
    legend_shown: set[str] = set()

    def _legend(trace: dict[str, Any], name: str) -> dict[str, Any]:
        trace["name"] = name
        trace["legendgroup"] = name
        trace["showlegend"] = name not in legend_shown
        legend_shown.add(name)
        return trace

    for i, b in enumerate(bars):
        by = -bar_gap * (i + 1)
        b["_y"] = by
        x0, x1 = int(b["x0"]), int(b["x1"])
        label, slug = b["label"], b.get("slug")
        region = b.get("region") or region_noun(b.get("orf_type"), default=DEFAULT_REGION_NOUN)
        region_cap = region_noun(b.get("orf_type"), cap=True, default=DEFAULT_REGION_NOUN)
        open_hint = "<br><i>click to open</i>" if slug else ""
        d0, d1 = b.get("diff_x0"), b.get("diff_x1")
        has_diff = d0 is not None and d1 is not None and int(d1) >= int(d0)
        lost = bool(b.get("diff_is_lost")) and has_diff

        if lost:
            n_lost = int(d1) - int(d0) + 1
            # Drawn up to x0 (not d1) so it meets the blue body flush, the way an
            # extension's green meets x=0; the hover still reports d0..d1.
            seg = _segment(
                int(d0),
                min(int(d1) + 1, x1),
                by,
                _TRUNC_FILL,
                "",
                [
                    f"<b>{label}</b><br>{region}: canonical residues {int(d0) + 1}–{int(d1) + 1}"
                    f" ({n_lost} aa){open_hint}"
                ],
                slug=slug,
                opacity=0.55,
            )
            traces.append(_legend(seg, region_cap))
        elif has_diff:
            # Extensions end their gained region at x=0 (the canonical start) so
            # it sits flush against the shared body; a whole-isoform separate ORF
            # is capped at its own C-terminus.
            g1 = min(int(d1) + 1, x1)
            # Counted off d0..d1 (inclusive), not the drawn span: a separate ORF's
            # cap at x1 would read one residue short.
            n_gained = int(d1) - int(d0) + 1
            seg = _segment(
                int(d0),
                g1,
                by,
                _EXT_FILL,
                "",
                [f"<b>{label}</b><br>{region}: {max(n_gained, 1)} aa{open_hint}"],
                slug=slug,
            )
            traces.append(_legend(seg, region_cap))

        # Blue spans what the isoform shares with the canonical.
        shared_x0 = x0 if lost or not has_diff else int(d1) + 1
        if shared_x0 <= x1:
            seg = _segment(
                shared_x0,
                x1,
                by,
                _ISO_COLOR,
                "",
                [f"<b>{label}</b><br>shared with canonical{open_hint}"],
                slug=slug,
            )
            traces.append(_legend(seg, "Isoform"))

        annotations.append(
            {
                "x": x_left,
                "y": by,
                "xref": "x",
                "yref": "y",
                "xanchor": "right",
                "xshift": -6,
                "text": label,
                "showarrow": False,
                "font": {"size": 9, "color": "#1d4ed8" if slug else "#475569"},
                # A clickable row label: gene.html listens for plotly_clickannotation
                # and reads the slug from ``name``.
                **(
                    {"name": slug, "captureevents": True, "hovertext": "open isoform"}
                    if slug
                    else {}
                ),
            }
        )
    lowest_bar = -bar_gap * len(bars)

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
    uploaded_drawn = False
    for i, c in enumerate(conseqs):
        ty = mut_base + i * 0.26
        xs, cols, sizes, opac, hover = [], [], [], [], []
        # Uploaded VCF hits share the consequence row — so a user's variant reads
        # against where the known pathogenics cluster — but get their own trace so
        # they can carry a distinct symbol and a single legend entry.
        up_xs, up_hover = [], []
        for v in by_conseq[c]:
            if (v.get("source") or "") == "uploaded":
                up_xs.append(v["pos"])
                up_hover.append(
                    f"<b>From your VCF</b><br>{v.get('uploaded_detail') or ''}<br>"
                    f"{v.get('protein_change') or ''}<br>{c}"
                )
                continue
            # Styled by significance only. Whether a variant is in an isoform's
            # differential region depends on the isoform, so the gene view makes
            # no such claim (the isoform pages do).
            is_path = is_pathogenic(v.get("significance"))
            xs.append(v["pos"])
            cols.append("#d62728" if is_path else CONSEQ_COLOR.get(c, "#94a3b8"))
            sizes.append(8 if is_path else 5)
            opac.append(1.0 if is_path else 0.7)
            hover.append(
                f"{v.get('variant_id', '?')}<br>{v.get('protein_change') or ''}<br>"
                f"{c}<br>{v.get('significance') or '—'} · {v.get('source') or ''}"
            )
        if xs:
            traces.append(
                {
                    "type": "scatter",
                    "mode": "markers",
                    "x": xs,
                    "y": [ty] * len(xs),
                    "marker": {"size": sizes, "color": cols, "opacity": opac, "symbol": "circle"},
                    "hovertext": hover,
                    "hoverinfo": "text",
                    "showlegend": False,
                }
            )
        if up_xs:
            traces.append(
                {
                    "type": "scatter",
                    "mode": "markers",
                    "x": up_xs,
                    "y": [ty] * len(up_xs),
                    "marker": {
                        "size": 11,
                        "color": CONSEQ_COLOR.get(c, "#94a3b8"),
                        "opacity": 1.0,
                        "symbol": "diamond",
                        # A dark outline is what separates these from the circles at
                        # a glance, since the fill still encodes the consequence.
                        "line": {"width": 1.6, "color": "#0f172a"},
                    },
                    "hovertext": up_hover,
                    "hoverinfo": "text",
                    # Exactly one legend entry across all rows: a novel symbol with
                    # no key leaves the reader guessing.
                    "name": "uploaded VCF",
                    "showlegend": not uploaded_drawn,
                    "legendgroup": "uploaded",
                }
            )
            uploaded_drawn = True
        _left_label(CONSEQ_SHORT.get(c, c)[:14], ty, size=9)
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
                _feature_box(
                    s["x0"],
                    s["x1"],
                    top,
                    bot,
                    fill,
                    "Domain (InterPro)",
                    "domain",
                    hover,
                    not legend_shown,
                )
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
                _feature_box(
                    d["x0"],
                    d["x1"],
                    top,
                    bot,
                    _DOMAIN_FILL,
                    "Domain (InterPro)",
                    "domain",
                    hover,
                    not legend_shown,
                )
            )
            legend_shown = True
            name = d.get("name") or ""
            chars_fit = int((d["x1"] - d["x0"]) / max(axis_len, 1) * _AXIS_CHARS)
            if d.get("interpro_id") and chars_fit >= 3:
                label = name if len(name) <= chars_fit else name[: max(1, chars_fit - 1)] + "…"
                annotations.append(
                    {
                        "x": d["x0"],
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
        y_cur = band_bottom - 0.13

    disorder = list(getattr(view, "disorder", []) or [])
    if disorder:
        top, bot = y_cur, y_cur - row_h * 0.7
        legend_shown = False
        for seg in disorder:
            hover = f"<b>Disordered region</b><br>frame residues {int(seg['x0'])}–{int(seg['x1'])}"
            traces.append(
                _feature_box(
                    seg["x0"],
                    seg["x1"],
                    top,
                    bot,
                    _DISORDER_FILL,
                    "Disordered (MobiDB)",
                    "disorder",
                    hover,
                    not legend_shown,
                )
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
        y_cur = cy - row_h * 0.35 - 0.13

    motifs = list(getattr(view, "motifs", []) or [])
    if motifs:
        motif_y = y_cur - row_h * 0.35
        legend_shown = False
        for m in motifs:
            mxs = _bar_samples(int(m["x0"]), int(m["x1"]))
            traces.append(
                {
                    "type": "scatter",
                    "name": "Motif",
                    "mode": "lines",
                    "x": mxs,
                    "y": [motif_y] * len(mxs),
                    "line": {"color": _MOTIF_COLOR, "width": 6},
                    "legendgroup": "motif",
                    "hovertext": [f"{m.get('name', '?')}"] * len(mxs),
                    "hoverinfo": "text",
                    "showlegend": not legend_shown,
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
            xs, sizes, colors, hover = [], [], [], []
            for m in t["marks"]:
                xs.append(m["residue"])
                sizes.append(_ie_size(m["log2_ie"]))
                # The canonical start rides along as a slate-gray dot (a within-gene
                # baseline); alternative starts stay blue.
                is_canon = bool(m.get("canonical"))
                colors.append(_CANON_COLOR if is_canon else _ISO_COLOR)
                try:
                    ie = f"{2 ** m['log2_ie']:.3g}"
                except (OverflowError, ValueError):
                    ie = "n/a"
                hover.append(f"<b>{t['sample']}</b><br>{m.get('label', '')}<br>IE {ie}")
            traces.append(
                {
                    "type": "scatter",
                    "mode": "markers",
                    "x": xs,
                    "y": [ly] * len(xs),
                    "marker": {
                        "size": sizes,
                        "color": colors,
                        "opacity": 0.85,
                        "line": {"width": 1, "color": "#fff"},
                    },
                    "hovertext": hover,
                    "hoverinfo": "text",
                    "showlegend": False,
                }
            )
            _left_label(t["sample"].replace("_", " "), ly)
        y_bottom = lane_base - (len(cell_lines) - 1) * lane_gap - 0.3

    y_hi = max(mut_top, canon_y + 0.2) + 0.3
    y_lo = y_bottom - 0.2
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
                    "text": "Protein residue (0 = canonical start)",
                    "font": {"size": 12, "color": "#4b5563"},
                },
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
    x0: float,
    x1: float,
    top: float,
    bot: float,
    fill: str,
    name: str,
    group: str,
    hover: str,
    show_legend: bool,
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


def _segment(
    x0: int,
    x1: int,
    y: float,
    color: str,
    name: str,
    hover: list[str],
    slug: str | None = None,
    opacity: float = 1.0,
    residue_hover: bool = False,
) -> dict[str, Any]:
    """One bar segment on a row: a thick line densified so its whole length hovers.

    ``hover`` is one string for the segment (``residue_hover`` appends the
    canonical-frame residue under the cursor, x=0 → residue 1). ``slug`` rides
    as ``customdata`` so a click opens that isoform's page (``gene.html``).
    """
    xs = _bar_samples(x0, x1)
    text = hover[0]
    trace: dict[str, Any] = {
        "type": "scatter",
        "name": name,
        "mode": "lines",
        "x": xs,
        "y": [y] * len(xs),
        "line": {"color": color, "width": 12},
        "opacity": opacity,
        "hovertext": [f"{text}<br>residue {int(x) + 1}" if residue_hover else text for x in xs],
        "hoverinfo": "text",
        "showlegend": False,
    }
    if slug:
        trace["customdata"] = [slug] * len(xs)
    return trace
