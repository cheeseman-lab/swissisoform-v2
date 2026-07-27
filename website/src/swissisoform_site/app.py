"""Flask routes for the SwissIsoform v2 viewer.

Read-only — every payload comes from ``data.load_all()`` which is cached at
the first request. There is no DB.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import types
from pathlib import Path
from typing import Any

from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    send_from_directory,
)
from markupsafe import Markup, escape

from swissisoform.site.evidence import (
    CRITERIA_METRIC_LABELS,
    format_metric,
    slice_criterion,
)
from swissisoform_site.data import (
    CARD_BADGES,
    CARD_GROUPS,
    CRITERIA_BY_ID,
    CRITERIA_FOR_PAGE,
    CRITERION_ABOUT,
    EXISTENCE_CRITERIA,
    FUNCTIONAL_CRITERIA,
    Isoform,
    _isoform_view,
    biophysics_card_for_isoform,
    category_verdicts_for_isoform,
    criterion_evidence_for,
    data_dir,
    llm_synthesis_for_isoform,
    load_all,
    sae_card_for_isoform,
    tis_slug,
    variant_rows_for_isoform,
    variant_url,
)
from swissisoform_site.genomics import interval_intersection
from swissisoform_site.plots import build_gene_protein_figure

# Cell line samples used by the transcript figure's bottom panel.
_CELL_LINE_SAMPLES = ("HeLa", "K562", "U2OS", "RPE1_Async", "RPE1_Que", "RPE1_Sen")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _log_llm_coverage(genes: dict, llm_dir: Path) -> None:
    """Print one summary line — easier to spot missing LLM JSONs than per-request logs."""
    log = logging.getLogger("swissisoform_site.coverage")
    n_iso = sum(len(g.isoforms) for g in genes.values())
    n_syn = 0
    n_category_files = 0
    n_category_cells = 0
    for gene in genes.values():
        for iso in gene.isoforms:
            slug = tis_slug(iso.tis_id)
            if (llm_dir / slug / "synthesis.json").exists():
                n_syn += 1
            cat_path = llm_dir / slug / "categories.json"
            if cat_path.exists():
                n_category_files += 1
                try:
                    payload = json.loads(cat_path.read_text())
                    if isinstance(payload, dict):
                        n_category_cells += len(payload)
                except Exception:
                    pass
    log.warning(
        "%d isoforms loaded; LLM coverage: categories %d/%d (%d/%d cells), synthesis %d/%d",
        n_iso,
        n_category_files,
        n_iso,
        n_category_cells,
        n_iso * 6,
        n_syn,
        n_iso,
    )


def create_app() -> Flask:
    """Build the Flask app. Factory pattern so tests can re-create cleanly."""
    app = Flask(__name__)

    # Slugify filter — must match data.py::tis_slug and the LLM dispatcher's
    # _tis_slug so that landing-page dropdown links resolve to the right
    # Flask route AND the right on-disk LLM output directory. Both call
    # sites use ``re.sub(r"[:.]+", "-", ...)`` — replace ONLY ``:`` and ``.``.
    # In particular DO NOT strip ``+`` (strand indicator on plus-strand
    # isoforms); URLs handle literal ``+`` fine in path segments.
    _slug_re = re.compile(r"[:.]+")

    @app.template_filter("slugify")
    def slugify(value: Any) -> str:
        return _slug_re.sub("-", str(value or "unknown"))

    # Linkify ``PMID:NNN`` citations in the gene mechanistic narrative. Match each
    # PMID token (not the enclosing bracket) so multi-PMID brackets like
    # ``[PMID:1, PMID:2, PMID:3]`` link every entry, leaving the brackets/commas
    # as literal text. Escape first (XSS-safe), then wrap each in a PubMed link.
    _pmid_re = re.compile(r"PMID:\s*(\d+)")

    @app.template_filter("pmid_links")
    def pmid_links(value: Any) -> Markup:
        safe = str(escape(value or ""))
        linked = _pmid_re.sub(
            r'<a href="https://pubmed.ncbi.nlm.nih.gov/\1/" target="_blank" '
            r'rel="noopener">PMID:\1</a>',
            safe,
        )
        return Markup(linked)

    # Surface the metric label dict + formatter so evidence-tile partials
    # can render human-readable rows without re-importing from the script.
    app.jinja_env.globals.update(
        CRITERIA_METRIC_LABELS=CRITERIA_METRIC_LABELS,
        format_metric=format_metric,
        variant_url=variant_url,
        CARD_GROUPS=CARD_GROUPS,
    )

    # JSON-friendly NaN cleaner for the API endpoint
    def _clean(obj: Any) -> Any:
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        return obj

    # ---------- Routes ----------

    @app.get("/healthz")
    def healthz() -> Any:
        return jsonify({"ok": True})

    @app.get("/")
    def index() -> Any:
        genes = load_all()
        return render_template(
            "index.html",
            genes=list(genes.values()),
            n_genes=len(genes),
            existence_criteria=EXISTENCE_CRITERIA,
            functional_criteria=FUNCTIONAL_CRITERIA,
        )

    @app.get("/about")
    def about() -> Any:
        """Static glossary — the 15 CDLMPS evidence criteria + the metrics behind them."""
        return render_template(
            "about.html",
            criteria_by_id=CRITERIA_BY_ID,
            criterion_about=CRITERION_ABOUT,
        )

    @app.get("/genes/<gene_name>")
    def gene_page(gene_name: str) -> Any:
        """Render the gene overview: combined protein-residue figure + isoform cards.

        The figure is a single canonical bar plus one bar per isoform aligned on
        the shared region (``build_gene_protein_figure``), with variant rows
        above and domains / disorder / coil / motifs / per-cell-line initiation
        below — all deduplicated across the gene's isoforms. Isoform cards below
        link to the per-isoform deep-dive page.
        """
        genes = load_all()
        gene = genes.get(gene_name) or genes.get(gene_name.upper())
        if gene is None or not gene.isoforms:
            abort(404)

        view = _make_gene_protein_view(gene)
        gene_fig = build_gene_protein_figure(view)
        gene_fig_collapsed = build_gene_protein_figure(view, collapse_domains=True)

        # Group isoforms for the card list, mirroring the landing dropdown order.
        extensions = [i for i in gene.isoforms if i.orf_type == "extended"]
        truncations = [i for i in gene.isoforms if i.orf_type == "truncated"]
        others = [i for i in gene.isoforms if i.orf_type not in ("extended", "truncated")]

        return render_template(
            "gene.html",
            gene=gene,
            extensions=extensions,
            truncations=truncations,
            others=others,
            igv_figure_json=json.dumps(gene_fig),
            igv_figure_collapsed_json=json.dumps(gene_fig_collapsed),
        )

    @app.get("/genes/<gene_name>/isoforms/<tis_slug_str>")
    def isoform_page(gene_name: str, tis_slug_str: str) -> Any:
        """Render the V2 per-isoform page.

        Wires data + plot figure dicts + LLM JSONs into ``isoform.html``.
        Returns 404 on unknown gene or tis_slug.
        """
        genes = load_all()
        gene = genes.get(gene_name) or genes.get(gene_name.upper())
        if gene is None:
            abort(404)
        iso = next(
            (i for i in gene.isoforms if tis_slug(i.tis_id) == tis_slug_str),
            None,
        )
        if iso is None:
            abort(404)

        data_dir_path = data_dir()
        llm_dir = data_dir_path / "llm"
        synthesis = llm_synthesis_for_isoform(llm_dir=llm_dir, tis_slug=tis_slug_str)
        category_llms = category_verdicts_for_isoform(llm_dir=llm_dir, tis_slug=tis_slug_str)

        # Reconstruct the per-isoform record shape slice_criterion wants: a
        # ``{"_raw": ..., "scoring": {"criteria": {name: {"value", "reason"}}}, ...}``
        # blob derived from the V1 Isoform dataclass + parquet ``raw`` mirror.
        combined_criteria = {
            name: {"value": iso.criteria.get(name), "reason": iso.reasons.get(name)}
            for name in iso.criteria
        }
        iso_record = {
            "tis_id": iso.tis_id,
            "gene": {"name": gene.name},
            "orf_type": iso.orf_type,
            "differential_sequence": iso.differential_sequence,
            "diff_space": iso.diff_space,
            "isoform_length_aa": iso.isoform_len,
            "canonical_length_aa": iso.canonical_len,
            "scoring": {"criteria": combined_criteria},
            "_raw": iso.raw or {},
        }

        # Per-criterion slices still drive the always-visible tile headlines; the
        # LLM interpretation is now per-category (category_llms), not per-tile.
        criterion_slices: dict[str, dict] = {}
        for c in CRITERIA_FOR_PAGE:
            cid = c["id"]
            criterion_slices[cid] = slice_criterion(iso_record, cid)

        variant_rows = variant_rows_for_isoform(data_dir_path / "variants_long.parquet", iso.tis_id)

        # Split the mutation table into the differential (isoform-unique) region
        # and the shared canonical core — they carry different meaning, and only
        # the canonical-frame shared core is AlphaMissense-scorable. Within each
        # section, order N→C by isoform residue.
        def _vsort(r: dict[str, Any]) -> tuple[int, float]:
            p = r.get("isoform_protein_pos")
            try:
                return (0, float(p))
            except (TypeError, ValueError):
                return (1, 0.0)

        variant_rows_unique = sorted(
            (r for r in variant_rows if r.get("in_isoform_unique")), key=_vsort
        )
        variant_rows_shared = sorted(
            (r for r in variant_rows if not r.get("in_isoform_unique")), key=_vsort
        )

        sae = sae_card_for_isoform(iso)
        bio = biophysics_card_for_isoform(iso)

        return render_template(
            "isoform.html",
            isoform=_isoform_view(iso, gene),
            sae=sae,
            bio=bio,
            criterion_evidence=criterion_evidence_for(iso),
            criteria=CRITERIA_FOR_PAGE,
            card_groups=CARD_GROUPS,
            card_badges=CARD_BADGES,
            criteria_by_id=CRITERIA_BY_ID,
            category_llms=category_llms,
            criterion_slices=criterion_slices,
            synthesis=synthesis,
            variant_rows=variant_rows,
            variant_rows_unique=variant_rows_unique,
            variant_rows_shared=variant_rows_shared,
            canonical_cif=iso.canonical_cif,
            isoform_cif=iso.isoform_cif,
            canonical_colors=iso.canonical_colors,
            isoform_colors=iso.isoform_colors,
            canonical_pae=iso.canonical_pae,
            isoform_pae=iso.isoform_pae,
            diff_start=iso.diff_start,
            diff_end=iso.diff_end,
            diff_space=iso.diff_space,
            canonical_length_aa=iso.canonical_len,
            isoform_length_aa=iso.isoform_len,
        )

    @app.get("/api/data.json")
    def api_data() -> Any:
        """Dump every gene record + its LLM blob as a single JSON document.

        Intended for downloads, not for high-frequency programmatic queries —
        no pagination, no caching headers.
        """
        genes = load_all()
        payload: dict[str, Any] = {}
        for name, g in genes.items():
            payload[name] = {
                "name": g.name,
                "uniprot_id": g.uniprot_id,
                "uniprot_url": g.uniprot_url,
                "function": g.function,
                "location": g.location,
                "keywords": g.keywords,
                "canonical_len": g.canonical_len,
                "canonical_cif": g.canonical_cif,
                "llm": g.llm,
                "isoforms": [_isoform_to_dict(i) for i in g.isoforms],
            }
        return jsonify(_clean(payload))

    @app.get("/structures/<path:filename>")
    def structures(filename: str) -> Any:
        """Serve baked .cif files from ``<DATA_DIR>/structures/``."""
        root = data_dir() / "structures"
        # send_from_directory enforces the path stays within ``root``
        if not (root / filename).is_file():
            abort(404)
        return send_from_directory(root, filename, mimetype="chemical/x-mmcif")

    @app.get("/structure-colors/<path:filename>")
    def structure_colors(filename: str) -> Any:
        """Serve precomputed per-residue colour maps from ``<DATA_DIR>/structures/colors/``.

        The 3Dmol folding viewer fetches these and applies them — the diff-region
        recolouring is decided offline (scripts/export/build_folding_colors.py), not live.
        """
        root = data_dir() / "structures" / "colors"
        if not (root / filename).is_file():
            abort(404)
        return send_from_directory(root, filename, mimetype="application/json")

    @app.get("/structure-pae/<path:filename>")
    def structure_pae(filename: str) -> Any:
        """Serve precomputed PAE heatmap JSONs from ``<DATA_DIR>/structures/pae/``.

        The folding panel's canvas renderer fetches these lazily and draws the
        L×L predicted-aligned-error map (precomputed offline by
        swissisoform.export.pae).
        """
        root = data_dir() / "structures" / "pae"
        if not (root / filename).is_file():
            abort(404)
        return send_from_directory(root, filename, mimetype="application/json")

    @app.errorhandler(404)
    def not_found(e: Any) -> tuple[Any, int]:
        return render_template("404.html", message=str(e)), 404

    # Startup coverage summary — one log line so missing LLM JSONs are obvious
    # before any clicking through tiles.
    _log_llm_coverage(load_all(), data_dir() / "llm")

    return app


# InterProScan member DBs grouped by the kind of feature they call, so the
# figure can render each kind in its own track instead of stacking them all into
# one green smear. Anything unlisted is treated as a folded domain.
_DISORDER_DBS = {"MobiDB-lite", "MobiDB"}
_COILED_COIL_DBS = {"COILS", "Coils"}


def _merge_hit_intervals(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge overlapping/adjacent hit records into regions, keeping members."""
    regions: list[dict[str, Any]] = []
    for r in sorted(recs, key=lambda rr: (rr["start"], rr["end"])):
        if regions and r["start"] <= regions[-1]["end"] + 1:
            reg = regions[-1]
            reg["end"] = max(reg["end"], r["end"])
            reg["members"].append(r["hit"])
        else:
            regions.append({"start": r["start"], "end": r["end"], "members": [r["hit"]]})
    return regions


def _depth_segments(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sweep-line coverage depth over hit records (1-based inclusive coords).

    Returns contiguous ``{start, end, depth}`` segments where ``depth`` is the
    number of hits covering that stretch — used to shade the collapsed domain
    bar by overlap density, recreating the original stacked-transparency look.
    """
    if not recs:
        return []
    bounds = sorted({r["start"] for r in recs} | {r["end"] + 1 for r in recs})
    segs: list[dict[str, Any]] = []
    for a, b in zip(bounds, bounds[1:]):
        lo, hi = a, b - 1
        if hi < lo:
            continue
        depth = sum(1 for r in recs if r["start"] <= lo and r["end"] >= hi)
        if depth <= 0:
            continue
        if segs and segs[-1]["depth"] == depth and segs[-1]["end"] + 1 >= lo:
            segs[-1]["end"] = hi
        else:
            segs.append({"start": lo, "end": hi, "depth": depth})
    return segs


def _classify_interproscan_hits(ips_hits: Any) -> dict[str, list[dict[str, Any]]]:
    """Split InterProScan hits by feature type and collapse redundancy.

    Every member DB is kept — nothing is filtered out. Hits are bucketed into
    ``domain`` / ``disorder`` / ``coiled_coil`` by their source DB, then hits of
    the same type that overlap in residue space are merged into one region so a
    dozen DBs describing the same domain render as a single labelled box (with
    every contributing signature listed on hover), not a green smear.

    Coordinates convert from the pipeline's 0-based ``pos``/``end`` to the
    1-based residue axis the figure uses.
    """
    empty = {"domains": [], "disorder": [], "coiled_coil": []}
    if ips_hits is None:
        return dict(empty)
    buckets: dict[str, list[dict[str, Any]]] = {"domain": [], "disorder": [], "coiled_coil": []}
    try:
        for h in list(ips_hits):
            if not isinstance(h, dict):
                continue
            start = h.get("start", h.get("pos"))
            end = h.get("end")
            if start is None or end is None:
                continue
            db = h.get("db")
            kind = (
                "disorder"
                if db in _DISORDER_DBS
                else "coiled_coil"
                if db in _COILED_COIL_DBS
                else "domain"
            )
            buckets[kind].append({"hit": h, "start": int(start) + 1, "end": int(end) + 1})
    except (TypeError, ValueError):
        return dict(empty)

    # One box per distinct InterPro entry (member DBs collapse under their
    # entry; unmapped signatures keep their own box), preserving sub-domain
    # structure. Overlapping entries are row-stacked by the figure, not merged,
    # so e.g. a chromo domain and a chromo-shadow domain stay distinct even when
    # a whole-protein family signature spans both.
    domains: list[dict[str, Any]] = []
    by_entry: dict[str, list[dict[str, Any]]] = {}
    for rec in buckets["domain"]:
        h = rec["hit"]
        key = h.get("interpro_id") or f"sig:{h.get('name') or h.get('db')}"
        by_entry.setdefault(key, []).append(rec)
    for recs in by_entry.values():
        # Merge only overlapping occurrences of the SAME entry; a single entry
        # appearing at two separate loci stays two boxes.
        for reg in _merge_hit_intervals(recs):
            members = reg["members"]
            mapped = [m for m in members if m.get("interpro_id")]
            best = mapped[0] if mapped else members[0]
            name = best.get("interpro_description") or best.get("name") or best.get("db") or "domain"
            if name in ("-", "—"):
                name = best.get("name") or best.get("db") or "domain"
            domains.append(
                {
                    "name": name,
                    "interpro_id": best.get("interpro_id"),
                    "start": reg["start"],
                    "end": reg["end"],
                    "dbs": sorted({m.get("db") for m in members if m.get("db")}),
                    "n_sig": len(members),
                }
            )
    domains.sort(key=lambda d: (d["start"], d["end"]))

    # Collapsed representation: one box per contiguous domain region (all member
    # DBs merged), for the compact state of the page's domain toggle.
    domains_merged: list[dict[str, Any]] = []
    for reg in _merge_hit_intervals(buckets["domain"]):
        members = reg["members"]
        mapped = [m for m in members if m.get("interpro_id")]
        best = (
            max(mapped, key=lambda m: len(m.get("interpro_description") or ""))
            if mapped
            else members[0]
        )
        name = best.get("interpro_description") or best.get("name") or best.get("db") or "domain"
        if name in ("-", "—"):
            name = best.get("name") or best.get("db") or "domain"
        domains_merged.append(
            {
                "name": name,
                "interpro_id": best.get("interpro_id"),
                "start": reg["start"],
                "end": reg["end"],
                "dbs": sorted({m.get("db") for m in members if m.get("db")}),
                "n_sig": len(members),
            }
        )

    def _simple(regs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"start": r["start"], "end": r["end"]} for r in regs]

    return {
        "domains": domains,
        "domains_merged": domains_merged,
        "domain_segments": _depth_segments(buckets["domain"]),
        "disorder": _simple(_merge_hit_intervals(buckets["disorder"])),
        "coiled_coil": _simple(_merge_hit_intervals(buckets["coiled_coil"])),
    }


# --------------------------------------------------------------------------- #
# Gene-page combined protein-residue view — one canonical bar + one bar per
# isoform, with variants / domains / features deduplicated across isoforms in
# the canonical residue frame.
# --------------------------------------------------------------------------- #


def _pathogenic(sig: Any) -> bool:
    return str(sig or "").lower().startswith(("pathogenic", "likely"))


def _union_intervals(interval_lists: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    """Merge several interval lists into one sorted, merged list."""
    merged: list[tuple[int, int]] = []
    for lst in interval_lists:
        merged.extend((int(s), int(e)) for s, e in lst if e > s)
    if not merged:
        return []
    merged.sort()
    out: list[tuple[int, int]] = [merged[0]]
    for s, e in merged[1:]:
        ls, le = out[-1]
        if s <= le:
            out[-1] = (ls, max(le, e))
        else:
            out.append((s, e))
    return out


def _frame_domain_clusters(occurrences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedup residue-frame domain occurrences into gene-level union glyphs.

    ``occurrences`` items: ``{key, name, interpro_id, x0, x1, label}`` (canonical
    residue-frame coords). Occurrences of the same InterPro entry whose spans
    overlap collapse into one glyph at their union extent. Returns
    ``[{name, interpro_id, x0, x1, isoforms}]`` sorted by start.
    """
    clusters: list[dict[str, Any]] = []
    for occ in occurrences:
        span = [(occ["x0"], occ["x1"])]
        placed = False
        for cl in clusters:
            if cl["key"] == occ["key"] and interval_intersection(cl["union"], span):
                cl["union"] = _union_intervals([cl["union"], span])
                cl["labels"].add(occ["label"])
                cl["interpro_id"] = cl["interpro_id"] or occ["interpro_id"]
                placed = True
                break
        if not placed:
            clusters.append(
                {
                    "key": occ["key"],
                    "name": occ["name"],
                    "interpro_id": occ["interpro_id"],
                    "union": list(span),
                    "labels": {occ["label"]},
                }
            )

    out: list[dict[str, Any]] = []
    for cl in clusters:
        lo = min(s for s, _ in cl["union"])
        hi = max(e for _, e in cl["union"])
        out.append(
            {
                "name": cl["name"],
                "interpro_id": cl["interpro_id"],
                "x0": lo,
                "x1": hi,
                "isoforms": sorted(cl["labels"]),
            }
        )
    out.sort(key=lambda d: d["x0"])
    return out


def _make_gene_protein_view(gene: Any) -> types.SimpleNamespace:
    """Residue-frame combined view consumed by ``build_gene_protein_figure``.

    One canonical bar (residues ``1..canonical_len``) plus one bar per isoform,
    aligned on the shared region against the gene canonical. Extensions and
    truncations share the canonical **C-terminus**, so each isoform's residue
    ``r`` maps to the canonical frame by ``r + (canonical_len - iso_len)`` —
    anchoring isoform residue ``iso_len`` to canonical residue ``canonical_len``.
    (Using ``diff_end`` instead would misplace truncation variants by a residue,
    because the initiator-Met boundary makes ``iso_len + diff_end`` one past
    ``canonical_len``.) Extensions then reach left of residue 1; a truncation's
    lost N-terminus shades the canonical bar. uORF / altORF isoforms have no
    shared region (``offset = 0``, whole isoform shaded).

    Variants / domains / disorder / coil / motifs / cell-line starts are mapped
    into that frame and deduplicated across isoforms.
    """
    can_len = int(getattr(gene, "canonical_len", 0) or 0)
    bars: list[dict[str, Any]] = []
    var_by_id: dict[str, dict[str, Any]] = {}
    domain_occ: list[dict[str, Any]] = []
    disorder_iv: list[tuple[int, int]] = []
    coil_iv: list[tuple[int, int]] = []
    motifs: dict[tuple, dict[str, Any]] = {}
    cell_by_sample: dict[str, list] = {}
    x_left = 1.0

    for iso in gene.isoforms:
        raw = getattr(iso, "raw", None) or {}
        diff_end = int(getattr(iso, "diff_end", 0) or 0)
        iso_len = int(getattr(iso, "isoform_len", 0) or 0) or 1
        can_len_i = int(getattr(iso, "canonical_len", 0) or 0) or can_len
        orf_type = (iso.orf_type or "").lower()
        diff_space = (getattr(iso, "diff_space", "") or "").lower()
        is_trunc = diff_space == "canonical" or orf_type == "truncated"
        has_shared = is_trunc or 0 < diff_end < iso_len

        if has_shared and can_len_i:
            # Anchor the shared C-terminus (identical in both proteins) to the
            # canonical C-terminus: isoform residue iso_len ↔ canonical residue
            # canonical_len. This is exact for extensions AND truncations —
            # unlike ``diff_end``, whose initiator-Met boundary is off by one on
            # truncations and would misplace shared variants by a residue.
            offset = can_len_i - iso_len
            x0, x1 = 1 + offset, iso_len + offset
            if is_trunc:
                diff_x0, diff_x1, diff_on_canon = 1, x0, True  # lost N-term on canonical
            else:
                diff_x0, diff_x1, diff_on_canon = x0, 0, False  # extension left of residue 1
        else:  # uORF / altORF / no shared region — whole isoform differential
            offset = 0
            x0, x1 = 1, iso_len
            diff_x0, diff_x1, diff_on_canon = x0, x1, False

        label = f"{iso.orf_type} · {iso.start_codon}"
        bars.append(
            {
                "label": label, "x0": x0, "x1": x1, "orf_type": iso.orf_type,
                "is_trunc": is_trunc, "diff_x0": diff_x0, "diff_x1": diff_x1,
                "diff_on_canonical": diff_on_canon, "slug": tis_slug(iso.tis_id),
            }
        )
        x_left = min(x_left, float(x0))

        # Classified InterProScan features in isoform-residue space, offset into
        # the canonical frame. (Classify directly rather than via the full
        # per-isoform protein adapter, which would also recompute variants and
        # per-sibling cell-line tracks we don't use here.)
        features = _classify_interproscan_hits(raw.get("isoform_interproscan_hits"))
        for d in features["domains"]:
            iid = d.get("interpro_id")
            domain_occ.append(
                {
                    "key": f"ipr:{iid}" if iid else f"sig:{d.get('name')}",
                    "name": d.get("name") or "domain",
                    "interpro_id": iid,
                    "x0": int(d["start"]) + offset,
                    "x1": int(d["end"]) + offset,
                    "label": label,
                }
            )
        for seg in features["disorder"]:
            disorder_iv.append((int(seg["start"]) + offset, int(seg["end"]) + offset))
        for seg in features["coiled_coil"]:
            coil_iv.append((int(seg["start"]) + offset, int(seg["end"]) + offset))
        # Motifs come straight off the raw hit column (0-based pos/end → 1-based).
        motif_hits = raw.get("isoform_motifs_hits")
        for m in (list(motif_hits)[:30] if motif_hits is not None else []):
            if not isinstance(m, dict):
                continue
            ms = m.get("start", m.get("pos"))
            if ms is None:
                continue
            try:
                mx0 = int(ms) + 1 + offset
                mx1 = int(m.get("end", ms)) + 1 + offset
            except (TypeError, ValueError):
                continue
            motifs[(m.get("name"), mx0, mx1)] = {
                "name": m.get("name", "motif"), "x0": mx0, "x1": mx1,
            }

        # Variants → canonical frame, deduped by variant_id (pathogenic wins).
        for v in getattr(iso, "variants_all", None) or []:
            if not isinstance(v, dict):
                continue
            # Retained/extension variants carry an isoform-protein position (mapped
            # to the canonical frame via ``offset``). A truncation's lost-region
            # (unique) variants aren't in the isoform protein, so they carry only a
            # canonical-frame ``protein_pos`` — that IS the display frame, no offset.
            pos_iso = v.get("isoform_protein_pos")
            pos_canon = v.get("protein_pos")
            try:
                if pos_iso is not None:
                    fr = int(float(pos_iso)) + 1 + offset
                elif pos_canon is not None:
                    fr = int(float(pos_canon)) + 1
                else:
                    continue
            except (TypeError, ValueError):
                continue
            vid = v.get("variant_id") or (
                f"{v.get('chrom')}-{v.get('genomic_pos')}-{v.get('ref')}-{v.get('alt')}"
            )
            rec = {
                "variant_id": vid, "pos": fr,
                "consequence": v.get("isoform_consequence") or v.get("consequence") or "other",
                "significance": v.get("clinical_significance"), "hgvsp": v.get("hgvsp"),
                "source": v.get("source"), "in_unique": bool(v.get("in_isoform_unique")),
            }
            prev = var_by_id.get(vid)
            if prev is None:
                var_by_id[vid] = rec
            else:
                # A variant shows up as "unique" if it lies in ANY isoform's
                # differential region. Preserve that across the pathogenic
                # upgrade below — ``prev.update(rec)`` would otherwise clobber it
                # with the current isoform's (possibly shared) flag.
                merged_unique = prev["in_unique"] or rec["in_unique"]
                if _pathogenic(rec["significance"]) and not _pathogenic(prev["significance"]):
                    prev.update(rec)
                prev["in_unique"] = merged_unique

        # Cell-line initiation: each isoform's start sits at its bar's left edge.
        for sample in _CELL_LINE_SAMPLES:
            val = raw.get(f"expr_{sample}_initiation_efficiency")
            try:
                val = float(val)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(val) or val <= 0:
                continue
            cell_by_sample.setdefault(sample, []).append(
                {"residue": x0, "log2_ie": math.log2(val), "label": label}
            )

    disorder = [{"x0": s, "x1": e} for s, e in _union_intervals([disorder_iv])]
    coils = [{"x0": s, "x1": e} for s, e in _union_intervals([coil_iv])]
    cell_lines = [
        {"sample": s, "marks": cell_by_sample[s]} for s in _CELL_LINE_SAMPLES if s in cell_by_sample
    ]

    domains = _frame_domain_clusters(domain_occ)
    # Coverage-depth chunks over the deduped domains — the collapsed lane shades
    # each stretch by how many distinct domains overlap it (reuses the sweep-line
    # depth helper, which works on ``start``/``end`` records).
    domain_segments = [
        {"x0": s["start"], "x1": s["end"], "depth": s["depth"]}
        for s in _depth_segments([{"start": d["x0"], "end": d["x1"]} for d in domains])
    ]
    variants = list(var_by_id.values())
    motif_list = list(motifs.values())

    # Anchor the canonical start at x=0 (residue 1 → 0) so extensions read as
    # negative positions: shift every computed coordinate by -1 (a global
    # translation — relative positions and dedup are unchanged).
    for b in bars:
        b["x0"] -= 1
        b["x1"] -= 1
        for k in ("diff_x0", "diff_x1"):
            if b[k] is not None:
                b[k] -= 1
    for rec in variants:
        rec["pos"] -= 1
    for coll in (domains, domain_segments, disorder, coils, motif_list):
        for it in coll:
            it["x0"] -= 1
            it["x1"] -= 1
    for t in cell_lines:
        for m in t["marks"]:
            m["residue"] -= 1
    x_left -= 1

    return types.SimpleNamespace(
        canonical_len=can_len,
        bars=bars,
        variants=variants,
        domains=domains,
        domain_segments=domain_segments,
        disorder=disorder,
        coiled_coil=coils,
        motifs=motif_list,
        cell_lines=cell_lines,
        x_left=x_left,
    )


def _isoform_to_dict(iso: Isoform) -> dict[str, Any]:
    """Project an Isoform to a JSON-shaped dict for the API endpoint."""
    return {
        "tis_id": iso.tis_id,
        "transcript_id": iso.transcript_id,
        "chrom": iso.chrom,
        "position": iso.position,
        "strand": iso.strand,
        "start_codon": iso.start_codon,
        "orf_type": iso.orf_type,
        "aa_len": iso.aa_len,
        "canonical_len": iso.canonical_len,
        "isoform_len": iso.isoform_len,
        "differential_sequence": iso.differential_sequence,
        "diff_start": iso.diff_start,
        "diff_end": iso.diff_end,
        "diff_space": iso.diff_space,
        "kozak_context": iso.kozak_context,
        "existence_score": iso.existence_score,
        "existence_evaluable": iso.existence_evaluable,
        "functional_score": iso.functional_score,
        "functional_evaluable": iso.functional_evaluable,
        "criteria": iso.criteria,
        "reasons": iso.reasons,
        "localization": {
            "canonical": iso.localization_canonical,
            "isoform": iso.localization_isoform,
            "changed": iso.localization_changed,
        },
        "conservation": {
            "phylop_unique_region_mean": iso.phylop_unique,
            "phylop_shared_region_mean": iso.phylop_shared,
            "phylop_enrichment": iso.phylop_enrichment,
        },
        "structure": {
            "plddt_isoform_mean": iso.plddt_isoform,
            "plddt_diffregion_mean": iso.plddt_diffregion,
            "tm_score": iso.tm_score,
            "rmsd_global": iso.rmsd_global,
            "isoform_cif": iso.isoform_cif,
            "canonical_cif": iso.canonical_cif,
        },
        "variants": {
            "n_total": iso.n_variants_total,
            "n_pathogenic_in_unique_region": iso.n_variants_pathogenic_unique,
            "pathogenic_in_unique_region": iso.pathogenic_variants,
        },
    }


# Module-level app for gunicorn / `flask --app swissisoform_site.app`
app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5050)), debug=True)
