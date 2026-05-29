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
    redirect,
    render_template,
    send_from_directory,
    url_for,
)
from scripts.site.build_evidence_records import slice_criterion

from swissisoform_site.data import (
    CRITERIA_FOR_PAGE,
    EXISTENCE_CRITERIA,
    FUNCTIONAL_CRITERIA,
    MODALITIES_FOR_PAGE,
    Isoform,
    _isoform_view,
    data_dir,
    llm_criterion_for_isoform,
    llm_synthesis_for_isoform,
    load_all,
    load_transcript_skeletons,
    tis_slug,
)
from swissisoform_site.plots import build_protein_figure, build_transcript_figure

# Cell line samples used by the transcript figure's bottom panel.
_CELL_LINE_SAMPLES = ("HeLa", "K562", "U2OS", "RPE1_Async", "RPE1_Que", "RPE1_Sen")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _log_llm_coverage(genes: dict, llm_dir: Path, modalities: list) -> None:
    """Print one summary line — easier to spot 'why are these tabs empty?' than per-request logs."""
    log = logging.getLogger("swissisoform_site.coverage")
    n_iso = sum(len(g.isoforms) for g in genes.values())
    n_syn = 0
    n_ex = 0
    n_fn = 0
    n_ex_cells = 0
    n_fn_cells = 0
    for gene in genes.values():
        for iso in gene.isoforms:
            slug = tis_slug(iso.tis_id)
            if (llm_dir / slug / "synthesis.json").exists():
                n_syn += 1
            ex_path = llm_dir / slug / "existence.json"
            fn_path = llm_dir / slug / "functional.json"
            if ex_path.exists():
                n_ex += 1
                try:
                    payload = json.loads(ex_path.read_text())
                    n_ex_cells += sum(1 for m in modalities if m["key"] in payload)
                except Exception:
                    pass
            if fn_path.exists():
                n_fn += 1
                try:
                    payload = json.loads(fn_path.read_text())
                    n_fn_cells += sum(1 for m in modalities if m["key"] in payload)
                except Exception:
                    pass
    total_ex_slots = sum(1 for m in modalities if m["has_existence"]) * n_iso
    total_fn_slots = sum(1 for m in modalities if m["has_functional"]) * n_iso
    log.warning(
        "%d isoforms loaded; LLM coverage: synthesis %d/%d, existence %d/%d (%d/%d cells), "
        "functional %d/%d (%d/%d cells)",
        n_iso,
        n_syn,
        n_iso,
        n_ex,
        n_iso,
        n_ex_cells,
        total_ex_slots,
        n_fn,
        n_iso,
        n_fn_cells,
        total_fn_slots,
    )


def create_app() -> Flask:
    """Build the Flask app. Factory pattern so tests can re-create cleanly."""
    app = Flask(__name__)

    # Slugify filter — strips ``:`` and ``.`` from tis_ids so they're safe
    # as DOM ids and CSS selectors.
    _slug_re = re.compile(r"[^A-Za-z0-9_-]+")

    @app.template_filter("slugify")
    def slugify(value: Any) -> str:
        return _slug_re.sub("-", str(value)).strip("-")

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

    @app.get("/genes/<gene_name>")
    def gene_page(gene_name: str) -> Any:
        """Deprecated — V1 per-gene overview removed.

        Redirects to the gene's first (sorted-by-tis_id) isoform's V2 page so
        old bookmarks and inbound links still resolve to something useful.
        """
        genes = load_all()
        gene = genes.get(gene_name) or genes.get(gene_name.upper())
        if gene is None or not gene.isoforms:
            abort(404)
        iso = sorted(gene.isoforms, key=lambda i: i.tis_id)[0]
        return redirect(
            url_for("isoform_page", gene_name=gene.name, tis_slug_str=tis_slug(iso.tis_id)),
            code=302,
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
        skeletons = load_transcript_skeletons(data_dir_path / "transcript_skeletons.parquet")
        skeleton = skeletons.get(iso.transcript_id) if iso.transcript_id else None

        llm_dir = data_dir_path / "llm"
        synthesis = llm_synthesis_for_isoform(llm_dir=llm_dir, tis_slug=tis_slug_str)

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

        criterion_slices: dict[str, dict] = {}
        criterion_llms: dict[str, dict | None] = {}
        for c in CRITERIA_FOR_PAGE:
            cid = c["id"]
            criterion_slices[cid] = slice_criterion(iso_record, cid)
            criterion_llms[cid] = llm_criterion_for_isoform(
                llm_dir=llm_dir, tis_slug=tis_slug_str, criterion_id=cid
            )

        transcript_fig = build_transcript_figure(
            _make_transcript_adapter(iso, gene), skeleton, overlays={}
        )
        protein_fig = build_protein_figure(_make_protein_adapter(iso), overlays={})

        return render_template(
            "isoform.html",
            isoform=_isoform_view(iso, gene),
            criteria=CRITERIA_FOR_PAGE,
            criterion_slices=criterion_slices,
            criterion_llms=criterion_llms,
            synthesis=synthesis,
            transcript_figure_json=json.dumps(transcript_fig),
            protein_figure_json=json.dumps(protein_fig),
            canonical_cif=iso.canonical_cif,
            isoform_cif=iso.isoform_cif,
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

    @app.errorhandler(404)
    def not_found(e: Any) -> tuple[Any, int]:
        return render_template("404.html", message=str(e)), 404

    # Startup coverage summary — one log line so missing LLM JSONs are obvious
    # before any clicking through tabs.
    _log_llm_coverage(load_all(), data_dir() / "llm", MODALITIES_FOR_PAGE)

    return app


def _make_transcript_adapter(iso: Isoform, gene: Any) -> types.SimpleNamespace:
    """Convert a V1 ``Isoform`` into the duck-typed input ``build_transcript_figure`` expects.

    Pulls all sibling TIS on the same transcript and their per-cell-line
    initiation-efficiency bars (log2-transformed, finite-only).
    """
    siblings = [s for s in gene.isoforms if s.transcript_id == iso.transcript_id]
    all_tis = [
        {"tis_id": s.tis_id, "genomic_pos": s.position, "orf_type": s.orf_type} for s in siblings
    ]
    cell_line_bars: dict[str, dict[str, float]] = {}
    for s in siblings:
        s_raw = getattr(s, "raw", None) or {}
        bars: dict[str, float] = {}
        for sample in _CELL_LINE_SAMPLES:
            v = s_raw.get(f"expr_{sample}_initiation_efficiency")
            if v is None:
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(v) or v <= 0:
                continue
            bars[sample] = math.log2(v)
        if bars:
            cell_line_bars[s.tis_id] = bars
    return types.SimpleNamespace(
        focal_tis_id=iso.tis_id,
        tis_id=iso.tis_id,
        transcript_id=iso.transcript_id,
        all_tis_on_transcript=all_tis,
        cell_line_bars=cell_line_bars,
    )


def _make_protein_adapter(iso: Isoform) -> types.SimpleNamespace:
    """Convert a V1 ``Isoform`` into the duck-typed input ``build_protein_figure`` expects.

    Variants come from ``iso.pathogenic_variants`` (already filtered to the
    unique region). Domains and motifs come from raw parquet columns and
    degrade to empty lists if the shape is unfamiliar.
    """
    raw = getattr(iso, "raw", None) or {}

    domains: list[dict[str, Any]] = []
    # ``raw`` values can be numpy arrays — their truthiness is ambiguous so we
    # check ``is None`` explicitly instead of ``or []``.
    ips_hits = raw.get("isoform_interproscan_hits")
    if ips_hits is None:
        ips_hits = []
    try:
        for h in list(ips_hits)[:30]:
            if not isinstance(h, dict):
                continue
            start = h.get("start", h.get("pos"))
            end = h.get("end")
            if start is None or end is None:
                continue
            domains.append(
                {
                    "name": (
                        h.get("name")
                        or h.get("interpro_description")
                        or h.get("description")
                        or h.get("db")
                        or "domain"
                    ),
                    "start": int(start),
                    "end": int(end),
                }
            )
    except (TypeError, ValueError):
        domains = []

    motifs: list[dict[str, Any]] = []
    motif_hits = raw.get("isoform_motifs_hits")
    if motif_hits is None:
        motif_hits = []
    try:
        for m in list(motif_hits)[:30]:
            if not isinstance(m, dict):
                continue
            start = m.get("start", m.get("pos"))
            if start is None:
                continue
            end = m.get("end", start)
            motifs.append(
                {
                    "name": m.get("name", "motif"),
                    "start": int(start),
                    "end": int(end),
                }
            )
    except (TypeError, ValueError):
        motifs = []

    variants: list[dict[str, Any]] = []
    for v in getattr(iso, "pathogenic_variants", []) or []:
        if not isinstance(v, dict):
            continue
        pos = v.get("protein_pos")
        if pos is None:
            pos = v.get("isoform_protein_pos")
        if pos is None:
            continue
        try:
            pos_int = int(float(pos))
        except (TypeError, ValueError):
            continue
        variants.append(
            {
                "variant_id": v.get("variant_id") or v.get("id") or "?",
                "protein_pos": pos_int,
                "hgvsp": v.get("hgvsp"),
                "clinical_significance": v.get("clinical_significance"),
            }
        )

    return types.SimpleNamespace(
        tis_id=iso.tis_id,
        orf_type=iso.orf_type,
        diff_space=getattr(iso, "diff_space", None),
        differential_sequence=getattr(iso, "differential_sequence", "") or "",
        isoform_length_aa=getattr(iso, "isoform_len", 0) or 0,
        canonical_length_aa=getattr(iso, "canonical_len", 0) or 0,
        variants_in_unique=variants,
        domains=domains,
        motifs=motifs,
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
