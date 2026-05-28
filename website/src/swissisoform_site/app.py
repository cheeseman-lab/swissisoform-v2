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
from pathlib import Path
from typing import Any

from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    send_from_directory,
)

from swissisoform_site.data import (
    EXISTENCE_CRITERIA,
    FUNCTIONAL_CRITERIA,
    MODALITIES_FOR_PAGE,
    Isoform,
    data_dir,
    llm_for_isoform,
    load_all,
    tis_slug,
)

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
        genes = load_all()
        gene = genes.get(gene_name) or genes.get(gene_name.upper())
        if gene is None:
            abort(404)

        # Per-isoform LLM narratives, keyed by tis_id, so the template can
        # avoid touching the raw blob.
        isoform_llm: dict[str, dict[str, Any] | None] = {
            iso.tis_id: llm_for_isoform(gene, iso.tis_id) for iso in gene.isoforms
        }

        return render_template(
            "gene.html",
            gene=gene,
            isoform_llm=isoform_llm,
            existence_criteria=EXISTENCE_CRITERIA,
            functional_criteria=FUNCTIONAL_CRITERIA,
            all_genes=sorted(genes.keys()),
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
