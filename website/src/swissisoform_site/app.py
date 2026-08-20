"""Flask routes for the SwissIsoform v2 viewer.

Almost entirely read-only — the gene/isoform payloads come from
``data.load_all()``, cached per worker at the first request. There is no DB.

The exception is the variant query: ``POST /api/variants/scan`` stores an uploaded
VCF under ``scanstore``'s temp directory, resolves it against the ORF index and
writes a digest beside the blob — the app's only write path. ``GET
/variants/<token>`` renders that digest, ``GET /api/variants/<token>.json`` returns
it raw, and ``GET /api/variants/status`` reports whether the scan path is
functional in this deployment (the index staged, the disk writable) since there is
no shell into the container.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import secrets
import types
from pathlib import Path
from typing import Any

from flask import (
    Flask,
    abort,
    g,
    has_request_context,
    jsonify,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from markupsafe import Markup, escape

from swissisoform.site.evidence import (
    CRITERIA_METRIC_LABELS,
    format_metric,
    slice_criterion,
)
from swissisoform.variantquery.scan import scan
from swissisoform_site import scanstore
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
    load_orf_index,
    sae_card_for_isoform,
    tis_slug,
    variant_rows_for_isoform,
    variant_url,
)
from swissisoform_site.genomics import interval_intersection
from swissisoform_site.plots import build_gene_protein_figure
from swissisoform_site.scanstore import ScanStoreError

# Cell line samples used by the transcript figure's bottom panel.
_CELL_LINE_SAMPLES = ("HeLa", "K562", "U2OS", "RPE1_Async", "RPE1_Que", "RPE1_Sen")

#: Upload ceiling. The real somatic VCFs are ~13 MB gzipped; 100 MB accepts
#: exome/somatic files while rejecting germline WGS, which has no business being
#: parsed synchronously inside a request.
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

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


#: Endpoints that participate in the variant-scan breadcrumb, and so should keep
#: ``?vcf=`` alive when one is active. Deliberately a whitelist: appending the token
#: to ``static`` or the structure-file routes would bust asset caching and write the
#: token into asset request logs for no benefit.
_SCAN_AWARE_ENDPOINTS = frozenset({"index", "gene_page", "isoform_page", "variants_page"})


def resolve_scan() -> tuple[str, dict[str, Any]]:
    """The request's active scan as ``(token, digest)``, resolved **once**.

    Memoised on ``g`` for two reasons. It is consulted by both the view and the URL
    injector, and the injector fires for every ``url_for`` — 17+ per page — so an
    un-cached lookup would mean that many disk reads per render.

    A token naming a scan that has expired or been swept resolves to ``("", {})``,
    which is what keeps a dead token from following the user around the site and lets
    every page render normally minus the breadcrumb. Scan context is always
    supplementary; it never fails a page.
    """
    if not has_request_context():
        return "", {}

    cached = getattr(g, "_swiss_scan", None)
    if cached is not None:
        return cached

    token = request.args.get("vcf") or ""
    # On the results page the token is a path segment, not a query arg — without
    # this that page's own outbound links would go out bare.
    if not token and request.endpoint == "variants_page":
        token = (request.view_args or {}).get("token") or ""

    digest: dict[str, Any] = {}
    if token:
        loaded = scanstore.load(token)
        if loaded.ok:
            digest = loaded.digest or {}
        else:
            token = ""

    cached = (token, digest)
    g._swiss_scan = cached
    return cached


def scan_hits(hits_key: str, hits_value: str) -> tuple[str, list[dict[str, Any]]]:
    """Active scan token plus the hits matching one field.

    Shared by the gene and isoform pages so their graceful-degradation behaviour
    cannot drift apart.

    Args:
        hits_key: Field of each hit to match on (``"gene"`` or ``"tis_id"``).
        hits_value: Value it must equal.

    Returns:
        ``(token, hits)``; ``("", [])`` when there is no usable scan.
    """
    token, digest = resolve_scan()
    if not token:
        return "", []
    hits = digest.get("hits", []) or []
    return token, [h for h in hits if h.get(hits_key) == hits_value]


def scan_context(hits_key: str, hits_value: str) -> tuple[str, int | None]:
    """As :func:`scan_hits`, but only the count — for the breadcrumb chips."""
    token, hits = scan_hits(hits_key, hits_value)
    return token, (len(hits) if token else None)


def _scan_debug_enabled() -> bool:
    """True when the full scan payload may be logged.

    Off by default on purpose. The digest carries variant positions and residues,
    and container logs are retained by the platform outside our control — the same
    reason coordinates never appear in a URL. Set
    ``SWISSISOFORM_SCAN_DEBUG=1`` only while testing with a synthetic VCF.
    """
    return os.environ.get("SWISSISOFORM_SCAN_DEBUG", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _log_scan(body: dict[str, Any], saved: Any) -> None:
    """Log a scan to stdout — i.e. to the platform's log console.

    The default line is deliberately redacted: counts, gene *names*, and ids, but
    no positions, no residues and no filename. That is enough to confirm from the
    logs that a scan ran and roughly what it found, without persisting variant
    coordinates anywhere we do not control.
    """
    counts = body.get("counts", {}) or {}
    logger.info(
        "scan token=%s key=%s cached=%s lines=%s alleles=%s non_pass=%s "
        "no_orf=%s off_contig=%s hits=%s genes=%s index=%s rejected=%s genes_hit=%s",
        body.get("vcf_id"),
        saved.key,
        saved.was_cached,
        counts.get("lines"),
        counts.get("alleles"),
        counts.get("skipped_non_pass"),
        counts.get("no_orf"),
        counts.get("off_catalog_contig"),
        counts.get("hits"),
        counts.get("genes_hit"),
        (body.get("provenance", {}) or {}).get("index_version"),
        counts.get("rejected"),
        [g.get("gene") for g in body.get("genes", []) or []],
    )
    if _scan_debug_enabled():
        logger.info(
            "scan response payload (SWISSISOFORM_SCAN_DEBUG=1)\n%s",
            json.dumps(body, indent=2, sort_keys=True, default=str),
        )


def create_app() -> Flask:
    """Build the Flask app. Factory pattern so tests can re-create cleanly."""
    app = Flask(__name__)

    # Werkzeug rejects a larger body before reading it, so an oversized upload
    # costs nothing but the 413.
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

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

    @app.url_defaults
    def _keep_scan_token(endpoint: str, values: dict[str, Any]) -> None:
        """Carry ``?vcf=`` through every navigation link automatically.

        Without this, the token has to be threaded by hand through 17 ``url_for``
        calls across 15 templates, and the breadcrumb breaks the moment one is
        missed. ``url_for`` puts any key absent from the URL rule into the query
        string, so this needs no route changes.

        Only a token that actually resolves is propagated — see ``resolve_scan``.
        Forwarding a dead one would make a stale ``?vcf=`` trail the user around the
        whole site, decorating every link with a scan that no longer exists.
        """
        if endpoint not in _SCAN_AWARE_ENDPOINTS:
            return
        # Never override an explicit value: a call site may pass vcf=None precisely
        # to drop the token, which gene.html does for its JS base.
        if "vcf" in values:
            return
        token, _digest = resolve_scan()
        if token:
            values["vcf"] = token

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

        scan_token, gene_scan_hits = scan_hits("gene", gene.name)
        view = _make_gene_protein_view(gene, uploaded=gene_scan_hits)
        gene_fig = build_gene_protein_figure(view)
        gene_fig_collapsed = build_gene_protein_figure(view, collapse_domains=True)

        # Group isoforms for the card list, mirroring the landing dropdown order.
        extensions = [i for i in gene.isoforms if i.orf_type == "extended"]
        truncations = [i for i in gene.isoforms if i.orf_type == "truncated"]
        others = [i for i in gene.isoforms if i.orf_type not in ("extended", "truncated")]

        scan_gene_hits = len(gene_scan_hits) if scan_token else None

        return render_template(
            "gene.html",
            scan_token=scan_token,
            scan_gene_hits=scan_gene_hits,
            # The click handler concatenates a path onto the base, so the query
            # string has to be kept separate — see gene.html.
            gene_path=url_for("gene_page", gene_name=gene.name, vcf=None),
            scan_qs=f"?vcf={scan_token}" if scan_token else "",
            gene=gene,
            extensions=extensions,
            truncations=truncations,
            others=others,
            protein_figure_json=json.dumps(gene_fig),
            protein_figure_collapsed_json=json.dumps(gene_fig_collapsed),
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

        # Same helper the gene page uses, keyed on this isoform rather than the
        # gene, so the breadcrumb survives one level deeper.
        scan_token, scan_isoform_hits = scan_context("tis_id", iso.tis_id)

        return render_template(
            "isoform.html",
            isoform=_isoform_view(iso, gene),
            scan_token=scan_token,
            scan_isoform_hits=scan_isoform_hits,
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
        # Named `record`, not `g` — `g` is Flask's request-global, imported above.
        for name, record in genes.items():
            payload[name] = {
                "name": record.name,
                "uniprot_id": record.uniprot_id,
                "uniprot_url": record.uniprot_url,
                "function": record.function,
                "location": record.location,
                "keywords": record.keywords,
                "canonical_len": record.canonical_len,
                "canonical_cif": record.canonical_cif,
                "llm": record.llm,
                "isoforms": [_isoform_to_dict(i) for i in record.isoforms],
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

    # ---------- Variant query ----------
    # The app's only write path. An uploaded VCF is stored under scanstore's temp
    # directory, resolved against the ORF index, and its digest written beside the
    # blob. The token returned here is what threads through the results and gene
    # pages so a scan survives navigation.

    @app.post("/api/variants/scan")
    def variants_scan() -> Any:
        """Accept a VCF upload, resolve it against the ORF index, return a token."""
        index = load_orf_index()
        if index is None:
            return jsonify(
                {
                    "error": "index_unavailable",
                    "message": (
                        "orf_index.parquet is not staged in this deployment; "
                        "run scripts/export/build_orf_index.py and re-stage."
                    ),
                }
            ), 503

        # Optional shared secret. Unset (the default, and every local dev run)
        # leaves the endpoint open; setting SWISSISOFORM_SCAN_TOKEN on a public
        # deployment closes it, since this is an unauthenticated upload path on a
        # world-reachable URL.
        required_token = os.environ.get("SWISSISOFORM_SCAN_TOKEN", "")
        if required_token:
            offered = request.headers.get("X-Scan-Token") or request.form.get("scan_token", "")
            if not secrets.compare_digest(offered, required_token):
                return jsonify(
                    {"error": "forbidden", "message": "missing or wrong scan token"}
                ), 403

        upload = request.files.get("vcf")
        if upload is None or not upload.filename:
            return jsonify(
                {"error": "no_file", "message": "attach a VCF as the 'vcf' form field"}
            ), 400

        # Sweeping here (rather than on a timer) keeps expiry enforcement off the
        # read path and out of background threads; it is rate-limited internally.
        scanstore.sweep()

        try:
            saved = scanstore.save(
                upload.stream, index_version=index.version, filename=upload.filename
            )
        except ScanStoreError as exc:
            logger.exception("scan upload failed")
            return jsonify({"error": "storage_failed", "message": str(exc)}), 507

        if saved.was_cached:
            # Same file, same index version — the finished digest is already on
            # disk, so the parse is skipped entirely.
            logger.info("scan reused cached digest key=%s", saved.key)
        else:
            result = scan(scanstore.source_path(saved.key), index)
            digest = result.to_dict()
            digest["provenance"] = {
                "vcf_sha256": saved.vcf_sha256,
                "index_version": index.version,
                "catalog_genes": index.n_genes,
                "catalog_isoforms": index.n_isoforms,
            }
            digest["filename"] = upload.filename
            scanstore.write_digest(saved.key, digest)

        loaded = scanstore.load(saved.token)
        if not loaded.ok:
            # Only reachable if the blob vanished between write and read.
            return jsonify({"error": "storage_failed", "message": "digest disappeared"}), 507

        payload = loaded.digest or {}
        response_body = {
            "vcf_id": saved.token,
            "redirect": f"/variants/{saved.token}",
            "was_cached": saved.was_cached,
            "counts": payload.get("counts", {}),
            "genes": payload.get("genes", []),
            "provenance": payload.get("provenance", {}),
            "expires_at": payload.get("expires_at", ""),
        }
        _log_scan(response_body, saved)
        return jsonify(response_body)

    @app.get("/variants/<token>")
    def variants_page(token: str) -> Any:
        """Render one scan's results: the funnel, the genes hit, and every hit."""
        loaded = scanstore.load(token)
        if loaded.expired:
            return render_template(
                "variants_gone.html",
                heading="Scan expired",
                message=(
                    "Uploaded VCFs are deleted after 24 hours. Upload the file "
                    "again to run a fresh scan."
                ),
            ), 410
        if not loaded.ok:
            return render_template(
                "variants_gone.html",
                heading="Scan not found",
                message=(
                    "That scan id is unknown — it may have expired, or the "
                    "deployment may have restarted since the upload."
                ),
            ), 404

        digest = _clean(loaded.digest) or {}
        counts = digest.get("counts", {}) or {}
        # Only genes the *displayed* catalogue knows about can be linked. The scan
        # index deliberately covers the whole catalogue, so a hit in a gene this
        # build has no page for is expected, not an error.
        known_genes = set(load_all().keys())
        return render_template(
            "variants.html",
            token=token,
            digest=digest,
            counts=counts,
            provenance=digest.get("provenance", {}) or {},
            genes=digest.get("genes", []) or [],
            hits=digest.get("hits", []) or [],
            known_genes=known_genes,
            was_cached=False,
            passing=max((counts.get("alleles") or 0) - (counts.get("skipped_non_pass") or 0), 0),
            # Alleles that landed in an ORF, derived by subtraction so the funnel
            # stays monotonic. counts["hits"] is NOT usable here: it counts
            # (variant, isoform) pairs, so one allele in three isoforms is 3 hits
            # and the last step would appear to grow. Subtraction is also
            # independent of the hit-list cap.
            in_orf_alleles=max(
                (counts.get("alleles") or 0)
                - (counts.get("skipped_non_pass") or 0)
                - (counts.get("off_catalog_contig") or 0)
                - (counts.get("no_orf") or 0),
                0,
            ),
            raw_json=json.dumps(digest, indent=2, sort_keys=True),
        )

    @app.get("/api/variants/status")
    def variants_status() -> Any:
        """Report whether the scan path is actually functional in this deployment.

        Exists because the two things most likely to differ between local and the
        Railway container — was ``orf_index.parquet`` staged, and is the ephemeral
        disk writable — cannot be told apart from a failed upload, and there is no
        shell into the container.
        """
        index = load_orf_index()
        probe_dir = scanstore.scan_dir()
        writable = False
        write_error = ""
        try:
            probe_dir.mkdir(parents=True, exist_ok=True)
            probe = probe_dir / f".writeprobe-{os.getpid()}"
            probe.write_text("ok")
            writable = probe.read_text() == "ok"
            probe.unlink(missing_ok=True)
        except OSError as exc:
            write_error = str(exc)

        return jsonify(
            {
                "index_loaded": index is not None,
                "index_version": index.version if index else "",
                "catalog_genes": index.n_genes if index else 0,
                "catalog_isoforms": index.n_isoforms if index else 0,
                "catalog_intervals": index.n_intervals if index else 0,
                "displayed_genes": len(load_all()),
                "scan_dir": str(probe_dir),
                "scan_dir_writable": writable,
                "scan_dir_error": write_error,
                "ttl_hours": scanstore.ttl_hours(),
                "budget_bytes": scanstore.budget_bytes(),
                "max_upload_bytes": app.config.get("MAX_CONTENT_LENGTH"),
                "scan_token_required": bool(os.environ.get("SWISSISOFORM_SCAN_TOKEN")),
                "debug_logging": _scan_debug_enabled(),
            }
        )

    @app.get("/api/variants/<token>.json")
    def variants_digest(token: str) -> Any:
        """Return one scan's digest: 200, 404 if unknown, 410 if past its TTL."""
        loaded = scanstore.load(token)
        if loaded.expired:
            return jsonify(
                {
                    "error": "expired",
                    "message": "this scan has expired; upload the VCF again",
                }
            ), 410
        if not loaded.ok:
            return jsonify({"error": "not_found", "message": "unknown scan id"}), 404
        response = jsonify(_clean(loaded.digest))
        # Uploaded variant data must never be indexed.
        response.headers["X-Robots-Tag"] = "noindex, nofollow"
        return response

    @app.errorhandler(413)
    def upload_too_large(e: Any) -> tuple[Any, int]:
        """JSON rather than HTML — the only client for this is the uploader."""
        # Read the live config, not the module constant, so the message stays
        # truthful if the limit is overridden.
        limit_mb = (app.config.get("MAX_CONTENT_LENGTH") or MAX_UPLOAD_BYTES) // (1024 * 1024)
        return jsonify(
            {
                "error": "too_large",
                "message": f"VCF exceeds the {limit_mb} MB upload limit",
            }
        ), 413

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
            name = (
                best.get("interpro_description") or best.get("name") or best.get("db") or "domain"
            )
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


def hgvsp_from_hit(aa_ref: str, aa_alt: str, residue: int, consequence: str) -> str:
    """HGVS protein notation for one scan hit, or ``""`` when none is derivable.

    An uploaded variant has no upstream annotator — the pipeline's own variants carry
    whatever VEP / ClinVar / COSMIC wrote — so the notation is built here from the
    classifier's amino acids. Three-letter codes and the same conventions those
    sources use, since both kinds share a tooltip:

    ========================  ==========================
    missense                  ``p.Arg253Glu``
    synonymous                ``p.Arg253=``
    stop gained               ``p.Arg253Ter``
    stop lost                 ``p.Ter253Arg``
    start lost                ``p.Leu1?``
    multi-residue (MNV)       ``p.Phe225_Glu226delinsSerLys``
    ========================  ==========================

    **The residue is numbered against the ORF the hit names, not a transcript.** The
    same nucleotide is a different residue in every ORF containing it, so the caller
    must show the frame alongside — a bare ``p.Arg253Glu`` next to a ClinVar string
    would otherwise read as the same coordinate system when it is not.

    Args:
        aa_ref: Reference residue(s), one letter each; empty for indels, which the
            classifier resolves by length without reading sequence.
        aa_alt: Alternate residue(s), same length as *aa_ref*.
        residue: 0-based residue of the first affected codon.
        consequence: The classifier's term, which decides the notation's shape.

    Returns:
        The notation, or ``""`` when there are no amino acids to name.
    """
    from Bio.Data.IUPACData import protein_letters_1to3

    def three(one: str) -> str:
        return "Ter" if one == "*" else protein_letters_1to3.get(one.upper(), one)

    if not aa_ref or len(aa_ref) != len(aa_alt):
        return ""

    first = residue + 1
    if consequence == "start_lost":
        # What changed is whether the codon still initiates, not the residue — the
        # same "?" VEP writes for p.Met1?.
        return f"p.{three(aa_ref[0])}{first}?"
    if len(aa_ref) > 1:
        last = first + len(aa_ref) - 1
        inserted = "".join(three(a) for a in aa_alt)
        return f"p.{three(aa_ref[0])}{first}_{three(aa_ref[-1])}{last}delins{inserted}"
    if aa_ref == aa_alt:
        return f"p.{three(aa_ref)}{first}="
    return f"p.{three(aa_ref)}{first}{three(aa_alt)}"


def _uploaded_variant_records(hits: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Scan hits → figure records, in the same shape the ClinVar variants use.

    Sharing the shape means they land on the same consequence rows, so an uploaded
    variant can be read against where the known pathogenics cluster. ``source`` is
    what the figure keys the distinct marker off.

    ``x`` comes from ``frame.plotly_x``, the same conversion the fixture's
    ``expect_x`` column pins — the residue is meaningless without the frame, since
    a truncation's lost region is numbered against the canonical protein.
    """
    if not hits:
        return []

    from swissisoform.variantquery.frame import plotly_x

    index = load_orf_index()
    # Keyed by (variant, x, consequence): one variant inside N isoforms yields N
    # hits, and shared-region hits all map to the SAME canonical x — so without this
    # they stack into one visible diamond with only one reachable hover. Distinct
    # variants that happen to collide at an x stay separate.
    merged: dict[tuple[str, int, str], dict[str, Any]] = {}
    isoform_counts: dict[tuple[str, int, str], set[str]] = {}

    for hit in hits:
        residue = hit.get("residue")
        if residue is None:
            # Classified but unplaceable (e.g. an indel crossing an intron). It has
            # no residue, so it has nowhere to sit on a protein axis.
            continue
        record = index.by_tis_id(hit.get("tis_id", "")) if index else None
        if record is None:
            continue
        x = plotly_x(record, int(residue), hit.get("frame", ""))
        if x is None:
            continue

        change = f"{hit.get('ref', '')}>{hit.get('alt', '')}"
        # Namespaced so it can never collide with a ClinVar/gnomAD id.
        variant_id = f"vcf:{hit.get('chrom')}:{hit.get('pos')}:{change}"
        consequence = hit.get("consequence") or "other"
        key = (variant_id, x, consequence)
        isoform_counts.setdefault(key, set()).add(hit.get("tis_id", ""))

        if key in merged:
            # A hit in the unique region of ANY isoform makes the mark prominent.
            merged[key]["in_unique"] |= hit.get("region") == "unique"
            continue
        # HGVS protein notation, in the same three-letter style the annotated
        # variants carry, so both read alike on the same tooltip.
        protein_change = hgvsp_from_hit(
            hit.get("aa_ref") or "", hit.get("aa_alt") or "", residue, consequence
        )
        if not protein_change:
            # Absent is a real answer for an indel — the classifier resolves those by
            # length without reading sequence — so say which, rather than leaving a
            # blank line in the tooltip that reads as a bug.
            protein_change = f"residue {residue + 1} — no amino-acid change resolved"
            note = hit.get("consequence_note") or ""
            if note:
                protein_change += f" ({note})"
        # Name the frame in the same breath as the number. An uploaded variant is
        # numbered against one ORF, while the ClinVar string beside it on the tooltip
        # counts against a transcript — identical-looking notations, different
        # coordinate systems, and nothing else on the mark says so.
        frame = hit.get("frame") or ""
        if frame:
            protein_change += f" · {frame} frame"

        merged[key] = {
            "variant_id": variant_id,
            "pos": x + 1,  # the caller applies the global -1 shift
            "consequence": consequence,
            "significance": None,
            "protein_change": protein_change,
            "source": "uploaded",
            "in_unique": hit.get("region") == "unique",
            "uploaded_detail": (
                f"{hit.get('chrom')}:{hit.get('pos'):,} {change} · line {hit.get('line_no')}"
            ),
        }

    for key, record in merged.items():
        n = len(isoform_counts[key])
        if n > 1:
            record["uploaded_detail"] += f" · in {n} isoforms"
    return list(merged.values())


def _make_gene_protein_view(
    gene: Any, uploaded: list[dict[str, Any]] | None = None
) -> types.SimpleNamespace:
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
    # Representative canonical-start IE per sample (max non-null over the gene's
    # isoform rows — rows sharing a canonical Tid carry identical values; max is
    # a stable pick when isoforms map to different canonical Tids).
    canon_ie_by_sample: dict[str, float] = {}
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
                # Lost N-terminus = canonical residues [1, x0-1]; residue x0 is the
                # FIRST retained (shared-core) residue where the isoform body begins,
                # so the lost-region overlay ends at x0-1 and does not bleed one
                # residue into the shared core on the canonical bar.
                diff_x0, diff_x1, diff_on_canon = 1, x0 - 1, True
            else:
                diff_x0, diff_x1, diff_on_canon = x0, 0, False  # extension left of residue 1
        else:  # uORF / altORF / no shared region — whole isoform differential
            offset = 0
            x0, x1 = 1, iso_len
            diff_x0, diff_x1, diff_on_canon = x0, x1, False

        label = f"{iso.orf_type} · {iso.start_codon}"
        bars.append(
            {
                "label": label,
                "x0": x0,
                "x1": x1,
                "orf_type": iso.orf_type,
                "is_trunc": is_trunc,
                "diff_x0": diff_x0,
                "diff_x1": diff_x1,
                "diff_on_canonical": diff_on_canon,
                "slug": tis_slug(iso.tis_id),
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
        for m in list(motif_hits)[:30] if motif_hits is not None else []:
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
                "name": m.get("name", "motif"),
                "x0": mx0,
                "x1": mx1,
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
                "variant_id": vid,
                "pos": fr,
                "consequence": v.get("isoform_consequence") or v.get("consequence") or "other",
                "significance": v.get("clinical_significance"),
                "protein_change": v.get("hgvsp"),
                "source": v.get("source"),
                "in_unique": bool(v.get("in_isoform_unique")),
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
                val = None
            if val is not None and math.isfinite(val) and val > 0:
                cell_by_sample.setdefault(sample, []).append(
                    {"residue": x0, "log2_ie": math.log2(val), "label": label}
                )
            # Canonical-start IE baseline for this sample — tracked independently
            # of whether this isoform has an alt dot in the lane, so the canonical
            # dot shows wherever the canonical start's IE is measured.
            cval = raw.get(f"canonical_expr_{sample}_initiation_efficiency")
            try:
                cval = float(cval)
            except (TypeError, ValueError):
                cval = None
            if cval is not None and math.isfinite(cval) and cval > 0:
                prev = canon_ie_by_sample.get(sample)
                if prev is None or cval > prev:
                    canon_ie_by_sample[sample] = cval

    # One gray canonical-start dot per sample at the canonical start (residue 1 →
    # x=0 after the global shift below), sized by the canonical IE.
    for sample, cval in canon_ie_by_sample.items():
        cell_by_sample.setdefault(sample, []).append(
            {
                "residue": 1,
                "log2_ie": math.log2(cval),
                "label": "canonical start",
                "canonical": True,
            }
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
    # Uploaded VCF hits are appended AFTER the dedupe, never through it: var_by_id
    # merges by variant_id with pathogenic-wins, which would absorb an uploaded hit
    # into a ClinVar record (or be absorbed by one) and lose its provenance.
    variants.extend(_uploaded_variant_records(uploaded))
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
