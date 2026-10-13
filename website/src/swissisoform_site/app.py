"""Flask routes for the SwissIsoform v2 viewer.

Almost entirely read-only — the gene/isoform payloads come from
``data.load_all()``, cached per worker at the first request. There is no DB.

The exception is the variant query: ``POST /api/variants/scan`` stages an uploaded
VCF to a temp file for the length of the request, resolves it against the ORF index
and returns the whole result (``scanview``). Nothing about a scan is kept on the
server: the browser stores the response in ``sessionStorage``, and ``GET /variants``
plus every gene/isoform page draw scan context from there. ``GET
/api/variants/status`` reports whether the scan path is functional in this
deployment (the index staged), since there is no shell into the container.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import secrets
import tempfile
import types
from pathlib import Path
from typing import Any

from flask import (
    Flask,
    abort,
    jsonify,
    make_response,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from markupsafe import Markup, escape
from swissisoform.clinical.significance import is_pathogenic
from swissisoform.site.evidence import (
    CRITERIA_METRIC_LABELS,
    format_metric,
    slice_criterion,
)
from swissisoform.variantquery.frame import x_offset_residues
from swissisoform.variantquery.portable import PORTABLE_FORMAT
from swissisoform.variantquery.scan import scan
from swissisoform.variantquery.vcf import VcfLimitExceeded
from werkzeug.middleware.proxy_fix import ProxyFix

from swissisoform_site import throttle
from swissisoform_site.browserscan import PYODIDE_URL, scanner_archive
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
    criterion_evidence_for,
    data_dir,
    llm_synthesis_for_isoform,
    load_all,
    load_orf_index,
    orf_index_version,
    portable_orf_index,
    sae_card_for_isoform,
    tis_slug,
    variant_rows_for_isoform,
    variant_url,
)
from swissisoform_site.genomics import interval_intersection
from swissisoform_site.plots import build_gene_protein_figure
from swissisoform_site.plots.protein import CONSEQ_COLOR, CONSEQ_SHORT
from swissisoform_site.scanview import (
    LIMIT_RESPONSES,
    SCAN_SCHEMA,
    StagingError,
    build_response,
    staged_upload,
    with_frame,
)

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


def _log_scan(body: dict[str, Any]) -> None:
    """Log a scan to stdout — i.e. to the platform's log console.

    The default line is deliberately redacted: counts, gene *names*, and a prefix of
    the upload's hash, but no positions, no residues and no filename. That is enough
    to confirm from the logs that a scan ran and roughly what it found, without
    persisting variant coordinates anywhere we do not control.
    """
    counts = body.get("counts", {}) or {}
    provenance = body.get("provenance", {}) or {}
    logger.info(
        "scan sha=%s lines=%s alleles=%s non_pass=%s "
        "no_orf=%s off_contig=%s hits=%s genes=%s index=%s rejected=%s genes_hit=%s",
        (provenance.get("vcf_sha256") or "")[:12],
        counts.get("lines"),
        counts.get("alleles"),
        counts.get("skipped_non_pass"),
        counts.get("no_orf"),
        counts.get("off_catalog_contig"),
        counts.get("hits"),
        counts.get("genes_hit"),
        provenance.get("index_version"),
        counts.get("rejected"),
        [g.get("gene") for g in body.get("genes", []) or []],
    )
    if _scan_debug_enabled():
        logger.info(
            "scan response payload (SWISSISOFORM_SCAN_DEBUG=1)\n%s",
            json.dumps(body, indent=2, sort_keys=True, default=str),
        )


def _dir_writable(path: Path, *, create: bool = False) -> tuple[bool, str]:
    """Whether a file can be written, read back and removed in ``path``.

    For ``/api/variants/status``: a scan stages its upload in the temp dir and
    charges the throttle in its own dir, and an unwritable one fails every upload
    in a way the uploader cannot tell from any other failure.

    Returns:
        ``(writable, error)``; ``error`` is the OSError text, or "".
    """
    try:
        if create:
            path.mkdir(parents=True, exist_ok=True)
        probe = path / f".writeprobe-{os.getpid()}"
        probe.write_text("ok")
        writable = probe.read_text() == "ok"
        probe.unlink(missing_ok=True)
    except OSError as exc:
        return False, str(exc)
    return writable, ""


def _human_delay(seconds: int) -> str:
    """Render a retry delay the way a person would say it."""
    if seconds < 60:
        return f"{seconds} seconds"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{minutes} minute{'' if minutes == 1 else 's'}"
    hours = round(seconds / 3600)
    return f"{hours} hour{'' if hours == 1 else 's'}"


def create_app() -> Flask:
    """Build the Flask app. Factory pattern so tests can re-create cleanly."""
    app = Flask(__name__)

    # Werkzeug rejects a larger body before reading it, so an oversized upload
    # costs nothing but the 413.
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

    # Behind Railway's edge proxy ``remote_addr`` is the proxy hop, so every
    # client would share one throttle bucket. ``x_for=1`` trusts exactly one hop
    # and takes the rightmost X-Forwarded-For entry — the one the proxy itself
    # appended — so a client cannot prepend a header to get a fresh bucket.
    # Opt-in: with no proxy in front, trusting the header would let anyone spoof.
    if os.environ.get("SWISSISOFORM_TRUST_PROXY") == "1":
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

    @app.after_request
    def _referrer_policy(response: Any) -> Any:
        """Send no ``Referer`` to other origins.

        The pages carry outbound links (doi.org, UniProt, PubMed) and CDN
        subresources that load without a click. No URL here carries scan data any
        more — a scan lives in the browser's ``sessionStorage``, never in a query
        string — so this is defence in depth rather than a leak fix: ``same-origin``
        sends nothing cross-origin and keeps same-origin referrers intact, without
        depending on a browser default we do not control.
        """
        response.headers.setdefault("Referrer-Policy", "same-origin")
        return response

    @app.context_processor
    def _scan_versions() -> dict[str, str]:
        """Index version and scan schema, for ``<body data-…>`` on every page.

        ``scanstate.js`` compares both with what a stored scan carries: a schema it
        does not know is discarded, and a scan resolved against another index is
        flagged instead of drawn at coordinates that may no longer be right. Read
        from the server so neither value has a second copy in the JS.
        """
        return {"index_version": orf_index_version(), "scan_schema": SCAN_SCHEMA}

    # Slugify filter: the site's one slug rule (scanview.tis_slug), so landing-page
    # dropdown links resolve to the right Flask route AND the right on-disk LLM
    # output directory.
    @app.template_filter("slugify")
    def slugify(value: Any) -> str:
        return tis_slug(value)

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
        CONSEQ_COLOR=CONSEQ_COLOR,
        CONSEQ_SHORT=CONSEQ_SHORT,
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
        # Footer only: the home page must not load the whole index to name its URL.
        index_version = orf_index_version()
        # What vcf_drop.js hands the scan worker. Empty index URL when no versioned
        # index is staged: the drop zone then says scanning is unavailable.
        browser_scan = {
            "pyodide": PYODIDE_URL,
            "scanner": url_for("variants_scanner", version=scanner_archive()[0]),
            "index": (
                url_for("variants_index", version=index_version, fmt=PORTABLE_FORMAT)
                if index_version
                else ""
            ),
        }
        return render_template(
            "index.html",
            genes=list(genes.values()),
            n_genes=len(genes),
            existence_criteria=EXISTENCE_CRITERIA,
            functional_criteria=FUNCTIONAL_CRITERIA,
            browser_scan=browser_scan,
        )

    @app.get("/about")
    def about() -> Any:
        """Static glossary — the 16 CDLMPS evidence criteria + the metrics behind them."""
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

        # Uploaded-VCF markers are not drawn here: the scan lives in the browser,
        # and gene_overlay.js adds them to both figures before the first render.
        view = _make_gene_protein_view(gene)
        gene_fig = build_gene_protein_figure(view)
        gene_fig_collapsed = build_gene_protein_figure(view, collapse_domains=True)

        # Group isoforms for the card list, mirroring the landing dropdown order.
        extensions = [i for i in gene.isoforms if i.orf_type == "extended"]
        truncations = [i for i in gene.isoforms if i.orf_type == "truncated"]
        others = [i for i in gene.isoforms if i.orf_type not in ("extended", "truncated")]

        return render_template(
            "gene.html",
            # The click handler concatenates the isoform path onto this base.
            gene_path=url_for("gene_page", gene_name=gene.name),
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
        # The reads and their flags come from the Isoform, as on the index page,
        # so the two pages cannot disagree about a category.
        category_llms = iso.category_verdicts
        category_flags = iso.category_flags

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
            category_flags=category_flags,
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
    # The app's only write path, and it writes nothing that outlives the request: the
    # upload is staged to a temp file for scan(), deleted before the response is
    # built, and the result goes back whole for the browser to keep.

    @app.post("/api/variants/scan")
    def variants_scan() -> Any:
        """Accept a VCF upload, resolve it against the ORF index, return the result."""
        # First, before the index load and the token check: it is the cheapest
        # check, and it is the one that has to hold when the endpoint is hot.
        client = request.remote_addr or "unknown"
        verdict = throttle.check(client)
        if not verdict.allowed:
            logger.info(
                "scan throttled scope=%s retry_after=%ds", verdict.scope, verdict.retry_after
            )
            response = jsonify(
                {
                    "error": "rate_limited",
                    "message": (
                        f"Too many scans from this address. Try again in "
                        f"{_human_delay(verdict.retry_after)}."
                    ),
                }
            )
            response.headers["Retry-After"] = str(verdict.retry_after)
            return response, 429

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
            # Encoded: compare_digest on two str raises TypeError for non-ASCII,
            # which would be a 500 on the very configuration this protects.
            if not secrets.compare_digest(offered.encode("utf-8"), required_token.encode("utf-8")):
                return jsonify(
                    {"error": "forbidden", "message": "missing or wrong scan token"}
                ), 403

        upload = request.files.get("vcf")
        if upload is None or not upload.filename:
            return jsonify(
                {"error": "no_file", "message": "attach a VCF as the 'vcf' form field"}
            ), 400

        # Charged before the parse, on every upload: with no stored results there
        # is no cache hit to exempt.
        throttle.record(client)
        try:
            with staged_upload(upload.stream) as staged:
                result = scan(staged.path, index)
        except StagingError as exc:
            logger.exception("scan upload could not be staged")
            return jsonify({"error": "storage_failed", "message": str(exc)}), 507
        except VcfLimitExceeded as exc:
            logger.info("scan refused kind=%s", exc.kind)
            error, status = LIMIT_RESPONSES[exc.kind]
            return jsonify({"error": error, "message": str(exc)}), status

        # The staged file is already gone; everything below works from the result.
        body = build_response(result, index=index, vcf_sha256=staged.vcf_sha256)
        _log_scan(body)
        response = jsonify(body)
        # The body carries uploaded variant positions; keep it out of every cache.
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/variants")
    def variants_page() -> Any:
        """The scan results page — the same HTML for everyone.

        It carries no scan data: ``variants.js`` reads the scan this tab stored and
        fills the page in. Only site-wide facts are rendered here — the consequence
        palette, and which genes this build has a page for, since the scan index
        covers the whole catalogue and a hit in an undisplayed gene is expected.
        """
        response = make_response(
            render_template(
                "variants.html",
                known_genes=sorted(load_all().keys()),
                conseq_color=CONSEQ_COLOR,
                conseq_short=CONSEQ_SHORT,
            )
        )
        response.headers["X-Robots-Tag"] = "noindex, nofollow"
        return response

    @app.get("/api/variants/index/<version>/f<int:fmt>.bin.gz")
    def variants_index(version: str, fmt: int) -> Any:
        """The ORF index for the browser scan, in the packed binary ``portable`` file.

        Addressed by index version *and* file format, so a URL's bytes never change
        and the response is cached as immutable: a browser that cached an older
        format can never be handed it for newer code. A version or format this
        deployment does not hold is a 404, never a different index — a scan must
        not silently resolve against coordinates it did not ask for.
        """
        portable = portable_orf_index()
        if portable is None or portable[0] != version or fmt != PORTABLE_FORMAT:
            abort(404)
        response = make_response(portable[1])
        response.headers["Content-Type"] = "application/gzip"
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response

    @app.get("/api/variants/scanner/<version>.zip")
    def variants_scanner(version: str) -> Any:
        """The scan's Python package for the browser worker (``browserscan``).

        Version-keyed and immutable like the index; a version this build does not
        have is a 404, so a page from before a deploy fails loudly instead of
        running new code against its old expectations.
        """
        current, data = scanner_archive()
        if version != current:
            abort(404)
        response = make_response(data)
        response.headers["Content-Type"] = "application/zip"
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response

    @app.get("/api/variants/status")
    def variants_status() -> Any:
        """Report whether the scan path is actually functional in this deployment.

        Exists because whether ``orf_index.parquet`` was staged, and whether the
        disk a scan writes to is writable, cannot be told apart from a failed upload,
        and there is no shell into the container.
        """
        index = load_orf_index()
        tmp_dir = Path(tempfile.gettempdir())
        tmp_writable, tmp_error = _dir_writable(tmp_dir)
        throttle_dir = throttle.throttle_dir()
        throttle_writable, throttle_error = _dir_writable(throttle_dir, create=True)
        return jsonify(
            {
                "index_loaded": index is not None,
                "index_version": index.version if index else "",
                "catalog_genes": index.n_genes if index else 0,
                "catalog_isoforms": index.n_isoforms if index else 0,
                "catalog_intervals": index.n_intervals if index else 0,
                "displayed_genes": len(load_all()),
                "tmp_dir": str(tmp_dir),
                "tmp_dir_writable": tmp_writable,
                "tmp_dir_error": tmp_error,
                "throttle_dir": str(throttle_dir),
                "throttle_dir_writable": throttle_writable,
                "throttle_dir_error": throttle_error,
                "max_upload_bytes": app.config.get("MAX_CONTENT_LENGTH"),
                "scan_token_required": bool(os.environ.get("SWISSISOFORM_SCAN_TOKEN")),
                "debug_logging": _scan_debug_enabled(),
            }
        )

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

    @app.errorhandler(429)
    def rate_limited(e: Any) -> tuple[Any, int]:
        """JSON, like the 413 — the only client for a 429 here is the uploader."""
        return jsonify(
            {
                "error": "rate_limited",
                "message": "Too many scans from this address. Try again later.",
            }
        ), 429

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
    return is_pathogenic(sig)


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


def _isoform_x_offset(
    orf_index: Any, iso: Any, canonical_len: int, isoform_len: int, has_shared: bool
) -> int | float:
    """The isoform's shift into canonical-residue space, in residues.

    ``canonical_x_offset_nt`` is the mRNA distance between the two start codons, so
    dividing by 3 gives the shift for any ORF — including uORFs and altORFs, whose
    lack of a shared C-terminus leaves the right-alignment shortcut undefined. It is
    fractional when the ORF reads out of the canonical frame, which is honest: such a
    residue has no canonical counterpart.

    Falls back to right-alignment when the index is unstaged or predates the column.
    That fallback anchors isoform residue ``isoform_len`` to canonical residue
    ``canonical_len`` — deliberately not ``diff_end``, whose initiator-Met boundary is
    off by one on truncations and would misplace shared variants by a residue. It is
    exact wherever the two proteins share a C-terminus, and reverts to the old
    ``offset = 0`` placeholder otherwise.
    """
    record = orf_index.by_tis_id(getattr(iso, "tis_id", "") or "") if orf_index else None
    if record is not None and record.canonical_x_offset_nt is not None:
        return x_offset_residues(record.canonical_x_offset_nt)
    return canonical_len - isoform_len if has_shared and canonical_len else 0


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
    # Representative canonical-start IE per sample (max non-null over the gene's
    # isoform rows — rows sharing a canonical Tid carry identical values; max is
    # a stable pick when isoforms map to different canonical Tids).
    canon_ie_by_sample: dict[str, float] = {}
    x_left = 1.0
    orf_index = load_orf_index()

    for iso in gene.isoforms:
        raw = getattr(iso, "raw", None) or {}
        diff_end = int(getattr(iso, "diff_end", 0) or 0)
        iso_len = int(getattr(iso, "isoform_len", 0) or 0) or 1
        can_len_i = int(getattr(iso, "canonical_len", 0) or 0) or can_len
        orf_type = (iso.orf_type or "").lower()
        diff_space = (getattr(iso, "diff_space", "") or "").lower()
        is_trunc = diff_space == "canonical" or orf_type == "truncated"
        has_shared = is_trunc or 0 < diff_end < iso_len

        # Where the isoform sits in canonical-residue space. Read from the ORF
        # index, so the bar, everything anchored to it, and the uploaded variant
        # markers (``variantquery.frame.plotly_x``) are all placed by the SAME
        # stored number. Two formulas here is how uORF markers ended up hundreds
        # of residues off their own bar.
        offset = _isoform_x_offset(orf_index, iso, can_len_i, iso_len, has_shared)
        x0, x1 = 1 + offset, iso_len + offset
        if has_shared and is_trunc:
            # Lost N-terminus = canonical residues [1, x0-1]; residue x0 is the
            # FIRST retained (shared-core) residue where the isoform body begins,
            # so the lost-region overlay ends at x0-1 and does not bleed one
            # residue into the shared core on the canonical bar.
            diff_x0, diff_x1, diff_on_canon = 1, x0 - 1, True
        elif has_shared:
            diff_x0, diff_x1, diff_on_canon = x0, 0, False  # extension left of residue 1
        else:  # uORF / altORF — no shared region, so the whole isoform is differential
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
                # Canonical whatever placed the mark: hgvsp is the source database's
                # own string, numbered against its canonical transcript, even where
                # the x above came from the isoform-frame position.
                "protein_change": with_frame(v.get("hgvsp"), "canonical"),
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
