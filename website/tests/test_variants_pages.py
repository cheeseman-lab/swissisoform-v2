"""What a browser receives for the variant query: pages, breadcrumbs, the figure.

The scan itself is held in the browser (``scanstate.js``), so the server-rendered
part is fixed scaffolding — the drop zone, the empty ``/variants`` page, the hidden
breadcrumb slots — and the scan content is drawn by JS from the POST response.
Tests here check the scaffolding and the response; ``variants.js`` rendering is
verified by hand. The figure tests run ``gene_overlay.js`` under Node on the real
fixture's hits (see ``jsrun``).

Skipped without the ``cheeseman_test`` run and ``ecf_data/test.vcf``, which live
outside the repository.
"""

from __future__ import annotations

import io
import json
import re
import shutil
from pathlib import Path

import pytest
from jsrun import apply_uploaded_hits, needs_node

WEBSITE_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = WEBSITE_ROOT.parent / "data" / "output" / "cheeseman_test"
FIXTURE_VCF = Path("/lab/barcheese01/ating/ecf_data/test.vcf")
STAGED_FILES = ("all_paired.parquet", "variants_long.parquet", "orf_index.parquet")

pytestmark = pytest.mark.skipif(
    not (FIXTURE_VCF.is_file() and all((RUN_DIR / f).is_file() for f in STAGED_FILES)),
    reason="needs the cheeseman_test run with orf_index.parquet built, plus ecf_data/test.vcf",
)

CBX1_ISOFORM = "/genes/CBX1/isoforms/chr17-48101392---GTG-ENST00000225603-9"


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    staged = tmp_path_factory.mktemp("sitedata")
    for name in STAGED_FILES:
        shutil.copy(RUN_DIR / name, staged / name)
    (staged / "llm").mkdir()
    return staged


@pytest.fixture
def client(data_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("SWISSISOFORM_DATA_DIR", str(data_dir))
    monkeypatch.setenv("SWISSISOFORM_THROTTLE_DIR", str(tmp_path / "throttle"))
    monkeypatch.delenv("SWISSISOFORM_SCAN_TOKEN", raising=False)
    monkeypatch.delenv("SWISSISOFORM_SCAN_DEBUG", raising=False)
    # The throttle is off for the rest of the suite: these tests post several
    # uploads each, and only the throttle's own tests should feel it.
    monkeypatch.setenv("SWISSISOFORM_SCAN_RATE_HOURLY", "0")
    monkeypatch.setenv("SWISSISOFORM_SCAN_RATE_DAILY", "0")
    monkeypatch.delenv("SWISSISOFORM_TRUST_PROXY", raising=False)

    from swissisoform_site import data as site_data
    from swissisoform_site.app import create_app

    site_data.load_all.cache_clear()
    site_data.load_orf_index.cache_clear()
    site_data.orf_index_version.cache_clear()
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client
    site_data.load_all.cache_clear()
    site_data.load_orf_index.cache_clear()
    site_data.orf_index_version.cache_clear()


def _post(client, name: str = "test.vcf", **kwargs):
    with FIXTURE_VCF.open("rb") as handle:
        return client.post(
            "/api/variants/scan",
            data={"vcf": (handle, name)},
            content_type="multipart/form-data",
            **kwargs,
        )


@pytest.fixture
def scan(client) -> dict:
    """The POST response — exactly what vcf_drop.js stores for the tab."""
    return _post(client).get_json()


def _with_scan(monkeypatch, mutate) -> None:
    """Patch the route's scan() so the ScanResult can be bent into a case."""
    from swissisoform.variantquery.scan import scan as real_scan

    import swissisoform_site.app as site_app

    def patched(path, index, **kwargs):
        result = real_scan(path, index, **kwargs)
        mutate(result)
        return result

    monkeypatch.setattr(site_app, "scan", patched)


# ----------------------------------------------------------------------
# Landing page: the restructure must not break the existing controls
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "hook",
    [
        'id="gene-search"',
        'id="kw-input"',
        'id="tag-input"',
        'id="cat-toggles"',
        'id="sort-mode"',
        'id="result-count"',
    ],
)
def test_existing_filter_hooks_survive_the_layout_change(client, hook: str) -> None:
    """The controls moved into a wrapper div; every JS id hook must still be there.

    All landing-page filtering is client-side JS bound to these ids, so losing one
    silently breaks a facet with no server-side error.
    """
    assert hook in client.get("/").data.decode()


def test_controls_and_drop_zone_are_siblings_in_one_row(client) -> None:
    """Drop zone to the right of the controls, both inside .search-and-refine."""
    body = client.get("/").data.decode()
    row = body.split('class="search-and-refine"', 1)[1]
    controls = row.index('class="vq-controls"')
    drop = row.index('class="vq-drop"')
    assert controls < drop, "the drop zone should follow the controls column"


def test_drop_zone_is_wired_to_its_script(client) -> None:
    body = client.get("/").data.decode()
    assert 'id="vq-drop"' in body
    assert 'id="vq-file"' in body
    assert "js/vcf_drop.js" in body


def test_drop_zone_states_the_retention_policy(client) -> None:
    """Users uploading variant data should be told what happens to it, on the widget."""
    page = client.get("/").data.decode()
    assert "not kept" in page
    assert "this browser tab" in page


def test_the_drop_zone_names_the_assembly(client) -> None:
    """An hg19 upload succeeds and reports nonsense, so the requirement is stated
    where the file is chosen, not only in the methods prose."""
    assert "GRCh38" in client.get("/").data.decode()


# ----------------------------------------------------------------------
# The results page: the same scaffolding for everyone
# ----------------------------------------------------------------------


def test_results_page_is_scaffolding_with_no_scan_in_it(client, scan) -> None:
    """The page cannot know the scan; variants.js fills it from sessionStorage."""
    response = client.get("/variants")
    assert response.status_code == 200
    page = response.data.decode()
    for slot in ("scan-header", "scan-funnel", "scan-genes", "scan-hits", "scan-empty"):
        assert f'id="{slot}"' in page, slot
    assert "js/variants.js" in page
    assert scan["provenance"]["vcf_sha256"] not in page


def test_results_page_carries_the_site_wide_facts(client) -> None:
    """Which genes link, and the palette the figure uses — one vocabulary."""
    page = client.get("/variants").data.decode()
    config = json.loads(re.search(r'id="vq-config">(.*?)</script>', page, re.S).group(1))
    assert "CBX1" in config["known_genes"]
    from swissisoform_site.plots.protein import CONSEQ_COLOR, CONSEQ_SHORT

    assert config["conseq_color"] == CONSEQ_COLOR
    assert config["conseq_short"] == CONSEQ_SHORT
    assert config["isoform_url"] == "/genes/__GENE__/isoforms/__SLUG__"


def test_results_page_is_not_indexable(client) -> None:
    response = client.get("/variants")
    assert response.headers["X-Robots-Tag"] == "noindex, nofollow"
    assert '<meta name="robots" content="noindex, nofollow">' in response.data.decode()


def test_old_token_urls_are_gone(client) -> None:
    assert client.get("/variants/neverminted").status_code == 404
    assert client.get("/api/variants/neverminted.json").status_code == 404


# ----------------------------------------------------------------------
# What the results page draws, checked on the response it draws from
# ----------------------------------------------------------------------


def test_funnel_is_monotonic(scan) -> None:
    """``counts["hits"]`` counts (variant, isoform) pairs, so the funnel's last
    allele step is derived by subtraction instead (16 alleles must not become 23)."""
    c, d = scan["counts"], scan["derived"]
    steps = [c["lines"], c["alleles"], d["passing"], d["in_orf_alleles"]]
    assert steps == sorted(steps, reverse=True), steps


def test_hits_outnumber_variants_in_the_fixture(scan) -> None:
    """The one-to-many case the page explains ("one per (variant, isoform) pair")."""
    n_variants = sum(g["n_variants"] for g in scan["genes"])
    assert scan["counts"]["hits"] > n_variants
    assert {"CBX1", "CDC34", "MAD2L1"} <= {g["gene"] for g in scan["genes"]}


def test_a_partial_scan_is_flagged_in_the_response(client, monkeypatch) -> None:
    """A budget stop qualifies every number; variants.js banners on counts.stopped."""

    def stop(result):
        result.counts.stopped = "time"
        result.counts.lines = 12345

    _with_scan(monkeypatch, stop)
    counts = _post(client, "partial.vcf").get_json()["counts"]
    assert counts["stopped"] == "time"
    assert counts["lines"] == 12345


def test_a_complete_scan_is_not_flagged(scan) -> None:
    assert scan["counts"]["stopped"] == ""


@pytest.mark.parametrize(
    ("consequences", "suspect", "percent"),
    [
        ({"reference_mismatch": 40, "missense_variant": 10}, True, 80),
        ({"reference_mismatch": 1, "missense_variant": 49}, False, 2),
        # Two of two is 100% and means nothing — the floor exists to stop that alarm.
        ({"reference_mismatch": 2}, False, 100),
        # 8 of 12 REF-checked hits, not 8 of 812: intronic spans and indels never had
        # their REF compared, so they must not dilute the fraction.
        (
            {
                "reference_mismatch": 8,
                "missense_variant": 4,
                "intronic": 500,
                "frameshift_variant": 300,
            },
            True,
            67,
        ),
    ],
)
def test_the_assembly_check(client, monkeypatch, consequences, suspect, percent) -> None:
    """The tally the classifier already computes is the only assembly check there is."""

    def tally(result):
        result.counts.consequences = dict(consequences)

    _with_scan(monkeypatch, tally)
    assembly = _post(client).get_json()["derived"]["assembly"]
    assert assembly["suspect"] is suspect
    assert assembly["percent"] == percent


def test_hits_carry_the_classifier_output(scan) -> None:
    """The table shows term, amino acids and residue for every hit."""
    hits = scan["hits"]
    assert hits
    for h in hits:
        assert h["consequence"]
        assert {"aa_ref", "aa_alt", "residue", "frame", "consequence_note"} <= set(h)
    assert any(h["aa_ref"] and h["aa_alt"] for h in hits)


def test_a_per_hit_note_reaches_the_response(client, monkeypatch) -> None:
    """The note is why the amino acids beside it are blank; the page shows it inline."""
    import dataclasses

    from swissisoform.variantquery.scan import _TERM_NOTES

    note = _TERM_NOTES["reference_mismatch"]

    def decorate(result):
        result.hits[0] = dataclasses.replace(
            result.hits[0],
            consequence="reference_mismatch",
            aa_ref="",
            aa_alt="",
            consequence_note=note,
        )

    _with_scan(monkeypatch, decorate)
    hit = _post(client, "noted.vcf").get_json()["hits"][0]
    assert hit["consequence_note"] == note
    assert note in hit["protein_change"]


# ----------------------------------------------------------------------
# Breadcrumbs: server leaves hidden slots, scanstate.js fills them
# ----------------------------------------------------------------------


def _crumbs(body: str) -> str:
    return re.search(r'<nav class="crumbs">(.*?)</nav>', body, re.S).group(1)


def test_gene_page_has_hidden_scan_slots(client) -> None:
    nav = _crumbs(client.get("/genes/CBX1").data.decode())
    assert re.search(r'class="js-scan-crumb" hidden><a href="/variants">Variant scan</a>', nav)
    assert 'class="vq-gene-meta js-scan-hits" data-gene="CBX1" hidden' in nav


def test_isoform_page_has_hidden_scan_slots(client) -> None:
    nav = _crumbs(client.get(CBX1_ISOFORM).data.decode())
    assert 'href="/genes/CBX1"' in nav, "the gene level must always link back"
    assert 'href="/variants"' in nav
    assert 'data-tis="chr17:48101392:-:GTG:ENST00000225603.9" hidden' in nav


def test_no_page_puts_scan_state_in_a_url(client) -> None:
    """The scan lives in the tab, never in a query string; an old ?vcf is inert."""
    for path in ("/", "/genes/CBX1", CBX1_ISOFORM, "/genes/CBX1?vcf=old", "/variants"):
        body = client.get(path).data.decode()
        urls = re.findall(r'(?:href|src)="([^"]*)"', body)
        assert not [u for u in urls if "vcf=" in u], f"{path}: {urls}"


def test_figure_click_handler_builds_a_bare_isoform_url(client) -> None:
    body = client.get("/genes/CBX1").data.decode()
    assert re.search(r'const GENE_PATH = "/genes/CBX1"', body)
    assert 'GENE_PATH + "/isoforms/" + encodeURIComponent(slug);' in body
    assert "SCAN_QS" not in body


def test_every_page_states_what_a_stored_scan_is_checked_against(client) -> None:
    from swissisoform_site.data import load_orf_index
    from swissisoform_site.scanview import SCAN_SCHEMA

    body = client.get("/").data.decode()
    assert f'data-index-version="{load_orf_index().version}"' in body
    assert f'data-scan-schema="{SCAN_SCHEMA}"' in body
    assert "js/scanstate.js" in body


# ----------------------------------------------------------------------
# Status endpoint — the Railway debugging surface
# ----------------------------------------------------------------------


def test_status_reports_the_index(client) -> None:
    status = client.get("/api/variants/status").get_json()
    assert status["index_loaded"] is True
    assert len(status["index_version"]) == 16
    assert status["catalog_genes"] == 9
    assert status["catalog_isoforms"] == 18


def test_status_reports_a_missing_index(client, monkeypatch) -> None:
    monkeypatch.setattr("swissisoform_site.app.load_orf_index", lambda: None)
    status = client.get("/api/variants/status").get_json()
    assert status["index_loaded"] is False
    assert status["catalog_genes"] == 0


def test_status_reflects_the_flags(client, monkeypatch) -> None:
    assert client.get("/api/variants/status").get_json()["scan_token_required"] is False
    monkeypatch.setenv("SWISSISOFORM_SCAN_TOKEN", "s3cret")
    monkeypatch.setenv("SWISSISOFORM_SCAN_DEBUG", "1")
    status = client.get("/api/variants/status").get_json()
    assert status["scan_token_required"] is True
    assert status["debug_logging"] is True


# ----------------------------------------------------------------------
# Optional upload gate
# ----------------------------------------------------------------------


def test_upload_is_open_when_no_token_is_configured(client) -> None:
    """Unset must stay open, or every local dev run breaks."""
    response = client.post(
        "/api/variants/scan",
        data={"vcf": (io.BytesIO(FIXTURE_VCF.read_bytes()), "test.vcf")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200


def test_configured_token_is_required(client, monkeypatch) -> None:
    monkeypatch.setenv("SWISSISOFORM_SCAN_TOKEN", "s3cret")
    body = {"vcf": (io.BytesIO(FIXTURE_VCF.read_bytes()), "test.vcf")}
    denied = client.post("/api/variants/scan", data=body, content_type="multipart/form-data")
    assert denied.status_code == 403
    assert denied.get_json()["error"] == "forbidden"

    allowed = client.post(
        "/api/variants/scan",
        data={"vcf": (io.BytesIO(FIXTURE_VCF.read_bytes()), "test.vcf")},
        content_type="multipart/form-data",
        headers={"X-Scan-Token": "s3cret"},
    )
    assert allowed.status_code == 200


def test_wrong_token_is_refused(client, monkeypatch) -> None:
    monkeypatch.setenv("SWISSISOFORM_SCAN_TOKEN", "s3cret")
    response = client.post(
        "/api/variants/scan",
        data={"vcf": (io.BytesIO(FIXTURE_VCF.read_bytes()), "test.vcf")},
        content_type="multipart/form-data",
        headers={"X-Scan-Token": "wrong"},
    )
    assert response.status_code == 403


# ----------------------------------------------------------------------
# Logging: what reaches the platform's console
# ----------------------------------------------------------------------


def test_default_log_line_carries_counts_but_no_variant_positions(client, caplog) -> None:
    """Container logs are retained by the platform, so positions must stay out."""
    import logging

    with caplog.at_level(logging.INFO, logger="swissisoform_site.app"):
        _post(client)
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert re.search(r"hits=\d+", logged), logged
    assert "CBX1" in logged, "gene names are useful and not identifying"
    for leak in ("48101329", "residue", "test.vcf", "0/1:"):
        assert leak not in logged, f"{leak!r} reached the log by default"


def test_debug_flag_logs_the_full_payload(client, caplog, monkeypatch) -> None:
    """Opt-in, for verifying the response on the platform console during testing."""
    import logging

    monkeypatch.setenv("SWISSISOFORM_SCAN_DEBUG", "1")
    with caplog.at_level(logging.INFO, logger="swissisoform_site.app"):
        _post(client)
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "SWISSISOFORM_SCAN_DEBUG=1" in logged
    assert '"schema"' in logged


# ----------------------------------------------------------------------
# Uploaded variants on the gene figure — gene_overlay.js on the fixture's hits
# ----------------------------------------------------------------------


def gene_figure(client, gene: str) -> dict:
    """The Plotly figure dict the gene page inlines, before any overlay."""
    body = client.get(f"/genes/{gene}").data.decode()
    match = re.search(r"applyUploadedHits\((\{.*?\}), SWISS_HITS\);", body, re.S)
    assert match, "no GENE_PROTEIN_FIG on the page"
    return json.loads(match.group(1))


def overlaid(client, scan: dict, gene: str) -> dict:
    """What the browser draws for ``gene`` with this scan stored."""
    hits = [h for h in scan["hits"] if h["gene"] == gene]
    return apply_uploaded_hits(gene_figure(client, gene), hits)


def uploaded_traces(figure: dict) -> list[dict]:
    return [t for t in figure["data"] if t.get("name") == "uploaded VCF"]


def test_the_server_figure_has_no_uploaded_marks(client) -> None:
    assert uploaded_traces(gene_figure(client, "CBX1")) == []


@needs_node
def test_uploaded_marks_appear_with_a_scan(client, scan) -> None:
    traces = uploaded_traces(overlaid(client, scan, "CBX1"))
    assert traces
    assert sum(len(t["x"]) for t in traces) >= 2


@needs_node
def test_uploaded_marks_are_visually_distinct(client, scan) -> None:
    """Diamond plus a dark outline — the fill still encodes the consequence."""
    marker = uploaded_traces(overlaid(client, scan, "CBX1"))[0]["marker"]
    assert marker["symbol"] == "diamond"
    assert marker["line"]["width"] >= 1
    assert marker["opacity"] == 1.0
    assert marker["size"] > 9, "must outsize the ClinVar circles"


@needs_node
def test_exactly_one_legend_entry_however_many_rows(client, scan) -> None:
    """CDC34's hits span three consequence rows; the key must appear once."""
    traces = uploaded_traces(overlaid(client, scan, "CDC34"))
    assert len(traces) >= 3, "expected marks on several consequence rows"
    assert sum(1 for t in traces if t["showlegend"]) == 1


@needs_node
def test_marks_land_on_the_row_matching_their_consequence(client, scan) -> None:
    """The point of shipping the CDS: EIF2B1's unique-region SNV is silent.

    It must sit on the synonymous row (grey #94a3b8), not missense (amber #d97706).
    """
    colours = {t["marker"]["color"] for t in uploaded_traces(overlaid(client, scan, "EIF2B1"))}
    assert "#94a3b8" in colours, colours
    assert "#d97706" not in colours, "a silent variant must not be drawn as missense"


@needs_node
def test_hover_names_the_source_and_the_vcf_line(client, scan) -> None:
    hovers = [h for t in uploaded_traces(overlaid(client, scan, "CBX1")) for h in t["hovertext"]]
    assert hovers
    assert all("From your VCF" in h for h in hovers)
    assert any(re.search(r"line \d+", h) for h in hovers)
    # The nucleotide change in the detail line (escaped: hover text is HTML).
    assert any(re.search(r"chr\d+:[\d,]+ [ACGT]+&gt;[ACGT]+", h) for h in hovers), hovers
    # HGVS three-letter notation, matching how the annotated variants read...
    assert any(re.search(r"p\.[A-Z][a-z]{2}\d+([A-Z][a-z]{2}|Ter|=|\?)", h) for h in hovers)
    # ...naming the frame it counts against.
    assert all("frame" in h for h in hovers if re.search(r"p\.[A-Z][a-z]{2}\d+", h)), hovers


@needs_node
def test_one_variant_in_several_isoforms_is_a_single_mark(client, scan) -> None:
    """Shared-region hits map to the SAME canonical x in every isoform."""
    traces = uploaded_traces(overlaid(client, scan, "CBX1"))
    hovers = [h for t in traces for h in t["hovertext"]]
    assert any("in 5 isoforms" in h for h in hovers), hovers
    for trace in traces:
        by_x: dict = {}
        for x, hover in zip(trace["x"], trace["hovertext"]):
            by_x.setdefault(x, set()).add(hover)
        for x, variants in by_x.items():
            assert len(variants) == trace["x"].count(x), x


@needs_node
def test_an_absent_amino_acid_change_explains_itself_in_the_hover(client, scan) -> None:
    hovers = [h for t in uploaded_traces(overlaid(client, scan, "CDC34")) for h in t["hovertext"]]
    absent = [h for h in hovers if "no amino-acid change resolved" in h]
    assert absent, hovers
    assert re.search(r"residue \d+", absent[0])
    assert "| |" not in absent[0].replace("<br>", " | "), "no empty tooltip field"


@needs_node
def test_stop_gained_and_inframe_deletion_reach_the_figure(client, scan) -> None:
    cdc34 = [h for t in uploaded_traces(overlaid(client, scan, "CDC34")) for h in t["hovertext"]]
    assert any("inframe_deletion" in h and "residue 2" in h for h in cdc34), cdc34
    # CDC34's CTG start: a third-base change ablates it.
    assert any("start_lost" in h for h in cdc34), cdc34
    cbx1 = [h for t in uploaded_traces(overlaid(client, scan, "CBX1")) for h in t["hovertext"]]
    assert any("stop_gained" in h for h in cbx1), cbx1


@needs_node
def test_both_variant_populations_name_their_frame(client, scan) -> None:
    """Scan hits and annotated variants are numbered in different coordinate systems;
    both tooltips say which, so adjacent ones cannot read as the same thing."""
    hovers = [
        h
        for trace in overlaid(client, scan, "CBX1")["data"]
        for h in (trace.get("hovertext") or [])
        if isinstance(h, str)
    ]
    with_notation = [h for h in hovers if re.search(r"p\.[A-Z][a-z]{2}\d+", h)]
    assert with_notation, "no protein notation on any mark"
    assert not [h for h in with_notation if "frame" not in h]
    assert any("canonical frame" in h for h in with_notation)


# ----------------------------------------------------------------------
# Referrer policy, caching, and the methods text
# ----------------------------------------------------------------------


def test_every_response_carries_a_same_origin_referrer_policy(client) -> None:
    """Defence in depth: no URL carries scan data, but outbound links and CDN
    subresources still should not receive our URLs."""
    for path in ("/", "/genes/CBX1", "/healthz", "/variants"):
        response = client.get(path)
        assert response.headers.get("Referrer-Policy") == "same-origin", path


def test_the_base_template_carries_the_meta_tag(client) -> None:
    """Belt-and-braces: a page saved to disk keeps the policy the header set."""
    page = client.get("/").data.decode()
    assert '<meta name="referrer" content="same-origin">' in page


def test_the_scan_response_is_never_cached(client) -> None:
    """It carries uploaded variant positions; a cache entry would outlive the tab."""
    assert _post(client).headers.get("Cache-Control") == "no-store"


def test_the_methods_do_not_promise_a_consequence_the_code_cannot_assign(client) -> None:
    """``stop_lost`` is unreachable by construction, so the methods must not list it.

    The stored CDS carries no trailing stop and an ORF's exons span exactly
    ``aa_len * 3`` nucleotides, so the stop codon sits one base past the last mapped
    position: a variant there maps nowhere in the ORF and is counted ``intronic``.
    Measured on cheeseman_test — zero ``stop_lost`` rows across 10,764 variants in
    both frames.

    The classifier keeps the branch for a CDS with an *internal* stop, which a
    well-formed ORF does not have; what changes here is the published description,
    which promised a term no reader will ever see.
    """
    page = client.get("/about").data.decode()
    assert "Annotated ORFs exclude their stop codon" in page
    # Named only as the thing that is *not* assigned, never in the assignable list.
    assignable = page.split("assign one of")[1].split(".")[0]
    assert "stop_lost" not in assignable, assignable
    assert "start_lost" in assignable


def test_the_frame_helper_leaves_an_absent_notation_alone() -> None:
    """No notation means nothing to qualify; a bare '· canonical frame' would be noise."""
    from swissisoform_site.app import with_frame

    assert with_frame("", "canonical") == ""
    assert with_frame(None, "canonical") == ""
    assert with_frame("p.Met1Val", "") == "p.Met1Val"
    assert with_frame("p.Met1Val", "isoform") == "p.Met1Val · isoform frame"
