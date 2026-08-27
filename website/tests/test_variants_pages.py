"""The HTML side of the variant query: drop zone, results page, breadcrumb.

Complements ``test_variants_routes.py`` (which covers the JSON API) by asserting
what a browser actually receives. Skipped without the ``cheeseman_test`` run,
which lives outside the repository.
"""

from __future__ import annotations

import io
import json
import re
import shutil
from pathlib import Path

import pytest

WEBSITE_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = WEBSITE_ROOT.parent / "data" / "output" / "cheeseman_test"
FIXTURE_VCF = Path("/lab/barcheese01/ating/ecf_data/test.vcf")
STAGED_FILES = ("all_paired.parquet", "variants_long.parquet", "orf_index.parquet")

pytestmark = pytest.mark.skipif(
    not (FIXTURE_VCF.is_file() and all((RUN_DIR / f).is_file() for f in STAGED_FILES)),
    reason="needs the cheeseman_test run with orf_index.parquet built, plus ecf_data/test.vcf",
)


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
    monkeypatch.setenv("SWISSISOFORM_SCAN_DIR", str(tmp_path / "scans"))
    monkeypatch.delenv("SWISSISOFORM_SCAN_TTL_HOURS", raising=False)
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
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client
    site_data.load_all.cache_clear()
    site_data.load_orf_index.cache_clear()


@pytest.fixture
def scan_token(client) -> str:
    with FIXTURE_VCF.open("rb") as handle:
        payload = client.post(
            "/api/variants/scan",
            data={"vcf": (handle, "test.vcf")},
            content_type="multipart/form-data",
        ).get_json()
    return payload["vcf_id"]


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
    """Users uploading variant data should be told it is deleted, on the widget."""
    assert "24" in client.get("/").data.decode()


# ----------------------------------------------------------------------
# Results page
# ----------------------------------------------------------------------


def test_results_page_renders_the_funnel_and_the_genes(client, scan_token) -> None:
    page = client.get(f"/variants/{scan_token}").data.decode()
    assert "Variant scan" in page
    assert "Genes hit" in page
    for gene in ("CBX1", "CDC34", "MAD2L1"):
        assert gene in page


def test_funnel_is_monotonic(client, scan_token) -> None:
    """Each step must be <= the one before it.

    ``counts["hits"]`` counts (variant, isoform) pairs, so using it as the last
    step made the funnel appear to grow (16 alleles -> 23 "hits"). The page derives
    in-ORF *alleles* by subtraction instead.
    """
    page = client.get(f"/variants/{scan_token}").data.decode()
    steps = [int(n.replace(",", "")) for n in re.findall(r'vq-step-num">([\d,]+)<', page)]
    assert len(steps) == 5
    # The last step is a gene count, not an allele count, so exclude it.
    alleles = steps[:4]
    assert alleles == sorted(alleles, reverse=True), f"funnel not monotonic: {steps}"


def test_results_page_explains_hits_versus_variants(client, scan_token) -> None:
    """The two numbers differ for a good reason; the page has to say so."""
    page = client.get(f"/variants/{scan_token}").data.decode()
    assert "one per (variant, isoform) pair" in page


def test_hits_table_is_labelled_hits_not_variants(client, scan_token) -> None:
    """More rows than variants — calling the table "Variants" contradicted the funnel.

    Both numbers are read back from the digest rather than hardcoded, so extending
    the fixture does not break this; what is asserted is that they differ and that
    each is labelled with the right noun.
    """
    digest = client.get(f"/api/variants/{scan_token}.json").get_json()
    n_hits = digest["counts"]["hits"]
    n_variants = sum(g["n_variants"] for g in digest["genes"])
    assert n_hits > n_variants, "the fixture must exercise the one-to-many case"

    page = re.sub(r"\s+", " ", client.get(f"/variants/{scan_token}").data.decode())
    assert f"<h2>Hits ({n_hits})</h2>" in page
    assert f"{n_variants} variants, one row per isoform" in page


def test_gene_rows_show_both_counts(client, scan_token) -> None:
    """The list has to distinguish variants from hits, or 7 vs 4 looks like a bug."""
    page = client.get(f"/variants/{scan_token}").data.decode()
    row = re.search(r'>CBX1</a>\s*<span class="vq-gene-meta">(.*?)</span>', page, re.S)
    assert row, "CBX1 row not found"
    text = re.sub(r"\s+", " ", row.group(1))
    assert re.search(r"\d+ variants? · \d+ hits? across \d+ isoforms?", text), text


def test_hits_table_names_the_isoform(client, scan_token) -> None:
    """Sibling rows for one variant differ only by isoform, so it must be a column."""
    page = client.get(f"/variants/{scan_token}").data.decode()
    assert "<th>Isoform TIS</th>" in page
    # Every row links to its isoform page.
    assert page.count("/isoforms/") >= 23


def test_results_page_names_the_catalogue_it_searched(client, scan_token) -> None:
    """Without this, zero hits is indistinguishable from a misconfigured index."""
    page = client.get(f"/variants/{scan_token}").data.decode()
    assert "isoforms" in page
    assert "index" in page


def test_gene_links_carry_the_scan_token(client, scan_token) -> None:
    page = client.get(f"/variants/{scan_token}").data.decode()
    assert re.search(rf'href="/genes/CBX1\?vcf={re.escape(scan_token)}"', page)


def test_results_page_exposes_the_scan_digest(client, scan_token) -> None:
    """The digest inline so the backend's output is inspectable in-browser.

    Everything except the hit list, which is what makes the block unbounded: it is
    rebuilt on every view of the token, so it is replaced by a count and a link.
    """
    page = client.get(f"/variants/{scan_token}").data.decode()
    assert "Scan JSON (counts, genes, provenance)" in page
    assert "&#34;counts&#34;" in page or '"counts"' in page
    assert "hits_omitted" in page
    assert f"/api/variants/{scan_token}.json" in page


def test_hit_table_is_capped_and_says_so(client, scan_token, monkeypatch) -> None:
    """A germline exome can reach scan.DEFAULT_MAX_HITS (20,000) rows.

    Rendering all of them — plus a pretty-printed digest — on every view is the
    defect; the cap must both apply and be visible, so a truncated table is never
    mistaken for the whole result.
    """
    import swissisoform_site.app as site_app

    monkeypatch.setattr(site_app, "PAGE_MAX_HITS", 1)
    page = client.get(f"/variants/{scan_token}").data.decode()

    total = len(json.loads(client.get(f"/api/variants/{scan_token}.json").data)["hits"])
    assert total > 1, "fixture needs more than one hit for the cap to mean anything"
    assert f"Showing the first 1 of {total:,}" in page
    # The heading still reports the true total, not the rendered subset.
    assert f"Hits ({total})" in page


def test_hit_table_is_not_capped_when_it_fits(client, scan_token) -> None:
    page = client.get(f"/variants/{scan_token}").data.decode()
    assert "Showing the first" not in page


def test_results_page_is_not_indexable(client, scan_token) -> None:
    """The JSON twin has always sent this; the page rendering the same hits did not.

    A token is a capability, and it travels in a URL — pasted into a chat, an issue
    tracker or a synced browser, all of which are crawled.
    """
    response = client.get(f"/variants/{scan_token}")
    assert response.headers["X-Robots-Tag"] == "noindex, nofollow"
    assert '<meta name="robots" content="noindex, nofollow">' in response.data.decode()


def test_dead_token_pages_are_not_indexable(client, scan_token, monkeypatch) -> None:
    """404 and 410 are reached by token too, so they carry the header as well."""
    unknown = client.get("/variants/neverminted")
    assert unknown.status_code == 404
    assert unknown.headers["X-Robots-Tag"] == "noindex, nofollow"

    monkeypatch.setenv("SWISSISOFORM_SCAN_TTL_HOURS", "0")
    expired = client.get(f"/variants/{scan_token}")
    assert expired.status_code == 410
    assert expired.headers["X-Robots-Tag"] == "noindex, nofollow"


def test_unknown_scan_renders_a_page_not_a_traceback(client) -> None:
    response = client.get("/variants/neverminted")
    assert response.status_code == 404
    assert "Scan not found" in response.data.decode()


def test_expired_scan_renders_410_with_an_explanation(client, scan_token, monkeypatch) -> None:
    monkeypatch.setenv("SWISSISOFORM_SCAN_TTL_HOURS", "0")
    response = client.get(f"/variants/{scan_token}")
    assert response.status_code == 410
    body = response.data.decode()
    assert "Scan expired" in body
    assert "24 hours" in body


# ----------------------------------------------------------------------
# Gene page breadcrumb
# ----------------------------------------------------------------------


def test_gene_page_links_back_to_the_scan(client, scan_token) -> None:
    body = client.get(f"/genes/CBX1?vcf={scan_token}").data.decode()
    assert f"/variants/{scan_token}" in body
    assert re.search(r"\d+ variants? from the uploaded VCF", body)


def test_gene_page_without_a_token_has_no_breadcrumb(client) -> None:
    body = client.get("/genes/CBX1").data.decode()
    assert "from the uploaded VCF" not in body


def test_gene_page_ignores_a_stale_token_rather_than_failing(client) -> None:
    """A gene page must render on its own; an expired scan only loses the crumb."""
    response = client.get("/genes/CBX1?vcf=doesnotexist")
    assert response.status_code == 200
    assert "from the uploaded VCF" not in response.data.decode()


# ----------------------------------------------------------------------
# The breadcrumb trail: index / [variant scan] / gene / isoform
# ----------------------------------------------------------------------

CBX1_ISOFORM = "/genes/CBX1/isoforms/chr17-48101392---GTG-ENST00000225603-9"


def crumbs(client, path: str) -> tuple[str, str]:
    """Return the page body and its breadcrumb as flattened text."""
    body = client.get(path).data.decode()
    match = re.search(r'<nav class="crumbs">(.*?)</nav>', body, re.S)
    trail = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", match.group(1))).strip() if match else ""
    return body, trail


def test_isoform_crumb_links_its_gene_even_without_a_scan(client) -> None:
    """Pre-existing gap: the gene level used to be plain text, so an isoform page
    had no way back to its gene at all."""
    body, trail = crumbs(client, CBX1_ISOFORM)
    assert 'href="/genes/CBX1"' in body
    assert trail.startswith("Index / CBX1 /")
    assert "Variant scan" not in trail


def test_isoform_crumb_shows_all_four_levels_with_a_scan(client, scan_token) -> None:
    body, trail = crumbs(client, f"{CBX1_ISOFORM}?vcf={scan_token}")
    assert re.match(r"Index / Variant scan / CBX1 / chr17:48101392", trail), trail
    assert f"/variants/{scan_token}" in body, "no link back to the scan"
    assert f'href="/genes/CBX1?vcf={scan_token}"' in body, "no link back to the gene"


def test_isoform_page_counts_hits_on_this_isoform(client, scan_token) -> None:
    """Per-isoform, not per-gene: CBX1 has 7 gene hits spread over 5 isoforms."""
    _, trail = crumbs(client, f"{CBX1_ISOFORM}?vcf={scan_token}")
    match = re.search(r"(\d+) variants? from the uploaded VCF hit this isoform", trail)
    assert match, trail
    assert 1 <= int(match.group(1)) < 7


def test_isoform_page_ignores_a_stale_token(client) -> None:
    body, trail = crumbs(client, f"{CBX1_ISOFORM}?vcf=doesnotexist")
    assert "Variant scan" not in trail
    assert 'href="/genes/CBX1"' in body, "the gene link must survive a dead token"


def test_gene_page_isoform_cards_carry_the_token(client, scan_token) -> None:
    """Five cards; missing the token on any one silently ends the trail."""
    body = client.get(f"/genes/CBX1?vcf={scan_token}").data.decode()
    cards = set(re.findall(r'href="(/genes/CBX1/isoforms/[^"]+)"', body))
    assert len(cards) == 5
    assert all(f"vcf={scan_token}" in c for c in cards)


def test_figure_click_handler_keeps_path_and_query_separate(client, scan_token) -> None:
    """The handler concatenates a path onto its base, so the base must stay bare.

    A base already carrying "?vcf=..." would yield
    /genes/CBX1?vcf=TOK/isoforms/<slug> — the gene page with junk in the query.
    """
    body = client.get(f"/genes/CBX1?vcf={scan_token}").data.decode()
    gene_path = re.search(r'const GENE_PATH = "([^"]*)"', body).group(1)
    scan_qs = re.search(r'const SCAN_QS = "([^"]*)"', body).group(1)
    assert gene_path == "/genes/CBX1", gene_path
    assert scan_qs == f"?vcf={scan_token}"
    assert "encodeURIComponent(slug) + SCAN_QS" in body


def test_results_page_gene_links_are_not_doubly_parameterised(client, scan_token) -> None:
    """The template used to append ?vcf by hand; with the injector that doubled it."""
    body = client.get(f"/variants/{scan_token}").data.decode()
    links = set(re.findall(r'href="(/genes/[^"]+)"', body))
    assert links
    for link in links:
        assert link.count("?") <= 1, f"malformed URL: {link}"
        assert f"vcf={scan_token}" in link


def test_asset_urls_never_gain_the_token(client, scan_token) -> None:
    """Injecting into static/structure URLs would bust caching and log the token."""
    body = client.get(f"/genes/CBX1?vcf={scan_token}").data.decode()
    assets = re.findall(r'(?:href|src)="(/(?:static|structures|structure-[a-z]+)/[^"]*)"', body)
    assert assets, "expected some asset URLs on the gene page"
    assert not [a for a in assets if "vcf=" in a]


def test_index_links_keep_the_scan_alive(client, scan_token) -> None:
    """Index is in the trail: its gene cards must not silently drop the token."""
    body = client.get(f"/?vcf={scan_token}").data.decode()
    cards = set(re.findall(r'href="(/genes/[^"/]+\?[^"]*)"', body))
    assert cards
    assert all(f"vcf={scan_token}" in c for c in cards)


def test_untokenised_pages_stay_clean(client) -> None:
    """The injector must be a no-op with no scan — no ?vcf= on any generated URL.

    Checks href/src attributes rather than the raw body: the page legitimately
    contains the literal text "?vcf=" inside a JS comment explaining the split.
    """
    for path in ("/", "/genes/CBX1", CBX1_ISOFORM):
        body = client.get(path).data.decode()
        urls = re.findall(r'(?:href|src)="([^"]*)"', body)
        assert not [u for u in urls if "vcf=" in u], f"{path}: {urls}"


def test_a_dead_token_is_not_propagated(client) -> None:
    """A stale ?vcf must not be forwarded, or it trails the user around the site."""
    body = client.get("/genes/CBX1?vcf=doesnotexist").data.decode()
    urls = re.findall(r'(?:href|src)="([^"]*)"', body)
    assert not [u for u in urls if "vcf=" in u], "a dead token leaked into links"


# ----------------------------------------------------------------------
# Status endpoint — the Railway debugging surface
# ----------------------------------------------------------------------


def test_status_reports_the_index_and_a_real_write_probe(client) -> None:
    status = client.get("/api/variants/status").get_json()
    assert status["index_loaded"] is True
    assert len(status["index_version"]) == 16
    assert status["catalog_genes"] == 9
    assert status["catalog_isoforms"] == 18
    # An actual write-and-delete, not a stat: this is the thing most likely to
    # differ between a laptop and a container.
    assert status["scan_dir_writable"] is True
    assert status["scan_dir_error"] == ""
    assert status["ttl_hours"] == 24.0


def test_status_reports_an_unwritable_scan_dir(client, monkeypatch, tmp_path) -> None:
    """The failure this endpoint exists to diagnose must actually be detected."""
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(0o500)
    monkeypatch.setenv("SWISSISOFORM_SCAN_DIR", str(blocked / "scans"))
    try:
        status = client.get("/api/variants/status").get_json()
        assert status["scan_dir_writable"] is False
        assert status["scan_dir_error"]
    finally:
        blocked.chmod(0o700)


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
        with FIXTURE_VCF.open("rb") as handle:
            client.post(
                "/api/variants/scan",
                data={"vcf": (handle, "test.vcf")},
                content_type="multipart/form-data",
            )
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
        with FIXTURE_VCF.open("rb") as handle:
            client.post(
                "/api/variants/scan",
                data={"vcf": (handle, "test.vcf")},
                content_type="multipart/form-data",
            )
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "SWISSISOFORM_SCAN_DEBUG=1" in logged
    assert '"redirect"' in logged


# ----------------------------------------------------------------------
# Uploaded variants on the gene figure
# ----------------------------------------------------------------------


def gene_figure(client, path: str) -> dict:
    """The Plotly figure dict the gene page inlines for the combined view."""
    body = client.get(path).data.decode()
    match = re.search(r"GENE_PROTEIN_FIG\s*=\s*(\{.*?\});", body, re.S)
    assert match, "no GENE_PROTEIN_FIG on the page"
    return json.loads(match.group(1))


def uploaded_traces(figure: dict) -> list[dict]:
    return [t for t in figure["data"] if t.get("marker", {}).get("symbol") == "diamond"]


def test_no_uploaded_marks_without_a_scan(client) -> None:
    figure = gene_figure(client, "/genes/CBX1")
    assert uploaded_traces(figure) == []
    assert not [t for t in figure["data"] if t.get("name") == "uploaded VCF"]


def test_uploaded_marks_appear_with_a_scan(client, scan_token) -> None:
    traces = uploaded_traces(gene_figure(client, f"/genes/CBX1?vcf={scan_token}"))
    assert traces
    assert sum(len(t["x"]) for t in traces) >= 2


def test_uploaded_marks_are_visually_distinct(client, scan_token) -> None:
    """Diamond plus a dark outline — the fill still encodes the consequence."""
    trace = uploaded_traces(gene_figure(client, f"/genes/CBX1?vcf={scan_token}"))[0]
    marker = trace["marker"]
    assert marker["symbol"] == "diamond"
    assert marker["line"]["width"] >= 1
    assert marker["opacity"] == 1.0
    assert marker["size"] > 9, "must outsize the ClinVar circles"


def test_exactly_one_legend_entry_however_many_rows(client, scan_token) -> None:
    """CDC34's hits span three consequence rows; the key must appear once."""
    figure = gene_figure(client, f"/genes/CDC34?vcf={scan_token}")
    traces = uploaded_traces(figure)
    assert len(traces) >= 3, "expected marks on several consequence rows"
    named = [t for t in figure["data"] if t.get("showlegend") and t.get("name") == "uploaded VCF"]
    assert len(named) == 1


def test_marks_land_on_the_row_matching_their_consequence(client, scan_token) -> None:
    """The point of shipping the CDS: EIF2B1's unique-region SNV is silent.

    It must sit on the synonymous row (grey #94a3b8), not missense (amber #d97706).
    A "SNV means missense" guess would have drawn it as a missense hit inside a
    differential region.
    """
    traces = uploaded_traces(gene_figure(client, f"/genes/EIF2B1?vcf={scan_token}"))
    assert traces
    colours = {t["marker"]["color"] for t in traces}
    assert "#94a3b8" in colours, colours
    assert "#d97706" not in colours, "a silent variant must not be drawn as missense"


def test_hover_names_the_source_and_the_vcf_line(client, scan_token) -> None:
    traces = uploaded_traces(gene_figure(client, f"/genes/CBX1?vcf={scan_token}"))
    hovers = [h for t in traces for h in t["hovertext"]]
    assert hovers
    assert all("From your VCF" in h for h in hovers)
    assert any(re.search(r"line \d+", h) for h in hovers)
    # The nucleotide change belongs in the detail line, and the protein change on
    # its own — a bug once put the protein change in both, losing the REF>ALT.
    assert any(re.search(r"chr\d+:[\d,]+ [ACGT]+>[ACGT]+", h) for h in hovers), hovers
    # HGVS three-letter notation, matching how the annotated variants read.
    assert any(re.search(r"p\.[A-Z][a-z]{2}\d+([A-Z][a-z]{2}|Ter|=|\?)", h) for h in hovers), hovers
    # ...and it must name the frame it counts against, since the identical-looking
    # ClinVar string beside it is numbered against a transcript instead.
    assert all(
        "frame" in h for h in hovers if re.search(r"p\.[A-Z][a-z]{2}\d+", h)
    ), hovers


def test_one_variant_in_several_isoforms_is_a_single_mark(client, scan_token) -> None:
    """Shared-region hits map to the SAME canonical x in every isoform.

    Without merging they stack into one visible diamond with only one reachable
    hover, so the count is stated in the tooltip instead.
    """
    traces = uploaded_traces(gene_figure(client, f"/genes/CBX1?vcf={scan_token}"))
    hovers = [h for t in traces for h in t["hovertext"]]
    assert any("in 5 isoforms" in h for h in hovers), hovers
    for trace in traces:
        counts = {}
        for x, hover in zip(trace["x"], trace["hovertext"]):
            counts.setdefault(x, set()).add(hover)
        for x, variants in counts.items():
            # More than one mark at an x is only allowed for genuinely different
            # variants (the fixture's multi-allelic row is two distinct alleles).
            assert len(variants) == len([1 for v in trace["x"] if v == x]), x


def test_a_stale_token_draws_nothing(client) -> None:
    figure = gene_figure(client, "/genes/CBX1?vcf=doesnotexist")
    assert uploaded_traces(figure) == []


def test_an_absent_amino_acid_change_explains_itself_in_the_hover(client, scan_token) -> None:
    """Indels carry no amino acids, so the tooltip must say so rather than blank out.

    The classifier resolves an indel from its length delta without reading sequence —
    the same answer an annotated variant gets — so there is a residue but no residue
    change. Leaving the field empty would read as a rendering bug.
    """
    traces = uploaded_traces(gene_figure(client, f"/genes/CDC34?vcf={scan_token}"))
    hovers = [h for t in traces for h in t["hovertext"]]
    absent = [h for h in hovers if "no amino-acid change resolved" in h]
    assert absent, hovers
    assert re.search(r"residue \d+", absent[0])
    assert "| |" not in absent[0].replace("<br>", " | "), "no empty tooltip field"


def test_stop_gained_and_inframe_deletion_reach_the_figure(client, scan_token) -> None:
    """Both rows were unreachable before the fixture gained these cases."""
    cdc34 = uploaded_traces(gene_figure(client, f"/genes/CDC34?vcf={scan_token}"))
    hovers = [h for t in cdc34 for h in t["hovertext"]]
    assert any("inframe_deletion" in h and "residue 2" in h for h in hovers), hovers
    # CDC34's CTG start: a first-base change keeps a near-cognate start, so it is
    # synonymous, while the third-base row ablates it.
    assert any("start_lost" in h for h in hovers), hovers

    cbx1 = uploaded_traces(gene_figure(client, f"/genes/CBX1?vcf={scan_token}"))
    cbx1_hovers = [h for t in cbx1 for h in t["hovertext"]]
    assert any("stop_gained" in h for h in cbx1_hovers), cbx1_hovers


def test_a_partial_scan_says_so_on_the_results_page(client, monkeypatch) -> None:
    """A budget stop qualifies every number in the funnel, so it has to be visible.

    Rendered as its own banner above the funnel rather than appended to the
    "Dropped" line: it is a caveat on all of those counts, not another one.
    """
    import swissisoform_site.app as site_app
    from swissisoform.variantquery.scan import scan as real_scan

    def stopped_scan(path, index, **kwargs):
        result = real_scan(path, index, **kwargs)
        result.counts.stopped = "time"
        result.counts.lines = 12345
        return result

    monkeypatch.setattr(site_app, "scan", stopped_scan)

    with FIXTURE_VCF.open("rb") as handle:
        token = client.post(
            "/api/variants/scan",
            data={"vcf": (handle, "partial.vcf")},
            content_type="multipart/form-data",
        ).get_json()["vcf_id"]

    page = client.get(f"/variants/{token}").data.decode()
    assert "This scan did not finish" in page
    assert "12,345 records" in page
    assert "on its time budget" in page


def test_a_complete_scan_shows_no_partial_banner(client, scan_token) -> None:
    page = client.get(f"/variants/{scan_token}").data.decode()
    assert "This scan did not finish" not in page


# ----------------------------------------------------------------------
# GRCh38 labelling, the REF-mismatch tally, and the classifier columns
# (PR #29 gates 2 and 6)
# ----------------------------------------------------------------------


def test_the_drop_zone_names_the_assembly(client) -> None:
    """An hg19 upload succeeds and reports nonsense, so the requirement is stated
    where the file is chosen, not only in the methods prose."""
    page = client.get("/").data.decode()
    assert "GRCh38" in page


def test_the_results_header_names_the_assembly(client, scan_token) -> None:
    page = client.get(f"/variants/{scan_token}").data.decode()
    assert "GRCh38" in page


def test_a_high_ref_mismatch_fraction_warns_about_the_assembly(client, monkeypatch) -> None:
    """The tally the classifier already computes is the only assembly check there is."""
    import swissisoform_site.app as site_app
    from swissisoform.variantquery.scan import scan as real_scan

    def mismatching(path, index, **kwargs):
        result = real_scan(path, index, **kwargs)
        result.counts.consequences = {"reference_mismatch": 40, "missense_variant": 10}
        return result

    monkeypatch.setattr(site_app, "scan", mismatching)
    with FIXTURE_VCF.open("rb") as handle:
        token = client.post(
            "/api/variants/scan",
            data={"vcf": (handle, "hg19ish.vcf")},
            content_type="multipart/form-data",
        ).get_json()["vcf_id"]

    page = client.get(f"/variants/{token}").data.decode()
    assert "may not be GRCh38" in page
    assert "80%" in page


def test_a_low_ref_mismatch_count_is_reported_without_the_warning(client, monkeypatch) -> None:
    """One or two mismatches are ordinary — reported, but not an alarm."""
    import swissisoform_site.app as site_app
    from swissisoform.variantquery.scan import scan as real_scan

    def one_mismatch(path, index, **kwargs):
        result = real_scan(path, index, **kwargs)
        result.counts.consequences = {"reference_mismatch": 1, "missense_variant": 49}
        return result

    monkeypatch.setattr(site_app, "scan", one_mismatch)
    with FIXTURE_VCF.open("rb") as handle:
        token = client.post(
            "/api/variants/scan",
            data={"vcf": (handle, "one_off.vcf")},
            content_type="multipart/form-data",
        ).get_json()["vcf_id"]

    page = client.get(f"/variants/{token}").data.decode()
    assert "may not be GRCh38" not in page
    assert "disagree with GRCh38" in page


def test_a_tiny_scan_does_not_trigger_the_assembly_warning(client, monkeypatch) -> None:
    """Two of two is 100% and means nothing — the floor exists to stop that alarm."""
    import swissisoform_site.app as site_app
    from swissisoform.variantquery.scan import scan as real_scan

    def two_of_two(path, index, **kwargs):
        result = real_scan(path, index, **kwargs)
        result.counts.consequences = {"reference_mismatch": 2}
        return result

    monkeypatch.setattr(site_app, "scan", two_of_two)
    with FIXTURE_VCF.open("rb") as handle:
        token = client.post(
            "/api/variants/scan",
            data={"vcf": (handle, "tiny.vcf")},
            content_type="multipart/form-data",
        ).get_json()["vcf_id"]

    assert "may not be GRCh38" not in client.get(f"/variants/{token}").data.decode()


def test_the_hit_table_shows_the_classifier_output(client, scan_token) -> None:
    """The shared classifier computes a term and amino acids for every hit; the page
    that presents the scan has to show them."""
    page = client.get(f"/variants/{scan_token}").data.decode()
    hits = json.loads(client.get(f"/api/variants/{scan_token}.json").data)["hits"]

    assert "<th>Consequence</th>" in page
    assert "<th>AA change</th>" in page

    terms = {h["consequence"] for h in hits}
    assert terms, "fixture must produce at least one classified hit"
    # Rendered with the figure's short labels, not the raw term.
    from swissisoform_site.plots.protein import CONSEQ_SHORT

    for term in terms:
        assert CONSEQ_SHORT.get(term, term.replace("_", " ")) in page

    changed = [h for h in hits if h.get("aa_ref") and h.get("aa_alt")]
    if changed:
        h = changed[0]
        assert f"{h['aa_ref']}&rarr;{h['aa_alt']}" in page


def test_a_per_hit_note_is_visible_not_only_a_tooltip(client, monkeypatch) -> None:
    """The note is the reason the amino acids beside it are blank; a title= would be
    unreachable on touch and invisible in print.

    The note is forced rather than fished out of the fixture, which need not contain
    one — a skip here would mean the rendering was never checked at all.
    """
    import dataclasses

    from markupsafe import escape

    import swissisoform_site.app as site_app
    from swissisoform.variantquery.scan import _TERM_NOTES
    from swissisoform.variantquery.scan import scan as real_scan

    note = _TERM_NOTES["reference_mismatch"]

    def with_a_note(path, index, **kwargs):
        result = real_scan(path, index, **kwargs)
        assert result.hits, "fixture must produce at least one hit to decorate"
        result.hits[0] = dataclasses.replace(
            result.hits[0],
            consequence="reference_mismatch",
            aa_ref="",
            aa_alt="",
            consequence_note=note,
        )
        return result

    monkeypatch.setattr(site_app, "scan", with_a_note)
    with FIXTURE_VCF.open("rb") as handle:
        token = client.post(
            "/api/variants/scan",
            data={"vcf": (handle, "noted.vcf")},
            content_type="multipart/form-data",
        ).get_json()["vcf_id"]

    page = client.get(f"/variants/{token}").data.decode()
    # Compared escaped: the note contains an apostrophe, and rendering it raw
    # would mean the template was not escaping hit text at all.
    assert str(escape(note)) in page
    assert "conseq-note" in page
    # The term has no short label, so it falls back to the readable form.
    assert "reference mismatch" in page


def test_intronic_and_indel_hits_do_not_dilute_the_mismatch_fraction(client, monkeypatch) -> None:
    """The classifier only reaches reference_mismatch on the substitution path.

    Counting intronic spans and indels in the denominator would hide a real
    assembly mismatch behind a pile of hits whose REF was never compared.
    """
    import swissisoform_site.app as site_app
    from swissisoform.variantquery.scan import scan as real_scan

    def diluted(path, index, **kwargs):
        result = real_scan(path, index, **kwargs)
        result.counts.consequences = {
            "reference_mismatch": 8,
            "missense_variant": 4,
            # Neither of these ever had a REF comparison.
            "intronic": 500,
            "frameshift_variant": 300,
        }
        return result

    monkeypatch.setattr(site_app, "scan", diluted)
    with FIXTURE_VCF.open("rb") as handle:
        token = client.post(
            "/api/variants/scan",
            data={"vcf": (handle, "diluted.vcf")},
            content_type="multipart/form-data",
        ).get_json()["vcf_id"]

    page = client.get(f"/variants/{token}").data.decode()
    # 8 of 12 REF-checked hits, not 8 of 812.
    assert "may not be GRCh38" in page
    assert "67%" in page


def test_the_table_and_the_figure_name_consequences_identically(client, scan_token) -> None:
    """One vocabulary: the table imports the map the figure draws with."""
    import swissisoform_site.app as site_app
    from swissisoform_site.plots.protein import CONSEQ_COLOR, CONSEQ_SHORT

    assert site_app.CONSEQ_COLOR is CONSEQ_COLOR
    assert site_app.CONSEQ_SHORT is CONSEQ_SHORT
    page = client.get(f"/variants/{scan_token}").data.decode()
    assert "conseq-chip" in page
