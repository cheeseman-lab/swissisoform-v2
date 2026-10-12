"""The variant-query endpoint, end to end through Flask, against the fixture VCF.

Exercises the contract the front end depends on: a VCF in, the whole result out
(the browser keeps it; the server keeps nothing), and the failure modes mapped onto
distinct status codes rather than a generic 500. ``test_scan_route.py`` covers the
same contract with a VCF it writes itself, for when this fixture is absent.

Skipped without the ``cheeseman_test`` run, which lives outside the repository.
"""

from __future__ import annotations

import gzip
import io
import shutil
from pathlib import Path

import pytest

WEBSITE_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = WEBSITE_ROOT.parent / "data" / "output" / "cheeseman_test"
FIXTURE_VCF = Path("/lab/barcheese01/ating/ecf_data/test.vcf")

STAGED_FILES = ("all_paired.parquet", "variants_long.parquet", "orf_index.parquet")

pytestmark = pytest.mark.skipif(
    not (FIXTURE_VCF.is_file() and all((RUN_DIR / f).is_file() for f in STAGED_FILES)),
    reason=(
        "needs the cheeseman_test run with orf_index.parquet built "
        "(python scripts/export/build_orf_index.py --run cheeseman_test) plus ecf_data/test.vcf"
    ),
)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    """A staged data dir shaped like the deployed image's ``/app/data``."""
    staged = tmp_path_factory.mktemp("sitedata")
    for name in STAGED_FILES:
        shutil.copy(RUN_DIR / name, staged / name)
    (staged / "llm").mkdir()
    return staged


@pytest.fixture
def client(data_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("SWISSISOFORM_DATA_DIR", str(data_dir))
    monkeypatch.setenv("SWISSISOFORM_THROTTLE_DIR", str(tmp_path / "throttle"))
    # The throttle is off for the rest of the suite: these tests post several
    # uploads each, and only the throttle's own tests should feel it.
    monkeypatch.setenv("SWISSISOFORM_SCAN_RATE_HOURLY", "0")
    monkeypatch.setenv("SWISSISOFORM_SCAN_RATE_DAILY", "0")
    monkeypatch.delenv("SWISSISOFORM_TRUST_PROXY", raising=False)

    from swissisoform_site import data as site_data
    from swissisoform_site.app import create_app

    # These loaders are lru_cached per worker; clear them so the tmp data dir is
    # actually read instead of a previous test's.
    site_data.load_all.cache_clear()
    site_data.load_orf_index.cache_clear()

    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client

    site_data.load_all.cache_clear()
    site_data.load_orf_index.cache_clear()


def _post(client, body: bytes, name: str = "test.vcf"):
    return client.post(
        "/api/variants/scan",
        data={"vcf": (io.BytesIO(body), name)},
        content_type="multipart/form-data",
    )


@pytest.fixture
def vcf_bytes() -> bytes:
    return FIXTURE_VCF.read_bytes()


# ----------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------


def test_scan_returns_the_funnel_and_no_token(client, vcf_bytes) -> None:
    payload = _post(client, vcf_bytes).get_json()
    assert "vcf_id" not in payload and "redirect" not in payload

    counts = payload["counts"]
    # Totals are derived, not literal, so adding fixture coverage does not break
    # this. The per-category counts below ARE literal: each names one specific
    # fixture row, and they are what makes a zero-hit scan explainable.
    assert counts["lines"] > 0
    assert counts["hits"] == sum(g["n_hits"] for g in payload["genes"])
    assert counts["skipped_non_pass"] == 1
    assert counts["off_catalog_contig"] == 1
    assert counts["no_orf"] == 2
    assert counts["rejected"] == {"sv_breakend": 1}


def test_scan_reports_which_catalogue_it_resolved_against(client, vcf_bytes) -> None:
    """Without this, a zero-hit result is indistinguishable from a wrong index."""
    provenance = _post(client, vcf_bytes).get_json()["provenance"]
    assert provenance["catalog_genes"] == 9
    assert provenance["catalog_isoforms"] == 18
    assert len(provenance["index_version"]) == 16
    assert len(provenance["vcf_sha256"]) == 64


def test_genes_rollup_is_ordered_and_complete(client, vcf_bytes) -> None:
    genes = _post(client, vcf_bytes).get_json()["genes"]
    # Ranked by distinct variants, then hit records, then name.
    keys = [(-g["n_variants"], -g["n_hits"], g["gene"]) for g in genes]
    assert keys == sorted(keys)


def test_gene_rollup_separates_variants_from_hit_records(client, vcf_bytes) -> None:
    """One variant inside N isoforms is 1 variant and N hits — never conflated.

    CBX1 has five isoforms in this catalogue, so its shared-core variant alone
    contributes five hit records.
    """
    genes = {g["gene"]: g for g in _post(client, vcf_bytes).get_json()["genes"]}
    cbx1 = genes["CBX1"]
    assert cbx1["n_hits"] > cbx1["n_variants"]
    assert cbx1["n_hits"] == cbx1["n_unique"] + cbx1["n_shared"]
    for g in genes.values():
        assert g["n_variants"] <= g["n_hits"]
        assert g["n_variants"] >= 1


def test_the_response_carries_the_hits(client, vcf_bytes) -> None:
    response = _post(client, vcf_bytes)
    assert response.headers["Cache-Control"] == "no-store"

    body = response.get_json()
    assert len(body["hits"]) == body["counts"]["hits"]
    hit = next(h for h in body["hits"] if h["gene"] == "MAD2L1" and h["frame"] == "canonical")
    assert (hit["residue"], hit["region"]) == (12, "unique")
    assert hit["x"] is not None


def test_the_response_contains_no_genotype_data(client, vcf_bytes) -> None:
    """What the browser stores must not carry the uploaded sample columns.

    The fixture is full-width (INFO / FORMAT / NORMAL / TUMOR), so this asserts
    against real genotype text rather than a hypothetical.
    """
    body = _post(client, vcf_bytes).get_data(as_text=True)
    for leak in ("zcc10-N", "zcc10-T", "GT:DP:FDP", "GT:DP:DP2", "0/1:", "0/0:", "SOMATIC"):
        assert leak not in body, f"{leak!r} was sent back in the scan response"


def test_gzipped_upload_resolves_identically(client, vcf_bytes) -> None:
    plain = _post(client, vcf_bytes).get_json()
    packed = _post(client, gzip.compress(vcf_bytes), name="test.vcf.gz").get_json()
    assert packed["counts"] == plain["counts"]


def test_the_filename_is_never_echoed(client, vcf_bytes) -> None:
    """The browser already has the filename; the server never hands one back.

    It used to be stored, and a filename written into a shared digest was served to
    the next uploader of the same bytes. Not storing it at all ends that class.
    """
    body = _post(client, vcf_bytes, name="patient_A_confidential.vcf").get_data(as_text=True)
    assert "patient_A" not in body


# ----------------------------------------------------------------------
# Failure modes, each with its own status
# ----------------------------------------------------------------------


def test_missing_file_is_400(client) -> None:
    response = client.post("/api/variants/scan", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert response.get_json()["error"] == "no_file"


def test_oversized_upload_is_413_json(client) -> None:
    """HTML would be useless here — the only caller is the uploader's fetch()."""
    client.application.config["MAX_CONTENT_LENGTH"] = 1024
    response = _post(client, b"x" * 4096)
    assert response.status_code == 413
    assert response.get_json()["error"] == "too_large"


def test_missing_index_reports_503_rather_than_matching_nothing(
    client, vcf_bytes, monkeypatch
) -> None:
    """A silently absent index would look like a VCF with no interesting variants."""
    # Patch only the name the route resolves; patching data.load_orf_index would
    # replace the cached loader the client fixture clears during teardown.
    monkeypatch.setattr("swissisoform_site.app.load_orf_index", lambda: None)
    response = _post(client, vcf_bytes)
    assert response.status_code == 503
    assert response.get_json()["error"] == "index_unavailable"


def test_existing_pages_still_render(client) -> None:
    """The new config and imports must not disturb the read-only routes."""
    assert client.get("/healthz").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/genes/CBX1").status_code == 200


# ----------------------------------------------------------------------
# Resource caps (PR #29 gate 1)
# ----------------------------------------------------------------------


def test_a_gzip_bomb_is_413_json_not_a_500(client, monkeypatch) -> None:
    """The compressed body passes MAX_CONTENT_LENGTH; the decompressed cap catches it."""
    monkeypatch.setenv("SWISSISOFORM_VCF_MAX_BYTES", str(64 * 1024))

    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as handle:
        handle.write(b"##fileformat=VCFv4.2\n")
        handle.write(b"A" * (4 * 1024 * 1024))
    body = buf.getvalue()
    assert len(body) < 1024 * 1024, "the bomb must pass the upload limit to be a test"

    response = _post(client, body, "bomb.vcf.gz")
    assert response.status_code == 413
    payload = response.get_json()
    # Distinct from the compressed-body limit, so the uploader can tell them apart.
    assert payload["error"] == "expands_too_large"
    assert payload["message"]


def test_a_newline_free_upload_is_422_json_not_a_500(client, monkeypatch) -> None:
    monkeypatch.setenv("SWISSISOFORM_VCF_MAX_LINE_BYTES", str(4 * 1024))

    response = _post(client, b"A" * (256 * 1024), "oneline.vcf")
    assert response.status_code == 422
    assert response.get_json()["error"] == "line_too_long"


# ----------------------------------------------------------------------
# Per-IP throttle
# ----------------------------------------------------------------------


def test_uploads_are_throttled_per_ip(client, vcf_bytes, monkeypatch) -> None:
    monkeypatch.setenv("SWISSISOFORM_SCAN_RATE_HOURLY", "2")

    for i in range(2):
        assert _post(client, vcf_bytes + f"\n#pad{i}\n".encode()).status_code == 200

    response = _post(client, vcf_bytes + b"\n#pad-final\n")
    assert response.status_code == 429
    payload = response.get_json()
    assert payload["error"] == "rate_limited"
    # vcf_drop.js renders payload.message verbatim, so it has to read as English.
    assert "Try again in" in payload["message"]
    assert int(response.headers["Retry-After"]) > 0


def test_an_identical_reupload_is_charged(client, vcf_bytes, monkeypatch) -> None:
    """With nothing stored there is no cache hit: every upload is parsed and charged."""
    monkeypatch.setenv("SWISSISOFORM_SCAN_RATE_HOURLY", "2")

    for _ in range(2):
        assert _post(client, vcf_bytes).status_code == 200
    assert _post(client, vcf_bytes).status_code == 429


def test_the_throttle_answers_before_the_index_is_needed(client, vcf_bytes, monkeypatch) -> None:
    """429 must not depend on the catalogue being loadable."""
    monkeypatch.setenv("SWISSISOFORM_SCAN_RATE_HOURLY", "1")
    assert _post(client, vcf_bytes).status_code == 200

    monkeypatch.setattr("swissisoform_site.app.load_orf_index", lambda: None)
    assert _post(client, vcf_bytes + b"\n#pad\n").status_code == 429


def test_forwarded_for_is_ignored_without_the_proxy_opt_in(client, vcf_bytes, monkeypatch) -> None:
    """Trusting the header unproxied would let anyone mint a fresh bucket."""
    monkeypatch.setenv("SWISSISOFORM_SCAN_RATE_HOURLY", "1")
    assert _post(client, vcf_bytes).status_code == 200

    response = client.post(
        "/api/variants/scan",
        data={"vcf": (io.BytesIO(vcf_bytes + b"\n#pad\n"), "test.vcf")},
        content_type="multipart/form-data",
        headers={"X-Forwarded-For": "9.9.9.9"},
    )
    assert response.status_code == 429


def test_forwarded_for_splits_buckets_when_the_proxy_is_trusted(
    data_dir, tmp_path, vcf_bytes, monkeypatch
) -> None:
    monkeypatch.setenv("SWISSISOFORM_DATA_DIR", str(data_dir))
    monkeypatch.setenv("SWISSISOFORM_THROTTLE_DIR", str(tmp_path / "throttle"))
    monkeypatch.setenv("SWISSISOFORM_SCAN_RATE_HOURLY", "1")
    monkeypatch.setenv("SWISSISOFORM_SCAN_RATE_DAILY", "0")
    monkeypatch.setenv("SWISSISOFORM_TRUST_PROXY", "1")

    from swissisoform_site import data as site_data
    from swissisoform_site.app import create_app

    site_data.load_all.cache_clear()
    site_data.load_orf_index.cache_clear()
    app = create_app()
    app.config["TESTING"] = True

    def post(body: bytes, forwarded: str):
        return app.test_client().post(
            "/api/variants/scan",
            data={"vcf": (io.BytesIO(body), "test.vcf")},
            content_type="multipart/form-data",
            headers={"X-Forwarded-For": forwarded},
        )

    try:
        assert post(vcf_bytes + b"\n#a\n", "9.9.9.9").status_code == 200
        assert post(vcf_bytes + b"\n#b\n", "9.9.9.9").status_code == 429
        # A different client is unaffected by the first one's budget.
        assert post(vcf_bytes + b"\n#c\n", "8.8.8.8").status_code == 200
    finally:
        site_data.load_all.cache_clear()
        site_data.load_orf_index.cache_clear()
