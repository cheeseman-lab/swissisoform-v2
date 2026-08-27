"""The two variant-query endpoints, end to end through Flask.

Exercises the contract the front end will depend on: a token in, a digest out,
and the four failure modes mapped onto distinct status codes (400 / 404 / 410 /
413) rather than a generic 500.

Skipped without the ``cheeseman_test`` run, which lives outside the repository.
"""

from __future__ import annotations

import gzip
import io
import json
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
    monkeypatch.setenv("SWISSISOFORM_SCAN_DIR", str(tmp_path / "scans"))
    monkeypatch.delenv("SWISSISOFORM_SCAN_TTL_HOURS", raising=False)
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


def test_scan_returns_a_token_and_the_funnel(client, vcf_bytes) -> None:
    payload = _post(client, vcf_bytes).get_json()
    assert payload["vcf_id"]
    assert payload["redirect"] == f"/variants/{payload['vcf_id']}"

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


def test_digest_route_returns_the_hits(client, vcf_bytes) -> None:
    token = _post(client, vcf_bytes).get_json()["vcf_id"]
    response = client.get(f"/api/variants/{token}.json")
    assert response.status_code == 200
    assert response.headers["X-Robots-Tag"] == "noindex, nofollow"

    digest = response.get_json()
    assert digest["vcf_id"] == token
    assert len(digest["hits"]) == digest["counts"]["hits"]
    hit = next(h for h in digest["hits"] if h["gene"] == "MAD2L1" and h["frame"] == "canonical")
    assert (hit["residue"], hit["region"]) == (12, "unique")


def test_persisted_digest_contains_no_genotype_data(client, vcf_bytes, tmp_path) -> None:
    """The 24 h artifact on disk must not carry the uploaded sample columns.

    The fixture is full-width (INFO / FORMAT / NORMAL / TUMOR), so this asserts
    against real genotype text rather than a hypothetical. Read straight off the
    filesystem, not through the API, because the file is what persists.
    """
    token = _post(client, vcf_bytes).get_json()["vcf_id"]

    from swissisoform_site import scanstore

    pointer = json.loads((scanstore.scan_dir() / "tokens" / f"{token}.json").read_text())
    on_disk = scanstore.digest_path(pointer["key"]).read_text()

    for leak in ("zcc10-N", "zcc10-T", "GT:DP:FDP", "GT:DP:DP2", "0/1:", "0/0:", "SOMATIC"):
        assert leak not in on_disk, f"{leak!r} was written to the scan digest"


def test_stored_source_keeps_the_upload_verbatim(client, vcf_bytes) -> None:
    """The blob is the upload byte-for-byte — that is what makes a re-scan honest.

    Genotypes do live here, which is exactly why no route serves this file back.
    """
    token = _post(client, vcf_bytes).get_json()["vcf_id"]

    from swissisoform_site import scanstore

    pointer = json.loads((scanstore.scan_dir() / "tokens" / f"{token}.json").read_text())
    stored = gzip.decompress(scanstore.source_path(pointer["key"]).read_bytes())
    assert stored == vcf_bytes


def test_gzipped_upload_resolves_identically(client, vcf_bytes) -> None:
    plain = _post(client, vcf_bytes).get_json()
    packed = _post(client, gzip.compress(vcf_bytes), name="test.vcf.gz").get_json()
    assert packed["counts"] == plain["counts"]


def test_reupload_skips_the_parse_but_mints_a_new_token(client, vcf_bytes) -> None:
    first = _post(client, vcf_bytes).get_json()
    second = _post(client, vcf_bytes).get_json()
    assert second["vcf_id"] != first["vcf_id"]
    assert second["counts"] == first["counts"]
    # Both tokens resolve to the same shared digest.
    assert client.get(f"/api/variants/{first['vcf_id']}.json").status_code == 200
    assert client.get(f"/api/variants/{second['vcf_id']}.json").status_code == 200

    # The blob really is shared — checked at the store, not from the response,
    # which must not say so (see test_response_never_reveals_a_cache_hit).
    from swissisoform_site import scanstore

    def _key(tok: str) -> str:
        return json.loads((scanstore.scan_dir() / "tokens" / f"{tok}.json").read_text())["key"]

    assert _key(first["vcf_id"]) == _key(second["vcf_id"])


def test_response_never_reveals_a_cache_hit(client, vcf_bytes) -> None:
    """Blobs are content-addressed and shared between uploaders.

    So "this was already scanned" is a fact about someone *else's* upload, and
    returning it lets anyone holding a candidate VCF confirm it was submitted —
    the confirm-by-upload oracle the capability token exists to prevent.
    """
    first = _post(client, vcf_bytes).get_json()
    second = _post(client, vcf_bytes).get_json()
    assert "was_cached" not in first
    assert "was_cached" not in second


def test_filename_is_token_scoped_not_blob_scoped(client, vcf_bytes) -> None:
    """A second uploader of the same bytes must see their OWN filename.

    The digest lives in the shared blob, so a filename written into it was served
    to everyone who uploaded those bytes afterwards — leaking what the first
    uploader called their file.
    """
    a = _post(client, vcf_bytes, name="patient_A_confidential.vcf").get_json()
    b = _post(client, vcf_bytes, name="patient_B.vcf").get_json()

    da = client.get(f"/api/variants/{a['vcf_id']}.json").get_json()
    db = client.get(f"/api/variants/{b['vcf_id']}.json").get_json()
    assert da["filename"] == "patient_A_confidential.vcf"
    assert db["filename"] == "patient_B.vcf"


# ----------------------------------------------------------------------
# Failure modes, each with its own status
# ----------------------------------------------------------------------


def test_missing_file_is_400(client) -> None:
    response = client.post("/api/variants/scan", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert response.get_json()["error"] == "no_file"


def test_unknown_token_is_404(client) -> None:
    response = client.get("/api/variants/doesnotexist.json")
    assert response.status_code == 404
    assert response.get_json()["error"] == "not_found"


def test_expired_scan_is_410_not_500(client, vcf_bytes, monkeypatch) -> None:
    token = _post(client, vcf_bytes).get_json()["vcf_id"]
    monkeypatch.setenv("SWISSISOFORM_SCAN_TTL_HOURS", "0")
    response = client.get(f"/api/variants/{token}.json")
    assert response.status_code == 410
    assert response.get_json()["error"] == "expired"


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


def test_traversal_token_is_refused(client) -> None:
    assert client.get("/api/variants/..%2F..%2Fetc%2Fpasswd.json").status_code == 404


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


def test_a_refused_upload_leaves_no_resolvable_token(client, monkeypatch) -> None:
    """No digest is written, so nothing should look like a finished scan."""
    monkeypatch.setenv("SWISSISOFORM_VCF_MAX_LINE_BYTES", str(4 * 1024))
    refused = _post(client, b"A" * (256 * 1024), "oneline.vcf")
    assert refused.status_code == 422
    assert "vcf_id" not in (refused.get_json() or {})


# ----------------------------------------------------------------------
# Per-IP throttle
# ----------------------------------------------------------------------


def test_uploads_are_throttled_per_ip(client, vcf_bytes, monkeypatch) -> None:
    monkeypatch.setenv("SWISSISOFORM_SCAN_RATE_HOURLY", "2")

    # Distinct bytes each time: an identical re-upload is a cache hit and free.
    for i in range(2):
        assert _post(client, vcf_bytes + f"\n#pad{i}\n".encode()).status_code == 200

    response = _post(client, vcf_bytes + b"\n#pad-final\n")
    assert response.status_code == 429
    payload = response.get_json()
    assert payload["error"] == "rate_limited"
    # vcf_drop.js renders payload.message verbatim, so it has to read as English.
    assert "Try again in" in payload["message"]
    assert int(response.headers["Retry-After"]) > 0


def test_a_cache_hit_does_not_spend_budget(client, vcf_bytes, monkeypatch) -> None:
    """Identical bytes skip the parse entirely, so they cost nothing to serve."""
    monkeypatch.setenv("SWISSISOFORM_SCAN_RATE_HOURLY", "2")

    for _ in range(5):
        assert _post(client, vcf_bytes).status_code == 200


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
    monkeypatch.setenv("SWISSISOFORM_SCAN_DIR", str(tmp_path / "scans"))
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
