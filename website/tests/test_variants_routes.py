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
    assert payload["was_cached"] is False

    counts = payload["counts"]
    assert counts["lines"] == 17
    assert counts["hits"] == 23
    # The three negatives stay distinguishable, which is what makes a zero-hit
    # scan explainable rather than blank.
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
    assert len(digest["hits"]) == 23
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
    assert second["was_cached"] is True
    assert second["vcf_id"] != first["vcf_id"]
    assert second["counts"] == first["counts"]
    # Both tokens resolve to the same shared digest.
    assert client.get(f"/api/variants/{first['vcf_id']}.json").status_code == 200
    assert client.get(f"/api/variants/{second['vcf_id']}.json").status_code == 200


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
