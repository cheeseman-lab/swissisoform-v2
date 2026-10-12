"""``POST /api/variants/scan`` end to end, with a VCF written by the test.

The older route tests need ``ecf_data/test.vcf``, which lives outside the repo. This
file needs only the ``cheeseman_test`` run, and builds its VCF from coordinates that
run's ORF index is known to cover, so the stateless contract — the whole result in
the response, nothing left on disk — is checked wherever that run is staged.
"""

from __future__ import annotations

import gzip
import io
import shutil
import tempfile
from pathlib import Path

import pytest

WEBSITE_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = WEBSITE_ROOT.parent / "data" / "output" / "cheeseman_test"
STAGED_FILES = ("all_paired.parquet", "variants_long.parquet", "orf_index.parquet")

pytestmark = pytest.mark.skipif(
    not all((RUN_DIR / f).is_file() for f in STAGED_FILES),
    reason="needs the cheeseman_test run with orf_index.parquet built",
)

#: CBX1's shared core: inside ORF exon [48071437, 48071579) on chr17 (0-based), so
#: 1-based 48071500 lands in every CBX1 isoform. The REF need not match the genome;
#: a mismatch is still a hit, classified ``reference_mismatch``.
VCF = (
    "##fileformat=VCFv4.2\n"
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    "chr17\t48071500\t.\tA\tG\t.\tPASS\t.\n"
    "chr17\t48071501\t.\tA\tG\t.\tLowQual\t.\n"
    "chrUn_KI270302v1\t100\t.\tA\tG\t.\tPASS\t.\n"
).encode()


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    staged = tmp_path_factory.mktemp("sitedata")
    for name in STAGED_FILES:
        shutil.copy(RUN_DIR / name, staged / name)
    (staged / "llm").mkdir()
    return staged


@pytest.fixture
def stage_dir(tmp_path, monkeypatch) -> Path:
    """Route the staged upload into a dir the test can inspect."""
    target = tmp_path / "staging"
    target.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(target))
    return target


@pytest.fixture
def client(data_dir, tmp_path, monkeypatch, stage_dir):
    monkeypatch.setenv("SWISSISOFORM_DATA_DIR", str(data_dir))
    monkeypatch.setenv("SWISSISOFORM_THROTTLE_DIR", str(tmp_path / "throttle"))
    monkeypatch.setenv("SWISSISOFORM_SCAN_RATE_HOURLY", "0")
    monkeypatch.setenv("SWISSISOFORM_SCAN_RATE_DAILY", "0")
    monkeypatch.delenv("SWISSISOFORM_SCAN_TOKEN", raising=False)

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


def _post(client, body: bytes, name: str = "mine.vcf"):
    return client.post(
        "/api/variants/scan",
        data={"vcf": (io.BytesIO(body), name)},
        content_type="multipart/form-data",
    )


def test_the_response_is_the_whole_result(client) -> None:
    response = _post(client, VCF)
    assert response.status_code == 200
    body = response.get_json()

    from swissisoform_site.scanview import SCAN_SCHEMA

    assert body["schema"] == SCAN_SCHEMA
    assert set(body) == {"schema", "counts", "genes", "provenance", "derived", "hits"}
    counts = body["counts"]
    assert counts["lines"] == 3
    assert counts["skipped_non_pass"] == 1
    assert counts["off_catalog_contig"] == 1
    assert counts["hits"] == len(body["hits"]) > 0
    assert {g["gene"] for g in body["genes"]} == {"CBX1"}
    assert body["derived"]["passing"] == 2
    assert body["derived"]["in_orf_alleles"] == 1


def test_hits_carry_what_only_the_index_knows(client) -> None:
    hits = _post(client, VCF).get_json()["hits"]
    for hit in hits:
        assert hit["gene"] == "CBX1"
        assert isinstance(hit["x"], (int, float))
        assert hit["protein_change"].endswith("frame")
        assert "vclass" not in hit and "transcript_id" not in hit


def test_no_token_no_redirect_no_filename(client) -> None:
    """The server has nothing to point back to, and the filename stays in the browser."""
    response = _post(client, VCF, name="patient_0423_tumour.vcf")
    text = response.get_data(as_text=True)
    body = response.get_json()
    for key in ("vcf_id", "redirect", "expires_at", "filename"):
        assert key not in body
    assert "patient_0423" not in text


def test_the_response_is_never_cached(client) -> None:
    assert _post(client, VCF).headers["Cache-Control"] == "no-store"


def test_nothing_is_left_on_disk(client, stage_dir) -> None:
    assert _post(client, VCF).status_code == 200
    assert list(stage_dir.iterdir()) == []


def test_a_refused_upload_leaves_nothing_either(client, stage_dir, monkeypatch) -> None:
    monkeypatch.setenv("SWISSISOFORM_VCF_MAX_LINE_BYTES", "64")
    # The cap trips on a line still waiting for its newline (vcf.iter_lines), so the
    # long line is left unterminated.
    long_line = VCF + b"chr17\t48071502\t.\tA\t" + b"G" * 200
    response = _post(client, long_line)
    assert response.status_code == 422
    assert response.get_json()["error"] == "line_too_long"
    assert list(stage_dir.iterdir()) == []


def test_gzipped_upload_resolves_identically(client) -> None:
    plain = _post(client, VCF).get_json()
    packed = _post(client, gzip.compress(VCF)).get_json()
    assert packed["hits"] == plain["hits"]
    assert packed["counts"] == plain["counts"]
    # The hash is of the bytes as uploaded, so the two differ.
    assert packed["provenance"]["vcf_sha256"] != plain["provenance"]["vcf_sha256"]


def test_every_upload_is_charged(client, monkeypatch) -> None:
    """No stored results means no cache hit, so a repeat upload is parsed and charged."""
    monkeypatch.setenv("SWISSISOFORM_SCAN_RATE_HOURLY", "1")
    monkeypatch.setenv("SWISSISOFORM_SCAN_RATE_DAILY", "0")
    assert _post(client, VCF).status_code == 200
    assert _post(client, VCF).status_code == 429


def test_token_routes_are_gone(client) -> None:
    assert client.get("/variants/abc").status_code == 404
    assert client.get("/api/variants/abc.json").status_code == 404


def test_status_no_longer_reports_a_store(client) -> None:
    status = client.get("/api/variants/status").get_json()
    assert status["index_loaded"] is True
    for key in ("scan_dir", "scan_dir_writable", "ttl_hours", "budget_bytes"):
        assert key not in status


def test_every_page_states_the_index_version(client) -> None:
    from swissisoform_site.data import load_orf_index

    version = load_orf_index().version
    html = client.get("/genes/CBX1").get_data(as_text=True)
    assert f'data-index-version="{version}"' in html
