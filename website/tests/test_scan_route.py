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
    site_data.portable_orf_index.cache_clear()
    site_data.orf_index_fingerprint.cache_clear()
    site_data.orf_index_version.cache_clear()
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client
    site_data.load_all.cache_clear()
    site_data.load_orf_index.cache_clear()
    site_data.portable_orf_index.cache_clear()
    site_data.orf_index_fingerprint.cache_clear()
    site_data.orf_index_version.cache_clear()


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


def test_an_unusable_temp_dir_is_a_507_not_a_500(client, stage_dir, monkeypatch) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(stage_dir / "missing"))
    response = _post(client, VCF)
    assert response.status_code == 507
    assert response.get_json()["error"] == "storage_failed"


def test_a_corrupt_gzip_is_a_422_that_says_so(client) -> None:
    truncated = gzip.compress(VCF * 50)[:-40]
    response = _post(client, truncated, name="mine.vcf.gz")
    assert response.status_code == 422
    body = response.get_json()
    assert body["error"] == "unreadable"
    assert body["message"].startswith("Could not read this file as a VCF:")
    assert "Traceback" not in body["message"]


def test_the_browser_entry_point_reports_a_corrupt_gzip_in_one_line(client, tmp_path) -> None:
    import json

    from swissisoform_site.data import load_orf_index
    from swissisoform_site.scanview import scan_file_json

    path = tmp_path / "mine.vcf.gz"
    path.write_bytes(gzip.compress(VCF * 50)[:-40])
    body = json.loads(scan_file_json(path, load_orf_index()))
    assert body["error"] == "unreadable"
    assert "\n" not in body["message"]


def test_the_browser_result_skips_the_hash_the_server_result_keeps(client, tmp_path) -> None:
    """Hashing would re-read the whole file in WebAssembly for a value nothing there reads."""
    import json

    from swissisoform_site.data import load_orf_index
    from swissisoform_site.scanview import scan_file_json

    path = tmp_path / "mine.vcf"
    path.write_bytes(VCF)
    browser = json.loads(scan_file_json(path, load_orf_index()))
    server = _post(client, VCF).get_json()
    assert browser["provenance"]["vcf_sha256"] is None
    assert len(server["provenance"]["vcf_sha256"]) == 64
    assert browser["hits"] == server["hits"] and browser["counts"] == server["counts"]


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


def test_status_probes_the_dirs_a_scan_writes(client, stage_dir, tmp_path) -> None:
    """Staging and the throttle still need disk, and there is no shell into the container."""
    status = client.get("/api/variants/status").get_json()
    assert status["tmp_dir"] == str(stage_dir)
    assert status["tmp_dir_writable"] is True and status["tmp_dir_error"] == ""
    assert status["throttle_dir"] == str(tmp_path / "throttle")
    assert status["throttle_dir_writable"] is True
    assert list(stage_dir.iterdir()) == []  # the probe cleans up after itself


def test_status_names_an_unwritable_temp_dir(client, stage_dir, monkeypatch) -> None:
    blocker = stage_dir / "file"
    blocker.write_text("")
    monkeypatch.setattr(tempfile, "tempdir", str(blocker / "sub"))
    status = client.get("/api/variants/status").get_json()
    assert status["tmp_dir_writable"] is False
    assert status["tmp_dir_error"]


def test_every_page_states_the_index_version(client) -> None:
    from swissisoform_site.data import load_orf_index

    version = load_orf_index().version
    html = client.get("/genes/CBX1").get_data(as_text=True)
    assert f'data-index-version="{version}"' in html


def test_the_browser_index_is_served_by_fingerprint_and_rebuilds_the_same_index(client) -> None:
    from swissisoform.variantquery.portable import from_portable

    from swissisoform_site.data import load_orf_index, orf_index_fingerprint

    index = load_orf_index()
    response = client.get(f"/api/variants/index/{orf_index_fingerprint()}.json.gz")
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/gzip"
    assert "immutable" in response.headers["Cache-Control"]
    rebuilt = from_portable(response.data)
    assert rebuilt.version == index.version
    assert list(rebuilt.records) == list(index.records)


def test_another_fingerprint_is_not_served(client) -> None:
    """A stale page must not silently scan against data it did not ask for."""
    assert client.get("/api/variants/index/0000000000000000.json.gz").status_code == 404


def _restage(client, data_dir: Path, target: Path, monkeypatch, rewrite) -> None:
    """Point the running app at a copy of the data with ``orf_index.parquet`` rewritten."""
    import pyarrow.parquet as pq

    from swissisoform_site import data as site_data

    shutil.copytree(data_dir, target)
    table = pq.read_table(target / "orf_index.parquet")
    pq.write_table(rewrite(table), target / "orf_index.parquet")
    monkeypatch.setenv("SWISSISOFORM_DATA_DIR", str(target))
    for cached in (
        site_data.load_orf_index,
        site_data.orf_index_version,
        site_data.orf_index_fingerprint,
        site_data.portable_orf_index,
    ):
        cached.cache_clear()


def test_a_rebuild_with_the_same_coordinates_gets_a_new_index_url(
    client, data_dir, tmp_path, monkeypatch
) -> None:
    """The version keys coordinates only; the download URL must follow the bytes."""
    import pyarrow as pa

    from swissisoform_site.data import orf_index_fingerprint, orf_index_version

    old_url = f"/api/variants/index/{orf_index_fingerprint()}.json.gz"
    old_version = orf_index_version()

    def new_cds(table):
        cds = pa.array(["ATG" * 5] * table.num_rows, type=table.schema.field("orf_cds").type)
        return table.set_column(table.schema.get_field_index("orf_cds"), "orf_cds", cds)

    _restage(client, data_dir, tmp_path / "rebuilt", monkeypatch, new_cds)
    assert orf_index_version() == old_version
    assert client.get(old_url).status_code == 404
    page = client.get("/").get_data(as_text=True)
    assert f'data-index="/api/variants/index/{orf_index_fingerprint()}.json.gz"' in page
    assert orf_index_fingerprint() not in old_url


def test_an_unversioned_index_still_gets_a_working_index_url(
    client, data_dir, tmp_path, monkeypatch
) -> None:
    from swissisoform_site.data import orf_index_fingerprint, orf_index_version

    _restage(
        client,
        data_dir,
        tmp_path / "unversioned",
        monkeypatch,
        lambda table: table.replace_schema_metadata({}),
    )
    assert orf_index_version() == ""
    url = f"/api/variants/index/{orf_index_fingerprint()}.json.gz"
    assert f'data-index="{url}"' in client.get("/").get_data(as_text=True)
    assert client.get(url).status_code == 200
