"""The browser scan's downloads: the scan package zip and how pages point at it.

The decisive test runs the zip the way Pyodide does — alone on ``sys.path``, in a
fresh interpreter with no site-packages — and scans a VCF with it. If a module
were missing from :data:`SCANNER_MODULES`, or any of them imported pandas,
Biopython or pyarrow, the browser would fail; this fails first.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from swissisoform_site.browserscan import PYODIDE_URL, SCANNER_MODULES, scanner_archive

WEBSITE_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = WEBSITE_ROOT.parent / "data" / "output" / "cheeseman_test"
STAGED_FILES = ("all_paired.parquet", "variants_long.parquet", "orf_index.parquet")
needs_run = pytest.mark.skipif(
    not all((RUN_DIR / f).is_file() for f in STAGED_FILES),
    reason="needs the cheeseman_test run with orf_index.parquet built",
)

VCF = (
    "##fileformat=VCFv4.2\n"
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    "chr17\t48071500\t.\tA\tG\t.\tPASS\t.\n"
    "chr17\t48071501\t.\tA\tG\t.\tLowQual\t.\n"
)

_RUN_FROM_ZIP = """
import json, sys
sys.path[:] = [sys.argv[1]] + [p for p in sys.path if "site-packages" not in p]
from swissisoform_site.scanview import load_portable_index, scan_file_json
body = json.loads(scan_file_json(sys.argv[3], load_portable_index(sys.argv[2])))
heavy = {"pandas", "numpy", "Bio", "pyarrow", "pysam", "flask"}
loaded = sorted(heavy & {m.split(".")[0] for m in sys.modules})
outside = sorted(
    m.__file__ for m in list(sys.modules.values())
    if getattr(m, "__file__", None) and m.__name__.split(".")[0].startswith("swissisoform")
    and not m.__file__.startswith(sys.argv[1])
)
print(json.dumps({"body": body, "heavy": loaded, "outside": outside}))
"""


def test_the_archive_holds_exactly_the_scan_modules() -> None:
    _version, data = scanner_archive()
    names = set(zipfile.ZipFile(io.BytesIO(data)).namelist())
    assert len(names) == len(SCANNER_MODULES)
    assert "swissisoform/variantclass.py" in names
    assert "swissisoform_site/scanview.py" in names
    assert not any("validate" in n or n.endswith("app.py") for n in names)


def test_the_version_is_a_hash_of_stable_bytes() -> None:
    first = scanner_archive()
    scanner_archive.cache_clear()
    assert scanner_archive() == first
    assert len(first[0]) == 16


@needs_run
def test_the_archive_alone_runs_a_scan_with_no_heavy_dependency(tmp_path) -> None:
    from swissisoform.variantquery.load import load_index
    from swissisoform.variantquery.portable import to_portable

    archive = tmp_path / "scanner.zip"
    archive.write_bytes(scanner_archive()[1])
    index = tmp_path / "index.json.gz"
    index.write_bytes(to_portable(load_index(RUN_DIR / "orf_index.parquet")))
    vcf = tmp_path / "mine.vcf"
    vcf.write_text(VCF)

    done = subprocess.run(
        [sys.executable, "-I", "-S", "-c", _RUN_FROM_ZIP, str(archive), str(index), str(vcf)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout)
    assert out["heavy"] == [], out["heavy"]
    assert out["outside"] == [], "imported swissisoform code from outside the zip"
    body = out["body"]
    assert body["counts"]["lines"] == 2
    assert body["counts"]["skipped_non_pass"] == 1
    assert {g["gene"] for g in body["genes"]} == {"CBX1"}
    assert all(h["x"] is not None for h in body["hits"])


@pytest.fixture
def client(tmp_path, monkeypatch):
    staged = tmp_path / "data"
    staged.mkdir()
    for name in STAGED_FILES:
        shutil.copy(RUN_DIR / name, staged / name)
    (staged / "llm").mkdir()
    monkeypatch.setenv("SWISSISOFORM_DATA_DIR", str(staged))
    monkeypatch.setenv("SWISSISOFORM_THROTTLE_DIR", str(tmp_path / "throttle"))

    from swissisoform_site import data as site_data
    from swissisoform_site.app import create_app

    for cached in (
        site_data.load_all,
        site_data.load_orf_index,
        site_data.orf_index_version,
        site_data.portable_orf_index,
    ):
        cached.cache_clear()
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client
    for cached in (
        site_data.load_all,
        site_data.load_orf_index,
        site_data.orf_index_version,
        site_data.portable_orf_index,
    ):
        cached.cache_clear()


@needs_run
def test_the_scanner_is_served_by_version_and_immutable(client) -> None:
    version, data = scanner_archive()
    response = client.get(f"/api/variants/scanner/{version}.zip")
    assert response.status_code == 200
    assert response.data == data
    assert response.headers["Content-Type"] == "application/zip"
    assert "immutable" in response.headers["Cache-Control"]
    assert client.get("/api/variants/scanner/0000000000000000.zip").status_code == 404


@needs_run
def test_the_drop_zone_names_all_three_downloads(client) -> None:
    from swissisoform_site.data import load_orf_index

    page = client.get("/").data.decode()
    assert f'data-pyodide="{PYODIDE_URL}"' in page
    assert f'data-scanner="/api/variants/scanner/{scanner_archive()[0]}.zip"' in page
    assert f'data-index="/api/variants/index/{load_orf_index().version}.json.gz"' in page
    assert 'data-worker="/static/js/scan_worker.js"' in page
    assert "the file is not uploaded" in page
