"""What the browser needs to run the VCF scan itself (issue #33, step 2).

The scan is the same Python the server runs — ``variantquery`` + ``variantclass``
+ ``scanview`` — executed in a Web Worker under Pyodide, so a VCF never leaves the
user's machine. Two downloads, both version-keyed and cached as immutable:

* the scan package, a zip of exactly :data:`SCANNER_MODULES`, built here from the
  modules this process imports (so it is whatever the image ships, never a copy
  that can drift); its version is a hash of its bytes;
* the ORF index, as ``variantquery.portable`` gzipped JSON (``data.py``).

Pyodide itself comes from jsDelivr at a pinned version, as an ES module: Pyodide
314 refuses to boot in a classic worker, so ``scan_worker.js`` is a module worker
that imports ``pyodide.mjs``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import zipfile
from functools import lru_cache
from pathlib import Path

#: Pinned Pyodide release. Its interpreter must run the scan modules: they need
#: Python >= 3.11 and nothing outside the standard library.
PYODIDE_VERSION = "314.0.7"
PYODIDE_URL = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/"

#: The scan's import closure. Every module here must import only the standard
#: library and each other — ``tests/test_browserscan.py`` imports the zip alone,
#: in a fresh interpreter, to prove it.
SCANNER_MODULES = (
    "swissisoform",
    "swissisoform.contract",
    "swissisoform.coords",
    "swissisoform.variantclass",
    "swissisoform.variantquery",
    "swissisoform.variantquery.consequence",
    "swissisoform.variantquery.frame",
    "swissisoform.variantquery.index",
    "swissisoform.variantquery.portable",
    "swissisoform.variantquery.scan",
    "swissisoform.variantquery.spec",
    "swissisoform.variantquery.vcf",
    "swissisoform_site",
    "swissisoform_site.scanview",
)


def _archive_name(module: str, origin: Path) -> str:
    parts = module.split(".")
    if origin.name == "__init__.py":
        return "/".join(parts) + "/__init__.py"
    return "/".join(parts) + ".py"


@lru_cache(maxsize=1)
def scanner_archive() -> tuple[str, bytes]:
    """``(version, zip bytes)`` of :data:`SCANNER_MODULES`, built once per worker.

    Entries are written in a fixed order with a fixed timestamp, so the bytes —
    and the version, the first 16 hex of their SHA-256 — change only when the
    code does.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for module in SCANNER_MODULES:
            spec = importlib.util.find_spec(module)
            if spec is None or spec.origin is None:
                raise RuntimeError(f"scan module {module} is not importable here")
            origin = Path(spec.origin)
            info = zipfile.ZipInfo(_archive_name(module, origin), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, origin.read_bytes())
    data = buffer.getvalue()
    return hashlib.sha256(data).hexdigest()[:16], data
