"""The ORF index as gzipped JSON, readable with the standard library alone.

The browser scan (issue #33, step 2) runs this package under Pyodide, where
pyarrow would cost tens of MB to download, so the index it scans against is served
in this form instead of as ``orf_index.parquet``. It is the same records — the
columns :mod:`load` projects, one row per ORF — and :func:`from_portable` rebuilds
the same :class:`OrfIndex` the server builds from the parquet.

Layout (gzip of compact JSON)::

    {"format": 1, "version": "<index_version>",
     "columns": ["gene_name", "tis_id", …], "rows": [[…], …]}

Columnar rows rather than one object per ORF: the key names would otherwise repeat
6,000+ times. ``gzip`` is written with ``mtime=0`` so the bytes depend only on the
index, which keeps them cacheable under a version-keyed URL.
"""

from __future__ import annotations

import gzip
import json
from typing import Any

from swissisoform.variantquery.index import (
    CDS_COLUMNS,
    DERIVED_COLUMNS,
    INDEX_COLUMNS,
    OPTIONAL_COLUMNS,
    OrfIndex,
    OrfRecord,
)

#: Bump when the layout above changes; :func:`from_portable` refuses other values.
PORTABLE_FORMAT = 1

COLUMNS = INDEX_COLUMNS + OPTIONAL_COLUMNS + CDS_COLUMNS + DERIVED_COLUMNS

#: Column name → OrfRecord attribute, where they differ.
_ATTRIBUTE = {
    "unique_genomic_intervals": "unique_intervals",
    "shared_genomic_intervals": "shared_intervals",
}


def _cell(record: OrfRecord, column: str) -> Any:
    value = getattr(record, _ATTRIBUTE.get(column, column))
    if isinstance(value, tuple):
        return [list(pair) for pair in value]
    return value


def to_portable(index: OrfIndex) -> bytes:
    """Serialise ``index`` to gzipped JSON (see the module docstring)."""
    body = {
        "format": PORTABLE_FORMAT,
        "version": index.version,
        "columns": list(COLUMNS),
        "rows": [[_cell(record, column) for column in COLUMNS] for record in index.records],
    }
    raw = json.dumps(body, separators=(",", ":")).encode("utf-8")
    return gzip.compress(raw, compresslevel=6, mtime=0)


def from_portable(data: bytes) -> OrfIndex:
    """Rebuild the :class:`OrfIndex` from :func:`to_portable` output.

    Raises:
        ValueError: If the bytes are not this format, or a format this code does
            not know.
    """
    try:
        body = json.loads(gzip.decompress(data))
    except (OSError, EOFError, ValueError) as exc:  # EOFError: a truncated download
        raise ValueError(f"not a portable ORF index: {exc}") from exc
    if not isinstance(body, dict):
        raise ValueError("not a portable ORF index: the body is not an object")
    if body.get("format") != PORTABLE_FORMAT:
        raise ValueError(f"unknown portable ORF index format: {body.get('format')!r}")
    try:
        columns = body["columns"]
        return OrfIndex.from_records(
            (dict(zip(columns, row, strict=True)) for row in body["rows"]),
            version=str(body.get("version") or ""),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"not a portable ORF index: {exc!r}") from exc
