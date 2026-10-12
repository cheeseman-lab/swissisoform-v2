"""The ORF index as a compact binary file, readable with the standard library alone.

The browser scan (issue #33, step 2) runs this package under Pyodide, where
pyarrow would cost tens of MB to download, so the index it scans against is served
in this form instead of as ``orf_index.parquet``. It is the packed
:class:`~swissisoform.variantquery.index.OrfColumns` written out as they sit in
memory, so :func:`from_portable` rebuilds the same :class:`OrfIndex` the server
builds from the parquet with one ``array.frombytes`` per integer column — no
per-number Python objects at any point, which is what keeps the tab's peak memory
near the size of the data.

Layout (the whole file gzip-compressed, ``mtime=0`` so the bytes depend only on
the index and stay cacheable under a version-keyed URL)::

    b"SWISSORF"                      magic
    uint32 little-endian             length of the header
    header (UTF-8 JSON)              {"format": 2, "version", "n",
                                      "strings": {field: [str, …]},
                                      "blobs": {name: [offset, length]}}
    blobs, back to back              int64 little-endian arrays (ints, interval
                                     bounds and offsets, sequence-pool bounds and
                                     ids) and the UTF-8 sequence pool

Blob names are ``ints.<field>``, ``bounds.<field>``, ``offsets.<field>``,
``cds_ids.<field>``, ``cds_bounds`` and ``cds_pool``.
"""

from __future__ import annotations

import gzip
import json
import struct
import sys
from array import array

from swissisoform.variantquery.index import (
    CDS_FIELDS,
    INT_FIELDS,
    INTERVAL_FIELDS,
    STRING_FIELDS,
    OrfColumns,
    OrfIndex,
)

#: Bump when the layout above changes; :func:`from_portable` refuses other values.
PORTABLE_FORMAT = 2

MAGIC = b"SWISSORF"


def _le_bytes(values: array) -> bytes:
    if sys.byteorder == "little":
        return values.tobytes()
    swapped = array(values.typecode, values)
    swapped.byteswap()
    return swapped.tobytes()


def _from_le(data: bytes) -> array:
    values = array("q")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def to_portable(index: OrfIndex) -> bytes:
    """Serialise ``index`` (see the module docstring)."""
    columns = index.columns
    blobs: list[tuple[str, bytes]] = []
    for field in INT_FIELDS:
        blobs.append((f"ints.{field}", _le_bytes(columns.ints[field])))
    for field in INTERVAL_FIELDS:
        blobs.append((f"bounds.{field}", _le_bytes(columns.bounds[field])))
        blobs.append((f"offsets.{field}", _le_bytes(columns.offsets[field])))
    for field in CDS_FIELDS:
        blobs.append((f"cds_ids.{field}", _le_bytes(columns.cds_ids[field])))
    blobs.append(("cds_bounds", _le_bytes(columns.cds_bounds)))
    blobs.append(("cds_pool", columns.cds_pool.encode("utf-8")))

    table: dict[str, list[int]] = {}
    position = 0
    for name, data in blobs:
        table[name] = [position, len(data)]
        position += len(data)
    header = json.dumps(
        {
            "format": PORTABLE_FORMAT,
            "version": index.version,
            "n": columns.n,
            "strings": {field: columns.strings[field] for field in STRING_FIELDS},
            "blobs": table,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    body = b"".join([MAGIC, struct.pack("<I", len(header)), header] + [d for _, d in blobs])
    return gzip.compress(body, compresslevel=6, mtime=0)


def from_portable(data: bytes) -> OrfIndex:
    """Rebuild the :class:`OrfIndex` from :func:`to_portable` output.

    Raises:
        ValueError: If the bytes are not this format, or a format this code does
            not know.
    """
    try:
        body = gzip.decompress(data)
    except (OSError, EOFError) as exc:
        raise ValueError(f"not a portable ORF index: {exc}") from exc
    if not body.startswith(MAGIC):
        raise ValueError("not a portable ORF index: bad magic")
    (header_len,) = struct.unpack_from("<I", body, len(MAGIC))
    start = len(MAGIC) + 4
    header = json.loads(body[start : start + header_len])
    if header.get("format") != PORTABLE_FORMAT:
        raise ValueError(f"unknown portable ORF index format: {header.get('format')!r}")
    base = start + header_len
    view = memoryview(body)

    def blob(name: str) -> memoryview:
        offset, length = header["blobs"][name]
        return view[base + offset : base + offset + length]

    strings = {}
    for field in STRING_FIELDS:
        pool: dict[str, str] = {}
        strings[field] = [pool.setdefault(v, v) for v in header["strings"][field]]
    columns = OrfColumns(
        strings=strings,
        ints={field: _from_le(blob(f"ints.{field}")) for field in INT_FIELDS},
        bounds={field: _from_le(blob(f"bounds.{field}")) for field in INTERVAL_FIELDS},
        offsets={field: _from_le(blob(f"offsets.{field}")) for field in INTERVAL_FIELDS},
        cds_pool=str(blob("cds_pool"), "utf-8"),
        cds_bounds=_from_le(blob("cds_bounds")),
        cds_ids={field: _from_le(blob(f"cds_ids.{field}")) for field in CDS_FIELDS},
    )
    return OrfIndex.from_columns(columns, version=str(header.get("version") or ""))
