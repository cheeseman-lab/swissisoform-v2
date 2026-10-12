"""Read ``orf_index.parquet`` into an :class:`OrfIndex`.

Split out from :mod:`index` so that module stays dependency-free: pyarrow is
imported inside the function, and only this file knows about parquet. Both
consumers (the CLI and the website's cached loader) go through here so the
``index_version`` is read the same way in both.

:func:`load_index` reads Arrow columns straight into the packed
:class:`~swissisoform.variantquery.index.OrfColumns` — an Arrow list-of-pairs
column already *is* a flat values buffer plus offsets — rather than materialising
one Python dict per row (``to_pylist``), which roughly doubled the peak memory of a
load on top of the Python-object index it then built.
"""

from __future__ import annotations

from array import array
from pathlib import Path
from typing import Any

from swissisoform.variantquery.index import (
    CDS_COLUMNS,
    CDS_FIELDS,
    DERIVED_COLUMNS,
    INDEX_COLUMNS,
    INT_FIELDS,
    INTERVAL_FIELDS,
    NULL_INT,
    OPTIONAL_COLUMNS,
    STRING_FIELDS,
    OrfColumns,
    OrfIndex,
    pool_sequences,
)

#: OrfRecord field → parquet column, where they differ.
_PARQUET_NAME = {
    "unique_intervals": "unique_genomic_intervals",
    "shared_intervals": "shared_genomic_intervals",
}

#: Parquet key-value metadata key holding the index version.
VERSION_METADATA_KEY = b"swissisoform_index_version"


def read_index_version(path: str | Path) -> str:
    """Read the ``index_version`` stamped into the parquet's metadata.

    Returns an empty string when absent — callers treat that as "unversioned"
    rather than failing, but a cached scan keyed on an empty version is not
    safely reusable across index rebuilds.
    """
    import pyarrow.parquet as pq

    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    raw = metadata.get(VERSION_METADATA_KEY)
    return raw.decode() if raw else ""


def _present(path: Path, wanted: tuple[str, ...]) -> list[str]:
    """Intersect wanted columns with what the file actually has, preserving order.

    Lets one loader read indexes built before a column existed — notably the CDS
    columns, which only appear when the builder was given a genome.
    """
    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(path).schema_arrow.names)
    return [c for c in wanted if c in available]


def load_index(path: str | Path) -> OrfIndex:
    """Load the ORF index from ``orf_index.parquet``."""
    import pyarrow.parquet as pq

    path = Path(path)
    columns = _present(path, INDEX_COLUMNS + OPTIONAL_COLUMNS + CDS_COLUMNS + DERIVED_COLUMNS)
    table = pq.read_table(path, columns=columns)
    version = read_index_version(path)
    packed = columns_from_table(table)
    if packed is None:
        # A shape the fast path does not handle (a null inside an interval list):
        # the row-by-row path coerces it exactly as before.
        return OrfIndex.from_records(table.to_pylist(), version=version)
    return OrfIndex.from_columns(packed, version=version)


def columns_from_table(table: Any) -> OrfColumns | None:
    """An Arrow table of index columns → :class:`OrfColumns`, without per-row dicts.

    Applies :meth:`OrfRecord.from_mapping`'s coercions column-wise: a null or
    absent string is ``""``, a null or absent integer is ``None``, a null interval
    list is empty, and rows without a chromosome are dropped (as
    :meth:`OrfIndex.from_records` drops them).

    Returns:
        The columns, or None when an interval list contains a null pair, which
        only the row-by-row path coerces.

    Raises:
        ValueError: If an interval is not a pair, as ``from_mapping`` raises.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    names = set(table.column_names)
    if "chrom" in names:
        table = table.filter(pc.fill_null(pc.not_equal(table["chrom"], ""), False))
    else:
        table = table.slice(0, 0)
    n = table.num_rows

    def column(field: str) -> Any:
        name = _PARQUET_NAME.get(field, field)
        return table[name].combine_chunks() if name in names else None

    strings: dict[str, list[str]] = {}
    for field in STRING_FIELDS:
        col = column(field)
        pool: dict[str, str] = {}
        values = col.to_pylist() if col is not None else [""] * n
        strings[field] = [pool.setdefault(v or "", v or "") for v in values]

    ints: dict[str, array] = {}
    for field in INT_FIELDS:
        col = column(field)
        if col is None:
            ints[field] = array("q", [NULL_INT]) * n
        else:
            filled = pc.fill_null(col.cast(pa.int64()), NULL_INT)
            ints[field] = array(
                "q", filled.to_numpy(zero_copy_only=False).astype("int64").tobytes()
            )

    bounds: dict[str, array] = {}
    offsets: dict[str, array] = {}
    for field in INTERVAL_FIELDS:
        col = column(field)
        if col is None:
            bounds[field], offsets[field] = array("q"), array("q", [0]) * (n + 1)
            continue
        per_row = pc.fill_null(pc.list_value_length(col), 0)
        pairs = pc.list_flatten(col)  # one list<int64> per interval
        if pairs.null_count:
            return None
        widths = pc.list_value_length(pairs)
        if len(pairs) and (pc.min(widths).as_py() != 2 or pc.max(widths).as_py() != 2):
            raise ValueError(f"{field}: every interval must have 2 elements")
        flat = pc.list_flatten(pairs).cast(pa.int64())
        bounds[field] = array("q", flat.to_numpy(zero_copy_only=False).astype("int64").tobytes())
        offs = array("q", [0])
        for count in per_row.to_pylist():
            offs.append(offs[-1] + count)
        offsets[field] = offs

    sequences = {}
    for field in CDS_FIELDS:
        col = column(field)
        sequences[field] = [v or "" for v in col.to_pylist()] if col is not None else [""] * n
    pool, cds_bounds, cds_ids = pool_sequences(sequences)
    del sequences

    return OrfColumns(strings, ints, bounds, offsets, pool, cds_bounds, cds_ids)


def load_index_from_paired(path: str | Path, *, version: str = "") -> OrfIndex:
    """Build an index directly from a run's ``all_paired.parquet``.

    Only for local checks and tests, and it carries **no CDS** — that is derived
    from the genome by the builder, not present in ``all_paired.parquet``. Column
    projection is mandatory: the full-catalogue file is 2.09 GB in a single row
    group, so an unprojected read materialises all of it.
    """
    import pyarrow.parquet as pq

    path = Path(path)
    table = pq.read_table(path, columns=_present(path, INDEX_COLUMNS + OPTIONAL_COLUMNS))
    return OrfIndex.from_records(table.to_pylist(), version=version)
