"""Read ``orf_index.parquet`` into an :class:`OrfIndex`.

Split out from :mod:`index` so that module stays dependency-free: pyarrow is
imported inside the function, and only this file knows about parquet. Both
consumers (the CLI and the website's cached loader) go through here so the
``index_version`` is read the same way in both.
"""

from __future__ import annotations

from pathlib import Path

from swissisoform.variantquery.index import CDS_COLUMNS, INDEX_COLUMNS, OrfIndex

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
    columns = _present(path, INDEX_COLUMNS + CDS_COLUMNS)
    table = pq.read_table(path, columns=columns)
    version = read_index_version(path)
    return OrfIndex.from_records(table.to_pylist(), version=version)


def load_index_from_paired(path: str | Path, *, version: str = "") -> OrfIndex:
    """Build an index directly from a run's ``all_paired.parquet``.

    Only for local checks and tests, and it carries **no CDS** — that is derived
    from the genome by the builder, not present in ``all_paired.parquet``. Column
    projection is mandatory: the full-catalogue file is 2.09 GB in a single row
    group, so an unprojected read materialises all of it.
    """
    import pyarrow.parquet as pq

    path = Path(path)
    table = pq.read_table(path, columns=_present(path, INDEX_COLUMNS))
    return OrfIndex.from_records(table.to_pylist(), version=version)
