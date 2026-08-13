"""Streaming VCF reader — plain or gzip, never loads the file into memory.

Kept deliberately dumb: it yields raw data lines with their line numbers and
leaves parsing to :mod:`spec` and counting to :mod:`scan`, so the funnel tallies
live in exactly one place.
"""

from __future__ import annotations

import gzip
import io
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_GZIP_MAGIC = b"\x1f\x8b"


@contextmanager
def open_vcf(path: str | Path) -> Iterator[io.TextIOBase]:
    """Open a VCF as text, transparently handling gzip.

    Detection is by **magic bytes**, not filename — uploads arrive named
    anything, and a mislabelled ``.vcf`` that is really gzipped would otherwise
    parse as one long line of binary.
    """
    path = Path(path)
    with path.open("rb") as probe:
        compressed = probe.read(2) == _GZIP_MAGIC

    if compressed:
        handle = gzip.open(path, "rt", encoding="utf-8", errors="replace")
    else:
        handle = path.open("rt", encoding="utf-8", errors="replace")
    try:
        yield handle
    finally:
        handle.close()


def iter_data_lines(path: str | Path) -> Iterator[tuple[int, str]]:
    """Yield ``(line_no, line)`` for every non-header line.

    ``line_no`` is 1-based over the **whole file** including headers, so it
    reproduces the record with ``sed -n '<line_no>p'``.
    """
    with open_vcf(path) as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.startswith("#"):
                continue
            if not line.strip():
                continue
            yield line_no, line
