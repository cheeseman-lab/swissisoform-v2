"""Streaming VCF reader — plain or gzip, never loads the file into memory.

Kept deliberately dumb: it yields raw data lines with their line numbers and
leaves parsing to :mod:`spec` and counting to :mod:`scan`, so the funnel tallies
live in exactly one place.

Two caps live here because this is the only place that sees the *decompressed*
stream. The web upload limit bounds the compressed body; gzip expands ~1000:1, so
a file that passes it can still be hundreds of gigabytes on the way out, and a
stream containing no newline at all would buffer whole before the first line is
yielded. Both raise :class:`VcfLimitExceeded` before the allocation happens, so
the caller can answer 4xx instead of dying.
"""

from __future__ import annotations

import gzip
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

logger = logging.getLogger(__name__)

_GZIP_MAGIC = b"\x1f\x8b"
_CHUNK = 1 << 20  # 1 MiB, matching the scan store's hashing chunk.

#: Ceiling on bytes read *out of* the decompressor. A 100 MB gzipped VCF inflates
#: to roughly 1-1.5 GB, so this clears real files with room to spare while still
#: stopping a bomb long before it becomes minutes of parsing.
DEFAULT_MAX_DECOMPRESSED_BYTES = 2 * 1024**3

#: Ceiling on one line. A 1000-sample VCF line runs ~50 KB, so this is far above
#: anything real; it exists so a newline-free stream fails fast rather than
#: buffering itself into memory.
DEFAULT_MAX_LINE_BYTES = 4 * 1024 * 1024


class VcfLimitExceeded(Exception):
    """A resource cap tripped while reading a VCF.

    Carries ``kind`` so a caller can map the two causes onto different responses
    without matching on the message text.
    """

    def __init__(self, kind: str, limit: int, message: str) -> None:
        """Build the error from the cap that tripped and a human message."""
        super().__init__(message)
        #: ``"decompressed_bytes"`` or ``"line_bytes"``.
        self.kind = kind
        #: The cap that was exceeded, in bytes.
        self.limit = limit


def _env_bytes(name: str, default: int) -> int:
    """Read a byte cap from the environment, falling back on anything unusable.

    ``0`` disables the cap. A bad value warns and uses the default rather than
    raising: a typo in a deployment variable should not take the site down.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("ignoring non-numeric %s=%r", name, raw)
        return default


def default_max_bytes() -> int:
    """Configured decompressed-byte cap, from ``SWISSISOFORM_VCF_MAX_BYTES``."""
    return _env_bytes("SWISSISOFORM_VCF_MAX_BYTES", DEFAULT_MAX_DECOMPRESSED_BYTES)


def default_max_line_bytes() -> int:
    """Configured single-line cap, from ``SWISSISOFORM_VCF_MAX_LINE_BYTES``."""
    return _env_bytes("SWISSISOFORM_VCF_MAX_LINE_BYTES", DEFAULT_MAX_LINE_BYTES)


@contextmanager
def open_vcf(path: str | Path) -> Iterator[BinaryIO]:
    """Open a VCF as **bytes**, transparently handling gzip.

    Detection is by **magic bytes**, not filename — uploads arrive named
    anything, and a mislabelled ``.vcf`` that is really gzipped would otherwise
    parse as one long line of binary.

    Binary rather than text because the caps in :func:`iter_lines` have to be
    applied to the raw stream: text-mode iteration decides where a line ends
    before anyone can object to how long it is.
    """
    path = Path(path)
    with path.open("rb") as probe:
        compressed = probe.read(2) == _GZIP_MAGIC

    handle: BinaryIO = gzip.open(path, "rb") if compressed else path.open("rb")
    try:
        yield handle
    finally:
        handle.close()


def iter_lines(
    path: str | Path,
    *,
    max_bytes: int | None = None,
    max_line_bytes: int | None = None,
) -> Iterator[str]:
    r"""Yield decoded lines, refusing to read or buffer past the caps.

    Lines are split on ``b"\n"`` *before* decoding, which is safe for UTF-8: every
    byte of a multi-byte sequence is ``>= 0x80``, so a newline can never fall
    inside one. A trailing ``\r`` is dropped, matching what text mode's universal
    newlines used to do for CRLF files.

    Args:
        path: VCF, plain or gzipped.
        max_bytes: Decompressed-byte ceiling; ``None`` reads the environment,
            ``0`` disables.
        max_line_bytes: Per-line ceiling; ``None`` reads the environment, ``0``
            disables.

    Yields:
        One line at a time, without its terminator.

    Raises:
        VcfLimitExceeded: As soon as either cap is passed.
    """
    # Named apart from the parameters deliberately: a module function called
    # ``max_line_bytes`` would be shadowed by the keyword of the same name.
    cap_total = default_max_bytes() if max_bytes is None else max_bytes
    cap_line = default_max_line_bytes() if max_line_bytes is None else max_line_bytes

    total = 0
    pending = b""

    with open_vcf(path) as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break

            total += len(chunk)
            if cap_total and total > cap_total:
                raise VcfLimitExceeded(
                    "decompressed_bytes",
                    cap_total,
                    f"VCF expands past the {_human_bytes(cap_total)} decompressed limit",
                )

            # split() rather than a find/slice loop: it is one C-level pass, and
            # the reader sits in front of every scanned record.
            parts = (pending + chunk).split(b"\n")
            # The tail has no newline yet, so it is the start of the next line.
            pending = parts.pop()

            # Checked on what is still pending, so a stream with no newline trips
            # here rather than after the whole thing is in memory.
            if cap_line and len(pending) > cap_line:
                raise VcfLimitExceeded(
                    "line_bytes",
                    cap_line,
                    f"VCF has a line longer than the {_human_bytes(cap_line)} limit "
                    "(or is not line-oriented text)",
                )

            # Inlined rather than calling _decode: this runs once per record of
            # every scanned file, and the call overhead alone was measurable.
            for part in parts:
                if part.endswith(b"\r"):
                    part = part[:-1]
                yield part.decode("utf-8", errors="replace")

    if pending:
        yield _decode(pending)


def _human_bytes(n: int) -> str:
    """Render a byte cap at whatever scale reads naturally."""
    for unit, size in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if n >= size:
            return f"{n / size:.1f} {unit}".replace(".0 ", " ")
    return f"{n} bytes"


def _decode(raw: bytes) -> str:
    """Decode one line, tolerating CRLF and invalid bytes the way text mode did."""
    if raw.endswith(b"\r"):
        raw = raw[:-1]
    return raw.decode("utf-8", errors="replace")


def iter_data_lines(
    path: str | Path,
    *,
    max_bytes: int | None = None,
    max_line_bytes: int | None = None,
) -> Iterator[tuple[int, str]]:
    """Yield ``(line_no, line)`` for every non-header line.

    ``line_no`` is 1-based over the **whole file** including headers, so it
    reproduces the record with ``sed -n '<line_no>p'``.

    Raises:
        VcfLimitExceeded: Propagated from :func:`iter_lines`.
    """
    for line_no, line in enumerate(
        iter_lines(path, max_bytes=max_bytes, max_line_bytes=max_line_bytes), start=1
    ):
        if line.startswith("#"):
            continue
        if not line.strip():
            continue
        yield line_no, line
