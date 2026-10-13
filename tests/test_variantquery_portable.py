"""The packed ORF index: the binary file the browser scans against, and the loaders.

Every way of building an index — the column-wise parquet loader the server uses,
the binary file the browser downloads, and the general row-by-row path — must
yield the same records and the same lookups, or a browser scan and a server scan
of one VCF would disagree.
"""

from __future__ import annotations

import copy
import gzip
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from swissisoform.variantquery.index import OrfIndex
from swissisoform.variantquery.portable import MAGIC, PORTABLE_FORMAT, from_portable, to_portable

OUTPUT = Path(__file__).resolve().parents[1] / "data" / "output"
REAL_INDEXES = [
    pytest.param(OUTPUT / run / "orf_index.parquet", id=run)
    for run in ("cheeseman_test", "full_catalog")
]


def _index() -> OrfIndex:
    return OrfIndex.from_records(
        [
            {
                "gene_name": "G",
                "tis_id": "chr1:101:+:ATG:ENST1",
                "transcript_id": "ENST1",
                "chrom": "chr1",
                "strand": "+",
                "orf_exons": [[100, 130], [200, 260]],
                "canonical_orf_exons": [[200, 260]],
                "unique_genomic_intervals": [[100, 130]],
                "shared_genomic_intervals": [[200, 260]],
                "canonical_len": 20,
                "isoform_len": 30,
                "diff_space": "isoform",
                "orf_type": "extended",
                "start_codon": "ATG",
                "canonical_start_codon": "ATG",
                "orf_cds": "ATG" * 30,
                "canonical_cds": "ATG" * 20,
                "canonical_x_offset_nt": -30,
                "canonical_per_tid_length": None,
            }
        ],
        version="v-test",
    )


def test_round_trip_keeps_every_record_and_the_version() -> None:
    index = _index()
    rebuilt = from_portable(to_portable(index))
    assert rebuilt.version == "v-test"
    assert list(rebuilt.records) == list(index.records)
    assert rebuilt.lookup("chr1", 101) == index.lookup("chr1", 101)


def test_the_bytes_depend_only_on_the_index() -> None:
    """mtime=0: the same index always serialises to the same bytes."""
    assert to_portable(_index()) == to_portable(_index())


def _with_header(data: bytes, **changes) -> bytes:
    body = gzip.decompress(data)
    (length,) = struct.unpack_from("<I", body, len(MAGIC))
    start = len(MAGIC) + 4
    header = json.loads(body[start : start + length])
    header.update(changes)
    encoded = json.dumps(header).encode()
    rest = body[start + length :]
    return gzip.compress(MAGIC + struct.pack("<I", len(encoded)) + encoded + rest)


def test_an_unknown_format_is_refused() -> None:
    with pytest.raises(ValueError, match="format"):
        from_portable(_with_header(to_portable(_index()), format=PORTABLE_FORMAT + 1))
    with pytest.raises(ValueError, match="not a portable"):
        from_portable(b"not gzip")


def test_the_old_json_layout_is_refused() -> None:
    """Format 1 was gzipped JSON rows; a stale cached copy must not load."""
    old = gzip.compress(json.dumps({"format": 1, "columns": [], "rows": []}).encode())
    with pytest.raises(ValueError, match="bad magic"):
        from_portable(old)


def test_a_none_survives_the_packed_integer_columns() -> None:
    (record,) = from_portable(to_portable(_index())).records
    assert record.canonical_per_tid_length is None
    assert record.canonical_x_offset_nt == -30


def _written(damage) -> bytes:
    """A file written from deliberately inconsistent columns, as a buggy writer would."""
    columns = copy.deepcopy(_index().columns)
    damage(columns)
    return to_portable(SimpleNamespace(columns=columns, version="v-test"))


def _malformed() -> dict[str, bytes]:
    good = to_portable(_index())
    body = gzip.decompress(good)
    (header_len,) = struct.unpack_from("<I", body, len(MAGIC))
    start = len(MAGIC) + 4
    header = json.loads(body[start : start + header_len])

    def with_header(new: object) -> bytes:
        raw = json.dumps(new).encode()
        return gzip.compress(MAGIC + struct.pack("<I", len(raw)) + raw + body[start + header_len :])

    no_blob = dict(header, blobs={k: v for k, v in header["blobs"].items() if k != "cds_pool"})
    past_end = dict(header, blobs=dict(header["blobs"], cds_pool=[0, len(body) * 2]))
    negative = dict(header, blobs=dict(header["blobs"], cds_pool=[-8, 8]))
    corrupt_stream = bytearray(good)
    for k in range(8):  # behind a valid gzip header: zlib.error, not OSError
        corrupt_stream[20 + k] ^= 0xFF
    return {
        "a corrupt deflate stream": bytes(corrupt_stream),
        "a negative blob offset": with_header(negative),
        "the wrong ORF count": with_header(dict(header, n=header["n"] + 1)),
        "an integer column one short": _written(lambda c: c.ints["isoform_len"].pop()),
        "offsets that decrease": _written(lambda c: c.offsets["orf_exons"].__setitem__(1, -1)),
        "a sequence id out of range": _written(lambda c: c.cds_ids["orf_cds"].__setitem__(0, 99)),
        "a string column one short": _written(lambda c: c.strings["gene_name"].pop()),
        "a list": gzip.compress(b"[]"),
        "truncated gzip": good[: len(good) // 2],
        "magic only": gzip.compress(MAGIC),
        "header not json": gzip.compress(MAGIC + struct.pack("<I", 1) + b"{"),
        "header a list": with_header([]),
        "a missing blob": with_header(no_blob),
        "a blob past the end": with_header(past_end),
    }


@pytest.mark.parametrize("case", sorted(_malformed()))
def test_anything_malformed_is_a_value_error(case: str) -> None:
    """The docstring's one exception type, so a caller needs one except clause."""
    with pytest.raises(ValueError, match="not a portable"):
        from_portable(_malformed()[case])


def _boundaries(index: OrfIndex):
    """Every exon edge of every ORF, plus a point just outside each one."""
    for record in index.records:
        for start, end in (*record.orf_exons, *record.canonical_orf_exons):
            for pos in (start, start + 1, (start + end) // 2 + 1, end, end + 1):
                yield record.chrom, pos


@pytest.mark.parametrize("path", REAL_INDEXES)
def test_every_way_of_loading_a_real_index_agrees(path: Path) -> None:
    """Fast column loader, binary round trip and the row path: identical answers."""
    if not path.is_file():
        pytest.skip(f"needs {path}")
    import pyarrow.parquet as pq

    from swissisoform.variantquery.load import load_index

    fast = load_index(path)
    slow = OrfIndex.from_records(pq.read_table(path).to_pylist(), version=fast.version)
    browser = from_portable(to_portable(fast))
    for other in (slow, browser):
        assert other.version == fast.version
        assert list(other.records) == list(fast.records)
        assert other.n_intervals == fast.n_intervals
    for chrom, pos in _boundaries(fast):
        expected = fast.lookup_span(chrom, pos, pos + 2)
        assert slow.lookup_span(chrom, pos, pos + 2) == expected
        assert browser.lookup_span(chrom, pos, pos + 2) == expected
