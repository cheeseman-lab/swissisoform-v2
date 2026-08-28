"""Resource caps on the VCF reader and the scan loop.

Deliberately free of the external fixture data the other variantquery suites are
gated on: every input here is built in ``tmp_path``, so the caps stay covered in
an environment that has no ``ecf_data`` and no built run.
"""

from __future__ import annotations

import gzip
import time
from pathlib import Path

import pytest

from swissisoform.variantquery.index import OrfIndex, OrfRecord
from swissisoform.variantquery.scan import scan
from swissisoform.variantquery.vcf import VcfLimitExceeded, iter_data_lines, iter_lines

HEADER = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"


def write_vcf(path: Path, n_records: int, *, gzipped: bool = False) -> Path:
    """A minimal well-formed VCF with ``n_records`` data lines."""
    body = HEADER + "".join(f"chr1\t{1000 + i}\t.\tA\tT\t.\tPASS\t.\n" for i in range(n_records))
    if gzipped:
        with gzip.open(path, "wb") as handle:
            handle.write(body.encode())
    else:
        path.write_bytes(body.encode())
    return path


@pytest.fixture
def index() -> OrfIndex:
    """One ORF on chr1 with no CDS, so classification takes the sequence-free path."""
    record = OrfRecord(
        gene_name="TESTG",
        tis_id="chr1:900:+:ATG:T1",
        transcript_id="T1",
        chrom="chr1",
        strand="+",
        orf_exons=((900, 2000),),
        canonical_orf_exons=((1200, 2000),),
        unique_intervals=((900, 1200),),
        shared_intervals=((1200, 2000),),
        canonical_len=266,
        isoform_len=366,
        diff_space="isoform",
        orf_type="extended",
    )
    return OrfIndex.from_records([record], version="test")


# ----------------------------------------------------------------------
# Reader caps
# ----------------------------------------------------------------------


def test_a_gzip_bomb_trips_the_decompressed_cap(tmp_path: Path) -> None:
    """The compressed size is tiny; the cap has to see the expanded stream."""
    path = tmp_path / "bomb.vcf.gz"
    with gzip.open(path, "wb") as handle:
        handle.write(b"A" * (8 * 1024 * 1024))  # ~8 MiB of one repeated byte

    assert path.stat().st_size < 64 * 1024, "the bomb should be small on disk"

    with pytest.raises(VcfLimitExceeded) as excinfo:
        list(iter_lines(path, max_bytes=1024 * 1024, max_line_bytes=0))
    assert excinfo.value.kind == "decompressed_bytes"
    assert excinfo.value.limit == 1024 * 1024


def test_the_cap_stops_reading_rather_than_reporting_afterwards(tmp_path: Path) -> None:
    """Refusal has to come before the work, or it is not a defence."""
    path = tmp_path / "big.vcf"
    path.write_bytes(b"chr1\t1\t.\tA\tT\t.\tPASS\t.\n" * 400_000)
    size = path.stat().st_size

    seen = 0
    with pytest.raises(VcfLimitExceeded):
        for _ in iter_lines(path, max_bytes=1024 * 1024, max_line_bytes=0):
            seen += 1

    # It yielded some lines but nowhere near the file, i.e. it gave up early.
    assert 0 < seen
    assert seen < size // 24 // 2


def test_a_newline_free_stream_trips_the_line_cap(tmp_path: Path) -> None:
    """``for line in handle`` would buffer this whole; the cap must catch it first."""
    path = tmp_path / "oneline.vcf"
    path.write_bytes(b"A" * (4 * 1024 * 1024))

    with pytest.raises(VcfLimitExceeded) as excinfo:
        list(iter_lines(path, max_bytes=0, max_line_bytes=64 * 1024))
    assert excinfo.value.kind == "line_bytes"


def test_the_plain_path_is_capped_too(tmp_path: Path) -> None:
    """Gzip is not the only route to a large stream."""
    path = tmp_path / "plain.vcf"
    path.write_bytes(b"chr1\t1\t.\tA\tT\t.\tPASS\t.\n" * 200_000)

    with pytest.raises(VcfLimitExceeded) as excinfo:
        list(iter_lines(path, max_bytes=64 * 1024, max_line_bytes=0))
    assert excinfo.value.kind == "decompressed_bytes"


def test_zero_disables_a_cap(tmp_path: Path) -> None:
    path = write_vcf(tmp_path / "small.vcf", 5)
    assert len(list(iter_lines(path, max_bytes=0, max_line_bytes=0))) == 7


def test_multibyte_utf8_spanning_a_chunk_boundary_survives(tmp_path: Path, monkeypatch) -> None:
    """Splitting bytes before decoding is only safe if this holds."""
    from swissisoform.variantquery import vcf as vcf_mod

    monkeypatch.setattr(vcf_mod, "_CHUNK", 8)
    path = tmp_path / "utf8.vcf"
    # 'é' is two bytes; with an 8-byte chunk it straddles a boundary.
    path.write_bytes("aaaaaaaébbb\nccc\n".encode())

    assert list(iter_lines(path, max_bytes=0, max_line_bytes=0)) == ["aaaaaaaébbb", "ccc"]


def test_crlf_and_a_missing_final_newline_read_the_same_as_before(tmp_path: Path) -> None:
    """Text mode used to normalise both; the binary reader has to keep doing it."""
    path = tmp_path / "crlf.vcf"
    path.write_bytes(b"#h\r\nchr1\t1\t.\tA\tT\t.\tPASS\t.\r\nchr1\t2\t.\tC\tG\t.\tPASS\t.")

    lines = list(iter_data_lines(path, max_bytes=0, max_line_bytes=0))
    assert [n for n, _ in lines] == [2, 3]
    assert not any(line.endswith("\r") for _, line in lines)


def test_line_numbers_still_count_headers(tmp_path: Path) -> None:
    """``line_no`` is reproducible with ``sed -n '<n>p'`` — headers included."""
    path = write_vcf(tmp_path / "n.vcf", 3)
    assert [n for n, _ in iter_data_lines(path, max_bytes=0, max_line_bytes=0)] == [3, 4, 5]


# ----------------------------------------------------------------------
# Scan budgets
# ----------------------------------------------------------------------


def test_the_record_budget_stops_the_scan_and_says_so(tmp_path: Path, index: OrfIndex) -> None:
    path = write_vcf(tmp_path / "many.vcf", 50)

    result = scan(path, index, max_records=10, max_seconds=0)

    assert result.counts.stopped == "records"
    assert result.counts.lines == 10
    assert result.counts.alleles == 10


def test_the_wall_clock_budget_stops_the_scan_and_says_so(
    tmp_path: Path, index: OrfIndex, monkeypatch
) -> None:
    path = write_vcf(tmp_path / "many.vcf", 50)

    # A clock that jumps past any deadline on its second read: the first call
    # sets the deadline, the next one is already past it.
    ticks = iter([0.0, 1e9])
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks, 1e9))

    result = scan(path, index, max_seconds=1.0, max_records=0)

    assert result.counts.stopped == "time"
    assert result.counts.lines == 0


def test_an_unbudgeted_scan_reports_no_stop(tmp_path: Path, index: OrfIndex) -> None:
    path = write_vcf(tmp_path / "many.vcf", 20)

    result = scan(path, index, max_records=0, max_seconds=0)

    assert result.counts.stopped == ""
    assert result.counts.lines == 20


def test_stopped_and_truncated_are_independent(tmp_path: Path, index: OrfIndex) -> None:
    """The hit cap bounds output and leaves the counts complete; a budget does not."""
    path = write_vcf(tmp_path / "many.vcf", 20)

    result = scan(path, index, max_hits=3, max_records=0, max_seconds=0)

    assert result.counts.truncated is True
    assert result.counts.stopped == ""
    assert len(result.hits) == 3
    # Counting continued past the cap — that is the contract truncated carries.
    assert result.counts.hits == 20


def test_a_stopped_scan_still_returns_a_well_formed_result(tmp_path: Path, index: OrfIndex) -> None:
    path = write_vcf(tmp_path / "many.vcf", 50)

    result = scan(path, index, max_records=5, max_seconds=0)
    payload = result.to_dict()

    assert payload["counts"]["stopped"] == "records"
    assert isinstance(payload["genes"], list)
    assert isinstance(payload["hits"], list)
    assert all(hit["gene"] == "TESTG" for hit in payload["hits"])


def test_a_reader_cap_propagates_out_of_scan(tmp_path: Path, index: OrfIndex) -> None:
    """scan() must not swallow it — only the caller knows how to report a bad input."""
    path = tmp_path / "oneline.vcf"
    path.write_bytes(b"A" * (256 * 1024))

    import swissisoform.variantquery.vcf as vcf_mod

    original = vcf_mod.default_max_line_bytes
    try:
        vcf_mod.default_max_line_bytes = lambda: 4096
        with pytest.raises(VcfLimitExceeded):
            scan(path, index)
    finally:
        vcf_mod.default_max_line_bytes = original
