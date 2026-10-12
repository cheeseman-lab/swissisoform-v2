"""The gzipped-JSON ORF index the browser scans against.

It must rebuild exactly the index the server builds from the parquet — the same
records and the same lookups — or a browser scan and a server scan of one VCF
would disagree.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from swissisoform.variantquery.index import OrfIndex
from swissisoform.variantquery.portable import PORTABLE_FORMAT, from_portable, to_portable

RUN_INDEX = (
    Path(__file__).resolve().parents[1] / "data" / "output" / "cheeseman_test" / "orf_index.parquet"
)


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


def test_an_unknown_format_is_refused() -> None:
    body = json.loads(gzip.decompress(to_portable(_index())))
    body["format"] = PORTABLE_FORMAT + 1
    with pytest.raises(ValueError, match="format"):
        from_portable(gzip.compress(json.dumps(body).encode()))
    with pytest.raises(ValueError, match="not a portable"):
        from_portable(b"not gzip")


@pytest.mark.skipif(not RUN_INDEX.is_file(), reason="needs the cheeseman_test orf_index")
def test_a_real_index_round_trips_record_for_record() -> None:
    from swissisoform.variantquery.load import load_index

    index = load_index(RUN_INDEX)
    rebuilt = from_portable(to_portable(index))
    assert rebuilt.version == index.version
    assert list(rebuilt.records) == list(index.records)
    assert rebuilt.n_intervals == index.n_intervals
