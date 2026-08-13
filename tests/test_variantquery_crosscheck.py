"""Pin ``frame.coding_offset`` against the pipeline's own position mapper.

``ConsequenceValidator.build_position_map_from_orf`` is the authority on
genomic→coding offsets, but it materialises a full ``{gpos: offset}`` dict and
lives in a module that imports biopython — neither of which the website image can
afford. :func:`swissisoform.variantquery.frame.coding_offset` is an O(exons)
arithmetic restatement of the same walk.

Two implementations of one convention will drift unless something holds them
together. This is that something: it replays every ORF in the ``cheeseman_test``
run through both and requires exact agreement at every base.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from swissisoform.variantquery.frame import coding_offset
from swissisoform.variantquery.load import load_index_from_paired

RUN_DIR = Path(__file__).resolve().parents[1] / "data" / "output" / "cheeseman_test"
PAIRED = RUN_DIR / "all_paired.parquet"

pytestmark = pytest.mark.skipif(
    not PAIRED.is_file(), reason="needs the cheeseman_test run (outside the repo)"
)


@pytest.fixture(scope="module")
def validator():
    from swissisoform.clinical.validate import ConsequenceValidator

    return ConsequenceValidator()


@pytest.fixture(scope="module")
def records():
    return load_index_from_paired(PAIRED).records


def test_coding_offset_agrees_with_the_validator_at_every_base(validator, records) -> None:
    """Exact agreement over every base of every ORF, both strands, both ORFs."""
    compared = 0
    for record in records:
        for exons in (record.orf_exons, record.canonical_orf_exons):
            if not exons:
                continue
            reference = validator.build_position_map_from_orf(list(exons), record.strand)
            for gpos, expected in reference.items():
                assert coding_offset(exons, record.strand, gpos) == expected, (
                    f"{record.tis_id} strand={record.strand} pos={gpos}"
                )
            compared += len(reference)
    assert compared > 10_000, f"only {compared} bases compared — fixture looks empty"


def test_positions_outside_the_orf_return_none(validator, records) -> None:
    """The dict has no entry; the arithmetic version must return None, not 0."""
    for record in records[:5]:
        if not record.orf_exons:
            continue
        reference = validator.build_position_map_from_orf(list(record.orf_exons), record.strand)
        first_start = record.orf_exons[0][0]
        last_end = record.orf_exons[-1][1]
        for outside in (first_start, last_end + 1):
            assert outside not in reference
            assert coding_offset(record.orf_exons, record.strand, outside) is None


def test_total_orf_length_is_three_times_the_protein_length(records) -> None:
    """A sanity check on the exon walk itself: the ORF must be a whole codon count."""
    for record in records:
        if not record.orf_exons:
            continue
        total = sum(end - start for start, end in record.orf_exons)
        assert total % 3 == 0, f"{record.tis_id} ORF is {total} nt, not a multiple of 3"
