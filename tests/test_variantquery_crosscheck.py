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

REPO = Path(__file__).resolve().parents[1]
RUN_DIR = REPO / "data" / "output" / "cheeseman_test"
PAIRED = RUN_DIR / "all_paired.parquet"
GENOME = REPO / "data" / "reference" / "Gencode_v49_GRCh38.primary_assembly.genome.fa"

pytestmark = pytest.mark.skipif(
    not PAIRED.is_file(), reason="needs the cheeseman_test run (outside the repo)"
)


@pytest.fixture(scope="module")
def validator():
    from swissisoform.clinical.validate import ConsequenceValidator

    return ConsequenceValidator()


@pytest.fixture(scope="module")
def genome_validator():
    """A validator wired to the genome, for the sequence-extraction comparison."""
    from swissisoform.clinical.validate import ConsequenceValidator

    if not GENOME.is_file():
        pytest.skip("needs the GRCh38 FASTA")
    return ConsequenceValidator(genome_fasta=str(GENOME))


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


# ----------------------------------------------------------------------
# Translation + consequence, against biopython on real ORFs
# ----------------------------------------------------------------------


def test_translate_agrees_with_biopython_on_every_orf(genome_validator, records) -> None:
    """Our stdlib codon table vs Bio.Seq.translate over real coding sequences."""
    from Bio.Seq import Seq

    from swissisoform.variantquery.consequence import translate

    checked = 0
    for record in records:
        for exons in (record.orf_exons, record.canonical_orf_exons):
            if not exons:
                continue
            cds = genome_validator.build_coding_sequence_from_orf(
                list(exons), record.strand, record.chrom
            )
            assert translate(cds) == str(Seq(cds).translate()), record.tis_id
            checked += 1
    assert checked >= 30, f"only {checked} ORFs compared"


def test_orf_length_is_three_times_the_protein(genome_validator, records) -> None:
    """The build-time invariant the index will assert, checked here on real data."""
    for record in records:
        if not record.orf_exons or record.isoform_len is None:
            continue
        cds = genome_validator.build_coding_sequence_from_orf(
            list(record.orf_exons), record.strand, record.chrom
        )
        assert len(cds) == record.isoform_len * 3, (
            f"{record.tis_id}: {len(cds)} nt vs isoform_len {record.isoform_len}"
        )


def test_fixture_variants_classify_against_real_orfs(genome_validator, records) -> None:
    """Every fixture variant × every ORF containing it, through the real CDS.

    Not asserting specific terms here — that is the fixture test's job. This asserts
    the classifier never crashes on real geometry and never returns a bare term with
    no explanation, which is how a silent mis-map would show up.
    """
    import csv

    from swissisoform.variantquery.consequence import OTHER, classify
    from swissisoform.variantquery.frame import resolve_residue
    from swissisoform.variantquery.index import OrfIndex

    expectations = Path("/lab/barcheese01/ating/ecf_data/test_expectations.tsv")
    if not expectations.is_file():
        pytest.skip("needs ecf_data/test_expectations.tsv")

    index = OrfIndex.from_records(records)
    with expectations.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    classified = 0
    for row in rows:
        if not row["expect_gene"]:
            continue
        chrom = row["chrom"] if row["chrom"].startswith("chr") else f"chr{row['chrom']}"
        pos = int(row["pos"])
        ref = row["ref"]
        for alt in row["alt"].split(","):
            if not set(alt) <= set("ACGT"):
                continue
            for record in index.lookup_span(chrom, pos, pos + len(ref) - 1):
                _res, frame, _g = resolve_residue(record, pos, pos + len(ref) - 1)
                exons = record.orf_exons if frame != "canonical" else record.canonical_orf_exons
                cds = genome_validator.build_coding_sequence_from_orf(
                    list(exons), record.strand, record.chrom
                )
                c = classify(
                    exons=exons,
                    strand=record.strand,
                    cds=cds,
                    pos=pos,
                    ref=ref,
                    alt=alt,
                )
                assert c.term, f"{chrom}:{pos} {ref}>{alt} on {record.tis_id}"
                if c.term == OTHER or not c.hgvsp:
                    assert c.note, f"unexplained refusal on {record.tis_id}: {c}"
                classified += 1
    assert classified >= 20, f"only {classified} (variant, ORF) pairs classified"
