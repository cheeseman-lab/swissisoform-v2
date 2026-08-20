"""Real variants, checked against the annotation their source database published.

Every row of ``real.vcf`` is a ClinVar / gnomAD / COSMIC variant that actually falls
inside a ``cheeseman_test`` ORF, and its expectation comes from the **source's own**
protein notation — COSMIC's ``ENSP00000215574.2:p.Ser129Phe``, gnomAD's VEP
``p.Lys181Ter``. So a regression here breaks a claim somebody else's pipeline made,
which is the one thing ``test_variantquery_fixture.py`` cannot do: its expectations
are computed by the code under test.

**The comparison happens in canonical-residue space.** A source annotates one
transcript, while a position routinely lands in several ORFs numbered differently —
so the check is ``plotly_x(...) == source_position - 1``, which passes a
canonical-frame residue through and shifts an isoform-frame one by
``canonical_len - isoform_len``. That also puts the figure's own coordinate
arithmetic under an external oracle.

``match_level`` says how strict a row can be:

* ``aa``    — substitution: position **and** both amino acids must agree.
* ``pos``   — indel or delins: the notation names a residue but our classifier
  reports no amino acids for indels, so only the position is comparable.
* ``class`` — the synthetic ``start_lost`` row; real data has no example, because
  every single substitution of CDC34's CTG start lands on another near-cognate start.

Skipped when the fixture or the run is unavailable (a fresh clone, or CI without
`/lab`), because both live outside the repository.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from swissisoform.variantquery.frame import plotly_x
from swissisoform.variantquery.load import load_index
from swissisoform.variantquery.scan import scan

FIXTURE_DIR = Path("/lab/barcheese01/ating/ecf_data")
VCF = FIXTURE_DIR / "real.vcf"
EXPECTATIONS = FIXTURE_DIR / "real_expectations.tsv"
RUN_DIR = Path(__file__).resolve().parents[1] / "data" / "output" / "cheeseman_test"
ORF_INDEX = RUN_DIR / "orf_index.parquet"

pytestmark = pytest.mark.skipif(
    not (VCF.is_file() and EXPECTATIONS.is_file() and ORF_INDEX.is_file()),
    reason="needs ecf_data/real.vcf + the cheeseman_test index (both outside the repo)",
)


@pytest.fixture(scope="module")
def index():
    return load_index(ORF_INDEX)


@pytest.fixture(scope="module")
def result(index):
    return scan(VCF, index)


@pytest.fixture(scope="module")
def expectations() -> list[dict[str, str]]:
    with EXPECTATIONS.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def hits_for(result, row) -> list:
    """Every hit the scan produced for one expectation row's allele."""
    chrom = row["chrom"] if row["chrom"].startswith("chr") else f"chr{row['chrom']}"
    return [
        h
        for h in result.hits
        if h.chrom == chrom
        and h.pos == int(row["pos"])
        and h.ref == row["ref"]
        and h.alt == row["alt"]
    ]


# ----------------------------------------------------------------------
# Every row resolves, and agrees with the database that annotated it
# ----------------------------------------------------------------------


def test_every_real_variant_is_found(result, expectations) -> None:
    """A variant the pipeline already annotated must be reachable by the scan."""
    assert expectations
    for row in expectations:
        assert hits_for(result, row), f"{row['category']}: no hit for {row['variant_id']}"


def test_the_expected_orf_and_residue_are_reproduced(result, expectations, index) -> None:
    """The row's own (frame, residue, x, region) must come back from the scan.

    Matched on the gene and frame the fixture recorded — the same allele legitimately
    produces other hits in other ORFs at other residues.
    """
    for row in expectations:
        want = (row["expect_frame"], int(row["expect_residue"]), row["expect_region"])
        got = {
            (h.frame, h.residue, h.region)
            for h in hits_for(result, row)
            if h.gene == row["expect_gene"]
        }
        assert want in got, f"{row['category']}: expected {want}, got {sorted(got)}"

        hit = next(
            h for h in hits_for(result, row)
            if h.gene == row["expect_gene"] and h.frame == row["expect_frame"]
            and h.residue == int(row["expect_residue"])
        )
        record = index.by_tis_id(hit.tis_id)
        assert plotly_x(record, hit.residue, hit.frame) == int(row["expect_x"])


def test_the_consequence_matches_the_source(result, expectations) -> None:
    for row in expectations:
        terms = {h.consequence for h in hits_for(result, row) if h.gene == row["expect_gene"]}
        assert row["expect_consequence"] in terms, f"{row['category']}: got {sorted(terms)}"


def test_amino_acids_match_the_published_notation(result, expectations, index) -> None:
    """For substitutions, our residues must be the ones the source named.

    This is the assertion with an outside oracle behind it: ``expect_aa_ref`` /
    ``expect_aa_alt`` were parsed out of the database's ``hgvsp``, not produced here.
    """
    checked = 0
    for row in expectations:
        if row["match_level"] != "aa":
            continue
        hit = next(
            h for h in hits_for(result, row)
            if h.gene == row["expect_gene"] and h.frame == row["expect_frame"]
            and h.residue == int(row["expect_residue"])
        )
        assert (hit.aa_ref, hit.aa_alt) == (row["expect_aa_ref"], row["expect_aa_alt"]), (
            f"{row['category']}: {row['source_hgvsp']} says "
            f"{row['expect_aa_ref']}>{row['expect_aa_alt']}, we say {hit.aa_ref}>{hit.aa_alt}"
        )
        # The source's residue must fall inside our slice, and be the one that
        # changed. A substitution spanning a codon boundary alters two residues and
        # the source names only the differing one — COSMIC writes CDC34's ACC>ATT as
        # p.Pro116Ser while we report NP>NS starting one residue earlier. Comparing
        # the slice's first position against the source's would call that a
        # disagreement when the two agree exactly about which residue became what.
        record = index.by_tis_id(hit.tis_id)
        start = plotly_x(record, hit.residue, hit.frame)
        source_pos = int("".join(c for c in row["source_hgvsp"].split(":")[-1] if c.isdigit()))
        offset = (source_pos - 1) - start
        assert 0 <= offset < len(hit.aa_ref), (
            f"{row['category']}: {row['source_hgvsp']} names residue {source_pos}, "
            f"outside our slice at canonical {start}..{start + len(hit.aa_ref) - 1}"
        )
        assert (hit.aa_ref[offset], hit.aa_alt[offset]) == (
            row["expect_aa_ref"][offset],
            row["expect_aa_alt"][offset],
        )
        checked += 1
    assert checked >= 6, f"only {checked} rows carried an amino-acid oracle"


def test_indel_rows_report_a_position_and_no_amino_acids(result, expectations) -> None:
    """Indels are classified by length delta, so amino acids are absent by design."""
    checked = 0
    for row in expectations:
        if row["expect_consequence"] not in (
            "frameshift_variant", "inframe_deletion", "inframe_insertion"
        ):
            continue
        hit = next(
            h for h in hits_for(result, row)
            if h.gene == row["expect_gene"] and h.residue == int(row["expect_residue"])
        )
        assert (hit.aa_ref, hit.aa_alt) == ("", "")
        assert hit.residue is not None
        checked += 1
    assert checked >= 3, f"only {checked} indel rows"


# ----------------------------------------------------------------------
# Coverage: the fixture has to be broad enough to be worth running
# ----------------------------------------------------------------------


def test_every_drawable_consequence_row_has_a_real_example(result) -> None:
    """The gene figure has one row per term; each needs an example on real data."""
    seen = set(result.counts.consequences)
    for term in (
        "missense_variant",
        "synonymous_variant",
        "stop_gained",
        "start_lost",
        "frameshift_variant",
        "inframe_insertion",
        "inframe_deletion",
    ):
        assert term in seen, f"no row produces {term}: {sorted(seen)}"


def test_all_four_variant_classes_are_present(expectations) -> None:
    """snv / MNV / insertion / deletion — the MNV rows are the ones that used to
    come back as an unclassified ``"mnv"`` consequence before the codon walk widened.
    """
    assert {row["vclass"] for row in expectations} >= {"snv", "MNV", "insertion", "deletion"}


def test_both_regions_and_both_frames_are_exercised(expectations) -> None:
    assert {row["expect_region"] for row in expectations} >= {"unique", "shared"}
    assert {row["expect_frame"] for row in expectations} >= {"isoform", "canonical"}


def test_the_fixture_spans_most_of_the_run(expectations) -> None:
    """A per-gene regression must not hide behind a gene the fixture never touches."""
    genes = {row["expect_gene"] for row in expectations}
    assert len(genes) >= 7, f"only {len(genes)} genes covered: {sorted(genes)}"


def test_all_three_databases_are_represented(expectations) -> None:
    """Each source writes its notation differently; all three must parse."""
    assert {row["source"] for row in expectations} >= {"ClinVar", "gnomAD", "COSMIC"}
