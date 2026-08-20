"""Acceptance test: ``ecf_data/test.vcf`` against the ``cheeseman_test`` run.

The fixture is 17 hand-placed records with genuine GRCh38 reference bases,
covering both strands, both ORF types, both coordinate frames, all four variant
classes, interval boundaries, three kinds of negative, and three malformed
inputs. ``test_expectations.tsv`` carries the expected gene / frame / residue / x
per row, so these assertions come from the file rather than from numbers retyped
into Python.

Skipped when the fixture or the run is unavailable (a fresh clone, or CI without
`/lab`), because both live outside the repository.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from swissisoform.variantquery.frame import plotly_x
from swissisoform.variantquery.load import load_index
from swissisoform.variantquery.scan import scan
from swissisoform.variantquery.spec import SV_BREAKEND

FIXTURE_DIR = Path("/lab/barcheese01/ating/ecf_data")
VCF = FIXTURE_DIR / "test.vcf"
EXPECTATIONS = FIXTURE_DIR / "test_expectations.tsv"
RUN_DIR = Path(__file__).resolve().parents[1] / "data" / "output" / "cheeseman_test"
PAIRED = RUN_DIR / "all_paired.parquet"
ORF_INDEX = RUN_DIR / "orf_index.parquet"

pytestmark = pytest.mark.skipif(
    not (VCF.is_file() and EXPECTATIONS.is_file() and PAIRED.is_file()),
    reason="needs ecf_data/test.vcf + the cheeseman_test run (both outside the repo)",
)


@pytest.fixture(scope="module")
def index():
    """The built index, which is what production loads.

    ``all_paired.parquet`` carries no coding sequence — that is derived from the
    genome by ``build_orf_index.py`` — so building from it would exercise the
    sequence-free fallback rather than the real path.
    """
    if not ORF_INDEX.is_file():
        pytest.skip(
            "needs orf_index.parquet — "
            "python scripts/export/build_orf_index.py --run cheeseman_test"
        )
    return load_index(ORF_INDEX)


@pytest.fixture(scope="module")
def expectations() -> list[dict[str, str]]:
    with EXPECTATIONS.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


@pytest.fixture(scope="module")
def result(index):
    return scan(VCF, index)


@pytest.fixture(scope="module")
def result_all_filters(index):
    return scan(VCF, index, pass_only=False)


def _positive_rows(expectations: list[dict[str, str]]) -> list[dict[str, str]]:
    """Rows expected to resolve to a residue under the default PASS filter."""
    return [
        row
        for row in expectations
        if row["expect_gene"] and row["expect_residue"] and row["filter"] == "PASS"
    ]


# ----------------------------------------------------------------------
# Every positional row resolves to its expected gene / frame / residue / x
# ----------------------------------------------------------------------


def test_every_expected_hit_is_present(result, expectations) -> None:
    """Each row's (gene, frame, residue, region) must appear among the hits.

    Asserted as membership, not equality: a position often falls inside several
    isoforms of the same gene, and the fixture pins one representative per
    (gene, orf_type). Extra hits on sibling isoforms are correct behaviour.
    """
    rows = _positive_rows(expectations)
    # A floor rather than an exact count: adding fixture coverage should not break
    # this test, but losing rows silently should.
    assert len(rows) >= 12, f"positional PASS rows dropped to {len(rows)}"

    actual = {(h.gene, h.frame, h.residue, h.region, h.pos) for h in result.hits}
    for row in rows:
        wanted = (
            row["expect_gene"],
            row["expect_frame"],
            int(row["expect_residue"]),
            row["expect_region"],
            int(row["pos"]),
        )
        assert wanted in actual, f"{row['category']}: {wanted} not among the hits"


def test_plotly_x_matches_the_expected_figure_coordinate(result, expectations, index) -> None:
    """Residue → x is the one conversion the gene page will reuse."""
    by_tis = {r.tis_id: r for r in index.records}
    checked = 0
    for row in _positive_rows(expectations):
        wanted_x = int(row["expect_x"])
        for hit in result.hits:
            if (
                hit.gene == row["expect_gene"]
                and hit.pos == int(row["pos"])
                and hit.residue == int(row["expect_residue"])
                and hit.frame == row["expect_frame"]
            ):
                if plotly_x(by_tis[hit.tis_id], hit.residue, hit.frame) == wanted_x:
                    checked += 1
                    break
        else:
            pytest.fail(f"{row['category']}: no hit yields x={wanted_x}")
    assert checked == len(_positive_rows(expectations))


# ----------------------------------------------------------------------
# Interval boundaries — the convention, pinned end to end
# ----------------------------------------------------------------------


def test_first_and_last_orf_bases_are_inside(result, expectations) -> None:
    """CDC34's bar spans x ∈ [-55, 235]; its first/last ORF base must hit those."""
    boundaries = {
        row["category"]: row for row in expectations if row["category"].startswith("boundary_")
    }
    assert set(boundaries) == {"boundary_first_orf_base", "boundary_last_orf_base"}
    for row in boundaries.values():
        matches = [
            h
            for h in result.hits
            if h.pos == int(row["pos"]) and h.residue == int(row["expect_residue"])
        ]
        assert matches, f"{row['category']} at {row['chrom']}:{row['pos']} was not a hit"


def test_the_base_before_the_orf_start_is_outside(index) -> None:
    """``orf_start`` itself is the exclusive bound — only ``orf_start + 1`` is in."""
    first_in = 531767  # boundary_first_orf_base
    assert index.lookup("chr19", first_in), "first in-ORF base should hit"
    assert not index.lookup("chr19", first_in - 1), "exclusive start should not hit"


# ----------------------------------------------------------------------
# The three distinct negatives
# ----------------------------------------------------------------------


def test_negatives_produce_no_hits(result, expectations) -> None:
    """Intronic, exonic-untranslated and intergenic must all resolve to nothing."""
    negatives = [
        row
        for row in expectations
        if row["category"] in ("intronic", "exonic_untranslated", "intergenic")
    ]
    assert len(negatives) == 3
    hit_positions = {h.pos for h in result.hits}
    for row in negatives:
        assert int(row["pos"]) not in hit_positions, f"{row['category']} should not hit"


def test_intergenic_contig_is_reported_as_off_catalog(result) -> None:
    """chr1 carries no cheeseman_test ORF, so this is a contig miss, not an ORF miss.

    Keeping the two apart is what lets the UI distinguish a chromosome-naming
    problem from a genuine negative.
    """
    assert result.counts.off_catalog_contig == 1
    assert result.counts.no_orf == 2  # intronic + exonic-untranslated


# ----------------------------------------------------------------------
# Filtering and malformed input
# ----------------------------------------------------------------------


def test_non_pass_row_is_dropped_by_default(result, expectations) -> None:
    (row,) = [r for r in expectations if r["category"] == "non_pass"]
    assert result.counts.skipped_non_pass == 1
    assert int(row["pos"]) not in {h.pos for h in result.hits}


def test_non_pass_row_resolves_when_filtering_is_off(result_all_filters, expectations) -> None:
    """Same record, same expected residue — only the filter changes."""
    (row,) = [r for r in expectations if r["category"] == "non_pass"]
    matches = [
        h
        for h in result_all_filters.hits
        if h.pos == int(row["pos"]) and h.residue == int(row["expect_residue"])
    ]
    assert matches, "non-PASS row should resolve with pass_only=False"
    assert result_all_filters.counts.skipped_non_pass == 0


def test_multi_allelic_row_contributes_both_alleles(result, expectations) -> None:
    (row,) = [r for r in expectations if r["category"] == "multi_allelic"]
    alts = {h.alt for h in result.hits if h.pos == int(row["pos"])}
    assert {"A", "T"} <= alts


def test_breakend_row_is_rejected_with_its_own_reason(result) -> None:
    assert result.counts.rejected.get(SV_BREAKEND) == 1


# ----------------------------------------------------------------------
# Funnel arithmetic
# ----------------------------------------------------------------------


def test_funnel_accounts_for_every_allele(result, expectations) -> None:
    """Nothing may vanish: every parsed allele is filtered, missed, or hit.

    Alleles are counted as ``(line_no, alt)`` rather than ``(chrom, pos, ref,
    alt)`` — the fixture deliberately repeats ``chr19:531848 G>A`` on both the
    single-allele and the multi-allelic row, so a variant-tuple set would collapse
    two distinct records into one.
    """
    counts = result.counts
    # Derived, not hardcoded: one VCF record per expectation row. This also catches
    # the two fixture files drifting apart, which a literal count would not.
    assert counts.lines == len(expectations)
    alleles_with_hits = len({(h.line_no, h.alt) for h in result.hits})
    assert (
        counts.alleles
        == counts.skipped_non_pass + counts.off_catalog_contig + counts.no_orf + alleles_with_hits
    )


def test_genes_rollup_matches_the_hits(result) -> None:
    assert result.counts.genes_hit == len(result.genes)
    # n_hits is the (variant, isoform) pair count and must reconcile with the
    # global total; n_variants is the smaller distinct-allele count.
    assert sum(g.n_hits for g in result.genes) == result.counts.hits
    assert sum(g.n_variants for g in result.genes) <= result.counts.hits


# ----------------------------------------------------------------------
# Full-width input: the fixture carries INFO, FORMAT and a tumour/normal pair
# ----------------------------------------------------------------------


def test_fixture_really_is_full_width() -> None:
    """Guards the guard: if the fixture loses its sample columns, the next two
    tests would pass vacuously.
    """
    header = next(line for line in VCF.read_text().splitlines() if line.startswith("#CHROM"))
    columns = header.lstrip("#").split("\t")
    assert columns[:9] == [
        "CHROM",
        "POS",
        "ID",
        "REF",
        "ALT",
        "QUAL",
        "FILTER",
        "INFO",
        "FORMAT",
    ]
    assert columns[9:] == ["zcc10-N", "zcc10-T"]


def test_digest_carries_no_genotype_or_sample_data(result) -> None:
    """Read depths and genotypes must never reach the digest, hence never disk.

    The digest is the one artifact that persists for 24 h, so this is the
    boundary where uploaded genotype data would leak if a hit ever grew a
    "raw line" field.
    """
    serialised = json.dumps(result.to_dict())
    leaks = (
        "zcc10-N",  # sample names
        "zcc10-T",
        "GT:DP:FDP",  # Strelka's SNV FORMAT
        "GT:DP:DP2",  # Strelka's indel FORMAT
        "0/1:",  # genotype + colon-joined per-sample fields
        "0/0:",
        "SOMATIC",  # INFO flags
        "QSS_NT",
        "TESTCASE",
    )
    for leak in leaks:
        assert leak not in serialised, f"{leak!r} leaked into the scan digest"


def test_hit_records_expose_only_the_expected_fields(result) -> None:
    """A whitelist, so a future field addition has to be a deliberate decision."""
    assert result.hits
    for hit in result.hits:
        assert set(hit.to_dict()) == {
            "line_no",
            "chrom",
            "pos",
            "ref",
            "alt",
            "vclass",
            "gene",
            "tis_id",
            "transcript_id",
            "orf_type",
            "frame",
            "residue",
            "region",
            # Added deliberately with DIGEST_SCHEMA d3. None of these carries
            # anything from the VCF's sample columns — they are derived from the
            # ORF's own reference sequence.
            "consequence",
            "aa_ref",
            "aa_alt",
            "consequence_note",
        }


# ----------------------------------------------------------------------
# Consequence, on real ORFs
# ----------------------------------------------------------------------


def test_every_hit_carries_a_consequence_term(result) -> None:
    """The figure groups rows by term, so a blank one would be undrawable."""
    assert result.hits
    assert all(h.consequence for h in result.hits)


def test_synonymous_hits_exist_and_are_not_guessed_as_missense(result) -> None:
    """The case that justifies shipping the CDS.

    EIF2B1's unique-region SNV is silent (GAC->GAT). A "SNV means missense" guess
    would draw it as a missense hit inside a differential region — the exact false
    positive someone would act on. SRSF2's extension SNV is silent too.
    """
    by_gene = {}
    for hit in result.hits:
        by_gene.setdefault(hit.gene, set()).add(
            (hit.consequence, hit.aa_ref, hit.aa_alt, hit.residue)
        )

    assert ("synonymous_variant", "D", "D", 1) in by_gene["EIF2B1"]
    assert ("synonymous_variant", "R", "R", 17) in by_gene["SRSF2"]
    n_syn = sum(1 for h in result.hits if h.consequence == "synonymous_variant")
    assert n_syn >= 6, f"expected the silent hits to survive, got {n_syn}"


def test_indels_are_classified_by_length(result) -> None:
    trnt1 = [h for h in result.hits if h.gene == "TRNT1"]
    assert any(h.consequence == "frameshift_variant" for h in trnt1), trnt1
    ube2d2 = [h for h in result.hits if h.gene == "UBE2D2"]
    assert any(h.consequence == "inframe_insertion" for h in ube2d2), ube2d2


def test_multi_codon_mnv_reports_both_residues(result) -> None:
    """A 3-base substitution straddling a codon boundary changes two residues."""
    ube2m = [h for h in result.hits if h.gene == "UBE2M" and h.ref == "CAA"]
    assert ube2m
    hit = ube2m[0]
    assert (hit.aa_ref, hit.aa_alt) == ("FE", "SK")
    # The residue is the FIRST codon the span changes. This is minus-strand, so the
    # span's lowest coding offset — not the one POS maps to, which is a codon later.
    assert hit.residue == 224, hit


def test_a_start_codon_that_still_initiates_is_not_a_start_loss(result) -> None:
    """CDC34's TIS is CTG, and this C>T makes it TTG — still a near-cognate start.

    What decides a start-codon variant is whether the trinucleotide still initiates,
    not whether the first codon changed: TTG does, so the ORF survives and the
    question falls back to the residue, which is unchanged (Leu either way).
    """
    first_base = [h for h in result.hits if h.pos == 531767]
    assert first_base
    hit = first_base[0]
    assert hit.consequence == "synonymous_variant", hit
    assert (hit.aa_ref, hit.aa_alt, hit.residue) == ("L", "L", 0)


def test_residue_numbering_differs_per_orf_for_one_nucleotide(result) -> None:
    """Same substitution, different residue number in each ORF containing it.

    This is why ``residue`` travels with ``frame`` — the number is meaningless
    without knowing which protein it counts against.
    """
    same_pos = [h for h in result.hits if h.pos == 541549 and h.residue is not None]
    residues = {h.residue for h in same_pos}
    assert len(residues) > 1, f"expected per-ORF numbering, got {residues}"
    assert all(h.consequence == "synonymous_variant" for h in same_pos)


def test_stop_gained_is_reached_on_real_data(result) -> None:
    """The most clinically loaded row on the figure; previously untested.

    CBX1 is minus-strand, so a plus-strand G>A transition is C>T in mRNA sense,
    turning CGA into TGA.
    """
    hits = [h for h in result.hits if h.pos == 48101323]
    assert hits, "the stop_gained fixture row produced no hit"
    assert all(h.consequence == "stop_gained" for h in hits), hits
    assert all(h.aa_alt == "*" for h in hits), [h.aa_alt for h in hits]
    assert all(h.region == "unique" for h in hits), "a nonsense in the extension"


def test_inframe_deletion_is_classified_and_placed(result) -> None:
    """Class from the length delta, residue from the anchor — no sequence read.

    Amino acids stay empty for indels, exactly as they do for an annotated variant:
    the classifier resolves them by length rather than translating.
    """
    hits = [h for h in result.hits if h.pos == 531771]
    assert hits
    hit = hits[0]
    assert hit.consequence == "inframe_deletion"
    assert (hit.aa_ref, hit.aa_alt) == ("", "")
    assert hit.residue == 1, hit


def test_a_span_leaving_the_exon_still_gets_its_class(result) -> None:
    """The length delta needs no sequence, so the row is still right.

    Amino acids are absent because they are absent for every indel, not because this
    one crosses an intron — the classifier never translates an indel.
    """
    hits = [h for h in result.hits if h.pos == 532107]
    assert hits
    hit = hits[0]
    assert hit.consequence == "frameshift_variant", hit
    assert (hit.aa_ref, hit.aa_alt) == ("", "")
    assert hit.residue is not None, "still placeable — pos itself is in the exon"


def test_the_vocabulary_the_figure_draws_is_actually_exercised(result) -> None:
    """Guards the fixture's purpose: each figure row needs a real example."""
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
        assert term in seen, f"no fixture row produces {term}"
