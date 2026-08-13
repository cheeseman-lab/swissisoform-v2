"""VCF line parsing: normalisation, variant classes, and the awkward cases."""

from __future__ import annotations

import pytest

from swissisoform.variantquery.spec import (
    MALFORMED,
    SV_BREAKEND,
    UNSUPPORTED_ALT,
    Rejection,
    VariantSpec,
    normalize_chrom,
    parse_line,
)


def line(*fields: str) -> str:
    return "\t".join(fields)


# ----------------------------------------------------------------------
# Chromosome normalisation
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("17", "chr17"),
        ("chr17", "chr17"),
        (" 17 ", "chr17"),
        ("X", "chrX"),
        ("MT", "chrM"),
        ("chrMT", "chrM"),
        ("chrM", "chrM"),
    ],
)
def test_normalize_chrom(raw: str, expected: str) -> None:
    """The real hg38 somatic VCFs use bare names, so this runs on every line."""
    assert normalize_chrom(raw) == expected


# ----------------------------------------------------------------------
# Variant classification
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ref", "alt", "vclass"),
    [
        ("G", "A", "snv"),
        ("CAA", "TGG", "mnv"),
        ("T", "TTTC", "insertion"),
        ("CGT", "C", "deletion"),
    ],
)
def test_variant_class(ref: str, alt: str, vclass: str) -> None:
    specs = parse_line(line("17", "1000", ".", ref, alt, ".", "PASS"))
    assert len(specs) == 1
    assert isinstance(specs[0], VariantSpec)
    assert specs[0].vclass == vclass


def test_span_covers_the_ref_allele() -> None:
    """A deletion's affected span is wider than POS, which drives lookup_span."""
    (spec,) = parse_line(line("3", "3129083", ".", "CGT", "C", ".", "PASS"))
    assert isinstance(spec, VariantSpec)
    assert spec.span == (3129083, 3129085)


def test_snv_span_is_a_single_base() -> None:
    (spec,) = parse_line(line("3", "3129083", ".", "C", "T", ".", "PASS"))
    assert isinstance(spec, VariantSpec)
    assert spec.span == (3129083, 3129083)


# ----------------------------------------------------------------------
# FILTER handling
# ----------------------------------------------------------------------


@pytest.mark.parametrize(("field", "expected"), [("PASS", True), (".", True), ("QSS_ref", False)])
def test_is_pass(field: str, expected: bool) -> None:
    """``.`` means the caller did not filter — not that the record failed."""
    (spec,) = parse_line(line("17", "1000", ".", "G", "A", ".", field))
    assert isinstance(spec, VariantSpec)
    assert spec.is_pass is expected


@pytest.mark.parametrize(
    "field",
    [
        "QSS_ref;LOHFAIL;MGRB;PPFAIL;NCFAIL",  # zcc10's 3rd most common (10,448 records)
        "QSS_ref;LOHFAIL;MGRB",
        "LOHPASS;MGRB;NCFAIL",  # note: contains "PASS" as a substring
        "MGRB",  # zcc10's single most common (69,171 records)
    ],
)
def test_multi_tag_filters_do_not_pass(field: str) -> None:
    """54% of zcc10's records carry several semicolon-joined FILTER tags.

    Per the VCF spec any value other than ``PASS``/``.`` means the record failed
    that filter, so none of these may pass. ``LOHPASS;MGRB;NCFAIL`` is the trap:
    a substring or prefix test would wrongly admit it.
    """
    (spec,) = parse_line(line("17", "1000", ".", "G", "A", ".", field))
    assert isinstance(spec, VariantSpec)
    assert spec.filter_field == field
    assert spec.is_pass is False


# ----------------------------------------------------------------------
# The cases that must not silently pass through
# ----------------------------------------------------------------------


def test_multi_allelic_splits_rather_than_dropping_alleles() -> None:
    """``G>A,T`` must never resolve as if only the first ALT existed."""
    specs = parse_line(line("19", "531848", ".", "G", "A,T", ".", "PASS"))
    assert [s.alt for s in specs] == ["A", "T"]
    assert all(isinstance(s, VariantSpec) and s.pos == 531848 for s in specs)


def test_breakend_alt_is_rejected_with_an_sv_specific_reason() -> None:
    """GRIDSS SV VCFs sit beside the SNV ones and will get pasted in by mistake."""
    (result,) = parse_line(line("17", "7674893", ".", "C", "[19:531848[T", ".", "PASS"))
    assert isinstance(result, Rejection)
    assert result.reason == SV_BREAKEND


@pytest.mark.parametrize("alt", ["<DEL>", "<DUP:TANDEM>", "]13:123]A"])
def test_symbolic_alts_are_rejected(alt: str) -> None:
    (result,) = parse_line(line("17", "1000", ".", "C", alt, ".", "PASS"))
    assert isinstance(result, Rejection)
    assert result.reason == SV_BREAKEND


def test_missing_alt_is_unsupported_not_a_hit() -> None:
    """Strelka emits ALT ``.`` for no-call sites — 6,649 of them in zcc10."""
    (result,) = parse_line(line("17", "1000", ".", "C", ".", ".", "PASS"))
    assert isinstance(result, Rejection)
    assert result.reason == UNSUPPORTED_ALT


def test_spanning_deletion_placeholder_is_unsupported() -> None:
    (result,) = parse_line(line("17", "1000", ".", "C", "*", ".", "PASS"))
    assert isinstance(result, Rejection)
    assert result.reason == UNSUPPORTED_ALT


def test_multi_allelic_rejects_only_the_bad_allele() -> None:
    """A mixed line yields one entry per ALT, so the good allele survives."""
    results = parse_line(line("17", "1000", ".", "C", "T,<DEL>", ".", "PASS"))
    assert isinstance(results[0], VariantSpec)
    assert isinstance(results[1], Rejection)
    assert results[1].reason == SV_BREAKEND


@pytest.mark.parametrize(
    "bad",
    [
        "17\t1000\t.\tC",  # too few fields
        "17\tnot_a_number\t.\tC\tT\t.\tPASS",
        "17\t0\t.\tC\tT\t.\tPASS",  # VCF POS is 1-based
        "17\t1000\t.\tZZZ\tT\t.\tPASS",  # REF is not DNA
    ],
)
def test_malformed_lines_are_rejected(bad: str) -> None:
    (result,) = parse_line(bad)
    assert isinstance(result, Rejection)
    assert result.reason == MALFORMED


def test_id_column_dot_becomes_empty() -> None:
    (spec,) = parse_line(line("17", "1000", ".", "G", "A", ".", "PASS"))
    assert isinstance(spec, VariantSpec)
    assert spec.variant_id == ""


def test_id_column_is_preserved_when_present() -> None:
    (spec,) = parse_line(line("17", "1000", "rs123", "G", "A", ".", "PASS"))
    assert isinstance(spec, VariantSpec)
    assert spec.variant_id == "rs123"


# ----------------------------------------------------------------------
# Full-width records: INFO + FORMAT + per-sample columns
#
# Real VCFs are 10+ columns wide. FILTER sits at index 6 and everything after it
# is ignored, so a mis-indexed field would still parse — it would just silently
# read INFO or a genotype as the filter. These pin the width.
# ----------------------------------------------------------------------


def spec_line(*fields: str) -> str:
    """A record in the shape the VCF 4.x spec example uses."""
    return "\t".join(fields)


def test_filter_is_read_past_info_format_and_samples() -> None:
    """Ten columns wide — FILTER must still come from field 7, not INFO or a GT."""
    (spec,) = parse_line(
        spec_line(
            "20",
            "14370",
            "rs6054257",
            "G",
            "A",
            "29",
            "PASS",
            "NS=3;DP=14;AF=0.5;DB;H2",
            "GT:GQ:DP:HQ",
            "0|0:48:1:51,51",
            "1|0:48:8:51,51",
            "1/1:43:5:.,.",
        )
    )
    assert isinstance(spec, VariantSpec)
    assert spec.is_pass is True
    assert (spec.chrom, spec.pos, spec.ref, spec.alt) == ("chr20", 14370, "G", "A")
    assert spec.variant_id == "rs6054257"


def test_non_pass_filter_is_read_past_the_sample_columns() -> None:
    """The mirror case: a full-width record whose FILTER must NOT read as passing."""
    (spec,) = parse_line(
        spec_line(
            "20",
            "17330",
            ".",
            "T",
            "A",
            "3",
            "q10",
            "NS=3;DP=11;AF=0.017",
            "GT:GQ:DP:HQ",
            "0|0:49:3:58,50",
            "0|1:3:5:65,3",
            "0/0:41:3:.,.",
        )
    )
    assert isinstance(spec, VariantSpec)
    assert spec.filter_field == "q10"
    assert spec.is_pass is False


def test_full_width_multi_allelic_splits_both_alts() -> None:
    (a, b) = parse_line(
        spec_line(
            "20",
            "1110696",
            "rs6040355",
            "A",
            "G,T",
            "67",
            "PASS",
            "NS=2;DP=10;AF=0.333,0.667;AA=T;DB",
            "GT:GQ:DP:HQ",
            "1|2:21:6:23,27",
            "2|1:2:0:18,2",
            "2/2:35:4:.,.",
        )
    )
    assert (a.alt, b.alt) == ("G", "T")
    assert a.pos == b.pos == 1110696


def test_full_width_no_variant_record_is_rejected() -> None:
    """``ALT = .`` means "no variant called here" — there is nothing to place."""
    (result,) = parse_line(
        spec_line(
            "20",
            "1230237",
            ".",
            "T",
            ".",
            "47",
            "PASS",
            "NS=3;DP=13;AA=T",
            "GT:GQ:DP:HQ",
            "0|0:54:.:56,60",
            "0|0:48:4:51,51",
            "0/0:61:2:.,.",
        )
    )
    assert isinstance(result, Rejection)
    assert result.reason == UNSUPPORTED_ALT


def test_full_width_multi_allelic_insertions() -> None:
    specs = parse_line(
        spec_line(
            "20",
            "1234567",
            "microsat1",
            "G",
            "GA,GAC",
            "50",
            "PASS",
            "NS=3;DP=9;AA=G;AN=6;AC=3,1",
            "GT:GQ:DP",
            "0/1:.:4",
            "0/2:17:2",
            "1/1:40:3",
        )
    )
    assert [s.vclass for s in specs] == ["insertion", "insertion"]
    # A 1-base REF occupies a single position regardless of ALT length.
    assert all(s.span == (1234567, 1234567) for s in specs)


def test_full_width_mixed_deletion_and_substitution() -> None:
    """``AC -> A,ATG``: a deletion and an equal-length rewrite on one line."""
    deletion, other = parse_line(
        spec_line("X", "10", "rsTest", "AC", "A,ATG", "10", "PASS", ".", "GT", "0", "0/1", "0|2")
    )
    assert (deletion.vclass, deletion.alt) == ("deletion", "A")
    assert (other.vclass, other.alt) == ("insertion", "ATG")
    # REF spans two bases, so an indel here can overlap an exon boundary.
    assert deletion.span == (10, 11)


def test_dot_filter_on_a_full_width_record_counts_as_passing() -> None:
    """An unfiltered caller writes ``.``; dropping those would empty the scan."""
    (spec,) = parse_line(
        spec_line(
            "19",
            "111",
            ".",
            "A",
            "C",
            "9.6",
            ".",
            ".",
            "GT:HQ",
            "0|0:10,10",
            "0|0:10,10",
            "0/1:3,3",
        )
    )
    assert spec.is_pass is True


def test_sample_columns_are_not_retained_anywhere_on_the_spec() -> None:
    """Genotypes are the most identifying part of a VCF and must not be carried.

    :class:`VariantSpec` is what the digest is built from, so if genotype text
    cannot reach a spec it cannot reach disk.
    """
    (spec,) = parse_line(
        spec_line(
            "20",
            "14370",
            "rs6054257",
            "G",
            "A",
            "29",
            "PASS",
            "NS=3;DP=14;AF=0.5;DB;H2",
            "GT:GQ:DP:HQ",
            "0|0:48:1:51,51",
            "1|0:48:8:51,51",
            "1/1:43:5:.,.",
        )
    )
    serialised = repr(spec)
    for leak in ("GT:GQ", "0|0", "1|0", "1/1", "48:1:51", "NS=3", "AF=0.5"):
        assert leak not in serialised, f"{leak!r} leaked into the parsed spec"
