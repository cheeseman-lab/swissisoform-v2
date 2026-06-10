"""Invariant tests for the canonical-vs-alternative contract."""

from __future__ import annotations

import pytest

from swissisoform.contract import (
    ORFType,
    diff_region_rule,
    orf_type_from_ribotish,
)


@pytest.mark.parametrize(
    "tis_type, expected",
    [
        ("Annotated", ORFType.ANNOTATED),
        ("Annotated:Known", ORFType.ANNOTATED),
        ("Truncated", ORFType.TRUNCATED),
        ("Truncated:CDSFrameOverlap", ORFType.TRUNCATED),
        ("Extended", ORFType.EXTENDED),
        ("Extended:CDSFrameOverlap", ORFType.EXTENDED),
        ("5'UTR", ORFType.UORF),
        ("5'UTR:Known", ORFType.UORF),
        ("5'UTR:CDSFrameOverlap", ORFType.UOORF),
        ("3'UTR", ORFType.THREE_UTR_ORF),
        ("3'UTR:Known", ORFType.THREE_UTR_ORF),
        ("Internal", ORFType.INTERNAL_OUT_OF_FRAME),
        ("Internal:CDSFrameOverlap", ORFType.INTERNAL_OUT_OF_FRAME),
        ("Novel", ORFType.ALT_ORF),
        ("Novel:CDSFrameOverlap", ORFType.UOORF),
        ("uORF", ORFType.ALT_ORF),  # unknown prefix falls through to ALT_ORF
    ],
)
def test_orf_type_from_ribotish_is_total(tis_type: str, expected: ORFType) -> None:
    """Every known Ribo-TISH TisType prefix maps to an ORFType (no crash)."""
    assert orf_type_from_ribotish(tis_type) is expected


def test_truncation_lives_in_canonical_space() -> None:
    """Truncations carry their unique region in canonical coordinates."""
    rule = diff_region_rule(ORFType.TRUNCATED)
    assert rule.space == "canonical"
    assert rule.whole_isoform is False


def test_extension_lives_in_isoform_space() -> None:
    """Extensions carry their unique region in isoform coordinates."""
    rule = diff_region_rule(ORFType.EXTENDED)
    assert rule.space == "isoform"
    assert rule.whole_isoform is False


def test_uorf_is_whole_isoform() -> None:
    """uORFs are entirely unique — no shared region."""
    rule = diff_region_rule(ORFType.UORF)
    assert rule.space == "isoform"
    assert rule.whole_isoform is True


def test_annotated_has_no_diff_region() -> None:
    """Annotated TIS have no differential region."""
    rule = diff_region_rule(ORFType.ANNOTATED)
    assert rule.space == "none"
    assert rule.whole_isoform is False


def test_every_orf_type_has_a_rule() -> None:
    """diff_region_rule is total over ORFType — no KeyError / missing case."""
    for orf_type in ORFType:
        rule = diff_region_rule(orf_type)
        assert rule.space in ("isoform", "canonical", "none")
        assert isinstance(rule.whole_isoform, bool)
