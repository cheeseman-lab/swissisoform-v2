"""Invariant tests for the canonical-vs-alternative contract."""

from __future__ import annotations

import pytest

from swissisoform.contract import (
    NEAR_COGNATE_STARTS,
    START_LOST_ATG_NOTE,
    START_LOST_NEAR_COGNATE_NOTE,
    START_NOT_AN_INITIATOR_NOTE,
    START_STILL_NEAR_COGNATE_NOTE,
    START_STRENGTHENED_NOTE,
    ORFType,
    diff_region_rule,
    orf_type_from_ribotish,
    start_codon_effect,
)

_BASES = "ACGT"
_ALL_CODONS = [a + b + c for a in _BASES for b in _BASES for c in _BASES]


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


# ----------------------------------------------------------------------
# Start-codon rule (PR #29 gate 4)
# ----------------------------------------------------------------------


def test_the_near_cognate_set_is_exactly_atgs_neighbourhood():
    """The premise the whole asymmetry rests on, asserted rather than assumed.

    Because the set is ATG plus its nine single-base neighbours, a membership test
    on the mutated codon can never fire for an ATG start — which is why the rule
    has to read the direction of the change instead.
    """
    neighbours = {"ATG"[:i] + b + "ATG"[i + 1 :] for i in range(3) for b in _BASES}
    assert neighbours == set(NEAR_COGNATE_STARTS)


@pytest.mark.parametrize("alt", [c for c in _ALL_CODONS if c != "ATG"])
def test_every_substitution_away_from_atg_is_start_lost(alt):
    """All 63 of them, including the ones that land on another near-cognate."""
    override, note = start_codon_effect("ATG", alt)
    assert override == "start_lost"
    assert note == START_LOST_ATG_NOTE


def test_an_untouched_atg_is_not_start_lost():
    """A multi-base substitution can span codon 0 and leave the start intact."""
    assert start_codon_effect("ATG", "ATG") == (None, "")


@pytest.mark.parametrize("ref", sorted(NEAR_COGNATE_STARTS - {"ATG"}))
@pytest.mark.parametrize("alt", _ALL_CODONS)
def test_the_near_cognate_rule_is_total_and_directional(ref, alt):
    """Every near-cognate ref against every possible alt — no gaps, no surprises."""
    override, note = start_codon_effect(ref, alt)

    if alt == ref:
        assert (override, note) == (None, "")
    elif alt not in NEAR_COGNATE_STARTS:
        assert override == "start_lost"
        assert note == START_LOST_NEAR_COGNATE_NOTE
    elif alt == "ATG":
        # Gaining an ATG strengthens initiation; it is never start-loss.
        assert override is None
        assert note == START_STRENGTHENED_NOTE
    else:
        assert override is None
        assert note == START_STILL_NEAR_COGNATE_NOTE


def test_a_reference_that_cannot_initiate_says_so_instead_of_guessing():
    """Unreachable for a well-formed ORF; silence here would hide an annotation bug."""
    override, note = start_codon_effect("AAA", "AAG")
    assert override is None
    assert note == START_NOT_AN_INITIATOR_NOTE


def test_the_rule_is_case_insensitive():
    assert start_codon_effect("atg", "acg") == ("start_lost", START_LOST_ATG_NOTE)
