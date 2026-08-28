"""The sequence-free fallback — the only classification left in ``variantquery``.

Everything a coding sequence can answer now goes through
``ConsequenceValidator.classify_against_orf``; its cases, including the traps this
file used to own (minus-strand alleles, codon-straddling substitutions, REF
mismatch, the start-codon rule), live in ``test_clinical_validate_orf.py``.

What remains here is the path that has no pipeline counterpart: an index built with
``build_orf_index.py --no-cds`` carries no sequence, so a variant resolved against it
can only be classified by length.
"""

from __future__ import annotations

import pytest

from swissisoform.variantquery.consequence import (
    FRAMESHIFT,
    INFRAME_DELETION,
    INFRAME_INSERTION,
    OTHER,
    classify_without_sequence,
)


@pytest.mark.parametrize(
    "ref,alt,expected",
    [
        ("A", "AG", FRAMESHIFT),
        ("AC", "A", FRAMESHIFT),
        ("A", "AGGG", INFRAME_INSERTION),
        ("ACACG", "AG", INFRAME_DELETION),
        ("ATG", "A", FRAMESHIFT),
    ],
)
def test_class_comes_from_the_length_delta_alone(ref, alt, expected) -> None:
    """Frameshift vs in-frame needs no sequence, and matches the classifier's rule."""
    term, _note = classify_without_sequence(ref, alt)
    assert term == expected


def test_the_vcf_anchor_is_stripped_before_measuring_the_delta() -> None:
    """``CGT>C`` deletes two bases, not three: the leading C is padding.

    Measuring the raw alleles would call this a 3-to-1 substitution and get the
    frame arithmetic wrong.
    """
    assert classify_without_sequence("CGT", "C")[0] == FRAMESHIFT  # 2 bases deleted
    assert classify_without_sequence("CGTA", "C")[0] == INFRAME_DELETION  # 3 deleted


def test_a_substitution_is_refused_rather_than_guessed() -> None:
    """Missense, synonymous and stop_gained are indistinguishable without the codon.

    Guessing "missense" for every SNV would mislabel the synonymous ones — six of the
    fixture's hits — and, worse, silently downgrade a nonsense variant.
    """
    term, note = classify_without_sequence("G", "A")
    assert term == OTHER
    assert "codon" in note


def test_every_refusal_and_class_carries_a_note() -> None:
    """A bare term from this path would read as a real call rather than a fallback."""
    for ref, alt in (("A", "AG"), ("A", "AGGG"), ("ACACG", "AG"), ("G", "A")):
        _term, note = classify_without_sequence(ref, alt)
        assert note


# ----------------------------------------------------------------------
# The classifier's own note reaching a hit record (PR #29 gate 4)
# ----------------------------------------------------------------------


class TestHitNotes:
    """``_hit_fields`` turns a classifier result into the four fields a hit records.

    The note matters most where the term alone misleads: a near-cognate start
    upgraded to ATG classifies as ordinary missense, and without the note the page
    shows nothing to say the start got *stronger* rather than broken.
    """

    def test_the_classifier_note_wins_over_the_term_table(self):
        from swissisoform.contract import START_STRENGTHENED_NOTE
        from swissisoform.variantquery.scan import _hit_fields

        term, aa_ref, aa_alt, note = _hit_fields(
            {
                "consequence": "missense_variant",
                "aa_ref": "L",
                "aa_alt": "M",
                "note": START_STRENGTHENED_NOTE,
            }
        )
        assert (term, aa_ref, aa_alt) == ("missense_variant", "L", "M")
        assert note == START_STRENGTHENED_NOTE

    def test_the_term_table_still_covers_notes_the_classifier_does_not_carry(self):
        from swissisoform.variantquery.scan import _hit_fields

        _term, _aa_ref, _aa_alt, note = _hit_fields({"consequence": "intronic", "note": ""})
        assert note == "not inside this ORF's coding sequence"

    def test_an_unclassifiable_result_still_explains_itself(self):
        from swissisoform.variantquery.scan import _hit_fields

        term, _aa_ref, _aa_alt, note = _hit_fields({"consequence": None})
        assert term == "other"
        assert "could not be classified" in note
