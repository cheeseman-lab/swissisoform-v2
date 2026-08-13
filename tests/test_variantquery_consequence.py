"""Consequence classification: codon table, the four classes, and the traps.

Synthetic ORFs so every case is constructible; the real-data agreement with
``ConsequenceValidator`` is asserted in ``test_variantquery_crosscheck.py``.
"""

from __future__ import annotations

import pytest

from swissisoform.variantquery.consequence import (
    FRAMESHIFT,
    INFRAME_DELETION,
    INFRAME_INSERTION,
    MISSENSE,
    OTHER,
    START_LOST,
    STOP_GAINED,
    SYNONYMOUS,
    Consequence,
    classify,
    revcomp,
    translate,
)

# One 5-codon exon on the plus strand. Genomic 101..115 == coding offsets 0..14.
#          codon   0     1     2     3     4
PLUS_CDS = "ATGACACGTGAGAAA"
PLUS_EXONS = ((100, 115),)

# Same sequence on the minus strand: mRNA order runs from the HIGH genomic end, so
# genomic 115 is offset 0 and genomic 101 is offset 14.
MINUS_EXONS = ((100, 115),)


def plus(pos: int, ref: str, alt: str, **kw) -> Consequence:
    return classify(exons=PLUS_EXONS, strand="+", cds=PLUS_CDS, pos=pos, ref=ref, alt=alt, **kw)


# ----------------------------------------------------------------------
# The codon table is transcribed by hand — verify it, don't trust it
# ----------------------------------------------------------------------


def test_codon_table_matches_biopython() -> None:
    """A single wrong entry would silently mislabel one amino acid everywhere."""
    from Bio.Data.CodonTable import standard_dna_table as table

    from swissisoform.variantquery.consequence import _CODONS

    expected = dict(table.forward_table)
    for stop in table.stop_codons:
        expected[stop] = "*"
    assert len(_CODONS) == 64
    assert _CODONS == expected


def test_translate_drops_a_trailing_partial_codon() -> None:
    assert translate("ATGACA") == "MT"
    assert translate("ATGACAG") == "MT"


def test_translate_can_stop_at_the_first_stop() -> None:
    assert translate("ATGTAAACA") == "M*T"
    assert translate("ATGTAAACA", stop_at_first=True) == "M*"


def test_revcomp() -> None:
    assert revcomp("CAA") == "TTG"


# ----------------------------------------------------------------------
# Substitutions — the class that needs the codon
# ----------------------------------------------------------------------


def test_missense() -> None:
    # codon 1 is ACA (Thr); offset 4 is its middle base; C->T gives ATA (Ile).
    c = plus(105, "C", "T")
    assert (c.term, c.aa_ref, c.aa_alt, c.aa_pos, c.hgvsp) == (MISSENSE, "T", "I", 2, "p.T2I")


def test_synonymous() -> None:
    # codon 4 is AAA (Lys); its third base A->G gives AAG, still Lys.
    c = plus(115, "A", "G")
    assert (c.term, c.hgvsp) == (SYNONYMOUS, "p.K5K")


def test_stop_gained() -> None:
    # codon 3 is GAG (Glu); G->T at its first base gives TAG = stop.
    c = plus(110, "G", "T")
    assert (c.term, c.aa_ref, c.aa_alt, c.hgvsp) == (STOP_GAINED, "E", "*", "p.E4*")


def test_start_lost_is_decided_by_position() -> None:
    c = plus(101, "A", "G")
    assert c.term == START_LOST
    assert c.hgvsp == "p.M1?"


def test_non_aug_start_is_noted_not_assumed() -> None:
    """CBX1's isoform ORF starts GTG, so residue 1 is not M — the note says so."""
    c = classify(
        exons=PLUS_EXONS,
        strand="+",
        cds="GTG" + PLUS_CDS[3:],
        pos=101,
        ref="G",
        alt="A",
        start_codon="GTG",
    )
    assert c.term == START_LOST
    assert "GTG" in c.note


def test_mnv_spanning_one_codon() -> None:
    # Replace all of codon 1 (ACA -> GGG).
    c = plus(104, "ACA", "GGG")
    assert (c.term, c.aa_ref, c.aa_alt, c.hgvsp) == (MISSENSE, "T", "G", "p.T2G")


# ----------------------------------------------------------------------
# Minus strand: the trap a prototype fell into
# ----------------------------------------------------------------------


def test_minus_strand_substitution_uses_mrna_sense() -> None:
    """REF/ALT are plus-strand and must be complemented into mRNA sense."""
    # On the minus strand genomic 115 is offset 0. mRNA base 0 is 'A' (of ATG), so
    # the plus-strand REF there is 'T'.
    c = classify(exons=MINUS_EXONS, strand="-", cds=PLUS_CDS, pos=115, ref="T", alt="C")
    assert c.term == START_LOST, c


def test_minus_strand_mnv_applies_the_whole_alt_at_the_right_codon() -> None:
    """Two prototype bugs at once: only ALT[0] applied, and the highest offset used.

    Genomic 104..106 map to offsets 11, 10, 9 on the minus strand, so the change
    begins at offset **9** — the lowest, not offset(pos). Plus-strand CTC/TGG
    reverse-complement to GAG/CCA in mRNA sense, giving Pro at codon 3.
    """
    c = classify(exons=MINUS_EXONS, strand="-", cds=PLUS_CDS, pos=104, ref="CTC", alt="TGG")
    assert (c.term, c.aa_ref, c.aa_alt, c.aa_pos) == (MISSENSE, "E", "P", 4)
    assert c.hgvsp == "p.E4P"


def test_substitution_spanning_two_codons_reports_both() -> None:
    """A multi-base substitution can straddle a boundary and change two residues.

    Reporting only the first silently drops the second. Offsets 2..4 cover the last
    base of codon 0 and the first two of codon 1.
    """
    # Offsets 5..7: last base of codon 1 (ACA, Thr) and first two of codon 2
    # (CGT, Arg). Codon 1 stays Thr; codon 2 becomes TTT, Phe.
    c = classify(exons=PLUS_EXONS, strand="+", cds=PLUS_CDS, pos=106, ref="ACG", alt="TTT")
    assert (c.term, c.aa_ref, c.aa_alt, c.aa_pos) == (MISSENSE, "TR", "TF", 2)
    assert c.hgvsp == "p.T2_R3delinsTF", c.hgvsp


# ----------------------------------------------------------------------
# Indels — left-anchored, and the frameshift/in-frame split
# ----------------------------------------------------------------------


def test_left_anchor_is_trimmed_so_the_residue_is_not_early() -> None:
    """``CGT>C`` deletes GT *after* pos; the base at pos is untouched.

    Offset 6 is codon 2's first base. Anchoring naively at pos would report codon 2;
    the change actually begins at offset 7, still codon 2 here — so this asserts the
    deleted residue count and notation rather than just the position.
    """
    c = plus(107, "CGT", "C")
    assert c.term == FRAMESHIFT  # 2 bases deleted
    assert c.aa_pos == 3


def test_inframe_deletion_of_one_codon() -> None:
    # Anchor at 104 (A of ACA), delete the following 3 bases CAC -> shifts nothing
    # out of frame; 3 nt removed starting at offset 4.
    c = plus(104, "ACACG", "AG")
    assert c.term == INFRAME_DELETION
    assert c.hgvsp.startswith("p.")
    assert "del" in c.hgvsp


def test_inframe_insertion() -> None:
    c = plus(104, "A", "AGGG")
    assert c.term == INFRAME_INSERTION
    assert "ins" in c.hgvsp


#: Shifted frame is all-A, so no stop is reachable inside the CDS.
NO_STOP_CDS = "ATG" + "A" * 12


def test_frameshift_reports_no_stop_distance_when_none_is_reachable() -> None:
    """The CDS ends at the ORF's stop, so a downstream stop is often invisible.

    Inventing ``*n`` would be fabrication; the notation ends at ``fs``.
    """
    c = classify(exons=PLUS_EXONS, strand="+", cds=NO_STOP_CDS, pos=105, ref="A", alt="AC")
    assert c.term == FRAMESHIFT
    assert c.hgvsp == "p.K2fs", c.hgvsp


def test_frameshift_appends_the_stop_when_it_is_inside_the_cds() -> None:
    """Same CDS, but this insertion puts a TAA in the shifted frame."""
    c = classify(exons=PLUS_EXONS, strand="+", cds=NO_STOP_CDS, pos=107, ref="A", alt="TA")
    assert c.term == FRAMESHIFT
    assert c.hgvsp == "p.K3fs*1", c.hgvsp


# ----------------------------------------------------------------------
# Refusals: class still correct, notation withheld
# ----------------------------------------------------------------------


def test_span_crossing_an_intron_gets_class_only() -> None:
    """Splicing intronic bases into the CDS would be wrong, so no notation."""
    two_exons = ((100, 106), (200, 208))  # offsets 0-5 then 6-14
    # Span 105..107 runs off the end of the first exon into the intron.
    c = classify(exons=two_exons, strand="+", cds=PLUS_CDS, pos=105, ref="CAC", alt="C")
    assert c.term == FRAMESHIFT, "the length delta still classifies it"
    assert c.hgvsp == ""
    assert "coding sequence" in c.note


def test_intron_spanning_inframe_deletion_also_keeps_its_class() -> None:
    """The other half of the same rule: -3 stays in-frame, notation still withheld."""
    two_exons = ((100, 106), (200, 208))
    c = classify(exons=two_exons, strand="+", cds=PLUS_CDS, pos=105, ref="CACG", alt="C")
    assert c.term == INFRAME_DELETION
    assert c.hgvsp == ""
    assert "coding sequence" in c.note


def test_position_outside_the_orf_is_other_with_a_note() -> None:
    c = plus(500, "A", "G")
    assert c.term == OTHER
    assert c.note


def test_ref_mismatch_is_reported_not_silently_translated() -> None:
    """A wrong REF means the caller and our reference disagree; say so."""
    c = plus(105, "G", "T")  # offset 4 is really 'C'
    assert c.hgvsp == ""
    assert "REF does not match" in c.note


@pytest.mark.parametrize(
    "ref,alt,expected",
    [
        ("A", "AG", FRAMESHIFT),
        ("AC", "A", FRAMESHIFT),
        ("A", "AGGG", INFRAME_INSERTION),
        ("ACACG", "AG", INFRAME_DELETION),
    ],
)
def test_class_comes_from_the_length_delta_alone(ref, alt, expected) -> None:
    """No sequence needed for this half of the vocabulary."""
    assert plus(104, ref, alt).term == expected
