"""Tests for the ORF-frame entry point of ConsequenceValidator.

The canonical-CDS path is exercised by test_clinical_validate.py; the codon
logic is shared via _analyze_variant, so these tests focus on the new
ORF-frame position map + coding-sequence machinery and basic round-trip.
"""

from __future__ import annotations

from swissisoform.clinical.validate import ConsequenceValidator
from swissisoform.contract import START_LOST_ATG_NOTE


class TestPositionMapFromOrf:
    def test_plus_strand_single_exon(self):
        v = ConsequenceValidator()
        # 9 nt → 3 codons; 1-based gpos 101..109 map to coding 0..8.
        pm = v.build_position_map_from_orf([(100, 109)], "+")
        assert pm == {101: 0, 102: 1, 103: 2, 104: 3, 105: 4, 106: 5, 107: 6, 108: 7, 109: 8}

    def test_minus_strand_single_exon(self):
        v = ConsequenceValidator()
        # mRNA-order for minus strand: gpos 109 is the first (coding_pos 0).
        pm = v.build_position_map_from_orf([(100, 109)], "-")
        assert pm[109] == 0
        assert pm[108] == 1
        assert pm[101] == 8
        assert len(pm) == 9

    def test_plus_strand_multi_exon_crosses_introns(self):
        v = ConsequenceValidator()
        # 3 + 6 = 9 nt; exon1 [100,103) gives 101,102,103, exon2 [200,206) gives 201..206.
        pm = v.build_position_map_from_orf([(100, 103), (200, 206)], "+")
        assert pm[101] == 0 and pm[103] == 2  # exon 1 spans coding 0-2
        assert pm[201] == 3 and pm[206] == 8  # exon 2 spans coding 3-8
        # Intron is unmapped:
        assert 150 not in pm

    def test_minus_strand_multi_exon_descends(self):
        v = ConsequenceValidator()
        # exons given in plus-strand ascending; mRNA-order on minus reverses them.
        pm = v.build_position_map_from_orf([(100, 103), (200, 206)], "-")
        # First coding pos is the highest gpos of the highest exon.
        assert pm[206] == 0
        assert pm[201] == 5
        assert pm[103] == 6  # next exon in mRNA order
        assert pm[101] == 8

    def test_empty_exons_returns_empty_map(self):
        v = ConsequenceValidator()
        assert v.build_position_map_from_orf([], "+") == {}

    def test_cache_reuses_result(self):
        v = ConsequenceValidator()
        a = v.build_position_map_from_orf([(100, 109)], "+", orf_key="x")
        b = v.build_position_map_from_orf([(100, 109)], "+", orf_key="x")
        assert a is b  # same dict instance from cache


class TestValidateAgainstOrf:
    """Indel / MNV / intronic don't need a genome FASTA — they only need the position map."""

    def test_intronic_when_not_in_orf(self):
        v = ConsequenceValidator()
        res = v.validate_variant_against_orf(
            orf_exons=[(100, 103), (200, 206)], strand="+", chrom="chr1",
            genomic_pos=150, ref="A", alt="G",
        )
        assert res["consequence"] == "intronic"
        assert res["protein_pos"] is None

    def test_frameshift_indel(self):
        v = ConsequenceValidator()
        res = v.validate_variant_against_orf(
            orf_exons=[(100, 109)], strand="+", chrom="chr1",
            genomic_pos=104, ref="A", alt="AT",
        )
        assert res["consequence"] == "frameshift_variant"
        assert res["protein_pos"] == 1  # coding_pos 3 // 3

    def test_inframe_deletion(self):
        v = ConsequenceValidator()
        # 3-nt deletion (multiple of 3) → inframe_deletion.
        res = v.validate_variant_against_orf(
            orf_exons=[(100, 109)], strand="+", chrom="chr1",
            genomic_pos=104, ref="ATGC", alt="A",
        )
        assert res["consequence"] == "inframe_deletion"

    def test_snv_without_genome_returns_unvalidated(self):
        # No genome_fasta → can't extract codon → validated=False but protein_pos set.
        v = ConsequenceValidator()
        res = v.validate_variant_against_orf(
            orf_exons=[(100, 109)], strand="+", chrom="chr1",
            genomic_pos=105, ref="A", alt="G",
        )
        assert res["protein_pos"] == 1
        assert res["validated"] is False
        assert res["consequence"] is None  # SNV without coding_seq → null consequence

    def test_empty_orf_returns_null(self):
        v = ConsequenceValidator()
        res = v.validate_variant_against_orf(
            orf_exons=[], strand="+", chrom="chr1",
            genomic_pos=100, ref="A", alt="G",
        )
        assert res["validated"] is False
        assert res["protein_pos"] is None


class TestValidateVariantsBatchAgainstOrf:
    def test_writes_isoform_fields_in_place(self):
        v = ConsequenceValidator()
        variants = [
            {"genomic_pos": 105, "ref": "A", "alt": "G"},        # SNV in ORF, no genome → no aa
            {"genomic_pos": 150, "ref": "A", "alt": "G"},        # intronic
            {"genomic_pos": 104, "ref": "A", "alt": "AT"},       # frameshift in ORF
        ]
        v.validate_variants_against_orf(
            variants, orf_exons=[(100, 109)], strand="+", chrom="chr1",
        )
        assert variants[0]["isoform_protein_pos"] == 1
        assert variants[0]["isoform_consequence"] is None  # SNV w/o genome
        assert variants[1]["isoform_consequence"] == "intronic"
        assert variants[1]["isoform_protein_pos"] is None
        assert variants[2]["isoform_consequence"] == "frameshift_variant"
        assert variants[2]["isoform_protein_pos"] == 1
        # Canonical-frame keys are NOT touched
        assert "protein_pos" not in variants[0]


class TestSpanClassification:
    """The codon walk over a variant's whole span, and the traps it has to survive.

    Ported from ``test_variantquery_consequence.py`` when the scan stopped carrying a
    second classifier: the cases are the website's, the implementation is now this
    one. Synthetic ORFs so every case is constructible; agreement with real exon
    geometry is asserted in ``test_variantquery_crosscheck.py``.
    """

    # One 5-codon exon. Genomic 101..115 == coding offsets 0..14 on the plus strand;
    # on the minus strand mRNA order runs from the HIGH end, so 115 is offset 0.
    #          codon   0     1     2     3     4
    CDS = "ATGACACGTGAGAAA"
    EXONS = [(100, 115)]

    def classify(self, pos, ref, alt, *, strand="+", cds=None, exons=None):
        return ConsequenceValidator().classify_against_orf(
            orf_exons=exons or self.EXONS,
            strand=strand,
            cds=self.CDS if cds is None else cds,
            genomic_pos=pos,
            ref=ref,
            alt=alt,
        )

    # -- substitutions -------------------------------------------------

    def test_missense(self):
        # Codon 1 is ACA (Thr); offset 4 is its middle base, C->T gives ATA (Ile).
        out = self.classify(105, "C", "T")
        assert (out["consequence"], out["aa_ref"], out["aa_alt"], out["protein_pos"]) == (
            "missense_variant",
            "T",
            "I",
            1,
        )

    def test_synonymous(self):
        # Codon 4 is AAA (Lys); its third base A->G gives AAG, still Lys.
        out = self.classify(115, "A", "G")
        assert out["consequence"] == "synonymous_variant"

    def test_stop_gained(self):
        # Codon 3 is GAG (Glu); G->T at its first base gives TAG = stop.
        out = self.classify(110, "G", "T")
        assert (out["consequence"], out["aa_ref"], out["aa_alt"]) == ("stop_gained", "E", "*")

    def test_mnv_within_one_codon(self):
        out = self.classify(104, "ACA", "GGG")
        assert (out["consequence"], out["aa_ref"], out["aa_alt"]) == (
            "missense_variant",
            "T",
            "G",
        )

    def test_substitution_spanning_two_codons_reports_both_residues(self):
        """Reporting only the first codon silently drops the second one's change.

        Offsets 5..7 cover the last base of codon 1 (ACA, Thr) and the first two of
        codon 2 (CGT, Arg): Thr survives, Arg becomes Phe.
        """
        out = self.classify(106, "ACG", "TTT")
        assert (out["consequence"], out["aa_ref"], out["aa_alt"], out["protein_pos"]) == (
            "missense_variant",
            "TR",
            "TF",
            1,
        )

    # -- the start codon -----------------------------------------------

    def test_destroying_an_atg_start_is_start_lost_in_the_orf_frame(self):
        """ATG->GTG replaces the annotated initiator with a weaker one.

        GTG still initiates, but far less efficiently, and it is not the start this
        ORF is annotated at — so the proteoform is lost or heavily reduced. The
        amino-acid answer (M->V) is true and beside the point; the direction of the
        change at codon 0 is what carries the meaning.
        """
        out = self.classify(101, "A", "G")
        assert out["consequence"] == "start_lost"
        assert out["note"] == START_LOST_ATG_NOTE
        # The residue is still reported — it is the classification that changes.
        assert (out["aa_ref"], out["aa_alt"]) == ("M", "V")

    def test_a_start_that_stops_initiating_is_start_lost(self):
        """CTG->CTA leaves the near-cognate set, ablating the ORF's start.

        Uses a CTG start because it is the real case: CDC34 and many alternative TIS
        initiate at near-cognates, and a third-base change there can abolish the
        start while leaving the residue identical — which no missense score expresses.
        """
        out = self.classify(103, "G", "A", cds="CTG" + self.CDS[3:])
        assert out["consequence"] == "start_lost"

    # -- minus strand ---------------------------------------------------

    def test_minus_strand_substitution_reads_in_mrna_sense(self):
        """REF/ALT are plus-strand and must be complemented to compare.

        ``aa_ref == "M"`` is what proves it: reading the plus strand literally would
        translate this codon as anything but Met. The offset-0 position also makes it
        a start-loss (ATG->GTG), which is incidental to what is under test here but
        is asserted so the two facts cannot drift apart.
        """
        # Genomic 115 is offset 0; mRNA base 0 is the A of ATG, so plus-strand REF is T.
        out = self.classify(115, "T", "C", strand="-")
        assert out["aa_ref"] == "M"
        assert out["codon_ref"], "the codon must have been read to translate it"
        assert out["consequence"] == "start_lost"

    def test_minus_strand_multibase_uses_the_lowest_offset_and_the_whole_allele(self):
        """Two traps at once: applying only ALT[0], and anchoring at offset(pos).

        Genomic 104..106 map to offsets 11, 10, 9, so the change begins at offset
        **9** — the lowest, not the one POS maps to. Plus-strand CTC/TGG
        reverse-complement to GAG/CCA in mRNA sense, giving Pro at codon 3.
        """
        out = self.classify(104, "CTC", "TGG", strand="-")
        assert (out["consequence"], out["aa_ref"], out["aa_alt"], out["protein_pos"]) == (
            "missense_variant",
            "E",
            "P",
            3,
        )

    # -- indels: class, position, and the residues gained or lost -------

    def test_indels_are_classified_with_amino_acids(self):
        """The class comes from the length delta; the residues from translation.

        Both matter: the class alone cannot say which residues an in-frame deletion
        removes, and a frameshift's first affected residue is what any downstream
        reader needs to locate it.
        """
        for ref, alt, expected in (
            ("A", "AG", "frameshift_variant"),
            ("ACACG", "AG", "inframe_deletion"),
            ("A", "AGGG", "inframe_insertion"),
        ):
            out = self.classify(104, ref, alt)
            assert out["consequence"] == expected
            # Something is named in every case — which side depends on the class.
            assert out["aa_ref"] or out["aa_alt"]
            assert out["note"]

    def test_an_inframe_deletion_names_the_residue_it_removes(self):
        """CDS is ATG ACA CGT GAG AAA = M T R E K; this deletes codon 1's ACA.

        Anchor G at offset 2, deleted ACA at offsets 3-5.
        """
        out = self.classify(103, "GACA", "G")
        assert out["consequence"] == "inframe_deletion"
        assert out["protein_pos"] == 1
        assert out["aa_ref"] == "T"
        assert out["aa_alt"] == ""

    def test_a_frameshift_reports_the_stop_distance_only_when_one_is_reachable(self):
        out = self.classify(104, "A", "AG")
        assert out["consequence"] == "frameshift_variant"
        # The new frame usually runs past the annotated stop into 3'UTR the CDS does
        # not contain, so the note must not invent a distance it cannot see.
        assert "premature stop" in out["note"] or "no premature stop" in out["note"]

    def test_the_anchor_is_not_the_residue(self):
        """``protein_pos`` follows HGVS: the first *changed* base, not the padding one.

        The case has to straddle a codon boundary or it proves nothing. ``GAC>G`` at
        103 leaves the G at offset 2 (codon 0) untouched and deletes offsets 3-4,
        which are in codon **1**. Anchoring on the padding base reports codon 0 and
        disagrees by one with the ClinVar string displayed beside it.

        (The test this replaces used ``CGT>C`` at 107, where the anchor and the first
        changed base are both in codon 2 — so it asserted the anchor convention
        without ever exercising it.)

        Both arms stay consistent either way — they share ``_analyze_variant`` — so
        the choice is between agreeing with HGVS and agreeing with the padding base.
        """
        out = self.classify(103, "GAC", "G")
        assert out["consequence"] == "frameshift_variant"
        assert out["protein_pos"] == 1  # codon of offset 3, not of the anchor at 2

    def test_an_indel_inside_one_codon_is_unaffected_by_the_anchor_rule(self):
        """Where anchor and first changed base share a codon, both readings agree."""
        out = self.classify(101, "ATG", "A")
        assert out["protein_pos"] == 0

    def test_a_wrong_ref_indel_is_a_reference_mismatch(self):
        """Impossible before: indels returned before the reference was ever read.

        The bad base here is the *padding* base, which trimming removes before the
        changed-base check — so the whole supplied REF has to be verified, not just
        the part that changes.
        """
        out = self.classify(103, "TACA", "T")  # offset 2 is really G
        assert out["consequence"] == "reference_mismatch"
        assert not out["validated"]

    def test_a_deletion_running_off_the_exon_end_keeps_its_class_and_says_so(self):
        """Anchor-only mapping never noticed this; the span walk does.

        The class survives — a length delta needs no sequence — but the residues do
        not, because splicing there would translate bases outside the ORF. Before
        this the row looked fully resolved, which is the part that was wrong.
        """
        out = self.classify(114, "AAT", "A")
        assert out["consequence"] == "frameshift_variant"
        assert (out["aa_ref"], out["aa_alt"]) == (None, None)
        assert "leaves the ORF" in out["note"]

    # -- indels on the minus strand: the quadrant nothing ever covered ---

    def test_minus_strand_indel_numbers_from_the_first_translated_base(self):
        """The case that makes "trim the anchor and advance" wrong.

        On the minus strand mRNA order runs against genomic order, so the padding
        base — always the lowest genomic coordinate, because VCF writes REF on the
        plus strand — is the span's *last* base in translation order.

        Genomic 112 is offset 3 (codon 1); the deleted 113-114 are offsets 2 and 1,
        so the first translated changed base is offset 1, in codon **0**. Anchoring
        on the padding base would report codon 1, and advancing past it would move
        further the wrong way still.
        """
        out = self.classify(112, "TCA", "T", strand="-")
        assert out["consequence"] == "frameshift_variant"
        assert out["protein_pos"] == 0
        assert out["aa_ref"] == "M"

    def test_minus_strand_inframe_deletion_names_the_residue(self):
        """Deletes offsets 2,1,0 — the whole initiator codon."""
        out = self.classify(112, "TCAT", "T", strand="-")
        assert out["consequence"] == "inframe_deletion"
        assert out["protein_pos"] == 0
        assert (out["aa_ref"], out["aa_alt"]) == ("M", "")

    def test_minus_strand_insertion_is_placed_in_mrna_order(self):
        """An insertion lands *between* two mRNA positions, and which two is strandy.

        VCF inserts after genomic 112 in plus-strand terms. Reading the minus strand,
        mRNA walks 113 (offset 2) then the insert then 112 (offset 3), so the new
        material becomes offset 3 — codon 1. Mapping the genomic successor alone
        would give offset 2 and land a codon early, which is what
        ``min(offset(anchor), offset(anchor + 1)) + 1`` exists to avoid.
        """
        out = self.classify(112, "T", "TAA", strand="-")
        assert out["consequence"] == "frameshift_variant"
        assert out["protein_pos"] == 1

    # -- refusals -------------------------------------------------------

    def test_a_substitution_spanning_an_intron_is_not_translated(self):
        """Splicing intronic bases into the codons would invent a result.

        Reported as ``intronic`` — part of the span genuinely is. ``mnv`` names the
        variant *class* (``spec._classify``) and has no business in this field; it
        sat here only as a placeholder while the multi-codon walk was unimplemented.
        """
        two_exons = [(100, 106), (200, 208)]  # offsets 0-5, then 6-14
        out = self.classify(105, "CAC", "TGA", exons=two_exons)
        assert out["consequence"] == "intronic"
        assert out["aa_ref"] is None
        # The anchor itself is coding, so its residue stands.
        assert out["protein_pos"] == 1

    def test_position_outside_the_orf_is_intronic(self):
        out = self.classify(500, "A", "G")
        assert out["consequence"] == "intronic"
        assert out["protein_pos"] is None

    def test_ref_mismatch_is_reported_not_silently_translated(self):
        """A wrong REF means the caller and our reference disagree; say so."""
        out = self.classify(105, "G", "T")  # offset 4 is really 'C'
        assert out["consequence"] == "reference_mismatch"
        assert not out["validated"]
