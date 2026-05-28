"""Tests for the ORF-frame entry point of ConsequenceValidator.

The canonical-CDS path is exercised by test_clinical_validate.py; the codon
logic is shared via _analyze_variant, so these tests focus on the new
ORF-frame position map + coding-sequence machinery and basic round-trip.
"""

from __future__ import annotations

from swissisoform.clinical.validate import ConsequenceValidator


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
