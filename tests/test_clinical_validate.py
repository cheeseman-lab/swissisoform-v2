"""Tests for the variant consequence validation system.

Uses synthetic CDS data and injected coding sequences — no real genome
files are required.
"""

from __future__ import annotations

import pandas as pd

from swissisoform.clinical.validate import ConsequenceValidator
from swissisoform.contract import (
    START_LOST_ATG_NOTE,
    START_STILL_NEAR_COGNATE_NOTE,
    START_STRENGTHENED_NOTE,
)

# ---------------------------------------------------------------------------
# Synthetic CDS data
# ---------------------------------------------------------------------------

# Simple +strand gene: 2 CDS exons
# Exon 1: positions 100-108 (9 bp = 3 codons)
# Exon 2: positions 200-208 (9 bp = 3 codons)
# Total: 18 bp = 6 codons
CDS_DF = pd.DataFrame(
    [
        {
            "chromosome": "chr1",
            "start": 100,
            "end": 108,
            "strand": "+",
            "gene_id": "GENE1",
            "transcript_id": "TX1",
            "feature_type": "CDS",
        },
        {
            "chromosome": "chr1",
            "start": 200,
            "end": 208,
            "strand": "+",
            "gene_id": "GENE1",
            "transcript_id": "TX1",
            "feature_type": "CDS",
        },
        {
            "chromosome": "chr1",
            "start": 100,
            "end": 102,
            "strand": "+",
            "gene_id": "GENE1",
            "transcript_id": "TX1",
            "feature_type": "start_codon",
        },
    ]
)

# Minus strand gene: single CDS exon, positions 500-508
CDS_DF_MINUS = pd.DataFrame(
    [
        {
            "chromosome": "chr2",
            "start": 500,
            "end": 508,
            "strand": "-",
            "gene_id": "GENE2",
            "transcript_id": "TX2",
            "feature_type": "CDS",
        },
        {
            "chromosome": "chr2",
            "start": 506,
            "end": 508,
            "strand": "-",
            "gene_id": "GENE2",
            "transcript_id": "TX2",
            "feature_type": "start_codon",
        },
    ]
)


# ---------------------------------------------------------------------------
# Position map tests
# ---------------------------------------------------------------------------


class TestBuildPositionMap:
    """Tests for ConsequenceValidator.build_position_map."""

    def test_plus_strand(self):
        """Plus strand: positions 100-108 -> coding 0-8, 200-208 -> coding 9-17."""
        validator = ConsequenceValidator(cds_df=CDS_DF)
        pos_map = validator.build_position_map("TX1")

        # Exon 1
        assert pos_map[100] == 0
        assert pos_map[101] == 1
        assert pos_map[108] == 8

        # Exon 2
        assert pos_map[200] == 9
        assert pos_map[201] == 10
        assert pos_map[208] == 17

        # Total positions
        assert len(pos_map) == 18

    def test_minus_strand(self):
        """Minus strand: positions map right-to-left from canonical start."""
        validator = ConsequenceValidator(cds_df=CDS_DF_MINUS)
        pos_map = validator.build_position_map("TX2")

        # Minus strand with start_codon end=508: highest position maps first
        assert pos_map[508] == 0
        assert pos_map[507] == 1
        assert pos_map[500] == 8
        assert len(pos_map) == 9

    def test_empty_cds(self):
        """No CDS data -> empty map."""
        validator = ConsequenceValidator(
            cds_df=pd.DataFrame(
                columns=[
                    "chromosome",
                    "start",
                    "end",
                    "strand",
                    "gene_id",
                    "transcript_id",
                    "feature_type",
                ]
            )
        )
        assert validator.build_position_map("MISSING") == {}

    def test_none_cds(self):
        """None CDS DataFrame -> empty map."""
        validator = ConsequenceValidator(cds_df=None)
        assert validator.build_position_map("TX1") == {}

    def test_cached(self):
        """Second call returns cached result (same object)."""
        validator = ConsequenceValidator(cds_df=CDS_DF)
        map1 = validator.build_position_map("TX1")
        map2 = validator.build_position_map("TX1")
        assert map1 is map2


# ---------------------------------------------------------------------------
# Intronic / position-not-in-map test
# ---------------------------------------------------------------------------


class TestValidateIntronic:
    """Tests for positions that fall between exons."""

    def test_intronic_position(self):
        """Position 150 is between exon 1 (100-108) and exon 2 (200-208)."""
        validator = ConsequenceValidator(cds_df=CDS_DF)
        result = validator.validate_variant("TX1", 150, "A", "G")
        assert result["consequence"] == "intronic"
        assert result["validated"] is True
        assert result["protein_pos"] is None


# ---------------------------------------------------------------------------
# Indel tests
# ---------------------------------------------------------------------------


class TestValidateIndels:
    """Tests for insertion and deletion classification."""

    def test_frameshift_insertion(self):
        """ref='A', alt='AT' (1bp insertion) -> frameshift."""
        validator = ConsequenceValidator(cds_df=CDS_DF)
        result = validator.validate_variant("TX1", 100, "A", "AT")
        assert result["consequence"] == "frameshift_variant"
        assert result["protein_pos"] == 0  # coding_pos=0, 0//3=0
        assert result["validated"] is True

    def test_frameshift_deletion(self):
        """ref='AT', alt='A' (1bp deletion) -> frameshift."""
        validator = ConsequenceValidator(cds_df=CDS_DF)
        result = validator.validate_variant("TX1", 100, "AT", "A")
        assert result["consequence"] == "frameshift_variant"
        assert result["validated"] is True

    def test_inframe_insertion(self):
        """ref='A', alt='AAAA' (3bp insertion) -> inframe_insertion."""
        validator = ConsequenceValidator(cds_df=CDS_DF)
        result = validator.validate_variant("TX1", 100, "A", "AAAA")
        assert result["consequence"] == "inframe_insertion"
        assert result["validated"] is True

    def test_inframe_deletion(self):
        """ref='AAAA', alt='A' (3bp deletion) -> inframe_deletion."""
        validator = ConsequenceValidator(cds_df=CDS_DF)
        result = validator.validate_variant("TX1", 100, "AAAA", "A")
        assert result["consequence"] == "inframe_deletion"
        assert result["validated"] is True


# ---------------------------------------------------------------------------
# SNV tests (no genome — validated=False)
# ---------------------------------------------------------------------------


class TestValidateSNVNoGenome:
    """SNV validation without genome FASTA."""

    def test_snv_without_genome(self):
        """Without genome FASTA, SNV returns validated=False."""
        validator = ConsequenceValidator(cds_df=CDS_DF)
        result = validator.validate_variant("TX1", 100, "A", "G")
        assert result["validated"] is False
        assert result["protein_pos"] == 0


# ---------------------------------------------------------------------------
# SNV tests (with injected coding sequence)
# ---------------------------------------------------------------------------


class TestValidateSNVWithSequence:
    """SNV classification with manually injected coding sequences."""

    def _make_validator(self) -> ConsequenceValidator:
        """Create a validator with position map and injected coding sequence.

        Coding sequence: ATG GCC TAA GGG AAA TTT (18 bp = 6 codons)
        Codon 0: ATG (Met), Codon 1: GCC (Ala), Codon 2: TAA (Stop)
        Codon 3: GGG (Gly), Codon 4: AAA (Lys), Codon 5: TTT (Phe)
        """
        validator = ConsequenceValidator(cds_df=CDS_DF)
        validator.build_position_map("TX1")
        validator._coding_seq_cache["TX1"] = "ATGGCCTAAGGGAAATTT"
        return validator

    def test_missense(self):
        """GCC -> TCC at codon 1: Ala -> Ser = missense."""
        validator = self._make_validator()
        # Position 103 = coding_pos 3, codon 1 (GCC), offset 0
        result = validator.validate_variant("TX1", 103, "G", "T")
        assert result["consequence"] == "missense_variant"
        assert result["protein_pos"] == 1
        assert result["aa_ref"] == "A"  # Ala
        assert result["aa_alt"] == "S"  # Ser
        assert result["codon_ref"] == "GCC"
        assert result["codon_alt"] == "TCC"
        assert result["validated"] is True

    def _make_near_cognate_validator(self) -> ConsequenceValidator:
        """Same ORF but a CTG start — an alt-TIS isoform's near-cognate initiator."""
        validator = ConsequenceValidator(cds_df=CDS_DF)
        validator.build_position_map("TX1")
        validator._coding_seq_cache["TX1"] = "CTGGCCTAAGGGAAATTT"
        return validator

    def test_start_codon_ablated_is_start_lost_even_when_synonymous(self):
        """CTG -> CTA at codon 0: still Leu, but CTA cannot initiate.

        The whole point of deciding codon 0 on codon membership: a third-base
        change translates to the same residue while ablating the start, so the
        amino-acid classification (synonymous) is the opposite of the truth.
        """
        validator = self._make_near_cognate_validator()
        # Position 102 = coding_pos 2, codon 0 (CTG), offset 2.
        result = validator.validate_variant("TX1", 102, "G", "A")
        assert result["codon_ref"] == "CTG"
        assert result["codon_alt"] == "CTA"
        assert result["aa_ref"] == result["aa_alt"] == "L"  # would have been synonymous
        assert result["consequence"] == "start_lost"

    def test_start_codon_retained_keeps_the_literal_consequence(self):
        """CTG -> GTG at codon 0: still a near-cognate start, so not start_lost."""
        validator = self._make_near_cognate_validator()
        result = validator.validate_variant("TX1", 100, "C", "G")
        assert result["codon_alt"] == "GTG"
        assert result["consequence"] == "missense_variant"  # Leu -> Val
        # A lateral move between weak starts — the term alone would not say so.
        assert result["note"] == START_STILL_NEAR_COGNATE_NOTE

    def test_an_snv_that_destroys_an_atg_start_is_start_lost(self):
        """ATG -> ACG ablates the annotated initiator, so it is start-loss.

        This is the case a membership test cannot see. NEAR_COGNATE_STARTS is exactly
        ATG plus its nine single-base neighbours, so "is the mutated codon still in
        the set?" is always true for an ATG start — which labelled every classic
        pathogenic p.Met1? variant missense and kept it out of the LoF gate. Losing an
        ATG and gaining one are different events, so the rule reads the direction.
        """
        validator = self._make_validator()  # ATG start
        result = validator.validate_variant("TX1", 101, "T", "C")
        assert result["codon_alt"] == "ACG"
        assert result["consequence"] == "start_lost"
        assert result["note"] == START_LOST_ATG_NOTE

    def test_a_near_cognate_start_upgraded_to_atg_is_not_start_lost(self):
        """CTG -> ATG makes initiation *stronger*; calling it start-loss is backwards."""
        validator = self._make_near_cognate_validator()
        result = validator.validate_variant("TX1", 100, "C", "A")
        assert result["codon_alt"] == "ATG"
        assert result["consequence"] != "start_lost"
        assert result["note"] == START_STRENGTHENED_NOTE

    def test_start_lost_only_applies_to_codon_zero(self):
        """A downstream codon that happens to leave the set is untouched."""
        validator = self._make_validator()
        result = validator.validate_variant("TX1", 103, "G", "T")  # codon 1
        assert result["protein_pos"] == 1
        assert result["consequence"] == "missense_variant"

    def test_synonymous(self):
        """GCC -> GCT at codon 1: Ala -> Ala = synonymous."""
        validator = self._make_validator()
        # Position 105 = coding_pos 5, codon 1 (GCC), offset 2
        result = validator.validate_variant("TX1", 105, "C", "T")
        assert result["consequence"] == "synonymous_variant"
        assert result["protein_pos"] == 1
        assert result["aa_ref"] == "A"
        assert result["aa_alt"] == "A"
        assert result["validated"] is True

    def test_stop_gained(self):
        """TGG -> TAG: Trp -> Stop = stop_gained."""
        validator = ConsequenceValidator(cds_df=CDS_DF)
        validator.build_position_map("TX1")
        # Inject sequence where codon 1 = TGG (Trp)
        validator._coding_seq_cache["TX1"] = "ATGTGGAAAGGGAAATTT"
        # Position 104 = coding_pos 4, codon 1 (TGG), offset 1
        # TGG -> TAG = Stop
        result = validator.validate_variant("TX1", 104, "G", "A")
        assert result["consequence"] == "stop_gained"
        assert result["protein_pos"] == 1
        assert result["validated"] is True

    def test_stop_lost(self):
        """TAA -> TCA: Stop -> Ser = stop_lost."""
        validator = self._make_validator()
        # Codon 2 = TAA (Stop), positions 106-108
        # Position 107 = coding_pos 7, codon 2 (TAA), offset 1
        # TAA -> TCA = Ser
        result = validator.validate_variant("TX1", 107, "A", "C")
        assert result["consequence"] == "stop_lost"
        assert result["protein_pos"] == 2
        assert result["aa_ref"] == "*"
        assert result["aa_alt"] == "S"
        assert result["validated"] is True

    def test_reference_mismatch(self):
        """When ref allele doesn't match coding sequence -> reference_mismatch."""
        validator = self._make_validator()
        # Position 100 = coding_pos 0, codon 0 (ATG), offset 0 = 'A'
        # Claim ref is 'C' but coding seq has 'A'
        result = validator.validate_variant("TX1", 100, "C", "T")
        assert result["consequence"] == "reference_mismatch"
        assert result["validated"] is False


# ---------------------------------------------------------------------------
# Protein position calculation
# ---------------------------------------------------------------------------


class TestProteinPosition:
    """Tests for protein position derivation from coding position."""

    def test_protein_pos_from_coding_pos(self):
        """coding_pos=9 (first position of exon 2) -> protein_pos=3."""
        validator = ConsequenceValidator(cds_df=CDS_DF)
        validator.build_position_map("TX1")
        validator._coding_seq_cache["TX1"] = "ATGGCCTAAGGGAAATTT"
        # Position 200 = coding_pos 9, protein_pos = 9//3 = 3
        # Codon 3 = GGG (Gly), offset 0
        result = validator.validate_variant("TX1", 200, "G", "A")
        assert result["protein_pos"] == 3


# ---------------------------------------------------------------------------
# Batch validation
# ---------------------------------------------------------------------------


class TestValidateVariantsBatch:
    """Tests for the batch validate_variants method."""

    def test_batch_updates_variants(self):
        """Batch method updates variant dicts with validated consequences."""
        validator = ConsequenceValidator(cds_df=CDS_DF)
        validator.build_position_map("TX1")
        validator._coding_seq_cache["TX1"] = "ATGGCCTAAGGGAAATTT"

        variants = [
            {
                "variant_id": "v1",
                "genomic_pos": 103,
                "ref": "G",
                "alt": "T",
                "consequence": "unknown",
                "protein_pos": None,
            },
            {
                "variant_id": "v2",
                "genomic_pos": 150,
                "ref": "A",
                "alt": "G",
                "consequence": "unknown",
                "protein_pos": None,
            },
        ]

        result = validator.validate_variants(variants, "TX1")

        # First variant: missense
        assert result[0]["consequence"] == "missense_variant"
        assert result[0]["protein_pos"] == 1

        # Second variant: intronic (no protein_pos update since it's None)
        assert result[1]["consequence"] == "intronic"

    def test_batch_skips_incomplete_variants(self):
        """Variants missing genomic_pos/ref/alt are skipped."""
        validator = ConsequenceValidator(cds_df=CDS_DF)
        variants = [{"variant_id": "v1", "consequence": "unknown"}]
        result = validator.validate_variants(variants, "TX1")
        assert result[0]["consequence"] == "unknown"  # unchanged

    def test_batch_returns_same_list(self):
        """Batch method returns the same list object (mutated in place)."""
        validator = ConsequenceValidator(cds_df=CDS_DF)
        variants = [{"genomic_pos": 150, "ref": "A", "alt": "G", "consequence": "x"}]
        result = validator.validate_variants(variants, "TX1")
        assert result is variants


# ---------------------------------------------------------------------------
# GTF loader test
# ---------------------------------------------------------------------------


class TestLoadCdsFeatures:
    """Tests for the load_cds_features GTF reader."""

    def test_load_cds_features(self, tmp_path):
        """Parses CDS and start_codon from a minimal GTF."""
        gtf_content = (
            "##description: test\n"
            "chr1\tHAVANA\tgene\t100\t500\t.\t+\t.\t"
            'gene_id "G1"; transcript_id "T1";\n'
            "chr1\tHAVANA\tCDS\t100\t200\t.\t+\t0\t"
            'gene_id "G1"; transcript_id "T1";\n'
            "chr1\tHAVANA\tstart_codon\t100\t102\t.\t+\t0\t"
            'gene_id "G1"; transcript_id "T1";\n'
            "chr1\tHAVANA\texon\t100\t500\t.\t+\t.\t"
            'gene_id "G1"; transcript_id "T1";\n'
        )
        gtf_file = tmp_path / "test.gtf"
        gtf_file.write_text(gtf_content)

        from swissisoform.io.gtf import load_cds_features

        df = load_cds_features(str(gtf_file))

        assert len(df) == 2  # CDS + start_codon, not gene or exon
        assert set(df["feature_type"]) == {"CDS", "start_codon"}
        assert df.iloc[0]["chromosome"] == "chr1"
        assert df.iloc[0]["gene_id"] == "G1"
        assert df.iloc[0]["transcript_id"] == "T1"

    def test_load_cds_features_empty(self, tmp_path):
        """Empty GTF returns empty DataFrame with correct columns."""
        gtf_file = tmp_path / "empty.gtf"
        gtf_file.write_text("## empty\n")

        from swissisoform.io.gtf import load_cds_features

        df = load_cds_features(str(gtf_file))
        assert len(df) == 0
        assert "chromosome" in df.columns
        assert "transcript_id" in df.columns
