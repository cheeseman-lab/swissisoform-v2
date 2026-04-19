"""Tests for the massspec annotation module."""

from __future__ import annotations

import copy

from swissisoform.config import PipelineConfig
from swissisoform.models import TranslationInitiationSite, ORFType
from swissisoform.modules.massspec import MassSpecModule


class TestMassSpecModule:
    """Tests for MassSpecModule."""

    def test_tryptic_digest_simple(self, config):
        """Digest a protein with clear cleavage sites, verify expected peptides."""
        module = MassSpecModule(config)
        # MAAAAAALLLLLLLRBBBBBBBBR*
        # R at index 14 and R at index 23
        # 0-missed: MAAAAAALLLLLLR (0-15, 15 AA), BBBBBBBBBR (15-24, 9 AA) — wait,
        # need no K/R in internal fragments. Use only R as cleavage.
        # Protein with exactly two R cleavage sites and no K:
        protein = "MAAAAAALLLLLLLRBBBBBBBBR*"
        peptides = module._tryptic_digest(protein)
        pep_seqs = [p["peptide"] for p in peptides]
        # 0-missed cleavages:
        assert "MAAAAAALLLLLLLR" in pep_seqs  # pos 0-15, 15 AA
        assert "BBBBBBBBR" in pep_seqs         # pos 15-24, 9 AA
        # 1-missed cleavage:
        assert "MAAAAAALLLLLLLRBBBBBBBBR" in pep_seqs  # pos 0-24, 24 AA

    def test_tryptic_digest_kp_exception(self, config):
        """K followed by P should NOT be a cleavage site."""
        module = MassSpecModule(config)
        # MAAAAAAKPBBBBBBBR — K at index 7 followed by P, not a site.
        # R at index 16 is a site.
        # Only cleavage at R(16), so 0-missed: MAAAAAAKPBBBBBBBR (0-17, 17 AA)
        protein = "MAAAAAAKPBBBBBBBR"
        peptides = module._tryptic_digest(protein)
        pep_seqs = [p["peptide"] for p in peptides]
        # The whole protein should be one peptide (0-missed cleavage)
        assert "MAAAAAAKPBBBBBBBR" in pep_seqs
        # Should NOT have been split at the KP site
        assert "MAAAAAAK" not in pep_seqs

    def test_tryptic_digest_positions(self, config):
        """Verify pos and end fields match where peptide appears in sequence."""
        module = MassSpecModule(config)
        protein = "MAAAAAALLLLLLLRKKKKKKKR*"
        peptides = module._tryptic_digest(protein)
        seq = protein.rstrip("*").upper()
        for pep in peptides:
            assert seq[pep["pos"]:pep["end"]] == pep["peptide"]
            assert pep["length"] == pep["end"] - pep["pos"]

    def test_tryptic_digest_min_max_length(self, config):
        """Very short and very long peptides should be filtered out."""
        module = MassSpecModule(config)
        # Short peptides: MR (2 AA) at start, then a long stretch
        # R at index 1 → fragment "MR" (len 2, filtered), then rest
        protein = "MR" + "A" * 35 + "R"
        peptides = module._tryptic_digest(protein, missed_cleavages=0)
        for pep in peptides:
            assert pep["length"] >= 7
            assert pep["length"] <= 30
        # "MR" (len 2) should be filtered
        assert all(p["peptide"] != "MR" for p in peptides)
        # "A"*35 + "R" (len 36) should be filtered
        assert all(p["length"] <= 30 for p in peptides)

    def test_annotate_returns_hits_and_summary(self, config):
        """annotate should return dict with 'hits' and 'summary' keys."""
        module = MassSpecModule(config)
        protein = "MAAAAAALLLLLLLRKKKKKKKR*"
        result = module.annotate(protein)
        assert "hits" in result
        assert "summary" in result
        assert isinstance(result["hits"], list)
        assert isinstance(result["summary"], dict)
        assert result["summary"]["total_peptides"] == len(result["hits"])
        assert result["summary"]["total_peptides"] > 0
        assert result["summary"]["min_peptide_length"] is not None
        assert result["summary"]["max_peptide_length"] is not None

    def test_unique_to_isoform(self, config):
        """Peptides not found in canonical digest should be flagged unique."""
        module = MassSpecModule(config)
        # Isoform has an extra N-terminal region
        canonical = "MAAAAAALLLLLLLRKKKKKKKR*"
        isoform = "MXXXXXXLLLLLLLR" + canonical.rstrip("*") + "*"
        result = module.annotate(isoform, canonical_protein=canonical)
        unique_peps = [h for h in result["hits"] if h["unique_to_isoform"]]
        non_unique_peps = [h for h in result["hits"] if not h["unique_to_isoform"]]
        # There should be some unique peptides (from the MXXXXXXLLLLLLLR region)
        assert len(unique_peps) > 0
        # And some non-unique peptides (shared with canonical)
        assert len(non_unique_peps) > 0
        assert result["summary"]["unique_peptides"] == len(unique_peps)

    def test_unique_to_isoform_no_canonical(self, config):
        """Without canonical, unique_to_isoform is None (unknown) not False.

        Silently defaulting to False would hide the fact that the caller
        didn't provide a canonical, and would look identical to "peptide
        is in canonical".
        """
        module = MassSpecModule(config)
        protein = "MAAAAAALLLLLLLRKKKKKKKR*"
        result = module.annotate(protein)
        for hit in result["hits"]:
            assert hit["unique_to_isoform"] is None
        # Summary should also reflect "unknown"
        assert result["summary"]["unique_peptides"] is None

    def test_validated_peptides(self, config):
        """Peptides in validated_peptides dict should be marked validated."""
        protein = "MAAAAAALLLLLLLRKKKKKKKR*"
        # First get the peptides to know what to validate
        module_probe = MassSpecModule(config)
        probe_result = module_probe._tryptic_digest(protein)
        first_pep = probe_result[0]["peptide"]

        validated = {"TESTGENE": {first_pep}}
        module = MassSpecModule(config, validated_peptides=validated)
        result = module.annotate(protein, gene_name="TESTGENE")
        validated_hits = [h for h in result["hits"] if h["validated"]]
        assert len(validated_hits) >= 1
        assert result["summary"]["validated_peptides"] >= 1

    def test_empty_protein(self, config):
        """Empty string returns empty hits; unique/validated are None (unknown)
        when no canonical/gene was supplied."""
        module = MassSpecModule(config)
        result = module.annotate("")
        assert result["hits"] == []
        assert result["summary"]["total_peptides"] == 0
        # No canonical given → uniqueness is unknown
        assert result["summary"]["unique_peptides"] is None
        # No gene given → validation is unknown
        assert result["summary"]["validated_peptides"] is None
        assert result["summary"]["min_peptide_length"] is None
        assert result["summary"]["max_peptide_length"] is None

    def test_empty_protein_with_canonical(self, config):
        """Empty isoform with a canonical supplied: unique_peptides is 0 (known)."""
        module = MassSpecModule(config)
        result = module.annotate("", canonical_protein="MKKKKK*", gene_name="GENE")
        assert result["hits"] == []
        assert result["summary"]["unique_peptides"] == 0
        assert result["summary"]["validated_peptides"] == 0

    def test_run_no_sites_lost(self, synthetic_tis, config):
        """run() must preserve all input sites."""
        module = MassSpecModule(config)
        n_before = len(synthetic_tis)
        result = module.run(synthetic_tis)
        assert len(result) == n_before

    def test_module_scope(self, config):
        """Module SCOPE should be 'C'."""
        module = MassSpecModule(config)
        assert module.SCOPE == "C"

    def test_annotate_pure_function(self, config):
        """Two calls with same input should produce identical results."""
        module = MassSpecModule(config)
        protein = "MAAAAAALLLLLLLRKKKKKKKR*"
        result1 = module.annotate(protein)
        result2 = module.annotate(protein)
        assert result1 == result2
