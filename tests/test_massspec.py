"""Tests for the massspec annotation module."""

from __future__ import annotations

from swissisoform.assembly import install_initiator_met
from swissisoform.contract import ORFType
from swissisoform.evidence.d3_mass_spec.massspec import (
    MassSpecModule,
    diagnostic_peptides,
)
from swissisoform.models import DifferentialRegion


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
        assert "BBBBBBBBR" in pep_seqs  # pos 15-24, 9 AA
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
            assert seq[pep["pos"] : pep["end"]] == pep["peptide"]
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
        when no canonical/gene was supplied.
        """
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
        """Empty isoform with a canonical supplied: unique_peptides is 0 (known).

        ``validated_peptides`` stays None because PepQuery2 has not been
        precomputed for this gene (the cache dict passed at init was
        empty).  Only an actual PepQuery2 precompute should make
        ``pepquery_run`` True.
        """
        module = MassSpecModule(config)
        result = module.annotate("", canonical_protein="MKKKKK*", gene_name="GENE")
        assert result["hits"] == []
        assert result["summary"]["unique_peptides"] == 0
        assert result["summary"]["validated_peptides"] is None
        assert result["summary"]["pepquery_run"] is False

    def test_validated_peptides_marks_pepquery_run(self, config):
        """Providing a validated-peptides cache flips ``pepquery_run`` True."""
        protein = "MAAAAAALLLLLLLRKKKKKKKR*"
        module_probe = MassSpecModule(config)
        first_pep = module_probe._tryptic_digest(protein)[0]["peptide"]
        module = MassSpecModule(
            config, validated_peptides={"TESTGENE": {first_pep}}
        )
        result = module.annotate(protein, gene_name="TESTGENE")
        assert result["summary"]["pepquery_run"] is True
        assert result["summary"]["validated_peptides"] >= 1

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

    def test_pepquery_run_true_when_queried_but_zero_validated(self, config):
        """A gene whose peptides PepQuery queried but didn't validate must show
        ``pepquery_run=True, validated_peptides=0`` — not pepquery_run=False.

        Pre-fix, ``_regroup_by_gene`` skipped genes with no validated peptides,
        so MassSpecModule couldn't tell "queried + no MS evidence" from "never
        queried" — and E6 returned None instead of False for TRNT1 / CDC34 /
        SRSF2.
        """
        from swissisoform.evidence.d3_mass_spec.massspec import _regroup_by_gene
        peptide_to_genes = {
            "PEPONE": {"GENE_A"},
            "PEPTWO": {"GENE_A", "GENE_B"},
            "PEPTHREE": {"GENE_C"},  # GENE_C queried but PepQuery doesn't validate it
        }
        validated = {"PEPONE", "PEPTWO"}  # GENE_A and GENE_B get hits; GENE_C does not
        out = _regroup_by_gene(validated, peptide_to_genes)
        assert set(out.keys()) == {"GENE_A", "GENE_B", "GENE_C"}, (
            "every queried gene must appear in the output, including ones with "
            "zero validated peptides"
        )
        assert out["GENE_C"] == set(), "queried-but-unvalidated gene has empty set"
        assert out["GENE_A"] == {"PEPONE", "PEPTWO"}
        assert out["GENE_B"] == {"PEPTWO"}

        # And downstream: MassSpecModule on GENE_C now correctly reports
        # pepquery_run=True (it ran) + validated_peptides=0 (found nothing).
        module = MassSpecModule(config, validated_peptides=out)
        result = module.annotate("MAAAAAALLLLLLLRKKKKKKKR*", gene_name="GENE_C")
        assert result["summary"]["pepquery_run"] is True
        assert result["summary"]["validated_peptides"] == 0


class TestInstallInitiatorMet:
    """Part 1 — the initiator-Met install at assembly time."""

    def test_near_cognate_start_gets_met(self):
        # CBX1 CTG isoform begins with L (Leu) — should become M.
        assert install_initiator_met("LGATPPGDPTRR") == "MGATPPGDPTRR"
        # GTG isoform begins with V (Val).
        assert install_initiator_met("VRDATAATR") == "MRDATAATR"

    def test_atg_start_is_noop(self):
        assert install_initiator_met("MSRAAAA") == "MSRAAAA"

    def test_trailing_stop_preserved(self):
        assert install_initiator_met("LGATPP*") == "MGATPP*"

    def test_empty_returns_empty(self):
        assert install_initiator_met("") == ""
        assert install_initiator_met("*") == "*"


class TestDiagnosticPeptides:
    """Part 3 — diagnostic-region digest scope + NME variants."""

    # A protein with the CBX1-extension shape: M start, NME-substrate 2nd
    # residue (G), and tryptic sites so we get several peptides.
    PROT = "MGATPPGDPTRRKAAAAAAAAKLLLLLLLLLK"

    def test_orf_type_none_is_full_digest_no_nme(self):
        from swissisoform.evidence.d3_mass_spec.massspec import tryptic_digest

        full = {p["peptide"] for p in tryptic_digest(self.PROT)}
        out = diagnostic_peptides(self.PROT, orf_type=None)
        assert {p["peptide"] for p in out} == full
        assert all(p["nme"] is False for p in out)

    def test_truncated_keeps_only_start_peptides(self):
        out = diagnostic_peptides(self.PROT, orf_type=ORFType.TRUNCATED)
        # Every kept peptide starts at the new initiator (pos 0) or is its
        # Met-excised twin (pos 1) — nothing downstream of the first cleavage.
        assert {p["pos"] for p in out} <= {0, 1}
        assert all(p["peptide"].startswith("M") or p["nme"] for p in out)

    def test_extended_keeps_extension_overlap_including_junction(self):
        # Extension = isoform[0:10]; the junction peptide MGATPPGDPTR (0–11)
        # overlaps it and must be kept; the downstream KAAAA… peptide must not.
        dr = DifferentialRegion(isoform_start=0, isoform_end=10)
        out = diagnostic_peptides(self.PROT, orf_type=ORFType.EXTENDED, diff_region=dr)
        kept = {p["peptide"] for p in out}
        assert any(p.startswith("MGATPP") for p in kept)  # junction-spanning
        assert all(
            p["pos"] < 10 and p["end"] > 0 for p in out if not p["nme"]
        )

    def test_extended_without_diff_region_falls_back_to_full(self):
        full = {
            p["peptide"]
            for p in diagnostic_peptides(self.PROT, orf_type=None)
        }
        out = diagnostic_peptides(self.PROT, orf_type=ORFType.EXTENDED, diff_region=None)
        # Same peptide set as a full digest (NME variants aside).
        assert {p["peptide"] for p in out if not p["nme"]} == full

    def test_whole_isoform_is_full_digest(self):
        from swissisoform.evidence.d3_mass_spec.massspec import tryptic_digest

        full = {p["peptide"] for p in tryptic_digest(self.PROT)}
        out = diagnostic_peptides(self.PROT, orf_type=ORFType.UORF)
        # Whole-isoform ORFs keep the full digest (all unique by definition).
        assert full <= {p["peptide"] for p in out}

    def test_nme_variant_emitted_for_substrate_second_residue(self):
        out = diagnostic_peptides(self.PROT, orf_type=ORFType.TRUNCATED)
        nme = [p for p in out if p["nme"]]
        assert nme, "expected a Met-excised variant (residue 2 = G)"
        for p in nme:
            assert p["pos"] == 1
            assert not p["peptide"].startswith("M")
            # The excised twin is one residue shorter than its Met-retained form.
            assert p["length"] == p["end"] - p["pos"]

    def test_no_nme_when_second_residue_not_substrate(self):
        # Second residue K is not an NME substrate.
        prot = "MKAAAAAAAAR"
        out = diagnostic_peptides(prot, orf_type=ORFType.TRUNCATED)
        assert all(p["nme"] is False for p in out)

    def test_annotate_passes_orf_context(self, config):
        """annotate() scoped by orf_type matches the helper's isoform set."""
        module = MassSpecModule(config)
        dr = DifferentialRegion(isoform_start=0, isoform_end=10)
        result = module.annotate(
            self.PROT, orf_type=ORFType.EXTENDED, diff_region=dr
        )
        expected = {
            p["peptide"]
            for p in diagnostic_peptides(
                self.PROT, orf_type=ORFType.EXTENDED, diff_region=dr
            )
        }
        assert {h["peptide"] for h in result["hits"]} == expected
        assert all("nme" in h for h in result["hits"])
