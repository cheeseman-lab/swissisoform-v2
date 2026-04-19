"""Tests for the core_identity annotation module."""

from __future__ import annotations

from swissisoform.config import PipelineConfig, ScoringConfig
from swissisoform.models import DifferentialRegion, ORFType, TranslationInitiationSite
from swissisoform.modules.core_identity import CoreIdentityModule


class TestCoreIdentityModule:
    """Tests for CoreIdentityModule."""

    def test_no_sites_lost(self, synthetic_tis, config):
        """Output length must equal input length."""
        module = CoreIdentityModule(config)
        result = module.run(synthetic_tis)
        assert len(result) == len(synthetic_tis)

    def test_all_output_columns_present(self, synthetic_tis, config):
        """Every site must have all OUTPUT_COLUMNS in annotations."""
        module = CoreIdentityModule(config)
        result = module.run(synthetic_tis)
        prefix = f"{module.MODULE_NAME}_"
        expected_keys = [
            col[len(prefix) :] if col.startswith(prefix) else col for col in module.OUTPUT_COLUMNS
        ]
        for site in result:
            ann = site.isoform_annotations[module.MODULE_NAME]
            for key in expected_keys:
                assert key in ann, f"Missing key '{key}' for site {site.tis_id}"

    def test_annotated_site(self, synthetic_tis, config):
        """Annotated site: differential_length == 0, in_frame, no warning."""
        module = CoreIdentityModule(config)
        result = module.run(synthetic_tis)
        # First site is Annotated
        ann = result[0].isoform_annotations["core_identity"]
        assert ann["orf_type"] == "annotated"
        assert ann["differential_length_aa"] == 0
        assert ann["in_frame"] is True
        assert ann["large_truncation_warning"] is False

    def test_extension_site(self, synthetic_tis, config):
        """Extension: differential_length == 11, in_frame, isoform > canonical."""
        module = CoreIdentityModule(config)
        result = module.run(synthetic_tis)
        # Second site is Extension with "MRGSHHHHHGS" differential
        ann = result[1].isoform_annotations["core_identity"]
        assert ann["orf_type"] == "extended"
        assert ann["differential_length_aa"] == 11
        assert ann["in_frame"] is True
        assert ann["isoform_protein_length"] > ann["canonical_protein_length"]

    def test_truncation_no_warning(self, synthetic_tis, config):
        """Small truncation should NOT trigger large_truncation_warning."""
        module = CoreIdentityModule(config)
        result = module.run(synthetic_tis)
        # Site index 4 is Truncated +strand, differential "EEPQSDPSV" = 9 aa
        ann = result[4].isoform_annotations["core_identity"]
        assert ann["orf_type"] == "truncated"
        assert ann["in_frame"] is True
        assert ann["large_truncation_warning"] is False

    def test_large_truncation_warning(self, config):
        """A 250-aa differential on a TRUNCATED site should trigger warning."""
        site = TranslationInitiationSite(
            tis_id="chrX:999:+:ATG",
            gene_name="BIGGENE",
            transcript_id="ENST_BIG",
            chrom="chrX",
            position=999,
            strand="+",
            start_codon="ATG",
            orf_type=ORFType.TRUNCATED,
            canonical_protein="M" + "A" * 300 + "*",
            isoform_protein="M" + "A" * 50 + "*",
            diff_region=DifferentialRegion(sequence="A" * 250),
        )
        config_with_scoring = PipelineConfig(scoring=ScoringConfig(truncation_max_aa=200))
        module = CoreIdentityModule(config_with_scoring)
        result = module.run([site])
        ann = result[0].isoform_annotations["core_identity"]
        assert ann["large_truncation_warning"] is True

    def test_variant_ids_unique(self, synthetic_tis, config):
        """All variant_id values must be unique."""
        module = CoreIdentityModule(config)
        result = module.run(synthetic_tis)
        ids = [s.isoform_annotations["core_identity"]["variant_id"] for s in result]
        assert len(ids) == len(set(ids))

    def test_annotated_empty_differential(self, synthetic_tis, config):
        """Annotated site with empty differential should not crash."""
        module = CoreIdentityModule(config)
        result = module.run(synthetic_tis)
        ann = result[0].isoform_annotations["core_identity"]
        assert ann["differential_length_aa"] == 0

    def test_uorf_not_in_frame(self, synthetic_tis, config):
        """UORF should have in_frame == False."""
        module = CoreIdentityModule(config)
        result = module.run(synthetic_tis)
        # Site index 6 is uORF
        ann = result[6].isoform_annotations["core_identity"]
        assert ann["orf_type"] == "uorf"
        assert ann["in_frame"] is False

    def test_protein_length_excludes_stop_codon(self, synthetic_tis, config):
        """Protein length should exclude trailing '*'."""
        module = CoreIdentityModule(config)
        result = module.run(synthetic_tis)
        # First site canonical is "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPS*" = 37 residues
        ann = result[0].isoform_annotations["core_identity"]
        assert ann["canonical_protein_length"] == 37

    def test_uoorf_in_frame(self, synthetic_tis, config):
        """UoORF should have in_frame == True."""
        module = CoreIdentityModule(config)
        result = module.run(synthetic_tis)
        # Site index 7 is uoORF
        ann = result[7].isoform_annotations["core_identity"]
        assert ann["in_frame"] is True

    def test_module_scope(self, config):
        """Module SCOPE should be 'C'."""
        module = CoreIdentityModule(config)
        assert module.SCOPE == "C"

    def test_config_truncation_max_aa_default(self):
        """Default truncation_max_aa should be 200."""
        config = PipelineConfig()
        module = CoreIdentityModule(config)
        assert module.truncation_max_aa == 200

    def test_config_truncation_max_aa_custom(self):
        """Custom truncation_max_aa from ScoringConfig."""
        config = PipelineConfig(scoring=ScoringConfig(truncation_max_aa=100))
        module = CoreIdentityModule(config)
        assert module.truncation_max_aa == 100

    def test_annotate_site_returns_dict(self, synthetic_tis, config):
        """annotate_site returns a dict with all expected keys."""
        module = CoreIdentityModule(config)
        result = module.annotate_site(synthetic_tis[0])
        expected_keys = {
            "variant_id",
            "orf_type",
            "canonical_protein_length",
            "isoform_protein_length",
            "differential_length_aa",
            "in_frame",
            "large_truncation_warning",
        }
        assert isinstance(result, dict)
        assert set(result.keys()) == expected_keys

    def test_annotate_site_extension(self, synthetic_tis, config):
        """annotate_site on extension site returns correct values."""
        module = CoreIdentityModule(config)
        ann = module.annotate_site(synthetic_tis[1])
        assert ann["orf_type"] == "extended"
        assert ann["differential_length_aa"] == 11
        assert ann["in_frame"] is True
        assert ann["isoform_protein_length"] > ann["canonical_protein_length"]

    def test_annotate_site_does_not_mutate(self, synthetic_tis, config):
        """annotate_site must not write to site.isoform_annotations."""
        module = CoreIdentityModule(config)
        site = synthetic_tis[0]
        # Ensure isoform_annotations is empty before calling annotate_site
        assert module.MODULE_NAME not in site.isoform_annotations
        module.annotate_site(site)
        assert module.MODULE_NAME not in site.isoform_annotations
