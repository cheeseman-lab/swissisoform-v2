"""Tests for the generef (gene reference) annotation module."""

from __future__ import annotations

from swissisoform.modules.generef import _ANNOTATION_KEYS, GeneRefModule


class TestGeneRefModule:
    """Tests for GeneRefModule."""

    def test_no_sites_lost(self, synthetic_tis, config):
        """Output length must equal input length."""
        module = GeneRefModule(config)
        result = module.run(synthetic_tis)
        assert len(result) == len(synthetic_tis)

    def test_all_output_columns_present(self, synthetic_tis, config):
        """Every site must have all OUTPUT_COLUMNS in annotations."""
        module = GeneRefModule(config)
        result = module.run(synthetic_tis)
        prefix = f"{module.MODULE_NAME}_"
        expected_keys = [col.removeprefix(prefix) for col in module.OUTPUT_COLUMNS]
        for site in result:
            ann = site.isoform_annotations[module.MODULE_NAME]
            for key in expected_keys:
                assert key in ann, f"Missing key '{key}' for site {site.tis_id}"

    def test_no_annotations_all_none(self, synthetic_tis, config):
        """When gene_annotations is None/empty, all values are None."""
        module = GeneRefModule(config, gene_annotations=None)
        result = module.run(synthetic_tis)
        for site in result:
            ann = site.isoform_annotations["generef"]
            for key in _ANNOTATION_KEYS:
                assert ann[key] is None, f"Expected None for '{key}' on {site.tis_id}"

    def test_matching_gene_gets_annotations(self, synthetic_tis, config):
        """Sites whose gene matches the provided annotations get the values."""
        gene_data = {
            "TESTGENE_POS": {
                "uniprot_id": "P12345",
                "uniprot_function": "Transcription factor",
                "subcellular_location": "Nucleus",
                "hpa_protein_class": "Transcription factors",
                "hpa_subcellular_location": "Nucleoplasm",
                "depmap_mean_effect": -0.42,
                "omim_phenotypes": "Syndrome X",
            },
        }
        module = GeneRefModule(config, gene_annotations=gene_data)
        result = module.run(synthetic_tis)
        for site in result:
            ann = site.isoform_annotations["generef"]
            if site.gene_name == "TESTGENE_POS":
                assert ann["uniprot_id"] == "P12345"
                assert ann["uniprot_function"] == "Transcription factor"
                assert ann["subcellular_location"] == "Nucleus"

    def test_unmatched_gene_gets_none(self, synthetic_tis, config):
        """Sites whose gene is NOT in gene_annotations get None for all keys."""
        gene_data = {
            "TESTGENE_POS": {
                "uniprot_id": "P12345",
                "uniprot_function": "Transcription factor",
            },
        }
        module = GeneRefModule(config, gene_annotations=gene_data)
        result = module.run(synthetic_tis)
        for site in result:
            if site.gene_name != "TESTGENE_POS":
                ann = site.isoform_annotations["generef"]
                for key in _ANNOTATION_KEYS:
                    assert ann[key] is None, (
                        f"Expected None for '{key}' on unmatched gene {site.gene_name}"
                    )

    def test_all_genes_annotated(self, synthetic_tis, config):
        """When all 3 test genes have annotations, every site gets values."""
        gene_data = {
            "TESTGENE_POS": {"uniprot_id": "P111", "depmap_mean_effect": -0.1},
            "TESTGENE_NEG": {"uniprot_id": "P222", "depmap_mean_effect": -0.2},
            "TESTGENE_MULTI": {"uniprot_id": "P333", "depmap_mean_effect": -0.3},
        }
        module = GeneRefModule(config, gene_annotations=gene_data)
        result = module.run(synthetic_tis)
        for site in result:
            ann = site.isoform_annotations["generef"]
            assert ann["uniprot_id"] is not None, (
                f"Expected non-None uniprot_id for {site.gene_name}"
            )

    def test_module_scope_is_gene(self):
        """SCOPE must be 'G' for gene-level module."""
        assert GeneRefModule.SCOPE == "G"

    def test_partial_gene_data(self, synthetic_tis, config):
        """When only some fields are provided, missing ones are None."""
        gene_data = {
            "TESTGENE_POS": {
                "uniprot_id": "P99999",
                # All other keys deliberately missing.
            },
        }
        module = GeneRefModule(config, gene_annotations=gene_data)
        result = module.run(synthetic_tis)
        for site in result:
            if site.gene_name == "TESTGENE_POS":
                ann = site.isoform_annotations["generef"]
                assert ann["uniprot_id"] == "P99999"
                assert ann["uniprot_function"] is None
                assert ann["subcellular_location"] is None
