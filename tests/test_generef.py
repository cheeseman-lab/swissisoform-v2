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
                "function": "Transcription factor",
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
                assert ann["function"] == "Transcription factor"
                assert ann["subcellular_location"] == "Nucleus"

    def test_unmatched_gene_gets_none(self, synthetic_tis, config):
        """Sites whose gene is NOT in gene_annotations get None for all keys."""
        gene_data = {
            "TESTGENE_POS": {
                "uniprot_id": "P12345",
                "function": "Transcription factor",
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
                assert ann["function"] is None
                assert ann["subcellular_location"] is None


class TestAffinageFetch:
    """Tests for the Affinage-backed setup fetch (setup.generef.fetch_one)."""

    _GENE_PAYLOAD = {
        "narrative": {
            "mechanistic_narrative": "GENEX encodes a kinase that phosphorylates Y [PMID:1].",
            "mechanism_profile": {
                "molecular_activity": [
                    {"term_id": "GO:1", "term_label": "protein kinase activity"},
                    {"term_id": "GO:2", "term_label": "ATP binding"},
                ],
                "pathway": [
                    {"term_id": "R-1", "term_label": "Cell Cycle"},
                    {"term_id": "R-1", "term_label": "Cell Cycle"},  # dup → collapsed
                ],
                "localization": [
                    {"term_id": "GO:C1", "term_label": "cytosol"},
                    {"term_id": "GO:C2", "term_label": "nucleus"},
                ],
                "partners": ["SOMEGENE"],  # must NOT leak into keywords
            },
        }
    }

    def _patch(self, monkeypatch, *, gene_json, accession="P00001"):
        from swissisoform.setup import generef as g

        def fake_get_json(url, **kw):
            if url.startswith(g.AFFINAGE_API):
                return gene_json
            if url.startswith(g.UNIPROT_API):
                hits = [{"primaryAccession": accession}] if accession else []
                return {"results": hits}
            raise AssertionError(f"unexpected url {url}")

        monkeypatch.setattr(g, "_get_json", fake_get_json)

    def test_maps_all_fields(self, monkeypatch):
        from swissisoform.setup.generef import fetch_one

        self._patch(monkeypatch, gene_json=self._GENE_PAYLOAD)
        rec = fetch_one("GENEX")
        assert rec["uniprot_id"] == "P00001"
        assert rec["function"] == "GENEX encodes a kinase that phosphorylates Y [PMID:1]."
        # localization axis only, unique-preserving order
        assert rec["subcellular_location"] == "cytosol; nucleus"
        # molecular_activity + pathway, deduped, partners excluded
        assert rec["keywords"] == "protein kinase activity; ATP binding; Cell Cycle"

    def test_missing_gene_returns_none(self, monkeypatch):
        from swissisoform.setup.generef import fetch_one

        self._patch(monkeypatch, gene_json=None)  # Affinage 404 → _get_json None
        assert fetch_one("NOPE") is None

    def test_empty_profile_yields_none_fields(self, monkeypatch):
        from swissisoform.setup.generef import fetch_one

        self._patch(monkeypatch, gene_json={"narrative": {"mechanism_profile": {}}})
        rec = fetch_one("BARE")
        assert rec["uniprot_id"] == "P00001"  # accession still resolves
        assert rec["function"] is None
        assert rec["subcellular_location"] is None
        assert rec["keywords"] is None
