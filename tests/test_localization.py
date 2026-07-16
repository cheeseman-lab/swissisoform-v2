"""Tests for the localization annotation module."""

from __future__ import annotations

from swissisoform.evidence.l1_localization.localization import LocalizationModule


class TestLocalizationModule:
    """Tests for LocalizationModule."""

    def test_annotate_by_key_found(self, config):
        """Provide predictions for a key, verify values returned."""
        predictions = {
            "my_key": {
                "deeploc": "Cytoplasm",
                "deeploc_signals": "Signal peptide",
                "deeploc_membrane": "Soluble",
            },
        }
        module = LocalizationModule(config, predictions=predictions)
        ann = module.annotate_by_key("my_key")
        assert ann["deeploc_prediction"] == "Cytoplasm"
        assert ann["deeploc_signals"] == "Signal peptide"
        assert ann["deeploc_membrane"] == "Soluble"

    def test_annotate_by_key_not_found(self, config):
        """Query missing key, verify all None."""
        module = LocalizationModule(config, predictions={})
        ann = module.annotate_by_key("missing_key")
        assert ann["deeploc_prediction"] is None
        assert ann["deeploc_signals"] is None
        assert ann["deeploc_membrane"] is None

    def test_annotate_by_key_partial(self, config):
        """Provide only deeploc fields, missing fields are None."""
        predictions = {
            "partial_key": {
                "deeploc": "Nucleus",
                # signals / membrane intentionally omitted
            },
        }
        module = LocalizationModule(config, predictions=predictions)
        ann = module.annotate_by_key("partial_key")
        assert ann["deeploc_prediction"] == "Nucleus"
        assert ann["deeploc_signals"] is None
        assert ann["deeploc_membrane"] is None

    def test_run_no_sites_lost(self, synthetic_tis, config):
        """Output length must equal input length."""
        module = LocalizationModule(config)
        result = module.run(synthetic_tis)
        assert len(result) == len(synthetic_tis)

    def test_run_populates_annotations(self, synthetic_tis, config):
        """Each site has isoform_annotations['localization'] with all keys."""
        module = LocalizationModule(config)
        result = module.run(synthetic_tis)
        expected_keys = {
            "deeploc_prediction",
            "deeploc_signals",
            "deeploc_membrane",
        }
        for site in result:
            ann = site.isoform_annotations["localization"]
            assert set(ann.keys()) == expected_keys

    def test_run_matches_by_tis_id(self, synthetic_tis, config):
        """Provide prediction for one tis_id, verify it's found."""
        tis_id = synthetic_tis[0].tis_id
        predictions = {
            tis_id: {
                "deeploc": "Cytoplasm",
                "deeploc_signals": "Signal peptide",
                "deeploc_membrane": "Soluble",
            },
        }
        module = LocalizationModule(config, predictions=predictions)
        result = module.run(synthetic_tis)
        ann = result[0].isoform_annotations["localization"]
        assert ann["deeploc_prediction"] == "Cytoplasm"
        assert ann["deeploc_signals"] == "Signal peptide"
        assert ann["deeploc_membrane"] == "Soluble"

    def test_run_unmatched_gets_none(self, synthetic_tis, config):
        """tis_id not in predictions gets None values."""
        tis_id = synthetic_tis[0].tis_id
        predictions = {
            tis_id: {
                "deeploc": "Nucleus",
                "deeploc_signals": None,
                "deeploc_membrane": None,
            },
        }
        module = LocalizationModule(config, predictions=predictions)
        result = module.run(synthetic_tis)
        # Second site has no predictions
        ann = result[1].isoform_annotations["localization"]
        assert ann["deeploc_prediction"] is None
        assert ann["deeploc_signals"] is None

    def test_module_scope(self, config):
        """Module SCOPE should be 'C'."""
        module = LocalizationModule(config)
        assert module.SCOPE == "C"
