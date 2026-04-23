"""Tests for the TargetP annotation module."""

from __future__ import annotations

from swissisoform.modules.targetp import TargetPModule, _protein_hash


class TestTargetPModule:
    """Tests for TargetPModule."""

    def test_annotate_found(self, config):
        seq = "MLRARALLAAPASLRATP"
        predictions = {
            _protein_hash(seq): {
                "targetp_prediction": "mTP",
                "targetp_probability": 0.93,
                "targetp_sp_prob": 0.02,
                "targetp_mtp_prob": 0.93,
                "targetp_ctp_prob": None,
                "targetp_cleavage_site": "CS pos: 17-18. ProbCS: 0.81",
            },
        }
        module = TargetPModule(config, predictions=predictions)
        ann = module.annotate(seq)
        assert ann["targetp_prediction"] == "mTP"
        assert ann["targetp_probability"] == 0.93
        assert ann["targetp_mtp_prob"] == 0.93
        assert ann["targetp_sp_prob"] == 0.02
        assert ann["targetp_ctp_prob"] is None

    def test_annotate_missing(self, config):
        module = TargetPModule(config, predictions={})
        ann = module.annotate("MNOPQRST")
        for key in (
            "targetp_prediction",
            "targetp_probability",
            "targetp_sp_prob",
            "targetp_mtp_prob",
            "targetp_ctp_prob",
            "targetp_cleavage_site",
        ):
            assert ann[key] is None

    def test_run_no_sites_lost(self, synthetic_tis, config):
        module = TargetPModule(config)
        result = module.run(synthetic_tis)
        assert len(result) == len(synthetic_tis)

    def test_run_populates_annotations(self, synthetic_tis, config):
        module = TargetPModule(config)
        result = module.run(synthetic_tis)
        expected = {
            "targetp_prediction",
            "targetp_probability",
            "targetp_sp_prob",
            "targetp_mtp_prob",
            "targetp_ctp_prob",
            "targetp_cleavage_site",
        }
        for site in result:
            assert set(site.isoform_annotations["targetp"].keys()) == expected
