"""Tests for the SignalP annotation module."""

from __future__ import annotations

from swissisoform.modules.signalp import SignalPModule, _protein_hash


class TestSignalPModule:
    """Tests for SignalPModule."""

    def test_annotate_found(self, config):
        seq = "MKLLVVAAAGGSP"
        predictions = {
            _protein_hash(seq): {
                "signalp_prediction": "SP",
                "signalp_probability": 0.97,
                "signalp_cleavage_site": "CS pos: 25-26. ProbCS: 0.91",
            },
        }
        module = SignalPModule(config, predictions=predictions)
        ann = module.annotate(seq)
        assert ann["signalp_prediction"] == "SP"
        assert ann["signalp_probability"] == 0.97
        assert "CS pos" in ann["signalp_cleavage_site"]

    def test_annotate_missing(self, config):
        module = SignalPModule(config, predictions={})
        ann = module.annotate("MNOPQRST")
        assert ann["signalp_prediction"] is None
        assert ann["signalp_probability"] is None
        assert ann["signalp_cleavage_site"] is None

    def test_run_no_sites_lost(self, synthetic_tis, config):
        module = SignalPModule(config)
        result = module.run(synthetic_tis)
        assert len(result) == len(synthetic_tis)

    def test_run_populates_annotations(self, synthetic_tis, config):
        module = SignalPModule(config)
        result = module.run(synthetic_tis)
        expected = {
            "signalp_prediction",
            "signalp_probability",
            "signalp_cleavage_site",
        }
        for site in result:
            assert set(site.isoform_annotations["signalp"].keys()) == expected

    def test_stop_codon_strip_stable_hash(self, config):
        module = SignalPModule(
            config,
            predictions={
                _protein_hash("MKLLA"): {
                    "signalp_prediction": "OTHER",
                    "signalp_probability": 0.8,
                    "signalp_cleavage_site": None,
                },
            },
        )
        # The stored hash is computed without a trailing stop; module should
        # still find it when the input has one.
        ann_no_stop = module.annotate("MKLLA")
        ann_with_stop = module.annotate("MKLLA*")
        assert ann_no_stop == ann_with_stop
        assert ann_with_stop["signalp_prediction"] == "OTHER"
