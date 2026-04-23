"""Tests for the InterProScan annotation module."""

from __future__ import annotations

from swissisoform.modules.interproscan import InterProScanModule, _protein_hash


class TestInterProScanModule:
    """Tests for InterProScanModule."""

    def test_annotate_found(self, config):
        seq = "MKLLVVAAAGGSPQPQNNVPDSILKKLMV" * 5
        predictions = {
            _protein_hash(seq): {
                "hits": [
                    {
                        "name": "PF12345",
                        "pos": 10,
                        "end": 80,
                        "db": "Pfam",
                        "description": "DummyDom domain",
                        "score": 1e-25,
                        "interpro_id": "IPR001234",
                        "interpro_description": "Dummy domain family",
                    },
                ],
                "summary": {"n_hits": 1, "n_databases": 1, "n_interpro": 1},
            },
        }
        module = InterProScanModule(config, predictions=predictions)
        ann = module.annotate(seq)
        assert ann["summary"]["n_hits"] == 1
        hit = ann["hits"][0]
        assert hit["name"] == "PF12345"
        assert hit["pos"] == 10 and hit["end"] == 80
        assert hit["db"] == "Pfam"
        assert hit["interpro_id"] == "IPR001234"

    def test_annotate_missing(self, config):
        module = InterProScanModule(config, predictions={})
        ann = module.annotate("MNOPQRST")
        assert ann["hits"] == []
        assert ann["summary"] == {"n_hits": 0, "n_databases": 0, "n_interpro": 0}

    def test_run_no_sites_lost(self, synthetic_tis, config):
        module = InterProScanModule(config)
        result = module.run(synthetic_tis)
        assert len(result) == len(synthetic_tis)

    def test_run_populates_annotations(self, synthetic_tis, config):
        module = InterProScanModule(config)
        result = module.run(synthetic_tis)
        for site in result:
            ann = site.isoform_annotations["interproscan"]
            assert "hits" in ann and isinstance(ann["hits"], list)
            assert "summary" in ann and isinstance(ann["summary"], dict)

    def test_stop_codon_strip_stable_hash(self, config):
        seq = "MKLLA"
        predictions = {
            _protein_hash(seq): {
                "hits": [],
                "summary": {"n_hits": 0, "n_databases": 0, "n_interpro": 0},
            },
        }
        module = InterProScanModule(config, predictions=predictions)
        assert module.annotate(seq) == module.annotate(seq + "*")
