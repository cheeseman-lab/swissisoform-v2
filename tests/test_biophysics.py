"""Tests for the biophysics annotation module."""

from __future__ import annotations

from swissisoform.models import DifferentialRegion
from swissisoform.modules.biophysics import BiophysicsModule


class TestBiophysicsModule:
    """Tests for BiophysicsModule."""

    def test_no_sites_lost(self, synthetic_tis, config):
        """Output length must equal input length."""
        module = BiophysicsModule(config)
        result = module.run(synthetic_tis)
        assert len(result) == len(synthetic_tis)

    def test_all_output_columns_present(self, synthetic_tis, config):
        """Every site must have all OUTPUT_COLUMNS in annotations."""
        module = BiophysicsModule(config)
        result = module.run(synthetic_tis)
        prefix = f"{module.MODULE_NAME}_"
        expected_keys = [
            col[len(prefix) :] if col.startswith(prefix) else col for col in module.OUTPUT_COLUMNS
        ]
        for site in result:
            ann = site.isoform_annotations[module.MODULE_NAME]
            for key in expected_keys:
                assert key in ann, f"Missing key {key!r} in site {site.tis_id}"

    def test_empty_differential_returns_none(self, synthetic_tis, config):
        """Site[0] has empty diff_region.sequence — all values should be None."""
        module = BiophysicsModule(config)
        result = module.run(synthetic_tis)
        ann = result[0].isoform_annotations[module.MODULE_NAME]
        for key, val in ann.items():
            assert val is None, f"Expected None for {key!r} on empty diff, got {val}"

    def test_short_sequence_returns_none(self, synthetic_tis, config):
        """Sequences shorter than 3 AA should get all None."""
        module = BiophysicsModule(config)
        # Modify site[7] to have a 2-char differential
        synthetic_tis[7].diff_region = DifferentialRegion(sequence="MF")
        result = module.run(synthetic_tis)
        ann = result[7].isoform_annotations[module.MODULE_NAME]
        for key, val in ann.items():
            assert val is None, f"Expected None for {key!r} on short seq, got {val}"

    def test_extension_computes_values(self, synthetic_tis, config):
        """Site[1] (MRGSHHHHHGS, 11 AA) should have non-None biophysical values."""
        module = BiophysicsModule(config)
        result = module.run(synthetic_tis)
        ann = result[1].isoform_annotations[module.MODULE_NAME]
        for key in ("pI", "gravy", "disorder", "fraction_charged", "shannon_entropy"):
            assert ann[key] is not None, f"Expected non-None for {key!r}"

    def test_length_matches_differential(self, synthetic_tis, config):
        """Reported length should match len(differential_sequence) stripped of '*'."""
        module = BiophysicsModule(config)
        result = module.run(synthetic_tis)
        for site in result:
            ann = site.isoform_annotations[module.MODULE_NAME]
            seq = (site.diff_region.sequence if site.diff_region else "").rstrip("*")
            if len(seq) < 3:
                assert ann["length"] is None
            else:
                assert ann["length"] == len(seq), (
                    f"Length mismatch for {site.tis_id}: {ann['length']} != {len(seq)}"
                )

    def test_gravy_range(self, synthetic_tis, config):
        """GRAVY should be between -4.5 and 4.5 for all computed sites."""
        module = BiophysicsModule(config)
        result = module.run(synthetic_tis)
        for site in result:
            val = site.isoform_annotations[module.MODULE_NAME]["gravy"]
            if val is not None:
                assert -4.5 <= val <= 4.5, f"GRAVY out of range: {val}"

    def test_pi_range(self, synthetic_tis, config):
        """pI should be between 0 and 14 for all computed sites."""
        module = BiophysicsModule(config)
        result = module.run(synthetic_tis)
        for site in result:
            val = site.isoform_annotations[module.MODULE_NAME]["pI"]
            if val is not None:
                assert 0 <= val <= 14, f"pI out of range: {val}"

    def test_shannon_entropy_nonnegative(self, synthetic_tis, config):
        """Shannon entropy must be >= 0."""
        module = BiophysicsModule(config)
        result = module.run(synthetic_tis)
        for site in result:
            val = site.isoform_annotations[module.MODULE_NAME]["shannon_entropy"]
            if val is not None:
                assert val >= 0, f"Negative Shannon entropy: {val}"

    def test_fraction_charged_range(self, synthetic_tis, config):
        """Fraction charged should be between 0 and 1."""
        module = BiophysicsModule(config)
        result = module.run(synthetic_tis)
        for site in result:
            val = site.isoform_annotations[module.MODULE_NAME]["fraction_charged"]
            if val is not None:
                assert 0 <= val <= 1, f"fraction_charged out of range: {val}"

    def test_llps_score_range(self, synthetic_tis, config):
        """Composite LLPS score should be between 0 and 1."""
        module = BiophysicsModule(config)
        result = module.run(synthetic_tis)
        for site in result:
            val = site.isoform_annotations[module.MODULE_NAME]["llps_score"]
            if val is not None:
                assert 0 <= val <= 1, f"LLPS score out of range: {val}"

    def test_module_scope(self):
        """SCOPE should be 'C' (per-site classification)."""
        assert BiophysicsModule.SCOPE == "C"


class TestAnnotateMethod:
    """Tests for the annotate() ProteinModule interface."""

    def test_annotate_returns_dict(self, config):
        """annotate() should return a dict with all expected keys."""
        module = BiophysicsModule(config)
        result = module.annotate("MRGSHHHHHGS")
        assert isinstance(result, dict)
        prefix = f"{module.MODULE_NAME}_"
        expected_keys = [
            col[len(prefix) :] if col.startswith(prefix) else col for col in module.OUTPUT_COLUMNS
        ]
        for key in expected_keys:
            assert key in result, f"Missing key {key!r}"

    def test_annotate_empty_returns_none(self, config):
        """annotate('') should return all None values."""
        module = BiophysicsModule(config)
        result = module.annotate("")
        for key, val in result.items():
            assert val is None, f"Expected None for {key!r}, got {val}"

    def test_annotate_short_returns_none(self, config):
        """annotate('MF') (< 3 residues) should return all None values."""
        module = BiophysicsModule(config)
        result = module.annotate("MF")
        for key, val in result.items():
            assert val is None, f"Expected None for {key!r}, got {val}"

    def test_annotate_pure_function(self, config):
        """Calling annotate() twice with the same input gives identical results."""
        module = BiophysicsModule(config)
        result1 = module.annotate("MRGSHHHHHGS")
        result2 = module.annotate("MRGSHHHHHGS")
        assert result1 == result2

    def test_annotate_canonical_protein(self, config):
        """annotate() works on a full canonical protein sequence (not just diffs)."""
        module = BiophysicsModule(config)
        # A realistic canonical protein fragment (human ubiquitin, first 76 AA)
        ubiquitin = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
        result = module.annotate(ubiquitin)
        for key, val in result.items():
            assert val is not None, f"Expected non-None for {key!r} on canonical protein"
