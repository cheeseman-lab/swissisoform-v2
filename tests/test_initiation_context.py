"""Tests for the initiation_context annotation module."""

from __future__ import annotations

import copy

import pytest

from swissisoform.modules.initiation_context import (
    InitiationContextModule,
    hamming_distance_ambiguous,
)

# ---------------------------------------------------------------------------
# hamming_distance_ambiguous unit tests
# ---------------------------------------------------------------------------


class TestHammingDistanceAmbiguous:
    """Tests for the public hamming_distance_ambiguous function."""

    def test_identical_sequences(self):
        assert hamming_distance_ambiguous("ATCG", "ATCG") == 0

    def test_all_mismatches(self):
        assert hamming_distance_ambiguous("AAAA", "CCCC") == 4

    def test_r_matches_a(self):
        assert hamming_distance_ambiguous("R", "A") == 0

    def test_r_matches_g(self):
        assert hamming_distance_ambiguous("R", "G") == 0

    def test_r_does_not_match_c(self):
        assert hamming_distance_ambiguous("R", "C") == 1

    def test_case_insensitive(self):
        assert hamming_distance_ambiguous("atcg", "ATCG") == 0

    def test_weights_only_weighted_positions_count(self):
        # Mismatch at positions 0 and 3, but only position 3 has weight
        result = hamming_distance_ambiguous("AACG", "CACG", weights=[0, 1, 1, 1])
        assert result == 0  # mismatch at pos 0 has weight 0
        result2 = hamming_distance_ambiguous("AACG", "CACC", weights=[0, 1, 1, 1])
        assert result2 == 1  # mismatch at pos 3 has weight 1

    def test_perfect_kozak(self):
        # "gccgccAccATGG" matches consensus "gccgccRccATGG" perfectly (A is in R)
        assert hamming_distance_ambiguous("gccgccAccATGG", "gccgccRccATGG") == 0

    def test_unequal_lengths_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            hamming_distance_ambiguous("ATCG", "AT")


# ---------------------------------------------------------------------------
# InitiationContextModule tests
# ---------------------------------------------------------------------------


class TestInitiationContextModule:
    """Tests for the InitiationContextModule."""

    def test_all_output_columns_present(self, synthetic_tis, config):
        mod = InitiationContextModule(config)
        result = mod.run(synthetic_tis)
        for site in result:
            ann = site.annotations[mod.MODULE_NAME]
            prefix = f"{mod.MODULE_NAME}_"
            for col in mod.OUTPUT_COLUMNS:
                key = col[len(prefix):] if col.startswith(prefix) else col
                assert key in ann, f"Missing column {key} for {site.tis_id}"

    def test_no_sites_lost(self, synthetic_tis, config):
        mod = InitiationContextModule(config)
        result = mod.run(synthetic_tis)
        assert len(result) == len(synthetic_tis)

    def test_kozak_context_stored(self, synthetic_tis, config):
        mod = InitiationContextModule(config)
        result = mod.run(synthetic_tis)
        for site in result:
            ann = site.annotations["initiation_context"]
            assert ann["kozak_context"] == site.kozak_context

    def test_perfect_kozak_hamming_zero(self, config):
        """A site with perfect Kozak context should have all Hamming distances = 0."""
        from tests.conftest import SYNTHETIC_TIS

        site = copy.deepcopy(SYNTHETIC_TIS[0])
        site.kozak_context = "gccgccAccATGG"  # A matches R in consensus
        result = InitiationContextModule(config).run([site])
        ann = result[0].annotations["initiation_context"]
        assert ann["kozak_hamming_full"] == 0
        assert ann["kozak_hamming_major"] == 0
        assert ann["kozak_hamming_partial"] == 0

    def test_mismatch_at_minus3(self, config):
        """T at -3 (position 6) doesn't match R={A,G} → hamming_major == 1."""
        from tests.conftest import SYNTHETIC_TIS

        site = copy.deepcopy(SYNTHETIC_TIS[0])
        # Position 6 is the R position; change it to T (not in {A,G})
        site.kozak_context = "gccgccTccATGG"
        result = InitiationContextModule(config).run([site])
        ann = result[0].annotations["initiation_context"]
        assert ann["kozak_hamming_major"] == 1

    def test_none_kozak_context(self, config):
        """If kozak_context is None, all Hamming values should be None."""
        from tests.conftest import SYNTHETIC_TIS

        site = copy.deepcopy(SYNTHETIC_TIS[0])
        site.kozak_context = None
        result = InitiationContextModule(config).run([site])
        ann = result[0].annotations["initiation_context"]
        assert ann["kozak_hamming_full"] is None
        assert ann["kozak_hamming_major"] is None
        assert ann["kozak_hamming_partial"] is None

    def test_short_kozak_context(self, config):
        """If kozak_context is shorter than 13 chars, Hamming values should be None."""
        from tests.conftest import SYNTHETIC_TIS

        site = copy.deepcopy(SYNTHETIC_TIS[0])
        site.kozak_context = "ATGCCC"  # too short
        result = InitiationContextModule(config).run([site])
        ann = result[0].annotations["initiation_context"]
        assert ann["kozak_hamming_full"] is None
        assert ann["kozak_hamming_major"] is None
        assert ann["kozak_hamming_partial"] is None

    def test_annotated_no_crash(self, config):
        """Annotated TIS (empty differential) should not crash."""
        from tests.conftest import SYNTHETIC_TIS

        site = copy.deepcopy(SYNTHETIC_TIS[0])  # Annotated type
        assert site.differential_sequence == ""
        result = InitiationContextModule(config).run([site])
        assert len(result) == 1

    def test_gc_content(self, config):
        """GC content of 'GGGCCCAAATTT' (6 G/C out of 12) → 0.5."""
        from tests.conftest import SYNTHETIC_TIS

        site = copy.deepcopy(SYNTHETIC_TIS[0])
        site.kozak_context = "GGGCCCAAATTT"
        result = InitiationContextModule(config).run([site])
        ann = result[0].annotations["initiation_context"]
        assert ann["utr5_gc_content"] == pytest.approx(0.5)
