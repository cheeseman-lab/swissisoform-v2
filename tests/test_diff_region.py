"""Tests for differential-region derivation, focused on the initiator-Met
truncation path (near-cognate alt-TIS starts translated as an initiator Met).

See ``assembly.compute_diff_region`` / ``assembly._is_initiator_met_trunc``.
"""

from swissisoform.assembly import _is_initiator_met_trunc, compute_diff_region
from swissisoform.models import ORFType

# A unique, non-repeating canonical (GFP N-terminus) so prefix searches can't
# match spuriously. 66 residues, starts with the canonical Met.
CAN = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYG"
K = 10  # truncation start offset used across the cases


class TestIsInitiatorMetTrunc:
    def test_true_for_met_substituted_suffix(self):
        # iso = installed-M + canonical[K:]; canonical[K-1] != M so it's a real
        # substitution, not an exact suffix.
        iso = "M" + CAN[K:]
        assert CAN[K - 1] != "M"  # guard: genuine substitution
        assert _is_initiator_met_trunc(iso, CAN) is True

    def test_false_for_exact_suffix(self):
        # iso is already a byte-exact suffix → not the Met-substitution case.
        iso = CAN[K:]
        assert _is_initiator_met_trunc(iso, CAN) is False

    def test_false_when_not_met_start(self):
        iso = "V" + CAN[K + 1 :]
        assert _is_initiator_met_trunc(iso, CAN) is False

    def test_false_when_suffix_unrelated(self):
        assert _is_initiator_met_trunc("M" + "W" * 30, CAN) is False


class TestComputeDiffRegionTruncation:
    def test_initiator_met_truncation_is_verified_tier(self):
        iso = "M" + CAN[K:]
        dr = compute_diff_region(ORFType.TRUNCATED, iso, CAN, context="unit")
        assert dr.confidence == "initiator_met"
        # canonical_end = len(can) - (len(iso) - 1): the M is isoform-unique, so
        # the lost region is one longer than a naive length delta.
        assert dr.canonical_end == len(CAN) - (len(iso) - 1) == K
        assert dr.canonical_start == 0
        assert dr.sequence == CAN[:K]
        assert dr.isoform_start is None and dr.isoform_end is None

    def test_exact_suffix_truncation_stays_tail_verified(self):
        iso = CAN[K:]  # byte-exact suffix, no Met substitution
        dr = compute_diff_region(ORFType.TRUNCATED, iso, CAN, context="unit")
        assert dr.confidence == "tail_verified"
        assert dr.canonical_end == K
        assert dr.sequence == CAN[:K]

    def test_divergent_truncation_still_length_fallback(self):
        # Guard preserved: genuinely unrelated shorter sequence must NOT be
        # promoted to a verified tier.
        iso = "M" + "W" * 30
        dr = compute_diff_region(ORFType.TRUNCATED, iso, CAN, context="unit")
        assert dr.confidence == "length_fallback"
