"""Tests for the PairedComparison engine."""

from __future__ import annotations

import math

from swissisoform.compare.paired import PairedComparison


class TestCompareCategorical:
    """Tests for compare_categorical."""

    def test_same_values(self) -> None:
        result = PairedComparison.compare_categorical("nuclear", "nuclear")
        assert result["changed"] is False
        assert result["canonical"] == "nuclear"
        assert result["isoform"] == "nuclear"

    def test_different_values(self) -> None:
        result = PairedComparison.compare_categorical("nuclear", "cytoplasmic")
        assert result["changed"] is True
        assert result["canonical"] == "nuclear"
        assert result["isoform"] == "cytoplasmic"

    def test_empty_strings(self) -> None:
        result = PairedComparison.compare_categorical("", "")
        assert result["changed"] is False

    def test_one_empty(self) -> None:
        result = PairedComparison.compare_categorical("nuclear", "")
        assert result["changed"] is True


class TestCompareSets:
    """Tests for compare_sets."""

    def test_identical_sets(self) -> None:
        result = PairedComparison.compare_sets({"A", "B"}, {"A", "B"})
        assert result["gained"] == []
        assert result["lost"] == []
        assert result["shared"] == ["A", "B"]

    def test_disjoint_sets(self) -> None:
        result = PairedComparison.compare_sets({"A", "B"}, {"C", "D"})
        assert result["gained"] == ["C", "D"]
        assert result["lost"] == ["A", "B"]
        assert result["shared"] == []

    def test_overlapping_sets(self) -> None:
        result = PairedComparison.compare_sets({"A", "B", "C"}, {"B", "C", "D"})
        assert result["gained"] == ["D"]
        assert result["lost"] == ["A"]
        assert result["shared"] == ["B", "C"]

    def test_empty_sets(self) -> None:
        result = PairedComparison.compare_sets(set(), set())
        assert result["gained"] == []
        assert result["lost"] == []
        assert result["shared"] == []

    def test_empty_canonical(self) -> None:
        result = PairedComparison.compare_sets(set(), {"X"})
        assert result["gained"] == ["X"]
        assert result["lost"] == []

    def test_results_are_sorted(self) -> None:
        result = PairedComparison.compare_sets({"Z", "A"}, {"M", "A"})
        assert result["lost"] == ["Z"]
        assert result["shared"] == ["A"]
        assert result["gained"] == ["M"]


class TestCompareScalarRegions:
    """Tests for compare_scalar_regions."""

    def test_enriched(self) -> None:
        result = PairedComparison.compare_scalar_regions(10.0, 5.0)
        assert result["ratio"] == 2.0
        assert result["enriched"] is True

    def test_depleted(self) -> None:
        result = PairedComparison.compare_scalar_regions(3.0, 10.0)
        assert result["ratio"] == 0.3
        assert result["enriched"] is False

    def test_equal(self) -> None:
        result = PairedComparison.compare_scalar_regions(5.0, 5.0)
        assert result["ratio"] == 1.0
        assert result["enriched"] is False

    def test_zero_shared_nonzero_unique(self) -> None:
        """shared=0 makes the ratio undefined for "enrichment" purposes → None."""
        result = PairedComparison.compare_scalar_regions(5.0, 0.0)
        assert result["ratio"] == math.inf
        assert result["enriched"] is None

    def test_zero_both(self) -> None:
        """Both zero → ratio 0, but enrichment can't be called → None."""
        result = PairedComparison.compare_scalar_regions(0.0, 0.0)
        assert result["ratio"] == 0.0
        assert result["enriched"] is None

    def test_sign_changing_negative_unique_returns_none(self) -> None:
        """Negative unique (e.g. gravy in the extension) → enrichment undefined."""
        result = PairedComparison.compare_scalar_regions(-0.2, 0.3)
        assert result["enriched"] is None

    def test_sign_changing_negative_shared_returns_none(self) -> None:
        """Negative shared (e.g. disorder Top-IDP scale) → enrichment undefined."""
        result = PairedComparison.compare_scalar_regions(0.4, -0.1)
        assert result["enriched"] is None

    def test_values_preserved(self) -> None:
        result = PairedComparison.compare_scalar_regions(7.5, 2.5)
        assert result["unique"] == 7.5
        assert result["shared"] == 2.5


class TestCompareStructures:
    """Tests for compare_structures."""

    def test_full_metrics(self) -> None:
        canonical = {"plddt": 80.0}
        isoform = {
            "plddt": 65.0,
            "tm_score": 0.85,
            "rmsd_global": 1.2,
            "extension_contacts": 15,
        }
        result = PairedComparison.compare_structures(canonical, isoform)
        assert result["plddt_delta"] == -15.0
        assert result["tm_score"] == 0.85
        assert result["rmsd_global"] == 1.2
        assert result["extension_contacts"] == 15

    def test_missing_isoform_fields(self) -> None:
        result = PairedComparison.compare_structures({"plddt": 70.0}, {})
        assert result["plddt_delta"] == -70.0
        assert result["tm_score"] is None
        assert result["rmsd_global"] is None
        assert result["extension_contacts"] == 0

    def test_missing_canonical_plddt(self) -> None:
        result = PairedComparison.compare_structures({}, {"plddt": 50.0})
        assert result["plddt_delta"] == 50.0

    def test_both_empty(self) -> None:
        result = PairedComparison.compare_structures({}, {})
        assert result["plddt_delta"] == 0
        assert result["tm_score"] is None
        assert result["extension_contacts"] == 0
