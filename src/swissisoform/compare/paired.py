"""Paired comparison engine for canonical vs. isoform analysis.

Provides static methods for comparing categorical labels, set memberships,
scalar region enrichments, and structural prediction metrics between
canonical and isoform protein forms.
"""

from __future__ import annotations

import math


class PairedComparison:
    """Static methods for canonical-vs-isoform paired comparisons."""

    @staticmethod
    def compare_categorical(canonical: str, isoform: str) -> dict:
        """Compare two categorical labels.

        Args:
            canonical: Categorical value for the canonical form.
            isoform: Categorical value for the isoform.

        Returns:
            Dict with keys 'changed', 'canonical', 'isoform'.
        """
        return {
            "changed": canonical != isoform,
            "canonical": canonical,
            "isoform": isoform,
        }

    @staticmethod
    def compare_sets(canonical_set: set, isoform_set: set) -> dict:
        """Compare two sets and report gained, lost, and shared members.

        Args:
            canonical_set: Set of items for the canonical form.
            isoform_set: Set of items for the isoform.

        Returns:
            Dict with sorted lists for 'gained', 'lost', 'shared'.
        """
        return {
            "gained": sorted(isoform_set - canonical_set),
            "lost": sorted(canonical_set - isoform_set),
            "shared": sorted(canonical_set & isoform_set),
        }

    @staticmethod
    def compare_scalar_regions(unique_value: float, shared_value: float) -> dict:
        """Compare scalar measurements between unique and shared regions.

        Args:
            unique_value: Measurement in the isoform-unique region.
            shared_value: Measurement in the shared region.

        Returns:
            Dict with 'unique', 'shared', 'ratio', and 'enriched' keys.
            Ratio is inf when shared_value is zero and unique_value is nonzero.
        """
        if shared_value == 0:
            ratio = math.inf if unique_value != 0 else 0.0
        else:
            ratio = unique_value / shared_value
        return {
            "unique": unique_value,
            "shared": shared_value,
            "ratio": ratio,
            "enriched": ratio > 1.0,
        }

    @staticmethod
    def compare_structures(canonical_metrics: dict, isoform_metrics: dict) -> dict:
        """Compare structural prediction metrics between canonical and isoform.

        Args:
            canonical_metrics: Dict with at least optional 'plddt' key.
            isoform_metrics: Dict with optional 'plddt', 'tm_score',
                'rmsd_shared', 'extension_contacts' keys.

        Returns:
            Dict with 'plddt_delta', 'tm_score', 'rmsd_shared',
            'extension_contacts'.
        """
        canonical_plddt = canonical_metrics.get("plddt", 0)
        isoform_plddt = isoform_metrics.get("plddt", 0)
        return {
            "plddt_delta": isoform_plddt - canonical_plddt,
            "tm_score": isoform_metrics.get("tm_score"),
            "rmsd_shared": isoform_metrics.get("rmsd_shared"),
            "extension_contacts": isoform_metrics.get("extension_contacts", 0),
        }
