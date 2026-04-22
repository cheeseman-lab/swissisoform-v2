"""Module: Variant Intersection — genomic intersection of clinical hits.

Post-processes the variant hit list already attached to a TIS by
:class:`swissisoform.modules.clinical.ClinicalModule` and tags each hit
with whether its genomic position falls inside the isoform-unique ORF
region, the shared (canonical-overlapping) region, or outside the ORF
entirely. Requires :attr:`TIS.orf_exons` and
:attr:`TIS.canonical_orf_exons` (produced by the assembly layer's
Layer-2 walker).

This is a SiteModule because the decision depends on per-TIS exon
structure, which the purely protein-space ClinicalModule doesn't see.
Running it after clinical preserves clinical's simple protocol while
adding a first-class "does this variant sit in the isoform-unique
region?" signal for downstream scoring.

Outputs are aggregate counts (total / unique / shared / pathogenic-in-
unique) plus a copy of the hit list with intersection flags. The
original clinical output is left untouched.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from swissisoform.coords import interval_difference, interval_intersection
from swissisoform.models import TranslationInitiationSite

logger = logging.getLogger(__name__)


def _is_pathogenic(hit: dict[str, Any]) -> bool:
    """Return True when the hit carries a clinical_significance containing 'pathogenic'."""
    sig = hit.get("clinical_significance")
    return bool(sig) and "pathogenic" in str(sig).lower()


def _point_in_intervals(pos: int, intervals: list[tuple[int, int]]) -> bool:
    """True when *pos* lies in any half-open ``[start, end)`` interval."""
    for start, end in intervals:
        if start <= pos < end:
            return True
    return False


class VariantIntersectionModule:
    r"""Tag clinical hits by genomic membership in the isoform-unique ORF region.

    Runs after :class:`swissisoform.modules.clinical.ClinicalModule` in the
    pipeline (SiteModule stage). For each hit on ``site.isoform_annotations
    ['clinical'].hits`` that carries a ``genomic_pos`` field, it sets:

    - ``in_isoform_unique`` — genomic_pos ∈ (isoform ORF \ canonical ORF)
    - ``in_isoform_shared`` — genomic_pos ∈ (isoform ∩ canonical)
    - ``in_isoform`` — genomic_pos ∈ isoform ORF (= unique ∨ shared)

    and emits aggregate counts in ``summary``. Hits without a genomic
    position are passed through with ``None`` for every flag.

    Attributes:
        MODULE_NAME: ``"variant_intersection"``.
        OUTPUT_COLUMNS: Columns produced per TIS.
        SCOPE: ``"C"`` (per-candidate).
    """

    MODULE_NAME: str = "variant_intersection"
    OUTPUT_COLUMNS: list[str] = [
        "variant_intersection_hits",
        "variant_intersection_n_total",
        "variant_intersection_n_in_unique_region",
        "variant_intersection_n_in_shared_region",
        "variant_intersection_n_pathogenic_in_unique_region",
        "variant_intersection_n_pathogenic_in_shared_region",
        "variant_intersection_summary",
    ]
    SCOPE: str = "C"

    def annotate_site(self, site: TranslationInitiationSite) -> dict[str, Any]:
        """Tag clinical hits with genomic intersection flags.

        Args:
            site: A TIS whose ``isoform_annotations['clinical']`` has already
                been populated by :class:`ClinicalModule`. The TIS must also
                carry ``orf_exons`` and ``canonical_orf_exons`` from the
                assembly Layer-2 walker.

        Returns:
            Dict with ``hits`` (annotated copies) and aggregate ``summary``.
            When the skeleton is missing (``orf_exons`` empty) the module
            reports ``status="no_skeleton"`` and leaves every flag ``None``.
        """
        clinical = site.isoform_annotations.get("clinical") or {}
        raw_hits = clinical.get("hits") if isinstance(clinical, dict) else None
        if not isinstance(raw_hits, list):
            raw_hits = []

        if not site.orf_exons:
            return self._fallback(raw_hits, status="no_skeleton")

        unique = interval_difference(site.orf_exons, site.canonical_orf_exons)
        shared = interval_intersection(site.orf_exons, site.canonical_orf_exons)

        hits_out: list[dict[str, Any]] = []
        n_unique = 0
        n_shared = 0
        n_path_unique = 0
        n_path_shared = 0
        n_scored = 0

        for hit in raw_hits:
            tagged = deepcopy(hit)
            pos = hit.get("genomic_pos")
            if not isinstance(pos, int):
                tagged["in_isoform_unique"] = None
                tagged["in_isoform_shared"] = None
                tagged["in_isoform"] = None
                hits_out.append(tagged)
                continue

            in_unique = _point_in_intervals(pos, unique)
            in_shared = _point_in_intervals(pos, shared)
            tagged["in_isoform_unique"] = in_unique
            tagged["in_isoform_shared"] = in_shared
            tagged["in_isoform"] = in_unique or in_shared
            hits_out.append(tagged)

            n_scored += 1
            if in_unique:
                n_unique += 1
                if _is_pathogenic(hit):
                    n_path_unique += 1
            if in_shared:
                n_shared += 1
                if _is_pathogenic(hit):
                    n_path_shared += 1

        return {
            "hits": hits_out,
            "n_total": len(raw_hits),
            "n_in_unique_region": n_unique,
            "n_in_shared_region": n_shared,
            "n_pathogenic_in_unique_region": n_path_unique,
            "n_pathogenic_in_shared_region": n_path_shared,
            "summary": {
                "status": "ok",
                "n_scored": n_scored,
                "n_unscored": len(raw_hits) - n_scored,
                "unique_region_nt": sum(e - s for s, e in unique),
                "shared_region_nt": sum(e - s for s, e in shared),
            },
        }

    def run(
        self,
        tis_sites: list[TranslationInitiationSite],
    ) -> list[TranslationInitiationSite]:
        """Annotate every TIS and attach results to ``isoform_annotations``."""
        for site in tis_sites:
            site.isoform_annotations[self.MODULE_NAME] = self.annotate_site(site)
        return tis_sites

    @staticmethod
    def _fallback(
        raw_hits: list[dict[str, Any]], *, status: str
    ) -> dict[str, Any]:
        """Return a "no-intersection-computed" result with flags set to None."""
        hits_out: list[dict[str, Any]] = []
        for hit in raw_hits:
            tagged = deepcopy(hit)
            tagged["in_isoform_unique"] = None
            tagged["in_isoform_shared"] = None
            tagged["in_isoform"] = None
            hits_out.append(tagged)
        return {
            "hits": hits_out,
            "n_total": len(raw_hits),
            "n_in_unique_region": None,
            "n_in_shared_region": None,
            "n_pathogenic_in_unique_region": None,
            "n_pathogenic_in_shared_region": None,
            "summary": {
                "status": status,
                "n_scored": 0,
                "n_unscored": len(raw_hits),
            },
        }
