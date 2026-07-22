"""F6 — disease enrichment. Plumbing: swissisoform.varianteffect + clinical."""

from __future__ import annotations

from swissisoform.config import ScoringConfig
from swissisoform.evidence.common import CriterionResult, _annotation, _status_ok
from swissisoform.models import TranslationInitiationSite


def score(site: TranslationInitiationSite, cfg: ScoringConfig) -> CriterionResult:
    """F6: disease variants concentrate in the isoform-unique region.

    Scored on the per-nt density enrichment of disease variants (ClinVar +
    COSMIC) in the unique region vs the shared core
    (``variant_intersection.disease_enrichment_ratio``). ``True`` when the
    ratio is >= ``cfg.f6_disease_enrichment_min`` (1.0 — disease variation is
    at least as dense in the unique region as the shared core). This replaces
    the old mere-presence test, which fired on a single variant regardless of
    the shared-core background. ``None`` when the ratio is missing (e.g. zero
    region length or no shared-region disease density). Raw unique/shared
    disease counts are reported as context.
    """
    ann = _annotation(site, "variant_intersection")
    if not _status_ok(ann):
        return CriterionResult(
            "M2_clinical_variant_overlap", None, "variant_intersection not run"
        )
    ratio = ann.get("disease_enrichment_ratio") if ann else None
    if not isinstance(ratio, (int, float)):
        return CriterionResult(
            "M2_clinical_variant_overlap", None, "disease_enrichment_ratio unavailable"
        )
    n_unique = ann.get("n_disease_in_unique_region")
    n_shared = ann.get("n_disease_in_shared_region")
    return CriterionResult(
        "M2_clinical_variant_overlap",
        ratio >= cfg.f6_disease_enrichment_min,
        f"disease_enrichment_ratio={ratio:.2f} "
        f"(≥{cfg.f6_disease_enrichment_min}); "
        f"disease unique={n_unique}, shared={n_shared}",
    )
