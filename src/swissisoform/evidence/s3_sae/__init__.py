"""S3 — SAE interpretable features in the differential region. Plumbing: plm.sae_module."""

from __future__ import annotations

from swissisoform.config import ScoringConfig
from swissisoform.evidence.common import CriterionResult, _annotation
from swissisoform.models import TranslationInitiationSite

__all__ = ["score"]


def score(site: TranslationInitiationSite, cfg: ScoringConfig) -> CriterionResult:
    """S3: interpretable SAE features firing in the isoform-unique region.

    Reads ``site.isoform_annotations['sae']`` written by ``SAEFeatureModule``,
    which counts sparse-autoencoder features (ESM-C residual stream, top-K)
    that are differentially active in the isoform-unique region versus the
    shared core. The count (``n_unique_region_features``) is label-agnostic —
    the pipeline already applies a prevalence >= 2 filter — so a non-zero count
    means the differential region carries interpretable structure the core
    does not.

    ``True`` when ``n_unique_region_features >= cfg.s3_sae_min_unique_features``.
    Returns ``None`` when the SAE annotation is missing, its ``status`` is not
    ``ok`` (no cache / not run), or the count is unavailable — so an unrun SAE
    stage never masquerades as evidence-absent.

    Threshold is PROVISIONAL — set in threshold discussion.
    """
    ann = _annotation(site, "sae")
    if ann is None:
        return CriterionResult("S3_sae", None, "sae annotation missing")
    status = ann.get("status")
    if status != "ok":
        return CriterionResult("S3_sae", None, f"sae status={status}")
    n = ann.get("n_unique_region_features")
    if n is None:
        return CriterionResult("S3_sae", None, "n_unique_region_features unavailable")
    try:
        n = int(n)
    except (TypeError, ValueError):
        return CriterionResult("S3_sae", None, f"n_unique_region_features not numeric ({n!r})")
    return CriterionResult(
        "S3_sae",
        n >= cfg.s3_sae_min_unique_features,
        f"n_unique_region_features={n} (threshold {cfg.s3_sae_min_unique_features})",
    )
