"""S3 — presence of differential SAE features (isoform vs canonical). Plumbing: plm.sae_module."""

from __future__ import annotations

from swissisoform.config import ScoringConfig
from swissisoform.evidence.common import CriterionResult, _annotation
from swissisoform.models import TranslationInitiationSite

__all__ = ["score"]


def score(
    site: TranslationInitiationSite, cfg: ScoringConfig  # noqa: ARG001
) -> CriterionResult:
    """S3: are there any differential interpretable SAE features at all?

    Reads ``site.isoform_annotations['sae']`` written by ``SAEFeatureModule``.
    SAE features are a descriptive interpretability signal, so the honest scored
    claim is not "how much changed" (the LLM weighs that from the feature list)
    but simply whether the isoform uses interpretable features the canonical does
    not, or vice versa. It is a presence check with no magnitude threshold:

    * ``True``  — at least one feature is categorically gained or lost
      (``n_isoform_only + n_canonical_only > 0``).
    * ``False`` — zero differential features (the two proteins are identical in
      feature space).
    * ``None``  — the SAE step did not run (no cache / ``status != 'ok'``), or the
      gained/lost counts are unavailable — so an unrun stage never reads as
      evidence-absent.
    """
    ann = _annotation(site, "sae")
    if ann is None:
        return CriterionResult("S3_sae", None, "sae annotation missing")
    status = ann.get("status")
    if status != "ok":
        return CriterionResult("S3_sae", None, f"sae status={status}")
    iso_only = ann.get("n_isoform_only")
    can_only = ann.get("n_canonical_only")
    if not isinstance(iso_only, (int, float)) or not isinstance(can_only, (int, float)):
        return CriterionResult("S3_sae", None, "sae gained/lost feature counts unavailable")
    n_gained, n_lost = int(iso_only), int(can_only)
    n_diff = n_gained + n_lost
    return CriterionResult(
        "S3_sae",
        n_diff > 0,
        f"differential SAE features: {n_gained} gained + {n_lost} lost = {n_diff}",
    )
