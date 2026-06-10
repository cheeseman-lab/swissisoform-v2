"""E2 — mammalian conservation. Plumbing: swissisoform.conservation_frame."""

from __future__ import annotations

from swissisoform.config import ScoringConfig
from swissisoform.evidence.common import CriterionResult, _annotation, _status_ok
from swissisoform.models import TranslationInitiationSite


def score(site: TranslationInitiationSite, cfg: ScoringConfig) -> CriterionResult:
    """E2: mammalian amino-acid identity over the unique region exceeds threshold.

    Scored on ``mammalian_mean_pident`` (mean AA percent identity across the
    aligned mammalian orthologs), not ``frac_intact``. ``frac_intact`` is kept
    in the reason as context.
    """
    ann = _annotation(site, "conservation_frame")
    if not _status_ok(ann):
        return CriterionResult(
            "E2_mammalian_conservation", None, "conservation_frame not run"
        )
    val = ann.get("mammalian_mean_pident") if ann else None
    if val is None:
        return CriterionResult(
            "E2_mammalian_conservation", None, "mammalian_mean_pident unavailable"
        )
    passed = val >= cfg.e2_pident_min
    frac = ann.get("mammalian_frac_intact")
    frac_str = f"{frac:.2f}" if isinstance(frac, (int, float)) else "n/a"
    return CriterionResult(
        "E2_mammalian_conservation",
        passed,
        f"mean_pident={val:.2f} (threshold {cfg.e2_pident_min}); frac_intact={frac_str}",
    )
