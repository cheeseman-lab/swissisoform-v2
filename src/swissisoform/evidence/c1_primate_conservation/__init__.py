"""C1 — primate conservation. Plumbing: swissisoform.conservation_frame."""

from __future__ import annotations

from swissisoform.config import ScoringConfig
from swissisoform.evidence.common import CriterionResult, _annotation, _status_ok
from swissisoform.models import TranslationInitiationSite


def score(site: TranslationInitiationSite, cfg: ScoringConfig) -> CriterionResult:
    """C1: primate amino-acid identity over the unique region exceeds threshold.

    Scored on ``primate_mean_pident`` (mean AA percent identity of the aligned
    primate orthologs), not ``frac_intact`` — pident measures sequence
    conservation directly, whereas frac_intact only counts species with an
    intact frame. ``frac_intact`` is still surfaced in the reason as context.
    """
    ann = _annotation(site, "conservation_frame")
    if not _status_ok(ann):
        return CriterionResult("C1_primate_conservation", None, "conservation_frame not run")
    val = ann.get("primate_mean_pident") if ann else None
    if val is None:
        return CriterionResult(
            "C1_primate_conservation", None, "primate_mean_pident unavailable"
        )
    passed = val >= cfg.c1_pident_min
    frac = ann.get("primate_frac_intact")
    frac_str = f"{frac:.2f}" if isinstance(frac, (int, float)) else "n/a"
    return CriterionResult(
        "C1_primate_conservation",
        passed,
        f"mean_pident={val:.2f} (threshold {cfg.c1_pident_min}); frac_intact={frac_str}",
    )
