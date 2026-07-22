"""E5 — initiation efficiency. Plumbing: site.expression."""

from __future__ import annotations

from swissisoform.config import ScoringConfig
from swissisoform.evidence.common import CriterionResult
from swissisoform.models import TranslationInitiationSite


def score(site: TranslationInitiationSite, cfg: ScoringConfig) -> CriterionResult:
    """E5: max per-cell-line initiation efficiency exceeds threshold.

    Evaluates to ``None`` when no cell line carries an efficiency value —
    our upstream TIS counts don't always produce it.
    """
    efficiencies = [
        exp.initiation_efficiency
        for exp in site.expression.values()
        if exp.initiation_efficiency is not None
    ]
    if not efficiencies:
        return CriterionResult(
            "D2_initiation_efficiency", None, "no initiation_efficiency values"
        )
    best = max(efficiencies)
    passed = best >= cfg.initiation_efficiency_min
    return CriterionResult(
        "D2_initiation_efficiency",
        passed,
        f"max_efficiency={best:.3f} (threshold {cfg.initiation_efficiency_min})",
    )
