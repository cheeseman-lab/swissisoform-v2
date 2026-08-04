"""L2 — targeting change. Plumbing: swissisoform.signalp / swissisoform.targetp."""

from __future__ import annotations

from typing import Any

from swissisoform.config import ScoringConfig
from swissisoform.evidence.common import CriterionResult
from swissisoform.evidence.l2_targeting.signalp import SignalPModule, precompute_signalp
from swissisoform.evidence.l2_targeting.targetp import TargetPModule, precompute_targetp
from swissisoform.models import TranslationInitiationSite

__all__ = [
    "score",
    "SignalPModule",
    "precompute_signalp",
    "TargetPModule",
    "precompute_targetp",
]


def score(
    site: TranslationInitiationSite, cfg: ScoringConfig  # noqa: ARG001
) -> CriterionResult:
    """L2: targeting change — SignalP/TargetP disagree on canonical vs. isoform.

    Reads from ``site.comparison['signalp']`` / ``site.comparison['targetp']``
    written by the comparator (Scope A).  Returns ``None`` when neither
    module has produced a comparison (precompute not run), ``False`` when
    both ran but neither reports a category change, ``True`` when either
    does.
    """
    sp_cmp = site.comparison.get("signalp")
    tp_cmp = site.comparison.get("targetp")

    def _any_changed(cmp: dict[str, Any] | None) -> bool | None:
        if not isinstance(cmp, dict):
            return None
        changed = [k for k in cmp if k.endswith("_changed")]
        if not changed:
            return None
        return any(cmp.get(k) is True for k in changed)

    sp_state = _any_changed(sp_cmp)
    tp_state = _any_changed(tp_cmp)

    if sp_state is None and tp_state is None:
        return CriterionResult(
            "L2_targeting_change", None, "signalp/targetp comparisons not available"
        )
    if sp_state is True or tp_state is True:
        hits = []
        if sp_state is True:
            hits.append("signalp")
        if tp_state is True:
            hits.append("targetp")
        return CriterionResult("L2_targeting_change", True, f"changed in: {','.join(hits)}")
    return CriterionResult("L2_targeting_change", False, "no targeting change flagged")
