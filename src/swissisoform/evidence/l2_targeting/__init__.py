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
    written by the comparator (Scope A). ``True`` when either reports a
    category change; ``False`` only when both were evaluated and neither did;
    ``None`` when either could not be evaluated (no comparison, or every flag
    ``None``) and the other flags nothing — a gain on the unassessed side is
    still possible, so that is not a confident "no".
    """
    sp_cmp = site.comparison.get("signalp")
    tp_cmp = site.comparison.get("targetp")

    def _any_changed(cmp: dict[str, Any] | None) -> bool | None:
        if not isinstance(cmp, dict):
            return None
        flags = [cmp.get(k) for k in cmp if k.endswith("_changed")]
        if not flags or all(v is None for v in flags):
            # No flag, or none evaluable (the predictor did not run on a side).
            return None
        return any(v is True for v in flags)

    sp_state = _any_changed(sp_cmp)
    tp_state = _any_changed(tp_cmp)

    if sp_state is None and tp_state is None:
        return CriterionResult(
            "L2_targeting_change", None, "signalp/targetp comparisons not available"
        )
    if sp_state is not True and tp_state is not True and (sp_state is None or tp_state is None):
        # One predictor saw no change, the other could not be assessed. A gain on
        # the unassessed side is still possible, so this is not a confident "no".
        missing = "signalp" if sp_state is None else "targetp"
        return CriterionResult(
            "L2_targeting_change", None, f"no change flagged, but {missing} not evaluable"
        )
    if sp_state is True or tp_state is True:
        hits = []
        if sp_state is True:
            hits.append("signalp")
        if tp_state is True:
            hits.append("targetp")
        return CriterionResult("L2_targeting_change", True, f"changed in: {','.join(hits)}")
    return CriterionResult("L2_targeting_change", False, "no targeting change flagged")
