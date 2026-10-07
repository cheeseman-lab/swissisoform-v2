"""L2 — targeting change. Plumbing: swissisoform.signalp / swissisoform.targetp."""

from __future__ import annotations

from swissisoform.config import ScoringConfig
from swissisoform.evidence.common import NOT_IN_RUN, CriterionResult, predictor_change_state
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
    ``None`` when a predictor that is part of the run could not be evaluated
    (every flag ``None``: it missed a side) and the other flags nothing — a gain
    on the unassessed side is still possible, so that is not a confident "no".
    A predictor that is not part of the run (no comparison, or nothing produced
    on either side — see ``evidence.common.predictor_change_state``) is left out
    rather than counted as unassessed, so a run without TargetP still gives a
    confident ``False`` from SignalP alone.
    """
    states = {
        "signalp": predictor_change_state(site.comparison.get("signalp"), SignalPModule.RAN_FIELD),
        "targetp": predictor_change_state(site.comparison.get("targetp"), TargetPModule.RAN_FIELD),
    }
    present = {name: v for name, v in states.items() if v is not NOT_IN_RUN}
    if not present:
        return CriterionResult(
            "L2_targeting_change", None, "signalp/targetp comparisons not available"
        )
    hits = [name for name, v in present.items() if v is True]
    if hits:
        return CriterionResult("L2_targeting_change", True, f"changed in: {','.join(hits)}")
    unassessed = [name for name, v in present.items() if v is None]
    if unassessed:
        return CriterionResult(
            "L2_targeting_change",
            None,
            f"no change flagged, but {','.join(unassessed)} not evaluable",
        )
    return CriterionResult("L2_targeting_change", False, "no targeting change flagged")
