"""L1 — localization change. Plumbing: swissisoform.localization (via site.comparison)."""

from __future__ import annotations

from swissisoform.config import ScoringConfig
from swissisoform.evidence.common import NOT_IN_RUN, CriterionResult, predictor_change_state
from swissisoform.evidence.l1_localization.localization import (
    LocalizationModule,
    precompute_deeploc,
)
from swissisoform.models import TranslationInitiationSite

__all__ = [
    "score",
    "LocalizationModule",
    "precompute_deeploc",
]


def score(
    site: TranslationInitiationSite, cfg: ScoringConfig  # noqa: ARG001
) -> CriterionResult:
    """L1: isoform's localization features differ from canonical.

    Reads the DeepLoc comparator and ORs over its categorical change flags —
    prediction (top compartment), signals (sorting signals), and membrane
    association. The criterion is "localization features changed", not strictly
    "the predicted compartment changed", since a signals/membrane shift is
    functionally meaningful even without a top-compartment flip.
    """
    cmp = site.comparison.get("localization")
    state = predictor_change_state(cmp, LocalizationModule.RAN_FIELD)
    if state is NOT_IN_RUN:
        return CriterionResult("L1_localization_change", None, "localization comparison missing")
    if state is None:
        # DeepLoc ran but missed a side: nothing was evaluated, so this is not a
        # confident "unchanged".
        return CriterionResult("L1_localization_change", None, "no localization flag evaluable")
    if state is False:
        return CriterionResult(
            "L1_localization_change",
            False,
            "localization features unchanged (prediction/signals/membrane)",
        )
    changed_keys = [k for k in cmp if k.endswith("_changed") and cmp.get(k) is True]
    return CriterionResult(
        "L1_localization_change",
        True,
        f"localization features changed (prediction/signals/membrane): "
        f"{','.join(sorted(changed_keys))}",
    )
