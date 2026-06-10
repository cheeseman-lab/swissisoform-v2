"""F2 — localization change. Plumbing: swissisoform.localization (via site.comparison)."""

from __future__ import annotations

from swissisoform.config import ScoringConfig
from swissisoform.evidence.common import CriterionResult
from swissisoform.models import TranslationInitiationSite


def score(
    site: TranslationInitiationSite, cfg: ScoringConfig  # noqa: ARG001
) -> CriterionResult:
    """F2: isoform's localization features differ from canonical.

    Reads the DeepLoc comparator and ORs over its categorical change flags —
    prediction (top compartment), signals (sorting signals), and membrane
    association. The criterion is "localization features changed", not strictly
    "the predicted compartment changed", since a signals/membrane shift is
    functionally meaningful even without a top-compartment flip.
    """
    cmp = site.comparison.get("localization")
    if not isinstance(cmp, dict):
        return CriterionResult(
            "F2_localization_change", None, "localization comparison missing"
        )
    # The comparator emits ``{field}_changed`` keys for categorical shifts.
    changed_keys = [k for k in cmp if k.endswith("_changed") and cmp.get(k) is True]
    if not changed_keys:
        # Distinguish "comparator ran, no change" from "no comparator data"
        if any(k.endswith("_changed") for k in cmp):
            return CriterionResult(
                "F2_localization_change",
                False,
                "localization features unchanged (prediction/signals/membrane)",
            )
        return CriterionResult(
            "F2_localization_change", None, "no *_changed fields emitted"
        )
    return CriterionResult(
        "F2_localization_change",
        True,
        f"localization features changed (prediction/signals/membrane): "
        f"{','.join(sorted(changed_keys))}",
    )
