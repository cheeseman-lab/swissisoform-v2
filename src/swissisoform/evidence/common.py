"""Shared types and helpers for the evidence-scoring buckets.

``CriterionResult`` (per-criterion verdict), the ``Criterion`` callable type
alias, and the small annotation-reading helpers used by every bucket live here
so each ``score`` function imports them from one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from swissisoform.config import ScoringConfig
from swissisoform.models import TranslationInitiationSite


@dataclass
class CriterionResult:
    """Per-criterion result.

    Attributes:
        name: Stable identifier (``"C1_primate_conservation"``).
        value: ``True`` (evidence present), ``False`` (evidence absent),
            or ``None`` (cannot evaluate — upstream data missing).
        reason: Short free-text explanation for humans and audits.
    """

    name: str
    value: bool | None
    reason: str


# A criterion is a pure function of ``(site, cfg)``. We pass ``cfg`` even
# to criteria that currently don't read it so the signature stays uniform
# and future thresholds can be added without API churn.
Criterion = Callable[[TranslationInitiationSite, ScoringConfig], CriterionResult]


def _annotation(site: TranslationInitiationSite, name: str) -> dict[str, Any] | None:
    """Return ``site.isoform_annotations[name]`` if it's a dict, else ``None``."""
    ann = site.isoform_annotations.get(name)
    return ann if isinstance(ann, dict) else None


def _status_ok(ann: dict[str, Any] | None) -> bool:
    """True when the annotation has ``summary.status == 'ok'``.

    Used to gate criteria so an unrun module doesn't masquerade as
    evidence-absent (which would bias the score toward False).
    """
    if ann is None:
        return False
    summary = ann.get("summary")
    if not isinstance(summary, dict):
        return True  # no status field → assume annotation is valid
    return summary.get("status", "ok") == "ok"


# A predictor that is not part of this run, as opposed to one that ran and could
# not be assessed. See :func:`predictor_change_state`.
NOT_IN_RUN = object()


def predictor_change_state(cmp: dict[str, Any] | None, ran_field: str) -> object:
    """Reduce one predictor's comparison to a single change state.

    Shared by L1 (DeepLoc) and L2 (SignalP, TargetP). Returns:

    - ``True`` when any ``*_changed`` flag is ``True``;
    - ``None`` when every flag is ``None``: the predictor ran but missed a side,
      so a gain there is still possible;
    - ``False`` otherwise (at least one flag evaluated, none changed);
    - :data:`NOT_IN_RUN` when the predictor is not part of this run — no
      comparison at all (``--skip``), or *ran_field* empty on **both** sides (it
      is wired in but produced nothing: not installed, or its precompute failed,
      which only logs a warning). Treating that as unassessed would turn every
      other predictor's confident "no" into "unknown".
    """
    if not isinstance(cmp, dict):
        return NOT_IN_RUN
    flags = [v for k, v in cmp.items() if k.endswith("_changed")]
    if not flags:
        return NOT_IN_RUN
    ran_cols = (f"{ran_field}_canonical", f"{ran_field}_isoform")
    if all(c in cmp for c in ran_cols) and all(_missing(cmp[c]) for c in ran_cols):
        return NOT_IN_RUN
    if any(v is True for v in flags):
        return True
    if all(v is None for v in flags):
        return None
    return False


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and value != value)


def _score(results: list[CriterionResult]) -> tuple[int, int]:
    """Return ``(true_count, evaluable_count)`` over a list of criterion results."""
    evaluable = [r for r in results if r.value is not None]
    return sum(1 for r in evaluable if r.value), len(evaluable)
