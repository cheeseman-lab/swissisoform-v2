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
        name: Stable identifier (``"E1_primate_conservation"``).
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


def _score(results: list[CriterionResult]) -> tuple[int, int]:
    """Return ``(true_count, evaluable_count)`` over a list of criterion results."""
    evaluable = [r for r in results if r.value is not None]
    return sum(1 for r in evaluable if r.value), len(evaluable)
