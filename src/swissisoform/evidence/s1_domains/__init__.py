"""S1 — domain gain/loss. Plumbing: swissisoform.interproscan."""

from __future__ import annotations

from swissisoform.config import ScoringConfig
from swissisoform.evidence.common import CriterionResult
from swissisoform.evidence.s1_domains.interproscan import (
    InterProScanModule,
    is_real_domain,
    is_real_functional_domain,
    precompute_interproscan,
)
from swissisoform.models import TranslationInitiationSite

__all__ = [
    "score",
    "InterProScanModule",
    "precompute_interproscan",
    "is_real_domain",
    "is_real_functional_domain",
]


def score(
    site: TranslationInitiationSite, cfg: ScoringConfig  # noqa: ARG001
) -> CriterionResult:
    """S1: domain gain/loss — a REAL InterPro domain is gained or lost in the diff region.

    Symmetric across ORF types — the metric asks "is a functional domain
    gained or lost in the differential region?" and the interpretation flips
    by direction:

    * extension/uORF/altORF — diff region is added; a real domain starting
      there and absent from the canonical hit set is a GAINED domain.
    * truncation — diff region is lost; a real domain starting there and
      absent from the isoform hit set is a LOST canonical domain.

    Scored on ``cmp_interproscan_n_real_domains_changed_in_diff_region`` — the
    count the comparator emits of *filtered* real-InterPro domains (those with
    a real ``interpro_id`` and a non-disorder/non-structural database) that
    start in the diff region AND are absent from the other form's hit set
    (genuinely gained or lost, not merely repositioned).

    Returns ``None`` when:

    - the comparator data is missing (precompute not run), or
    - the source pane's InterProScan ``summary.status`` (surfaced by the
      comparator as ``hits_canonical_status``) is not ``ok`` — the
      canonical hit list we count domains from didn't actually complete, or
    - the comparator did not emit the real-domain count.
    """
    cmp = site.comparison.get("interproscan")
    if not isinstance(cmp, dict):
        return CriterionResult(
            "S1_domain_change", None, "interproscan comparison missing"
        )
    canonical_status = cmp.get("hits_canonical_status")
    if canonical_status != "ok":
        return CriterionResult(
            "S1_domain_change", None, f"interproscan status={canonical_status}"
        )
    n = cmp.get("n_real_domains_changed_in_diff_region")
    if n is None:
        return CriterionResult(
            "S1_domain_change",
            None,
            "n_real_domains_changed_in_diff_region unavailable",
        )
    try:
        n = int(n)
    except (TypeError, ValueError):
        return CriterionResult(
            "S1_domain_change",
            None,
            f"n_real_domains_changed_in_diff_region not numeric ({n!r})",
        )
    return CriterionResult(
        "S1_domain_change",
        n >= 1,
        f"n_real_domains_changed_in_diff_region={n}",
    )
