"""Derived tags — the escape hatch, and the reason it stays small.

Most tags are data: ``metric ⋈ cutoff``, a row in the frozen registry. Three
criteria are not, and no amount of registry schema makes them so:

``L1_localization_change`` / ``L2_targeting_change``
    Categorical. The scorers OR over whatever ``*_changed`` keys the comparator
    happened to emit, with a two-source ``None``-merge for L2 (undecidable only
    when *both* predictors are silent). The underlying quantities are compartment
    and targeting *labels* compared for inequality — there is no scalar to cut.
``P3_secondary_structure``
    ∃-over-a-list with a conjunction on a single element: some element in the
    differential region with ``length >= p3_min_sse_length`` **and** its own
    ``plddt_mean >= p3_min_sse_plddt``. ``max(length)`` loses the confidence test,
    ``max(plddt)`` loses the length test, and "longest among those passing pLDDT"
    is itself two thresholds rather than a cutoff.

Rather than reimplement any of that against the flat frame — where the null
semantics are the easy thing to get wrong, and getting them wrong is invisible —
a derived tag **calls the criterion function that already exists**. Those are pure
``(site, cfg) -> CriterionResult`` (``evidence/common.py:38``), so a derived tag's
value equals its criterion's value by construction, not by careful porting. The
cost is that derived tags need the ``TranslationInitiationSite`` objects, not just
the DataFrame.

Every criterion is listed, not only the three, so any criterion can be carried as
``derived`` — which is what makes moving one to a distribution-referenced cutoff a
one-row registry edit that is reversible.
"""

from __future__ import annotations

from swissisoform.config import ScoringConfig
from swissisoform.evidence import (
    c1,
    c2,
    c3,
    d1,
    d2,
    d3,
    l1,
    l2,
    m1,
    m2,
    p1,
    p2,
    p3,
    s1,
    s2,
    s3,
)
from swissisoform.evidence.common import Criterion, CriterionResult
from swissisoform.models import TranslationInitiationSite

# criterion_id -> the scorer that produces it. The ids are the strings the
# scorers put in ``CriterionResult.name``, which is not always derivable from the
# package name (M1's package is m1_germline_constraint; its id is
# M1_pathogenic_variant_enrichment, kept for back-compat). Verified by
# :func:`check_names`, which the registry builder runs.
SCORER_BY_CRITERION: dict[str, Criterion] = {
    "C1_primate_conservation": c1.score,
    "C2_mammalian_conservation": c2.score,
    "C3_phylop_coding_selection": c3.score,
    "D1_multi_cell_line": d1.score,
    "D2_initiation_efficiency": d2.score,
    "D3_mass_spec": d3.score,
    "L1_localization_change": l1.score,
    "L2_targeting_change": l2.score,
    "M1_pathogenic_variant_enrichment": m1.score,
    "M2_clinical_variant_overlap": m2.score,
    "P1_structured_extension": p1.score,
    "P2_shared_structural_change": p2.score,
    "P3_secondary_structure": p3.score,
    "S1_domain_change": s1.score,
    "S2_biophysics": s2.score,
    "S3_sae": s3.score,
}

# The three that must stay derived: no cutoff on any scalar expresses them.
IRREDUCIBLE: tuple[str, ...] = (
    "L1_localization_change",
    "L2_targeting_change",
    "P3_secondary_structure",
)


def score_criterion(
    criterion_id: str, site: TranslationInitiationSite, cfg: ScoringConfig
) -> CriterionResult:
    """Run one criterion's scorer on *site*.

    Raises:
        KeyError: ``criterion_id`` is not a known criterion — a registry naming a
            derived tag we cannot evaluate is a build error, not a missing value.
    """
    scorer = SCORER_BY_CRITERION[criterion_id]
    return scorer(site, cfg)


def check_names(site: TranslationInitiationSite, cfg: ScoringConfig) -> list[str]:
    """Return the keys whose scorer reports a different ``name``, on a probe site.

    The map above is hand-written, and a criterion rename would silently make a
    derived tag score the wrong thing. Running this against any site — including
    one with no annotations, since every scorer returns ``None`` with its own name
    rather than raising — turns that into a build-time failure.
    """
    bad: list[str] = []
    for criterion_id, scorer in SCORER_BY_CRITERION.items():
        if scorer(site, cfg).name != criterion_id:
            bad.append(criterion_id)
    return bad


__all__ = ["IRREDUCIBLE", "SCORER_BY_CRITERION", "check_names", "score_criterion"]
