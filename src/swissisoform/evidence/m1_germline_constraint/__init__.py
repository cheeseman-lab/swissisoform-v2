"""F5 — germline constraint. Plumbing: swissisoform.varianteffect + clinical."""

from __future__ import annotations

from swissisoform.config import ScoringConfig
from swissisoform.evidence.common import CriterionResult, _annotation
from swissisoform.models import TranslationInitiationSite


def score(site: TranslationInitiationSite, cfg: ScoringConfig) -> CriterionResult:
    """F5: germline tolerance / constraint over the isoform-unique region.

    Two complementary, independent signals that the unique region is under
    selective constraint in healthy humans:

    - **ESM-C constraint enrichment** (``plm_vep.constraint_enrichment``,
      unique/shared mean LLR ratio) — the protein language model finds the
      unique region's residues more constrained than the shared core.
    - **gnomAD depletion** (``variant_intersection.gnomad_depletion_ratio``,
      per-nt density of common variation unique/shared) — germline variation
      AVOIDS the unique region (a depletion ratio below 1 means constraint).

    ``True`` when EITHER ``constraint_enrichment >= cfg.f5_constraint_enrichment_min``
    OR ``gnomad_depletion_ratio < cfg.f5_depletion_ratio_max``. ``None`` only
    when BOTH inputs are missing/unevaluable; otherwise ``False`` when neither
    branch fires. This is germline tolerance/constraint — NOT a count of
    "damaging" variants.
    """
    plm = _annotation(site, "plm_vep")
    vi = _annotation(site, "variant_intersection")

    constraint = plm.get("constraint_enrichment") if isinstance(plm, dict) else None
    if not isinstance(constraint, (int, float)) or (
        isinstance(plm, dict) and plm.get("status") != "ok"
    ):
        constraint = None

    depletion = vi.get("gnomad_depletion_ratio") if isinstance(vi, dict) else None
    if not isinstance(depletion, (int, float)):
        depletion = None

    # The criterion *name* stays "M1_pathogenic_variant_enrichment" for
    # backward compatibility with the viewer / grounding units that key on it;
    # the *intent* is germline tolerance / constraint (see docstring).
    if constraint is None and depletion is None:
        return CriterionResult(
            "M1_pathogenic_variant_enrichment",
            None,
            "no plm_vep constraint_enrichment or gnomad_depletion_ratio",
        )

    constrained = constraint is not None and constraint >= cfg.f5_constraint_enrichment_min
    depleted = depletion is not None and depletion < cfg.f5_depletion_ratio_max
    passed = constrained or depleted

    constraint_str = f"{constraint:.2f}" if constraint is not None else "n/a"
    depletion_str = f"{depletion:.2f}" if depletion is not None else "n/a"
    return CriterionResult(
        "M1_pathogenic_variant_enrichment",
        passed,
        (
            f"constraint_enrichment={constraint_str} "
            f"(≥{cfg.f5_constraint_enrichment_min}); "
            f"gnomad_depletion_ratio={depletion_str} "
            f"(<{cfg.f5_depletion_ratio_max})"
        ),
    )
