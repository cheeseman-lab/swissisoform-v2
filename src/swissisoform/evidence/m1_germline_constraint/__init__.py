"""M1 — germline constraint. Plumbing: swissisoform.varianteffect + clinical."""

from __future__ import annotations

from swissisoform.config import ScoringConfig
from swissisoform.contract import NO_CANONICAL_BASELINE_ORFS
from swissisoform.evidence.common import CriterionResult, _annotation
from swissisoform.models import TranslationInitiationSite


def score(site: TranslationInitiationSite, cfg: ScoringConfig) -> CriterionResult:
    """M1: germline tolerance / constraint over the isoform-unique region.

    Two complementary, independent signals that the unique region is under
    selective constraint in healthy humans:

    - **ESM-C constraint delta** (``plm_vep.constraint_delta``, mean logP(wt)
      unique − shared) — the protein language model predicts the unique region's
      residues better than the shared core, i.e. finds them more conserved.
    - **gnomAD depletion** (``variant_intersection.gnomad_depletion_ratio``,
      per-nt density of common variation unique/shared) — germline variation
      AVOIDS the unique region (a depletion ratio below 1 means constraint).

    ``True`` when EITHER ``constraint_delta >= cfg.m1_constraint_delta_min``
    OR ``gnomad_depletion_ratio < cfg.m1_depletion_ratio_max``. ``None`` when the
    ORF type has no canonical-coding baseline (see below) or when BOTH inputs are
    missing/unevaluable; otherwise ``False`` when neither branch fires. This is
    germline tolerance/constraint — NOT a count of "damaging" variants.

    Both inputs are unique-vs-shared contrasts, so both need the unique region to
    be canonical coding sequence for the comparison to mean anything. On an
    extension or a separate ORF it never was: the gnomAD ratio then counts
    variation in never-coding nucleotides (shaped by UTR/splicing selection and
    coverage), and the ESM-C contrast measures a never-evolved sequence against a
    conserved core. Both fire for reasons unrelated to constraint — on
    cheeseman_test every extension scored 6x-412x — so the criterion abstains
    rather than contributing a confounded True to the functional score. The raw
    values stay in the output columns and stay queryable by the M tool loop.
    """
    if site.orf_type in NO_CANONICAL_BASELINE_ORFS:
        return CriterionResult(
            "M1_pathogenic_variant_enrichment",
            None,
            (
                f"not evaluable for orf_type={site.orf_type.value}: the unique region "
                "was never canonical coding sequence, so a unique-vs-shared "
                "constraint contrast has no baseline"
            ),
        )

    plm = _annotation(site, "plm_vep")
    vi = _annotation(site, "variant_intersection")

    constraint = plm.get("constraint_delta") if isinstance(plm, dict) else None
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
            "no plm_vep constraint_delta or gnomad_depletion_ratio",
        )

    constrained = constraint is not None and constraint >= cfg.m1_constraint_delta_min
    depleted = depletion is not None and depletion < cfg.m1_depletion_ratio_max
    passed = constrained or depleted

    constraint_str = f"{constraint:.2f}" if constraint is not None else "n/a"
    depletion_str = f"{depletion:.2f}" if depletion is not None else "n/a"
    return CriterionResult(
        "M1_pathogenic_variant_enrichment",
        passed,
        (
            f"constraint_delta={constraint_str} "
            f"(≥{cfg.m1_constraint_delta_min}); "
            f"gnomad_depletion_ratio={depletion_str} "
            f"(<{cfg.m1_depletion_ratio_max})"
        ),
    )
