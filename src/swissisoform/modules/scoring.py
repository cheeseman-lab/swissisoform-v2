"""Module: Evidence Scoring — CDLMPS categories, dual-axis roll-up.

Fifteen criteria per TIS, grouped into the six CDLMPS categories and derived
from the annotations other modules have already attached. The two-axis
roll-up is kept for back-compat: existence = Conservation + Detection,
functional = Localization + Mutation + Predicted-structure + Structural.

- **Existence (C + D)** — is this isoform a real biological entity?
- **Functional impact (L + M + P + S)** — does it change protein function?

Each criterion is a small function returning ``(value, reason)`` where
``value`` is ``True`` / ``False`` / ``None``. ``None`` means "cannot
evaluate" — the upstream annotation module didn't run or didn't produce
the field we need. This is surfaced via ``existence_evaluable`` /
``functional_evaluable`` counts so downstream code can tell a low score
driven by missing data apart from one driven by genuine non-evidence.

Criteria that depend on modules whose caches are not yet populated
(structure, PLM VEP) return ``None`` at evaluation time based on the
annotation status.

The 15 criterion ``score`` functions and their shared types/helpers live in
the ``swissisoform.evidence`` package — one folder per bucket, registered by
category in ``CATEGORY_CRITERIA``. They are imported here so
``EvidenceScoringModule`` keeps its public home in
``swissisoform.modules.scoring``.

Criteria
--------

Conservation (C):
    C1 primate amino-acid identity over the unique region
       (``conservation_frame.primate_mean_pident``)
    C2 mammalian amino-acid identity over the unique region
       (``conservation_frame.mammalian_mean_pident``)
    C3 absolute PhyloP unique-region mean above coding-selection threshold
       (``conservation``)

Detection (D):
    D1 Multi-cell-line support (``site.expression``)
    D2 Ribosome initiation efficiency threshold (``site.expression``)
    D3 Mass-spec validation — PepQuery2-validated unique peptide hits
       in public MS spectra (``massspec``).

Localization (L):
    L1 Localization features changed (``comparison['localization']``)
    L2 Targeting change (``comparison['signalp']`` / ``comparison['targetp']``)

Mutation Landscape (M):
    M1 Germline tolerance / constraint over the unique region
       (ESM-C ``plm_vep`` constraint delta OR gnomAD depletion via
       ``variant_intersection``; not evaluable without a canonical baseline)
    M2 Disease-variant density enrichment in unique region vs shared core
       (``variant_intersection.disease_enrichment_ratio``)

Predicted Structure (P):
    P1 Structured differential region — diff-region pLDDT above threshold
       (``structure``); folding only.
    P2 Shared-region structural change — retained region folds differently
       (``structure`` shared-region Cα RMSD, pLDDT-gated)

Structural Characteristics (S):
    S1 Real InterPro domain gained / lost in the diff region
       (``comparison['interproscan']``)
    S2 Whole-protein biophysical shift — isoform-vs-canonical whole-protein
       gravy / fraction-charged / disorder deltas (``comparison['biophysics']``)
    S3 Interpretable SAE features firing in the unique region
       (``isoform_annotations['sae']``)
"""

from __future__ import annotations

from typing import Any

from swissisoform.config import PipelineConfig, ScoringConfig
from swissisoform.evidence import (
    CATEGORY_CRITERIA,
    EXISTENCE_CRITERIA,
    FUNCTIONAL_CRITERIA,
    Criterion,
    CriterionResult,
)
from swissisoform.evidence import (
    c1_primate_conservation as _c1,
)
from swissisoform.evidence import (
    c2_mammalian_conservation as _c2,
)
from swissisoform.evidence import (
    c3_phylop_selection as _c3,
)
from swissisoform.evidence import (
    d1_reproducibility as _d1,
)
from swissisoform.evidence import (
    d2_initiation_efficiency as _d2,
)
from swissisoform.evidence import (
    d3_mass_spec as _d3,
)
from swissisoform.evidence import (
    l1_localization as _l1,
)
from swissisoform.evidence import (
    l2_targeting as _l2,
)
from swissisoform.evidence import (
    m1_germline_constraint as _m1,
)
from swissisoform.evidence import (
    m2_disease_enrichment as _m2,
)
from swissisoform.evidence import (
    p1_structure as _p1,
)
from swissisoform.evidence import (
    p2_shared_rmsd as _p2,
)
from swissisoform.evidence import (
    s1_domains as _s1,
)
from swissisoform.evidence import (
    s2_biophysics as _s2,
)
from swissisoform.evidence import (
    s3_sae as _s3,
)
from swissisoform.evidence.common import _score
from swissisoform.models import Gene, TranslationInitiationSite

# Backward-compatible aliases — the original private criterion functions are
# now ``score`` entry points in their bucket packages. Kept so existing
# imports (tests, ad-hoc callers) keep working under the CDLMPS ids.
_c1_primate_conservation = _c1.score
_c2_mammalian_conservation = _c2.score
_c3_phylop_coding_selection = _c3.score
_d1_multi_cell_line = _d1.score
_d2_initiation_efficiency = _d2.score
_d3_mass_spec = _d3.score
_l1_localization_change = _l1.score
_l2_targeting_change = _l2.score
_m1_pathogenic_variant_enrichment = _m1.score
_m2_clinical_variant_overlap = _m2.score
_p1_structured_extension = _p1.score
_p2_shared_structural_change = _p2.score
_s1_domain_change = _s1.score
_s2_biophysics = _s2.score
_s3_sae = _s3.score

__all__ = [
    "CATEGORY_CRITERIA",
    "EXISTENCE_CRITERIA",
    "FUNCTIONAL_CRITERIA",
    "Criterion",
    "CriterionResult",
    "EvidenceScoringModule",
]


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class EvidenceScoringModule:
    """Dual-axis evidence scoring over TIS annotations.

    **Run order is load-bearing.**  Several criteria (L1
    ``localization_change`` and S2 ``biophysics`` today; any future Scope-A consumer)
    read ``site.comparison``, which is populated by
    ``swissisoform.compare.comparator.compare_genes``.  This module
    must therefore run **after** ``compare_genes``, i.e. NOT inside
    ``AnnotationPipeline.site_modules``.  Use it as a standalone step:

    .. code-block:: python

        pipeline.run(genes)
        compare_genes(genes, scope_a_modules=[BiophysicsModule(cfg)])
        scoring_mod = EvidenceScoringModule(cfg)
        scoring_mod.run([s for g in genes for s in g.tis_sites])

    Reads from ``site.isoform_annotations`` and ``site.comparison``,
    evaluates 15 criteria (6 existence [C+D] + 9 functional [L+M+P+S]) and
    writes them onto ``site.isoform_annotations["scoring"]``.

    Attributes:
        MODULE_NAME: ``"scoring"``.
        OUTPUT_COLUMNS: Column names produced per TIS.
        SCOPE: ``"C"`` (per-candidate).
    """

    MODULE_NAME: str = "scoring"
    OUTPUT_COLUMNS: list[str] = [
        "scoring_existence_score",
        "scoring_existence_evaluable",
        "scoring_existence_high_confidence",
        "scoring_functional_score",
        "scoring_functional_evaluable",
        "scoring_functional_high_confidence",
        "scoring_criteria",
        "scoring_reasons",
        "scoring_summary",
    ]
    SCOPE: str = "C"

    def __init__(
        self,
        config: PipelineConfig,
        *,
        existence_criteria: list[Criterion] | None = None,
        functional_criteria: list[Criterion] | None = None,
    ) -> None:
        """Resolve ScoringConfig from *config*; optionally override criterion lists.

        Custom criterion lists let downstream callers run a subset for
        debugging or swap a stub for a real implementation without
        touching module internals.
        """
        self._scoring = config.scoring or ScoringConfig()
        self._existence = existence_criteria or EXISTENCE_CRITERIA
        self._functional = functional_criteria or FUNCTIONAL_CRITERIA

    # ------------------------------------------------------------------
    # SiteModule protocol
    # ------------------------------------------------------------------

    def annotate_site(self, site: TranslationInitiationSite) -> dict[str, Any]:
        """Evaluate every criterion and roll up two scores for *site*.

        Returns:
            Dict with per-axis score + evaluable count, a
            high-confidence boolean, the full criterion map, and a
            summary block. Criteria that can't be evaluated contribute
            ``None`` to the criterion map and are excluded from
            ``*_score`` / ``*_evaluable``.
        """
        existence = [crit(site, self._scoring) for crit in self._existence]
        functional = [crit(site, self._scoring) for crit in self._functional]

        existence_score, existence_evaluable = _score(existence)
        functional_score, functional_evaluable = _score(functional)

        criteria_map = {r.name: r.value for r in existence + functional}
        reasons_map = {r.name: r.reason for r in existence + functional}

        return {
            "existence_score": existence_score,
            "existence_evaluable": existence_evaluable,
            "existence_high_confidence": (
                existence_score >= self._scoring.existence_high_threshold
                and existence_evaluable >= self._scoring.existence_high_threshold
            ),
            "functional_score": functional_score,
            "functional_evaluable": functional_evaluable,
            "functional_high_confidence": (
                functional_score >= self._scoring.functional_high_threshold
                and functional_evaluable >= self._scoring.functional_high_threshold
            ),
            "criteria": criteria_map,
            "reasons": reasons_map,
            "summary": {
                "n_criteria_total": len(existence) + len(functional),
                "n_criteria_evaluable": existence_evaluable + functional_evaluable,
                "existence_axis_complete": existence_evaluable == len(self._existence),
                "functional_axis_complete": functional_evaluable == len(self._functional),
            },
        }

    def run(
        self,
        tis_sites: list[TranslationInitiationSite],
    ) -> list[TranslationInitiationSite]:
        """Annotate every TIS and attach to ``isoform_annotations``."""
        for site in tis_sites:
            site.isoform_annotations[self.MODULE_NAME] = self.annotate_site(site)
        return tis_sites

    # Gene-level adapter — the pipeline also dispatches ``annotate_gene``
    # for gene_modules. We expose it so scoring can optionally appear in
    # the gene_modules slot and run per-TIS internally.
    def annotate_gene(self, gene: Gene) -> dict[str, Any]:
        """Run per-TIS scoring over the gene and return a per-TIS map."""
        return {tis.tis_id: self.annotate_site(tis) for tis in gene.tis_sites}
