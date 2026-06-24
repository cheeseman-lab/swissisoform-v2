"""Module: Evidence Scoring — dual-axis existence + functional impact.

Two independent scores per TIS derived from the annotations other
modules have already attached:

- **Existence (E1–E6)** — is this isoform a real biological entity?
- **Functional impact (F1–F6)** — does it change protein function?

Each criterion is a small function returning ``(value, reason)`` where
``value`` is ``True`` / ``False`` / ``None``. ``None`` means "cannot
evaluate" — the upstream annotation module didn't run or didn't produce
the field we need. This is surfaced via ``existence_evaluable`` /
``functional_evaluable`` counts so downstream code can tell a low score
driven by missing data apart from one driven by genuine non-evidence.

Criteria that depend on modules whose caches are not yet populated
(structure, PLM VEP) return ``None`` at evaluation time based on the
annotation status.  F5 scores germline tolerance / constraint over the
isoform-unique region (ESM-C constraint enrichment + gnomAD depletion).

The 12 criterion ``score`` functions and their shared types/helpers live in
the ``swissisoform.evidence`` package — one folder per bucket. They are
imported here so ``EvidenceScoringModule`` keeps its public home in
``swissisoform.modules.scoring``.

Criteria
--------

Existence:
    E1 primate amino-acid identity over the unique region
       (``conservation_frame.primate_mean_pident``)
    E2 mammalian amino-acid identity over the unique region
       (``conservation_frame.mammalian_mean_pident``)
    E3 absolute PhyloP unique-region mean above coding-selection threshold
       (``conservation``)
    E4 Multi-cell-line support (``site.expression``)
    E5 Ribosome initiation efficiency threshold (``site.expression``)
    E6 Mass-spec validation — PepQuery2-validated unique peptide hits
       in public MS spectra (``massspec``).  Reports ``None`` until the
       PepQuery2 precompute lands.  In-silico tryptic
       detectability alone is NOT treated as evidence.

Functional impact:
    F1 Structured + biophysically-distinct differential region
       (``structure`` pLDDT AND ``comparison['biophysics']`` deltas)
    F2 Localization features changed (``comparison['localization']``)
    F3 Real InterPro domain gained / lost in the diff region
       (``comparison['interproscan']``)
    F4 Targeting change (``comparison['signalp']`` / ``comparison['targetp']``)
    F5 Germline tolerance / constraint over the unique region
       (ESM-C ``plm_vep`` constraint enrichment OR gnomAD depletion via
       ``variant_intersection``)
    F6 Disease-variant density enrichment in unique region vs shared core
       (``variant_intersection.disease_enrichment_ratio``)
"""

from __future__ import annotations

from typing import Any

from swissisoform.config import PipelineConfig, ScoringConfig
from swissisoform.evidence import (
    EXISTENCE_CRITERIA,
    FUNCTIONAL_CRITERIA,
    Criterion,
    CriterionResult,
)
from swissisoform.evidence import (
    e1_primate_conservation as _e1,
)
from swissisoform.evidence import (
    e2_mammalian_conservation as _e2,
)
from swissisoform.evidence import (
    e3_phylop_selection as _e3,
)
from swissisoform.evidence import (
    e4_reproducibility as _e4,
)
from swissisoform.evidence import (
    e5_initiation_efficiency as _e5,
)
from swissisoform.evidence import (
    e6_mass_spec as _e6,
)
from swissisoform.evidence import (
    f1_structure as _f1,
)
from swissisoform.evidence import (
    f2_localization as _f2,
)
from swissisoform.evidence import (
    f3_domains as _f3,
)
from swissisoform.evidence import (
    f4_targeting as _f4,
)
from swissisoform.evidence import (
    f5_germline_constraint as _f5,
)
from swissisoform.evidence import (
    f6_disease_enrichment as _f6,
)
from swissisoform.evidence.common import _score
from swissisoform.models import Gene, TranslationInitiationSite

# Backward-compatible aliases — the original private criterion functions are
# now ``score`` entry points in their bucket packages. Kept so existing
# imports (tests, ad-hoc callers) of ``_eN_*`` / ``_fN_*`` keep working.
_e1_primate_conservation = _e1.score
_e2_mammalian_conservation = _e2.score
_e3_phylop_coding_selection = _e3.score
_e4_multi_cell_line = _e4.score
_e5_initiation_efficiency = _e5.score
_e6_mass_spec = _e6.score
_f1_structured_extension = _f1.score
_f2_localization_change = _f2.score
_f3_domain_change = _f3.score
_f4_targeting_change = _f4.score
_f5_pathogenic_variant_enrichment = _f5.score
_f6_clinical_variant_overlap = _f6.score

__all__ = [
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

    **Run order is load-bearing.**  Several criteria (F2
    ``localization_change`` today; any future Scope-A consumer)
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
    evaluates 12 criteria (6 existence + 6 functional) and writes them onto
    ``site.isoform_annotations["scoring"]``.

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
