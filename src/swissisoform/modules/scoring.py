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
annotation status.  F5 scores per-variant damaging-effect evidence
(ESM-2 ΔLLR + AlphaMissense) over the isoform-unique region, read from
``varianteffect`` (built by ``VariantEffectModule``).

Criteria
--------

Existence:
    E1 primate reading-frame conservation (``conservation_frame``)
    E2 mammalian reading-frame conservation (``conservation_frame``)
    E3 PhyloP unique-region mean above coding-selection threshold
       (``conservation``)
    E4 Multi-cell-line support (``site.expression``)
    E5 Ribosome initiation efficiency threshold (``site.expression``)
    E6 Mass-spec validation — PepQuery2-validated unique peptide hits
       in public MS spectra (``massspec``).  Reports ``None`` until the
       PepQuery2 precompute lands (it has not).  In-silico tryptic
       detectability alone is NOT treated as evidence.

Functional impact:
    F1 Structured extension pLDDT (``structure``)
    F2 Localization change (``comparison['localization']``)
    F3 Domain gain / loss (``comparison['interproscan']``)
    F4 Targeting change (``comparison['signalp']`` / ``comparison['targetp']``)
    F5 Predicted-damaging variant enrichment in the unique region
       (per-variant ESM-2 ΔLLR + AlphaMissense; ``varianteffect``)
    F6 Clinical variant overlap in unique region
       (``variant_intersection``)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from swissisoform.config import PipelineConfig, ScoringConfig
from swissisoform.models import Gene, TranslationInitiationSite


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


# ---------------------------------------------------------------------------
# Existence criteria
# ---------------------------------------------------------------------------


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


def _e1_primate_conservation(
    site: TranslationInitiationSite, cfg: ScoringConfig
) -> CriterionResult:
    """E1: fraction of primate species with intact reading frame exceeds threshold."""
    ann = _annotation(site, "conservation_frame")
    if not _status_ok(ann):
        return CriterionResult("E1_primate_conservation", None, "conservation_frame not run")
    val = ann.get("primate_frac_intact") if ann else None
    if val is None:
        return CriterionResult("E1_primate_conservation", None, "primate_frac_intact unavailable")
    passed = val >= cfg.primate_frac_intact_min
    return CriterionResult(
        "E1_primate_conservation",
        passed,
        f"frac_intact={val:.2f} (threshold {cfg.primate_frac_intact_min})",
    )


def _e2_mammalian_conservation(
    site: TranslationInitiationSite, cfg: ScoringConfig
) -> CriterionResult:
    """E2: fraction of mammalian species with intact reading frame exceeds threshold."""
    ann = _annotation(site, "conservation_frame")
    if not _status_ok(ann):
        return CriterionResult(
            "E2_mammalian_conservation", None, "conservation_frame not run"
        )
    val = ann.get("mammalian_frac_intact") if ann else None
    if val is None:
        return CriterionResult(
            "E2_mammalian_conservation", None, "mammalian_frac_intact unavailable"
        )
    passed = val >= cfg.mammalian_frac_intact_min
    return CriterionResult(
        "E2_mammalian_conservation",
        passed,
        f"frac_intact={val:.2f} (threshold {cfg.mammalian_frac_intact_min})",
    )


def _e3_phylop_coding_selection(
    site: TranslationInitiationSite, cfg: ScoringConfig
) -> CriterionResult:
    """E3: PhyloP mean over the unique region exceeds coding-selection threshold."""
    ann = _annotation(site, "conservation")
    if ann is None:
        return CriterionResult("E3_phylop_coding_selection", None, "conservation not run")
    summary = ann.get("summary") if isinstance(ann, dict) else None
    status = summary.get("region_status") if isinstance(summary, dict) else None
    if status != "ok":
        return CriterionResult(
            "E3_phylop_coding_selection", None, f"region_status={status}"
        )
    val = ann.get("phylop_unique_region_mean")
    if val is None:
        return CriterionResult(
            "E3_phylop_coding_selection", None, "phylop_unique_region_mean unavailable"
        )
    passed = val >= cfg.phylop_coding_min
    return CriterionResult(
        "E3_phylop_coding_selection",
        passed,
        f"phylop={val:.2f} (threshold {cfg.phylop_coding_min})",
    )


def _e4_multi_cell_line(
    site: TranslationInitiationSite, cfg: ScoringConfig
) -> CriterionResult:
    """E4: TIS detected in at least ``min_cell_lines`` cell lines.

    Sensitivity note: with ``cfg.min_cell_lines == 1`` (used by single-sample
    diagnostic presets) this criterion is degenerate — every TIS in the
    dataset is detected in ≥1 cell line by construction, so E4 returns True
    universally and contributes a constant +1 to every existence score. For
    multi-sample / production runs raise it to 2-3 (the default is 3) so it
    actually discriminates.
    """
    n = len(site.expression)
    passed = n >= cfg.min_cell_lines
    return CriterionResult(
        "E4_multi_cell_line",
        passed,
        f"n_cell_lines={n} (threshold {cfg.min_cell_lines})",
    )


def _e5_initiation_efficiency(
    site: TranslationInitiationSite, cfg: ScoringConfig
) -> CriterionResult:
    """E5: max per-cell-line initiation efficiency exceeds threshold.

    Evaluates to ``None`` when no cell line carries an efficiency value —
    our upstream TIS counts don't always produce it.
    """
    efficiencies = [
        exp.initiation_efficiency
        for exp in site.expression.values()
        if exp.initiation_efficiency is not None
    ]
    if not efficiencies:
        return CriterionResult(
            "E5_initiation_efficiency", None, "no initiation_efficiency values"
        )
    best = max(efficiencies)
    passed = best >= cfg.initiation_efficiency_min
    return CriterionResult(
        "E5_initiation_efficiency",
        passed,
        f"max_efficiency={best:.3f} (threshold {cfg.initiation_efficiency_min})",
    )


def _e6_mass_spec(
    site: TranslationInitiationSite, cfg: ScoringConfig
) -> CriterionResult:
    """E6: PepQuery2-validated unique peptide(s) match public MS spectra.

    Requires the massspec module to have been initialised with a
    precomputed ``validated_peptides`` cache covering the gene.  When
    PepQuery2 has NOT been run (``summary.pepquery_run = False``) the
    criterion reports ``None`` — the isoform cannot be scored on mass
    spec until the precompute lands.  In-silico tryptic detectability
    alone is not evidence of proteomic observation.
    """
    ann = _annotation(site, "massspec")
    if ann is None:
        return CriterionResult("E6_mass_spec", None, "massspec not run")
    summary = ann.get("summary") if isinstance(ann, dict) else None
    if not isinstance(summary, dict) or not summary.get("pepquery_run"):
        return CriterionResult("E6_mass_spec", None, "pepquery2 not precomputed")
    hits = ann.get("hits")
    if not isinstance(hits, list):
        return CriterionResult("E6_mass_spec", None, "no hits field")
    n_validated_unique = sum(
        1 for h in hits
        if h.get("unique_to_isoform") is True and h.get("validated") is True
    )
    passed = n_validated_unique >= cfg.massspec_unique_peptides_min
    return CriterionResult(
        "E6_mass_spec",
        passed,
        f"n_validated_unique_peptides={n_validated_unique} "
        f"(threshold {cfg.massspec_unique_peptides_min})",
    )


# ---------------------------------------------------------------------------
# Functional criteria
# ---------------------------------------------------------------------------


def _f1_structured_extension(
    site: TranslationInitiationSite, cfg: ScoringConfig
) -> CriterionResult:
    """F1: structured differential region — diff-region pLDDT exceeds threshold.

    The metric is symmetric across ORF types — it asks "is the differential
    region structured?" — and the interpretation differs by direction:

    * extension/uORF/altORF — the diff region is the NEW N-terminal sequence
      the isoform adds; a high diff-region pLDDT indicates a well-folded
      addition (possibly a new functional domain).
    * truncation — the diff region is the canonical N-terminal sequence the
      isoform LOSES; a high diff-region pLDDT indicates that what was lost
      was structured (potentially functional). Same measurement, opposite
      narrative.

    Reads ``site.isoform_annotations['structure']`` written by
    ``StructureModule``.  Returns ``None`` when:

    - structure cache is empty (``status ∈ {no_cache, too_long, failed}``)
    - the backend produced only a scalar complex pLDDT, not per-residue
      (``status='uniform_plddt'``) — region-level statistics from a
      uniform fill aren't a real per-region measurement.

    Otherwise returns ``True`` when the mean pLDDT over the differential
    region is >= ``cfg.f1_plddt_threshold``, ``False`` otherwise. The
    threshold's scale must match the backend (Boltz-2 emits 0–1,
    AlphaFold-style emits 0–100).
    """
    ann = _annotation(site, "structure")
    if ann is None:
        return CriterionResult(
            "F1_structured_extension", None, "structure annotation missing"
        )
    status = ann.get("status")
    if status in ("no_cache", "too_long", "failed", "uniform_plddt"):
        return CriterionResult(
            "F1_structured_extension", None, f"structure status={status}"
        )
    plddt = ann.get("plddt_diffregion_mean")
    if plddt is None:
        return CriterionResult(
            "F1_structured_extension", None, "plddt_diffregion_mean unavailable"
        )
    threshold = cfg.f1_plddt_threshold
    passed = plddt >= threshold
    return CriterionResult(
        "F1_structured_extension",
        passed,
        f"plddt_diffregion_mean={plddt:.3f} (threshold {threshold})",
    )


def _f2_localization_change(
    site: TranslationInitiationSite, cfg: ScoringConfig  # noqa: ARG001
) -> CriterionResult:
    """F2: isoform's predicted subcellular location differs from canonical."""
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
            return CriterionResult("F2_localization_change", False, "no change flagged")
        return CriterionResult(
            "F2_localization_change", None, "no *_changed fields emitted"
        )
    return CriterionResult(
        "F2_localization_change", True, f"changed: {','.join(sorted(changed_keys))}"
    )


def _f3_domain_change(
    site: TranslationInitiationSite, cfg: ScoringConfig  # noqa: ARG001
) -> CriterionResult:
    """F3: domain gain/loss — InterProScan hit STARTS in the differential region.

    Symmetric across ORF types — the metric asks "is there a domain in the
    differential region?" and the interpretation flips by direction:

    * extension/uORF/altORF — diff region is added; an InterProScan hit
      starting there represents a GAINED domain.
    * truncation — diff region is lost; a hit starting there represents a
      LOST canonical domain.

    The comparator's positional subset emits every domain whose interval
    *overlaps* the diff region; for long domains a 1–2 aa boundary touch
    counts as overlap, which would call a body domain a "domain in the diff
    region" spuriously. This criterion tightens the rule to: the hit's
    ``pos`` (its start residue) must fall inside the diff window. For
    extensions the diff window is in isoform-frame; for truncations it's in
    canonical-frame — the comparator's ``hits_source_pane`` field tells us
    which side we're looking at.

    Returns ``None`` when:

    - the comparator data is missing (precompute not run), or
    - the IPS annotation's ``summary.status`` is not ``ok`` (precompute
      run but the scan didn't actually complete).
    """
    ips_ann = _annotation(site, "interproscan")
    if ips_ann is not None:
        status = ips_ann.get("summary", {}).get("status", "ok")
        if status != "ok":
            return CriterionResult(
                "F3_domain_change", None, f"interproscan status={status}"
            )

    cmp = site.comparison.get("interproscan")
    if not isinstance(cmp, dict):
        return CriterionResult(
            "F3_domain_change", None, "interproscan comparison missing"
        )
    hits = cmp.get("hits_in_diff_region")
    if not isinstance(hits, list):
        return CriterionResult(
            "F3_domain_change", None, "hits_in_diff_region unavailable"
        )

    dr = site.diff_region
    pane = cmp.get("hits_source_pane")
    if dr is None or pane is None:
        return CriterionResult(
            "F3_domain_change", None, "diff_region or source pane unavailable"
        )
    if pane == "isoform":
        start, end = dr.isoform_start, dr.isoform_end
    elif pane == "canonical":
        start, end = dr.canonical_start, dr.canonical_end
    else:
        return CriterionResult(
            "F3_domain_change", None, f"unknown hits_source_pane={pane!r}"
        )
    if start is None or end is None:
        return CriterionResult(
            "F3_domain_change", None, "diff coords missing for the source pane"
        )

    n_starting = sum(
        1
        for h in hits
        if isinstance(h.get("pos"), int) and start <= h["pos"] < end
    )
    return CriterionResult(
        "F3_domain_change",
        n_starting > 0,
        f"n_interproscan_hits_starting_in_diff_region={n_starting} "
        f"(of {len(hits)} overlapping)",
    )


def _f4_targeting_change(
    site: TranslationInitiationSite, cfg: ScoringConfig  # noqa: ARG001
) -> CriterionResult:
    """F4: targeting change — SignalP/TargetP disagree on canonical vs. isoform.

    Reads from ``site.comparison['signalp']`` / ``site.comparison['targetp']``
    written by the comparator (Scope A).  Returns ``None`` when neither
    module has produced a comparison (precompute not run), ``False`` when
    both ran but neither reports a category change, ``True`` when either
    does.
    """
    sp_cmp = site.comparison.get("signalp")
    tp_cmp = site.comparison.get("targetp")

    def _any_changed(cmp: dict[str, Any] | None) -> bool | None:
        if not isinstance(cmp, dict):
            return None
        changed = [k for k in cmp if k.endswith("_changed")]
        if not changed:
            return None
        return any(cmp.get(k) is True for k in changed)

    sp_state = _any_changed(sp_cmp)
    tp_state = _any_changed(tp_cmp)

    if sp_state is None and tp_state is None:
        return CriterionResult(
            "F4_targeting_change", None, "signalp/targetp comparisons not available"
        )
    if sp_state is True or tp_state is True:
        hits = []
        if sp_state is True:
            hits.append("signalp")
        if tp_state is True:
            hits.append("targetp")
        return CriterionResult("F4_targeting_change", True, f"changed in: {','.join(hits)}")
    return CriterionResult("F4_targeting_change", False, "no targeting change flagged")


def _f5_pathogenic_variant_enrichment(
    site: TranslationInitiationSite, cfg: ScoringConfig
) -> CriterionResult:
    """F5: predicted-damaging variant enrichment in the isoform-unique region.

    Scores per-variant, not on region means. Reads ``varianteffect`` (built by
    ``VariantEffectModule``), which attaches two complementary predictors to
    every clinical variant in the unique region:

    - ESM-2 masked-marginal ΔLLR = ``logP(alt) − logP(wt)`` at the variant's
      residue, and
    - AlphaMissense calibrated missense pathogenicity.

    A variant counts as damaging when AlphaMissense calls it
    ``likely_pathogenic`` **or** its ΔLLR ≤ ``cfg.f5_llr_damaging_threshold``.

    Returns:
        ``None`` when ``varianteffect`` didn't run or no scorable (missense,
        predictor-available) variant falls in the unique region — there is
        nothing to evaluate. ``True`` when at least
        ``cfg.f5_min_pathogenic_in_unique`` such variants are damaging,
        ``False`` when scorable variants exist but too few are damaging.
    """
    ve = _annotation(site, "varianteffect")
    if not _status_ok(ve):
        return CriterionResult(
            "F5_pathogenic_variant_enrichment", None, "varianteffect not run"
        )

    n_scorable_raw = ve.get("n_scorable_in_unique") if ve else None
    n_damaging_raw = ve.get("n_damaging_in_unique") if ve else None
    try:
        n_scorable = int(n_scorable_raw) if n_scorable_raw is not None else 0
        n_damaging = int(n_damaging_raw) if n_damaging_raw is not None else 0
    except (TypeError, ValueError):
        return CriterionResult(
            "F5_pathogenic_variant_enrichment",
            None,
            f"varianteffect counts not numeric ({n_scorable_raw!r}, {n_damaging_raw!r})",
        )

    if n_scorable == 0:
        return CriterionResult(
            "F5_pathogenic_variant_enrichment",
            None,
            "no scorable missense variants in unique region",
        )

    passed = n_damaging >= cfg.f5_min_pathogenic_in_unique
    n_am_path = ve.get("n_am_pathogenic_in_unique")
    mean_delta = ve.get("mean_delta_llr_unique")
    mean_delta_str = f"{mean_delta:.2f}" if isinstance(mean_delta, (int, float)) else "n/a"
    return CriterionResult(
        "F5_pathogenic_variant_enrichment",
        passed,
        (
            f"n_damaging={n_damaging}/{n_scorable} in unique "
            f"(≥{cfg.f5_min_pathogenic_in_unique}); "
            f"n_am_pathogenic={n_am_path}; mean_ΔLLR={mean_delta_str}"
        ),
    )


def _f6_clinical_variant_overlap(
    site: TranslationInitiationSite, cfg: ScoringConfig  # noqa: ARG001
) -> CriterionResult:
    """F6: any clinical variant sits in the isoform-unique region."""
    ann = _annotation(site, "variant_intersection")
    if not _status_ok(ann):
        return CriterionResult(
            "F6_clinical_variant_overlap", None, "variant_intersection not run"
        )
    n = ann.get("n_in_unique_region") if ann else None
    if n is None:
        return CriterionResult(
            "F6_clinical_variant_overlap", None, "n_in_unique_region unavailable"
        )
    return CriterionResult(
        "F6_clinical_variant_overlap",
        n > 0,
        f"n_variants_in_unique={n}",
    )


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


EXISTENCE_CRITERIA: list[Criterion] = [
    _e1_primate_conservation,
    _e2_mammalian_conservation,
    _e3_phylop_coding_selection,
    _e4_multi_cell_line,
    _e5_initiation_efficiency,
    _e6_mass_spec,
]


FUNCTIONAL_CRITERIA: list[Criterion] = [
    _f1_structured_extension,
    _f2_localization_change,
    _f3_domain_change,
    _f4_targeting_change,
    _f5_pathogenic_variant_enrichment,
    _f6_clinical_variant_overlap,
]


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
            "existence_high_confidence": existence_score >= self._scoring.existence_high_threshold,
            "functional_score": functional_score,
            "functional_evaluable": functional_evaluable,
            "functional_high_confidence": (
                functional_score >= self._scoring.functional_high_threshold
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


def _score(results: list[CriterionResult]) -> tuple[int, int]:
    """Return ``(true_count, evaluable_count)`` over a list of criterion results."""
    evaluable = [r for r in results if r.value is not None]
    return sum(1 for r in evaluable if r.value), len(evaluable)

