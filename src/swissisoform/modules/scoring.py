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
isoform-unique region (ESM-2 constraint enrichment + gnomAD depletion).

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
       (ESM-2 ``plm_vep`` constraint enrichment OR gnomAD depletion via
       ``variant_intersection``)
    F6 Disease-variant density enrichment in unique region vs shared core
       (``variant_intersection.disease_enrichment_ratio``)
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
    """E1: primate amino-acid identity over the unique region exceeds threshold.

    Scored on ``primate_mean_pident`` (mean AA percent identity of the aligned
    primate orthologs), not ``frac_intact`` — pident measures sequence
    conservation directly, whereas frac_intact only counts species with an
    intact frame. ``frac_intact`` is still surfaced in the reason as context.
    """
    ann = _annotation(site, "conservation_frame")
    if not _status_ok(ann):
        return CriterionResult("E1_primate_conservation", None, "conservation_frame not run")
    val = ann.get("primate_mean_pident") if ann else None
    if val is None:
        return CriterionResult(
            "E1_primate_conservation", None, "primate_mean_pident unavailable"
        )
    passed = val >= cfg.e1_pident_min
    frac = ann.get("primate_frac_intact")
    frac_str = f"{frac:.2f}" if isinstance(frac, (int, float)) else "n/a"
    return CriterionResult(
        "E1_primate_conservation",
        passed,
        f"mean_pident={val:.2f} (threshold {cfg.e1_pident_min}); frac_intact={frac_str}",
    )


def _e2_mammalian_conservation(
    site: TranslationInitiationSite, cfg: ScoringConfig
) -> CriterionResult:
    """E2: mammalian amino-acid identity over the unique region exceeds threshold.

    Scored on ``mammalian_mean_pident`` (mean AA percent identity across the
    aligned mammalian orthologs), not ``frac_intact``. ``frac_intact`` is kept
    in the reason as context.
    """
    ann = _annotation(site, "conservation_frame")
    if not _status_ok(ann):
        return CriterionResult(
            "E2_mammalian_conservation", None, "conservation_frame not run"
        )
    val = ann.get("mammalian_mean_pident") if ann else None
    if val is None:
        return CriterionResult(
            "E2_mammalian_conservation", None, "mammalian_mean_pident unavailable"
        )
    passed = val >= cfg.e2_pident_min
    frac = ann.get("mammalian_frac_intact")
    frac_str = f"{frac:.2f}" if isinstance(frac, (int, float)) else "n/a"
    return CriterionResult(
        "E2_mammalian_conservation",
        passed,
        f"mean_pident={val:.2f} (threshold {cfg.e2_pident_min}); frac_intact={frac_str}",
    )


def _e3_phylop_coding_selection(
    site: TranslationInitiationSite, cfg: ScoringConfig
) -> CriterionResult:
    """E3: absolute PhyloP over the unique region indicates purifying selection.

    Scored on the *absolute* mean PhyloP of the unique region exceeding
    ``cfg.e3_phylop_min`` (strong purifying selection at coding level). This is
    NOT a unique-vs-shared comparison — the claim is that the unique region is
    itself under coding-level constraint. ``phylop_enrichment`` (unique/shared)
    is reported as context only.
    """
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
    passed = val >= cfg.e3_phylop_min
    enrich = ann.get("phylop_enrichment")
    enrich_str = f"{enrich:.2f}" if isinstance(enrich, (int, float)) else "n/a"
    return CriterionResult(
        "E3_phylop_coding_selection",
        passed,
        f"phylop_unique={val:.2f} (threshold {cfg.e3_phylop_min}); "
        f"enrichment={enrich_str} (context)",
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

    ``True`` requires BOTH halves: the diff region is well-folded
    (mean pLDDT >= ``cfg.f1_plddt_threshold``) AND it is biophysically
    distinct from the shared core. Distinctness is read from the biophysics
    comparator deltas — any of ``|gravy_delta|``, ``|fraction_charged_delta|``,
    ``|disorder_delta|`` exceeding its provisional cutoff. A structured-but-
    biophysically-identical addition is not counted as a functional change.

    Returns ``None`` when structure status is not ``ok`` (cache empty /
    too long / failed / uniform fill) OR the biophysics comparison is missing.
    The threshold's scale must match the backend (Boltz-2 emits 0–1,
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

    bio = site.comparison.get("biophysics")
    if not isinstance(bio, dict):
        return CriterionResult(
            "F1_structured_extension", None, "biophysics comparison missing"
        )
    # Provisional distinctness cutoffs — CALIBRATE ON GENOME-WIDE RUN.
    gravy_d = bio.get("gravy_delta")
    charged_d = bio.get("fraction_charged_delta")
    disorder_d = bio.get("disorder_delta")
    distinct_flags = []
    if isinstance(gravy_d, (int, float)) and abs(gravy_d) >= 0.3:
        distinct_flags.append("gravy")
    if isinstance(charged_d, (int, float)) and abs(charged_d) >= 0.05:
        distinct_flags.append("charged")
    if isinstance(disorder_d, (int, float)) and abs(disorder_d) >= 0.05:
        distinct_flags.append("disorder")
    distinct = len(distinct_flags) > 0

    threshold = cfg.f1_plddt_threshold
    folded = plddt >= threshold
    passed = folded and distinct
    folded_str = "folded" if folded else "unfolded"
    distinct_str = ",".join(distinct_flags) if distinct_flags else "none"
    return CriterionResult(
        "F1_structured_extension",
        passed,
        f"plddt_diffregion_mean={plddt:.3f} ({folded_str}, threshold {threshold}); "
        f"biophysically_distinct={distinct_str}",
    )


def _f2_localization_change(
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


def _f3_domain_change(
    site: TranslationInitiationSite, cfg: ScoringConfig  # noqa: ARG001
) -> CriterionResult:
    """F3: domain gain/loss — a REAL InterPro domain is gained or lost in the diff region.

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
            "F3_domain_change", None, "interproscan comparison missing"
        )
    canonical_status = cmp.get("hits_canonical_status")
    if canonical_status != "ok":
        return CriterionResult(
            "F3_domain_change", None, f"interproscan status={canonical_status}"
        )
    n = cmp.get("n_real_domains_changed_in_diff_region")
    if n is None:
        return CriterionResult(
            "F3_domain_change",
            None,
            "n_real_domains_changed_in_diff_region unavailable",
        )
    try:
        n = int(n)
    except (TypeError, ValueError):
        return CriterionResult(
            "F3_domain_change",
            None,
            f"n_real_domains_changed_in_diff_region not numeric ({n!r})",
        )
    return CriterionResult(
        "F3_domain_change",
        n >= 1,
        f"n_real_domains_changed_in_diff_region={n}",
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
    """F5: germline tolerance / constraint over the isoform-unique region.

    Two complementary, independent signals that the unique region is under
    selective constraint in healthy humans:

    - **ESM-2 constraint enrichment** (``plm_vep.constraint_enrichment``,
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

    # The criterion *name* stays "F5_pathogenic_variant_enrichment" for
    # backward compatibility with the viewer / grounding units that key on it;
    # the *intent* is germline tolerance / constraint (see docstring).
    if constraint is None and depletion is None:
        return CriterionResult(
            "F5_pathogenic_variant_enrichment",
            None,
            "no plm_vep constraint_enrichment or gnomad_depletion_ratio",
        )

    constrained = constraint is not None and constraint >= cfg.f5_constraint_enrichment_min
    depleted = depletion is not None and depletion < cfg.f5_depletion_ratio_max
    passed = constrained or depleted

    constraint_str = f"{constraint:.2f}" if constraint is not None else "n/a"
    depletion_str = f"{depletion:.2f}" if depletion is not None else "n/a"
    return CriterionResult(
        "F5_pathogenic_variant_enrichment",
        passed,
        (
            f"constraint_enrichment={constraint_str} "
            f"(≥{cfg.f5_constraint_enrichment_min}); "
            f"gnomad_depletion_ratio={depletion_str} "
            f"(<{cfg.f5_depletion_ratio_max})"
        ),
    )


def _f6_clinical_variant_overlap(
    site: TranslationInitiationSite, cfg: ScoringConfig
) -> CriterionResult:
    """F6: disease variants concentrate in the isoform-unique region.

    Scored on the per-nt density enrichment of disease variants (ClinVar +
    COSMIC) in the unique region vs the shared core
    (``variant_intersection.disease_enrichment_ratio``). ``True`` when the
    ratio is >= ``cfg.f6_disease_enrichment_min`` (1.0 — disease variation is
    at least as dense in the unique region as the shared core). This replaces
    the old mere-presence test, which fired on a single variant regardless of
    the shared-core background. ``None`` when the ratio is missing (e.g. zero
    region length or no shared-region disease density). Raw unique/shared
    disease counts are reported as context.
    """
    ann = _annotation(site, "variant_intersection")
    if not _status_ok(ann):
        return CriterionResult(
            "F6_clinical_variant_overlap", None, "variant_intersection not run"
        )
    ratio = ann.get("disease_enrichment_ratio") if ann else None
    if not isinstance(ratio, (int, float)):
        return CriterionResult(
            "F6_clinical_variant_overlap", None, "disease_enrichment_ratio unavailable"
        )
    n_unique = ann.get("n_disease_in_unique_region")
    n_shared = ann.get("n_disease_in_shared_region")
    return CriterionResult(
        "F6_clinical_variant_overlap",
        ratio >= cfg.f6_disease_enrichment_min,
        f"disease_enrichment_ratio={ratio:.2f} "
        f"(≥{cfg.f6_disease_enrichment_min}); "
        f"disease unique={n_unique}, shared={n_shared}",
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


def _score(results: list[CriterionResult]) -> tuple[int, int]:
    """Return ``(true_count, evaluable_count)`` over a list of criterion results."""
    evaluable = [r for r in results if r.value is not None]
    return sum(1 for r in evaluable if r.value), len(evaluable)

