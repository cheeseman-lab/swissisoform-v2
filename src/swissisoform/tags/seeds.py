"""Hand-curated inputs to the candidate sweep.

Everything a human decided rather than a machine derived: how each of the sixteen
scored criteria maps onto a metric, where a cutoff has a real zero-point, which
ORF types a tag is even defined for, and the judgment tags that need the LLM tool
loop. Kept in one module so the sweep in :mod:`.candidates` stays mechanical and
auditable — if a proposal looks wrong, the reason is either here or in the data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from swissisoform.distributions import SEPARATE_ORF_TYPES

PAIRED_ORF_TYPES: tuple[str, ...] = ("extended", "truncated")
ALL_ORF_TYPES: tuple[str, ...] = PAIRED_ORF_TYPES + SEPARATE_ORF_TYPES


# ---------------------------------------------------------------------------
# Stream 1 — the sixteen scored criteria
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CriterionSeed:
    """One existing criterion, as a tag candidate.

    Attributes:
        criterion_id: The ``CRITERIA`` key (``"C1_primate_conservation"``).
        metric: Parquet column, or ``tx:<name>`` for a derived quantity.
        direction: ``">="`` or ``"<"`` — matching the scorer's own comparison.
        config_field: ``ScoringConfig`` field holding the current cutoff, so the
            sweep can reproduce today's fire rate before proposing a new one.
        label: Proposed tag label; the reviewer rewrites these.
        note: Why this criterion is shaped the way it is, where non-obvious.
    """

    criterion_id: str
    metric: str | None
    direction: str
    config_field: str | None
    label: str
    note: str = ""
    blocked: str = ""  # non-empty ⇒ carried into the table but not proposable
    literal: float | None = None  # cutoff hardcoded in the scorer, with no config field


# `metric=None` marks a criterion with no single numeric quantity: L1/L2 are
# categorical-change tests (their boolean form is picked up by the boolean
# stream), and M1/S2 are either-or roll-ups over several metrics, each of which
# is proposed separately below.
CRITERION_SEEDS: tuple[CriterionSeed, ...] = (
    CriterionSeed(
        "C1_primate_conservation", "isoform_conservation_frame_primate_mean_pident",
        ">=", "c1_pident_min", "Conserved across primates",
    ),
    CriterionSeed(
        "C2_mammalian_conservation", "isoform_conservation_frame_mammalian_mean_pident",
        ">=", "c2_pident_min", "Conserved across mammals",
    ),
    CriterionSeed(
        "C3_phylop_coding_selection", "isoform_conservation_phylop_unique_region_mean",
        ">=", "c3_phylop_min", "Unique region under purifying selection",
        "phyloP is signed: positive = conserved, negative = accelerated. The "
        "scorer reads c3_phylop_min (2.0); phylop_coding_min (1.0) is the looser "
        "descriptive band the tile labels with, not a score lever.",
    ),
    CriterionSeed(
        "D1_multi_cell_line", "tx:n_cell_lines", ">=", "min_cell_lines",
        "Detected in multiple cell lines",
        "Biased by the campaign's --drop-unsupported-tis HeLa restriction.",
    ),
    CriterionSeed(
        "D2_initiation_efficiency", "tx:max_initiation_efficiency", ">=",
        "initiation_efficiency_min", "Efficient start-site usage",
    ),
    CriterionSeed(
        "D3_mass_spec", "tx:n_validated_unique_peptides", ">=",
        "massspec_unique_peptides_min", "Mass-spec validated",
    ),
    CriterionSeed(
        "L1_localization_change", None, ">=", None, "Predicted localization changes",
        "Categorical; the cmp_localization_*_changed booleans carry it.",
    ),
    CriterionSeed(
        "L2_targeting_change", None, ">=", None, "N-terminal targeting changes",
        "Categorical; the cmp_signalp/targetp booleans carry it.",
    ),
    CriterionSeed(
        "M1_pathogenic_variant_enrichment", "isoform_plm_vep_constraint_enrichment",
        ">=", "m1_constraint_delta_min", "Unique region is constraint-enriched",
        "M1 is either-or over two independent signals; the gnomAD branch is a "
        "separate candidate. On extensions BOTH are uninformative — the region "
        "was never coding.",
        blocked=(
            "stale metric: PLMVEPModule emits constraint_delta (unique minus shared "
            "mean logP, cutoff 0.0) but full_catalog carries only "
            "isoform_plm_vep_constraint_enrichment, whose values are strictly "
            "positive and span orders of magnitude — a ratio, not a difference. "
            "The genome-wide run predates the rename, so the frozen distribution "
            "describes a different quantity than the scorer computes and no cutoff "
            "may be derived from it. M1 rests on its gnomAD branch until the run "
            "is regenerated."
        ),
    ),
    CriterionSeed(
        "M1_pathogenic_variant_enrichment", "isoform_variant_intersection_gnomad_depletion_ratio",
        "<", "m1_depletion_ratio_max", "Population variation avoids the unique region",
        "Lower = more constrained; runs opposite to the ESM-C branch.",
    ),
    CriterionSeed(
        "M2_clinical_variant_overlap", "isoform_variant_intersection_disease_enrichment_ratio",
        ">=", "m2_disease_enrichment_min", "Disease variants concentrate in the unique region",
    ),
    CriterionSeed(
        "P1_structured_extension", "isoform_structure_plddt_diffregion_mean", ">=",
        "p1_plddt_threshold", "Differential region is confidently folded",
    ),
    CriterionSeed(
        "P2_shared_structural_change", "isoform_structure_rmsd_shared", ">=",
        "p2_rmsd_shared_min", "Shared core refolds",
        "Only meaningful when tx:min_shared_plddt clears its gate.",
    ),
    CriterionSeed(
        "P3_secondary_structure", None, ">=", "p3_min_sse_length",
        "Secondary-structure element gained or lost",
        "Scored off the SSE element list, not a scalar column.",
    ),
    CriterionSeed(
        "S1_domain_change", "cmp_interproscan_n_real_domains_changed_in_diff_region",
        ">=", None, "InterPro domain gained or lost",
        "The >= 1 is hardcoded in the scorer; there is no ScoringConfig field.",
        literal=1.0,
    ),
    CriterionSeed(
        "S2_biophysics", "tx:abs_gravy_delta", ">=", "s2_gravy_delta_min",
        "Hydropathy shifts", "S2 is any-of-three; the other two are separate candidates.",
    ),
    CriterionSeed(
        "S2_biophysics", "tx:abs_fraction_charged_delta", ">=",
        "s2_fraction_charged_delta_min", "Charge shifts",
    ),
    CriterionSeed(
        "S2_biophysics", "tx:abs_disorder_delta", ">=", "s2_disorder_delta_min",
        "Disorder shifts",
    ),
    CriterionSeed(
        "S3_sae", "tx:sae_top_delta", ">=", "s3_top_delta_min",
        "Interpretable feature activation shifts",
        "Magnitude, not presence: two proteins always differ in hundreds of features.",
    ),
)


# ---------------------------------------------------------------------------
# Cutoff anchors — real zero-points, preferred over any percentile
# ---------------------------------------------------------------------------

# (name-pattern, anchor value, why). A ratio's null hypothesis is 1.0 and a
# delta's is 0.0; those are boundaries in the biology, not quantiles of this
# corpus, so they beat a swept percentile when the resulting rate is in band.
# Percent-identity deliberately has no anchor: 1.0 is a ceiling, not a null.
ANCHOR_RULES: tuple[tuple[re.Pattern[str], float, str], ...] = (
    (re.compile(r"(_ratio|_enrichment)$"), 1.0, "ratio null hypothesis"),
    (re.compile(r"_delta(_max)?$"), 0.0, "no change"),
    (re.compile(r"phylop"), 0.0, "neutral evolution"),
)


def anchor_for(metric: str) -> tuple[float, str] | None:
    """Return ``(value, reason)`` when *metric* has a principled zero-point.

    Magnitudes are excluded: 0.0 is the null for a *signed* delta, but ``|x| >= 0``
    is true of everything. For a two-sided quantity the useful cutoff is a
    distance from the null, not the null itself.
    """
    from swissisoform import metrics

    if metrics.is_magnitude(metric):
        return None
    bare = metric.split(":", 1)[-1]
    for pattern, value, reason in ANCHOR_RULES:
        if pattern.search(bare):
            return value, reason
    return None


# ---------------------------------------------------------------------------
# Validity overrides — undefined by construction, not missing data
# ---------------------------------------------------------------------------

# A metric whose *definition* does not apply to a stratum. Empirical fill cannot
# distinguish this from a data hole, and the difference matters: "not evaluable
# because there is no shared region" is a fact about the isoform, whereas a low
# fill rate is a fact about the pipeline.
VALIDITY_OVERRIDES: dict[str, tuple[str, ...]] = {
    # No shared region ⇒ no denominator, no retained core to compare.
    "isoform_variant_intersection_gnomad_depletion_ratio": PAIRED_ORF_TYPES,
    "isoform_variant_intersection_disease_enrichment_ratio": PAIRED_ORF_TYPES,
    "isoform_structure_rmsd_shared": PAIRED_ORF_TYPES,
    "isoform_structure_shared_region_len": PAIRED_ORF_TYPES,
    "isoform_structure_tm_score_shared": PAIRED_ORF_TYPES,
    "isoform_conservation_phylop_shared_region_mean": PAIRED_ORF_TYPES,
    "isoform_conservation_phylop_enrichment": PAIRED_ORF_TYPES,
    "isoform_conservation_phastcons_shared_region_mean": PAIRED_ORF_TYPES,
    "tx:min_shared_plddt": PAIRED_ORF_TYPES,
    "tx:sae_top_delta": PAIRED_ORF_TYPES,
    # The unique region of an extension was never coding, so neither germline
    # signal measures protein constraint there (see category-pass.txt).
    "isoform_plm_vep_constraint_enrichment": ("truncated",),
}


def validity_for(metric: str, null_pattern: str | None) -> tuple[str, ...]:
    """ORF types a metric is defined for.

    The hand override wins; otherwise the catalog's measured ``null_pattern`` is
    trusted — ``absent_for_separate_orfs`` means the paired types only.
    """
    override = VALIDITY_OVERRIDES.get(metric)
    if override is not None:
        return override
    if null_pattern == "absent_for_separate_orfs":
        return PAIRED_ORF_TYPES
    return ALL_ORF_TYPES


# ---------------------------------------------------------------------------
# Stream 3 — judgment tags that need the tool loop
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMSeed:
    """A tag no threshold can express, fired by the category tool loop.

    Attributes:
        tag_id: Stable id; becomes a member of the category's ``emit_tags`` enum.
        category: CDLMPS letter whose loop fires it.
        label: Sidebar label.
        question: What the model is being asked, in one line.
        reader: The existing tool that supplies the evidence.
        citation: The single number a fired tag must cite.
    """

    tag_id: str
    category: str
    label: str
    question: str
    reader: str
    citation: str
    valid_for: tuple[str, ...] = field(default=ALL_ORF_TYPES)


# Deliberately few. Every tag here is one the sweep cannot express as a cutoff on
# a scalar, and each confines the loop's non-reproducibility to where it is
# unavoidable. Their fire rates are blank until the loop runs on cheeseman50.
LLM_SEEDS: tuple[LLMSeed, ...] = (
    LLMSeed(
        "variant_hotspot_in_unique_region", "M", "Variant hotspot, not diffuse spread",
        "Do the unique-region variants cluster tightly, or spread evenly?",
        "variant_position_histogram", "fraction on the 10 busiest residues",
    ),
    LLMSeed(
        "start_codon_variant_cluster", "M", "Variants at the alternative start",
        "Do variants concentrate on the alternative start codon itself?",
        "query_variants", "variant count within the start-codon window",
    ),
    LLMSeed(
        "extension_docks_against_core", "P", "Extension packs against the core",
        "Does the extension contact a specific core surface, or merely sit near it?",
        "contacts", "number of core contact partners",
        valid_for=("extended",),
    ),
    LLMSeed(
        "conflicting_evidence", "P", "Confidence conflicts with geometry",
        "Does a structural signal survive its own confidence gate?",
        "plddt_profile", "pLDDT over the element in question",
    ),
)


# ---------------------------------------------------------------------------
# UI labels
# ---------------------------------------------------------------------------

# What a fired tag is called in the filter sidebar. Keyed by ``(metric,
# direction)`` because a metric read high and read low are opposite claims.
#
# House style, matching ``ISOFORM_TAG_VOCAB`` in the website's ``data.py``:
# a short noun phrase, sentence case, two to five words, stating what is TRUE of
# an isoform the tag fires on — not the column it came from and not the test.
# "Start codon conserved", never "primate_start_codon_conserved >= 0.34".
TAG_LABELS: dict[tuple[str, str], str] = {
    # ── C, conservation ────────────────────────────────────────────────
    ("isoform_conservation_frame_primate_mean_pident", ">="): "Conserved in primates",
    ("isoform_conservation_frame_mammalian_mean_pident", ">="): "Conserved in mammals",
    ("isoform_conservation_phylop_unique_region_mean", ">="): "Unique region constrained",
    ("isoform_conservation_frame_mammalian_start_codon_conserved", "<"): (
        "Start codon not conserved"
    ),
    ("isoform_conservation_phastcons_at_tis", "<"): "Unconserved start site",
    ("isoform_conservation_phastcons_kozak_mean", "<"): "Unconserved Kozak context",
    ("isoform_conservation_phastcons_unique_region_mean", "<"): "Unique region unconserved",
    ("isoform_conservation_phylop_at_tis", "<"): "Start site under drift",
    ("isoform_conservation_frame_summary.tree_loaded", "bool"): "Species tree loaded",
    # ── D, detection ───────────────────────────────────────────────────
    ("tx:n_cell_lines", ">="): "Seen in multiple cell lines",
    ("tx:max_initiation_efficiency", ">="): "Efficient start site",
    ("tx:n_validated_unique_peptides", ">="): "Mass-spec validated",
    ("isoform_massspec_hits__len", "<"): "No mass-spec peptides",
    ("cmp_massspec_hits_in_diff_region__len", ">="): "Peptides in unique region",
    ("tis_pvalue", ">="): "Weak detection p-value",
    # ── L, localization ────────────────────────────────────────────────
    ("cmp_localization_deeploc_prediction_changed", "bool"): "Localization changes",
    ("cmp_localization_deeploc_membrane_changed", "bool"): "Membrane association changes",
    ("cmp_localization_deeploc_signals_changed", "bool"): "Sorting signals change",
    ("cmp_signalp_signalp_prediction_changed", "bool"): "Signal peptide changes",
    ("cmp_signalp_signalp_cleavage_site_changed", "bool"): "Signal cleavage site moves",
    ("cmp_targetp_targetp_prediction_changed", "bool"): "Targeting changes",
    ("cmp_targetp_targetp_cleavage_site_changed", "bool"): "Transit peptide site moves",
    ("cmp_targetp_targetp_ctp_prob_changed", "bool"): "Chloroplast transit changes",
    ("isoform_localization_deeploc_prob_nucleus", ">="): "Predicted nuclear",
    ("isoform_localization_deeploc_prob_cytoplasm", "<"): "Predicted non-cytoplasmic",
    ("isoform_localization_deeploc_top_prob", ">="): "Confident localization call",
    ("isoform_targetp_targetp_probability", "<"): "Uncertain targeting call",
    # ── M, mutation landscape ──────────────────────────────────────────
    ("isoform_variant_intersection_gnomad_depletion_ratio", "<"): "Population variation avoided",
    ("isoform_variant_intersection_disease_enrichment_ratio", ">="): (
        "Disease variants concentrated"
    ),
    ("isoform_plm_vep_constraint_enrichment", ">="): "Unique region constraint-enriched",
    ("isoform_plm_vep_n_constrained_positions_unique", ">="): "Constrained residues gained",
    ("isoform_plm_vep_n_constrained_positions_shared", "<"): "Few constrained residues in core",
    ("isoform_plm_vep_mean_llr_unique_region", "<"): "Unique region poorly predicted",
    ("isoform_plm_vep_mean_llr_isoform", "<"): "Isoform poorly predicted",
    ("abs:isoform_varianteffect_mean_delta_llr_unique_gnomad", ">="): (
        "Population variants shift fitness"
    ),
    ("isoform_clinical_summary.by_consequence.inframe_insertion", ">="): (
        "Inframe insertions reported"
    ),
    # ── P, predicted structure ─────────────────────────────────────────
    ("isoform_structure_plddt_diffregion_mean", ">="): "Differential region folds",
    ("isoform_structure_rmsd_shared", ">="): "Shared core refolds",
    ("isoform_structure_tm_score", "<"): "Overall fold diverges",
    ("isoform_structure_sse_all_elements__len", "<"): "Little secondary structure",
    ("isoform_structure_plddt_isoform_mean", "<"): "Low-confidence fold",
    ("isoform_structure_ptm_isoform", ">="): "Confident isoform fold",
    ("isoform_structure_ptm_canonical", ">="): "Confident canonical fold",
    # ── S, structural characteristics ──────────────────────────────────
    ("cmp_interproscan_n_real_domains_changed_in_diff_region", ">="): "Domain gained or lost",
    ("cmp_motifs_hits_in_diff_region__len", ">="): "Motifs in unique region",
    ("tx:sae_top_delta", ">="): "Interpretable features shift",
    ("abs:isoform_sae_top_gained_delta_max", ">="): "Strong feature gained",
    ("tx:abs_gravy_delta", ">="): "Hydropathy shifts",
    ("tx:abs_fraction_charged_delta", ">="): "Charge shifts",
    ("tx:abs_disorder_delta", ">="): "Disorder shifts",
    ("cmp_biophysics_gravy_ratio", ">="): "Unique region hydrophobic",
    ("cmp_biophysics_aromaticity_ratio", ">="): "Unique region aromatic",
    ("cmp_biophysics_pipi_propensity_ratio", ">="): "Unique region pi-pi prone",
    ("cmp_biophysics_mean_window_entropy_ratio", ">="): "Unique region high entropy",
    ("cmp_biophysics_normalized_complexity_ratio", ">="): "Unique region complex",
    ("cmp_biophysics_shannon_entropy_ratio", ">="): "Unique region diverse",
    ("cmp_biophysics_length_ratio", ">="): "Long unique region",
    ("cmp_biophysics_aa_diversity_ratio", ">="): "Unique region varied",
    # Re-derived from the *_enriched booleans: same claim, distribution cutoff.
    ("cmp_biophysics_disorder_ratio", ">="): "Unique region disordered",
    ("cmp_biophysics_fraction_disorder_promoting_ratio", ">="): "Disorder-promoting residues",
    ("cmp_biophysics_fraction_charged_ratio", ">="): "Unique region charged",
    ("cmp_biophysics_pI_ratio", ">="): "Unique region more basic",
    ("cmp_biophysics_instability_index_ratio", ">="): "Unique region unstable",
    ("cmp_biophysics_llps_score_ratio", ">="): "Phase-separation prone",
    ("cmp_biophysics_prionlike_fraction_ratio", ">="): "Prion-like unique region",
    ("cmp_biophysics_fraction_lcr_ratio", ">="): "Low-complexity unique region",
    ("cmp_biophysics_rg_fg_density_ratio", ">="): "RG/FG-rich unique region",
    ("cmp_biophysics_longest_homopolymer_ratio", ">="): "Homopolymer run in unique region",
    ("abs:cmp_biophysics_longest_homopolymer_delta", ">="): "Homopolymer length shifts",
    ("cmp_biophysics_longest_homopolymer_unique", "<"): "No homopolymer run",
    ("cmp_biophysics_aromaticity_unique", "<"): "Unique region non-aromatic",
    ("cmp_biophysics_pI_unique", ">="): "Basic unique region",
    ("cmp_biophysics_pI_shared", ">="): "Basic shared core",
    ("cmp_biophysics_fraction_lcr_shared", "<"): "Low-complexity-poor core",
    ("cmp_biophysics_longest_homopolymer_shared", "<"): "No homopolymer in core",
    ("cmp_biophysics_rg_fg_density_shared", "<"): "RG/FG-poor core",
    ("isoform_sae_n_shared", "<"): "Few shared features",
    ("isoform_sae_top_gained_feature_index", ">="): "High gained-feature index",
    ("isoform_sae_top_lost_feature_index", ">="): "High lost-feature index",
    ("isoform_motifs_hits__len", "<"): "Few motifs overall",
}

_MOTIF_LABELS = {
    "SH3_ClassI": "SH3 class-I motifs",
    "SH3_ClassII_density_per100aa": "SH3 class-II motifs",
    "RG_rich": "RG-rich stretches",
    "EB1_SxIP_density_per100aa": "EB1 SxIP motifs",
    "density_1433_per100aa": "14-3-3 motifs",
    "CDK_Sites_SP_TP_density_per100aa": "CDK phosphosites",
    "ATM_Sites_SQ_TQ_density_per100aa": "ATM phosphosites",
}


def label_for(metric: str, direction: str, fallback: str = "") -> str:
    """Sidebar label for a tag, or a readable fallback built from the metric name.

    The fallback exists so a newly-swept metric still reads as English rather than
    a raw column name; anything surfacing it is a prompt to add a curated entry.
    """
    hit = TAG_LABELS.get((metric, direction or "bool"))
    if hit:
        return hit
    if metric.startswith("isoform_motifs_summary."):
        name = _MOTIF_LABELS.get(metric.split(".", 1)[1])
        if name:
            return f"{'Many' if direction == '>=' else 'Few'} {name}"
    bare = metric.split(":", 1)[-1].rsplit(".", 1)[-1]
    for prefix in ("cmp_", "isoform_", "canonical_"):
        bare = bare.removeprefix(prefix)
    words = bare.replace("__len", " count").replace("_", " ").strip()
    qualifier = "" if direction == "bool" else (" high" if direction == ">=" else " low")
    return (words[:1].upper() + words[1:] + qualifier) or fallback
