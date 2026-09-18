#!/usr/bin/env python
"""One-off (uncommitted): export EvidenceScoringModule cutoffs to CSV.

Pulls, per scoring criterion, the numeric cutoff and the decision logic that
turns a raw annotation value into a True/False/None verdict, and writes it to
``scoring_cutoffs.csv`` next to this script. Intended as the reference table for
building distribution plots of each cutoff against the genome-wide data.

The numeric thresholds are read *live* from an instantiated ``ScoringConfig``
(``getattr(cfg, field)``) so the CSV stays in sync with ``config.py`` instead of
drifting. Criteria with more than one lever (M1, P2, P3, S2) name a primary
``config_field`` plus ``secondary_config_fields``; both resolve live, so no
threshold is ever transcribed by hand into the prose columns. Everything that
can't be introspected from the code — the source annotation field, the comparison
operator, the None/gating conditions, the ORF-type asymmetries — is curated below
from the per-criterion ``score()`` functions in
``src/swissisoform/evidence/<bucket>/__init__.py``.

Covers all 16 criteria: 6 existence (C1-C3, D1-D3) + 10 functional (L1-L2,
M1-M2, P1-P3, S1-S3), plus the two high-confidence roll-ups.

The ``dist_*`` columns carry a five-number summary (min / Q1 / median / Q3 / max)
of where real isoforms fall for each cutoff, so the observed spread sits beside
the threshold it is judged against. Values come from the genome-wide
``full_catalog`` run via the ``SERIES`` registry in
``plot_cutoff_distributions.py`` — that module owns every parquet column name and
transform, and this one only consumes them. Three caveats, none of which fit in a
column: the run is HeLa-restricted (``--drop-unsupported-tis``), so D1 especially
is biased by construction; the spread is over rows where the value exists, not
rows the criterion actually scored, since each criterion's not-evaluable gates
are not applied; and P3 has no data in that run at all. Blank ``dist_*`` cells
mean no numeric quantity exists (L1/L2, roll-ups) or none was found (P3). Note
these are quartiles — ``cutoff_distributions_summary.csv`` reports p05/p95
instead, so the two tables are deliberately different.

Run:
    python figures/scoring_cutoffs/export_scoring_cutoffs.py
"""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

from plot_cutoff_distributions import DEFAULT_SHARDS, SERIES, load_frame, shard_files

from swissisoform.config import ScoringConfig

OUT_CSV = Path(__file__).with_name("scoring_cutoffs.csv")

# Five-number summary columns, appended last so the existing order is untouched.
DIST_FIELDS = ("dist_min", "dist_q1", "dist_median", "dist_q3", "dist_max")
_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)

FIELDNAMES = [
    "criterion_id",
    "legacy_id",
    "axis",
    "category",
    "name",
    "fires_when",
    "source_module",
    "source_field",
    "config_field",
    "threshold_value",
    "secondary_config_fields",
    "comparison",
    "none_conditions",
    "orf_type_handling",
    "provisional",
    *DIST_FIELDS,
]

# One entry per criterion. ``config_field`` is looked up on a live ScoringConfig;
# leave it None for categorical / hardcoded / threshold-free criteria and put the
# literal directly in ``threshold_value``. ``secondary_config_fields`` is a list
# of the remaining live-resolved levers for multi-threshold criteria.
#
# ``fires_when`` is the whole True condition in one line, written with ``{field}``
# placeholders that are ``str.format``-ed against the live config — so it reads as
# e.g. "plddt_diffregion_mean >= 0.7" without any number being typed by hand.
#
# ``dist_*`` is filled at write time from the genome-wide run, not curated here —
# see :func:`distribution_summary`.
CRITERIA = [
    # ---------------- EXISTENCE: Conservation (C) ----------------
    {
        "criterion_id": "C1_primate_conservation",
        "legacy_id": "E1",
        "axis": "existence",
        "category": "Conservation",
        "name": "Primate conservation (mean AA %identity, unique region)",
        "fires_when": "primate_mean_pident >= {c1_pident_min}",
        "source_module": "conservation_frame",
        "source_field": "primate_mean_pident",
        "config_field": "c1_pident_min",
        "comparison": (
            "val >= c1_pident_min -> True; else False. "
            "primate_frac_intact appears in the reason string only — "
            "primate_frac_intact_min (0.5) is display context, not a score lever"
        ),
        "none_conditions": (
            "conservation_frame not run (status not ok) or primate_mean_pident is None"
        ),
        "orf_type_handling": (
            "unique region = isoform_orf minus canonical_orf; TRUNCATED flips to "
            "canonical_orf minus isoform_orf. Empty unique region -> "
            "status='no_unique_region' -> None"
        ),
        "provisional": True,
    },
    {
        "criterion_id": "C2_mammalian_conservation",
        "legacy_id": "E2",
        "axis": "existence",
        "category": "Conservation",
        "name": "Mammalian conservation (mean AA %identity, unique region)",
        "fires_when": "mammalian_mean_pident >= {c2_pident_min}",
        "source_module": "conservation_frame",
        "source_field": "mammalian_mean_pident",
        "config_field": "c2_pident_min",
        "comparison": (
            "val >= c2_pident_min -> True; else False. "
            "mammalian_frac_intact_min (0.3) is display context, not a score lever"
        ),
        "none_conditions": "conservation_frame not run or mammalian_mean_pident is None",
        "orf_type_handling": "same unique-region flip as C1 (TRUNCATED reads canonical space)",
        "provisional": True,
    },
    {
        "criterion_id": "C3_phylop_coding_selection",
        "legacy_id": "E3",
        "axis": "existence",
        "category": "Conservation",
        "name": "PhyloP purifying selection over unique region",
        "fires_when": "phylop_unique_region_mean >= {c3_phylop_min}",
        "source_module": "conservation",
        "source_field": "phylop_unique_region_mean",
        "config_field": "c3_phylop_min",
        "comparison": (
            "val >= c3_phylop_min -> True; else False. Absolute mean PhyloP, NOT a "
            "unique-vs-shared ratio; phylop_enrichment is reason-string context only. "
            "Legacy phylop_coding_min (1.0) is unused"
        ),
        "none_conditions": (
            "conservation not run, summary.region_status != 'ok' (note: region_status, "
            "not the generic status), or value None"
        ),
        "orf_type_handling": (
            "same unique-region flip as C1; missing orf_exons/canonical_orf_exons -> "
            "region_status='no_skeleton' -> None"
        ),
        "provisional": False,
    },
    # ---------------- EXISTENCE: Detection (D) ----------------
    {
        "criterion_id": "D1_multi_cell_line",
        "legacy_id": "E4",
        "axis": "existence",
        "category": "Detection",
        "name": "Multi-cell-line reproducibility",
        "fires_when": "n_cell_lines >= {min_cell_lines}",
        "source_module": "(site.expression)",
        "source_field": "len(site.expression)",
        "config_field": "min_cell_lines",
        "comparison": (
            "n_cell_lines >= min_cell_lines -> True; else False (NEVER None). "
            "Counts cell lines only — no p-value or CPM filter is applied here"
        ),
        "none_conditions": (
            "never None (always evaluable); degenerate when min_cell_lines == 1 "
            "(True for every TIS by construction, a constant +1 on the existence score)"
        ),
        "orf_type_handling": "symmetric (expression-based, no ORF-type dependence)",
        "provisional": False,
    },
    {
        "criterion_id": "D2_initiation_efficiency",
        "legacy_id": "E5",
        "axis": "existence",
        "category": "Detection",
        "name": "Ribosome initiation efficiency (best across cell lines)",
        "fires_when": "max(initiation_efficiency) >= {initiation_efficiency_min}",
        "source_module": "(site.expression)",
        "source_field": "max(exp.initiation_efficiency)",
        "config_field": "initiation_efficiency_min",
        "comparison": "best >= initiation_efficiency_min -> True; else False (max, not mean)",
        "none_conditions": "no cell line carries an initiation_efficiency value",
        "orf_type_handling": "symmetric (expression-based, no ORF-type dependence)",
        "provisional": False,
    },
    {
        "criterion_id": "D3_mass_spec",
        "legacy_id": "E6",
        "axis": "existence",
        "category": "Detection",
        "name": "Mass-spec validated isoform-unique peptides (PepQuery2)",
        "fires_when": "n_validated_unique_peptides >= {massspec_unique_peptides_min}",
        "source_module": "massspec",
        "source_field": "count(hits: unique_to_isoform is True AND validated is True)",
        "config_field": "massspec_unique_peptides_min",
        "comparison": (
            "n_validated_unique >= massspec_unique_peptides_min -> True; else False. "
            "Strict 'is True' on both flags, so None (unknown) never counts. "
            "Digest literals are hardcoded: missed_cleavages=1, min_length=7, max_length=30"
        ),
        "none_conditions": (
            "massspec not run, summary.pepquery_run falsy/absent (search not precomputed), "
            "or hits is not a list"
        ),
        "orf_type_handling": (
            "extension: peptides overlapping the isoform diff region (junction peptide "
            "included); truncation: ONLY the pos==0 peptide (new start through first "
            "cleavage) — canonical-pane peptides are explicitly refused; "
            "uORF/uoORF/altORF/internal-OOF/3'UTR and unclassified: full digest. "
            "An NME (Met-excised) variant is added for any classified pos==0 peptide"
        ),
        "provisional": False,
    },
    # ---------------- FUNCTIONAL: Localization (L) ----------------
    {
        "criterion_id": "L1_localization_change",
        "legacy_id": "F2",
        "axis": "functional",
        "category": "Localization",
        "name": "Subcellular localization change (DeepLoc)",
        "fires_when": "any localization *_changed is True (no numeric cutoff)",
        "source_module": "comparison.localization",
        "source_field": "any key ending in '_changed' (prediction/signals/membrane)",
        "config_field": None,
        "threshold_value": "categorical (no cutoff)",
        "comparison": (
            "any *_changed is True -> True; keys present but none True -> False. "
            "The comparator emits changed=False when BOTH panes are missing, so a "
            "one-sided module failure reads as 'unchanged' rather than unevaluable. "
            "Continuous deeploc_top_prob / per-compartment probabilities are displayed "
            "but not scored"
        ),
        "none_conditions": "comparison.localization missing or no *_changed fields emitted",
        "orf_type_handling": "symmetric (whole-protein predictions on both panes)",
        "provisional": False,
    },
    {
        "criterion_id": "L2_targeting_change",
        "legacy_id": "F4",
        "axis": "functional",
        "category": "Localization",
        "name": "Signal/targeting-peptide change (SignalP + TargetP)",
        "fires_when": "any signalp/targetp *_changed is True (no numeric cutoff)",
        "source_module": "comparison.signalp / comparison.targetp",
        "source_field": "any key ending in '_changed' across both modules",
        "config_field": None,
        "threshold_value": "categorical (no cutoff)",
        "comparison": (
            "any *_changed True across signalp/targetp -> True (one available source "
            "suffices); both sources None -> None; else False — which includes "
            "'one ran and said no change, the other never ran'"
        ),
        "none_conditions": (
            "both comparison.signalp and comparison.targetp missing / no *_changed keys"
        ),
        "orf_type_handling": "symmetric (whole-protein predictions on both panes)",
        "provisional": False,
    },
    # ---------------- FUNCTIONAL: Mutation Landscape (M) ----------------
    {
        "criterion_id": "M1_pathogenic_variant_enrichment",
        "legacy_id": "F5",
        "axis": "functional",
        "category": "Mutation Landscape",
        "name": "Germline constraint / depletion in unique region (ESM-C + gnomAD)",
        "fires_when": (
            "constraint_enrichment >= {m1_constraint_enrichment_min} "
            "OR gnomad_depletion_ratio < {m1_depletion_ratio_max}"
        ),
        "source_module": "plm_vep + variant_intersection",
        "source_field": (
            "plm_vep.constraint_enrichment ; variant_intersection.gnomad_depletion_ratio"
        ),
        "config_field": "m1_constraint_enrichment_min",
        "secondary_config_fields": ["m1_depletion_ratio_max"],
        "comparison": (
            "OR of two branches pointing opposite directions: "
            "constrained = constraint_enrichment >= m1_constraint_enrichment_min; "
            "depleted = gnomad_depletion_ratio < m1_depletion_ratio_max (strict <). "
            "Constraint counts only when plm_vep top-level status == 'ok'; depletion "
            "needs only a numeric ratio (NOT status-gated, unlike M2). "
            "CAVEAT: when only one input exists the other branch is silently False, so "
            "a single available signal can produce a False that reads like a tested "
            "negative. Dormant: m1_min_pathogenic_in_unique, m1_llr_damaging_threshold"
        ),
        "none_conditions": (
            "ONLY when both inputs are missing (plm_vep status != ok AND no numeric "
            "depletion ratio). Ratios are None upstream when unique_nt<=0, shared_nt<=0, "
            "or shared density == 0 — so separate-ORF isoforms yield None"
        ),
        "orf_type_handling": (
            "unique/shared genomic intervals flip for TRUNCATED; PLM region space uses "
            "canonical protein coords for truncations, isoform coords otherwise "
            "(Annotated -> 'none'). Semantically interpretable ONLY on truncations: an "
            "extension's unique region was never coding, so ESM-C is out of distribution"
        ),
        "provisional": True,
    },
    {
        "criterion_id": "M2_clinical_variant_overlap",
        "legacy_id": "F6",
        "axis": "functional",
        "category": "Mutation Landscape",
        "name": "Disease-variant density enrichment (ClinVar + COSMIC, unique vs shared)",
        "fires_when": "disease_enrichment_ratio >= {m2_disease_enrichment_min}",
        "source_module": "variant_intersection",
        "source_field": "disease_enrichment_ratio",
        "config_field": "m2_disease_enrichment_min",
        "comparison": (
            "ratio >= m2_disease_enrichment_min -> True; else False. Per-nt density, "
            "unique vs shared. The default is a zero-point (parity), not a calibrated "
            "cutoff. Raw n_disease_in_{unique,shared}_region are reason-string context"
        ),
        "none_conditions": (
            "variant_intersection not run (summary status not ok) or ratio not numeric "
            "(zero region length / zero shared density)"
        ),
        "orf_type_handling": (
            "same unique/shared interval flip for TRUNCATED as M1; variants outside both "
            "coding regions are dropped before counting"
        ),
        "provisional": False,
    },
    # ---------------- FUNCTIONAL: Predicted Structure (P) ----------------
    {
        "criterion_id": "P1_structured_extension",
        "legacy_id": "F1",
        "axis": "functional",
        "category": "Predicted Structure",
        "name": "Differential region is confidently folded (mean pLDDT)",
        "fires_when": "plddt_diffregion_mean >= {p1_plddt_threshold}",
        "source_module": "structure",
        "source_field": "plddt_diffregion_mean",
        "config_field": "p1_plddt_threshold",
        "comparison": (
            "plddt >= p1_plddt_threshold -> True; else False. Scale-dependent: 0-1 for "
            "ESMFold2/Boltz-2 (0.70), would need 70.0 for AlphaFold-style backends"
        ),
        "none_conditions": (
            "structure status in {no_cache, too_long, failed, uniform_plddt} or plddt "
            "unavailable. NOTE 'partial' is NOT excluded here (it is for P2)"
        ),
        "orf_type_handling": (
            "diff-region pLDDT is read off the ISOFORM fold, or the CANONICAL fold when "
            "truncated = (orf_type=='truncated') or (isoform_start is None and "
            "canonical_start is not None). Measurement symmetric; narrative flips "
            "(gained fold vs lost fold)"
        ),
        "provisional": False,
    },
    {
        "criterion_id": "P2_shared_structural_change",
        "legacy_id": "F7",
        "axis": "functional",
        "category": "Predicted Structure",
        "name": "Shared-region structural change (Ca RMSD, Kabsch)",
        "fires_when": (
            "rmsd_shared >= {p2_rmsd_shared_min} A "
            "AND shared_region_len >= {p2_min_shared_len} aa "
            "AND min(plddt_shared iso, canon) >= {p2_plddt_min}"
        ),
        "source_module": "structure",
        "source_field": "rmsd_shared (gated by shared_region_len, plddt_shared_mean_{iso,canon})",
        "config_field": "p2_rmsd_shared_min",
        "secondary_config_fields": ["p2_min_shared_len", "p2_plddt_min"],
        "comparison": (
            "passed = rmsd_shared >= p2_rmsd_shared_min (A); reached only after 3 gates: "
            "rmsd_shared_status == 'ok', "
            "shared_region_len >= p2_min_shared_len (aa), "
            "min(plddt_shared_mean_isoform, plddt_shared_mean_canonical) >= p2_plddt_min. "
            "tm_score_shared is context only"
        ),
        "none_conditions": (
            "structure status unusable (no_cache/too_long/failed/uniform_plddt/partial — "
            "'partial' excluded here but not in P1), rmsd_shared_status != 'ok' "
            "(no_shared_region / unverified_alignment) or rmsd None, "
            "shared_region_len below the minimum, either shared pLDDT missing, or "
            "min(pLDDT) below the gate"
        ),
        "orf_type_handling": (
            "ONLY 'extended' and 'truncated' are ever evaluable — every other ORF type "
            "(uORF/uoORF/altORF/internal-OOF/3'UTR/Annotated) gets "
            "rmsd_shared_status='no_shared_region' -> None. Pairing: extension aligns "
            "iso[diff_end:] to full canonical; truncation aligns full isoform to "
            "can[diff_end:]; the initiator_met case drops the leading M"
        ),
        "provisional": True,
    },
    {
        "criterion_id": "P3_secondary_structure",
        "legacy_id": "(none)",
        "axis": "functional",
        "category": "Predicted Structure",
        "name": "Confident secondary-structure element in the differential region (P-SEA)",
        "fires_when": (
            "some unique/spans element has length >= {p3_min_sse_length} aa "
            "AND that element's plddt_mean >= {p3_min_sse_plddt}"
        ),
        "source_module": "structure",
        "source_field": "sse_all_elements (region in {unique, spans}) -> length, plddt_mean",
        "config_field": "p3_min_sse_length",
        "secondary_config_fields": ["p3_min_sse_plddt"],
        "comparison": (
            "True when SOME element with region in {unique, spans} has "
            "length >= p3_min_sse_length AND plddt_mean >= p3_min_sse_plddt — BOTH "
            "required, so a geometrically clean helix through a disordered stretch is "
            "not a finding. 'shared' elements are excluded. Never None once "
            "sse_status == 'ok': no qualifying element (or no elements at all) -> False. "
            "Uses the whole-protein scan at TRUE element length, not the window-clipped "
            "sse_diff_elements. Upstream floor structure/sse.py MIN_LENGTH="
            "{helix: 5, strand: 3} drops shorter elements before P3 ever sees them"
        ),
        "none_conditions": (
            "structure annotation missing, status in {no_cache, too_long, failed, "
            "uniform_plddt}, or sse_status != 'ok' (no_structure / no_diff_region)"
        ),
        "orf_type_handling": (
            "elements are read off the ISOFORM fold, or the CANONICAL fold when "
            "truncated (same predicate as P1) — the removed segment exists only there. "
            "Extension GAINS an element, truncation LOSES one"
        ),
        "provisional": True,
    },
    # ---------------- FUNCTIONAL: Structural Characteristics (S) ----------------
    {
        "criterion_id": "S1_domain_change",
        "legacy_id": "F3",
        "axis": "functional",
        "category": "Structural Characteristics",
        "name": "InterPro domain gained/lost in diff region",
        "fires_when": "n_real_domains_changed_in_diff_region >= 1 (hardcoded)",
        "source_module": "comparison.interproscan",
        "source_field": "n_real_domains_changed_in_diff_region",
        "config_field": None,
        "threshold_value": ">= 1 (hardcoded, no config field)",
        "comparison": (
            "n_real_domains_changed_in_diff_region >= 1 -> True; else False. A domain "
            "counts only when it (a) is a real functional domain — non-empty interpro_id "
            "and db not in DISORDER_STRUCTURAL_DBS {mobidb-lite, coils, low_complexity, "
            "signalp, phobius, tmhmm}; (b) STARTS inside the diff region; and (c) is "
            "absent from the other pane, so a merely repositioned domain is not counted"
        ),
        "none_conditions": (
            "comparison.interproscan missing, hits_canonical_status != 'ok' (S1 checks "
            "only the canonical pane), or count absent/non-numeric"
        ),
        "orf_type_handling": (
            "extension/uORF/altORF: gains counted on the ISOFORM pane; truncation: "
            "losses counted on the CANONICAL pane in canonical coords. "
            "diff_region is None -> None"
        ),
        "provisional": False,
    },
    {
        "criterion_id": "S2_biophysics",
        "legacy_id": "(none)",
        "axis": "functional",
        "category": "Structural Characteristics",
        "name": "Whole-protein biophysical shift (GRAVY / charge / disorder)",
        "fires_when": (
            "|gravy_delta| >= {s2_gravy_delta_min} "
            "OR |fraction_charged_delta| >= {s2_fraction_charged_delta_min} "
            "OR |disorder_delta| >= {s2_disorder_delta_min}"
        ),
        "source_module": "comparison.biophysics",
        "source_field": "gravy_delta ; fraction_charged_delta ; disorder_delta",
        "config_field": "s2_gravy_delta_min",
        "secondary_config_fields": [
            "s2_fraction_charged_delta_min",
            "s2_disorder_delta_min",
        ],
        "comparison": (
            "OR over three independent levers: |gravy_delta| >= s2_gravy_delta_min, "
            "|fraction_charged_delta| >= s2_fraction_charged_delta_min, "
            "|disorder_delta| >= s2_disorder_delta_min -> True; else False. "
            "Partial evaluability scores: 1 or 2 of the 3 deltas present is enough. "
            "Deltas are WHOLE-PROTEIN means (isoform - canonical), so a real but short "
            "diff region is length-diluted by design"
        ),
        "none_conditions": "comparison.biophysics missing, or zero of the three deltas numeric",
        "orf_type_handling": (
            "symmetric whole-protein means — needs no shared region, so unlike P2 it "
            "does score uORF/altORF"
        ),
        "provisional": True,
    },
    {
        "criterion_id": "S3_sae",
        "legacy_id": "(none)",
        "axis": "functional",
        "category": "Structural Characteristics",
        "name": "Differential SAE feature magnitude (ESM-C interpretability)",
        "fires_when": (
            "max(|top_gained_delta_max|, |top_lost_delta_max|) >= {s3_top_delta_min}"
        ),
        "source_module": "sae",
        "source_field": "max(|top_gained_delta_max|, |top_lost_delta_max|)",
        "config_field": "s3_top_delta_min",
        "comparison": (
            "top_delta >= s3_top_delta_min -> True; else False. Deltas are computed over "
            "the SHARED feature set (isoform-active AND canonical-active); "
            "n_isoform_only / n_canonical_only come from the symmetric difference and "
            "are reason-string context only. This REPLACED a presence check "
            "(n_isoform_only + n_canonical_only > 0) that was True for 100.0% of the "
            "6,462 isoforms in the full_catalog run. Upstream literals: "
            "DEFAULT_MIN_PREVALENCE=1 residue, DEFAULT_UNIQUE_TOP_N=30"
        ),
        "none_conditions": (
            "sae annotation missing, status != 'ok' (not_run / no_cache), or neither "
            "top delta is numeric"
        ),
        "orf_type_handling": "symmetric whole-protein feature comparison",
        # Calibrated on the genome-wide full_catalog run, not provisional:
        # |top delta| spans 0.36-30.85 (median 9.36); 10.0 fires on ~46%.
        "provisional": False,
    },
]

# Roll-up thresholds (not per-criterion; how True-counts become HC flags).
ROLLUP = [
    {
        "criterion_id": "(rollup)_existence_high_confidence",
        "legacy_id": "(rollup)",
        "axis": "existence",
        "category": "(roll-up)",
        "name": "existence_high_confidence flag (over 6 criteria: C1-C3, D1-D3)",
        "fires_when": (
            "existence_score >= {existence_high_threshold} "
            "AND existence_evaluable >= {existence_high_threshold}"
        ),
        "source_module": "EvidenceScoringModule",
        "source_field": "existence_score (count True) and existence_evaluable (count non-None)",
        "config_field": "existence_high_threshold",
        "comparison": (
            "existence_score >= existence_high_threshold AND "
            "existence_evaluable >= existence_high_threshold — the same field is used "
            "both as the score cutoff and as the evaluable-count floor"
        ),
        "none_conditions": "n/a (boolean flag); summary.existence_axis_complete = evaluable == 6",
        "orf_type_handling": "n/a",
        "provisional": False,
    },
    {
        "criterion_id": "(rollup)_functional_high_confidence",
        "legacy_id": "(rollup)",
        "axis": "functional",
        "category": "(roll-up)",
        "name": "functional_high_confidence flag (over 10 criteria: L1-L2, M1-M2, P1-P3, S1-S3)",
        "fires_when": (
            "functional_score >= {functional_high_threshold} "
            "AND functional_evaluable >= {functional_high_threshold}"
        ),
        "source_module": "EvidenceScoringModule",
        "source_field": "functional_score (count True) and functional_evaluable (count non-None)",
        "config_field": "functional_high_threshold",
        "comparison": (
            "functional_score >= functional_high_threshold AND "
            "functional_evaluable >= functional_high_threshold — the same field is used "
            "both as the score cutoff and as the evaluable-count floor"
        ),
        "none_conditions": "n/a (boolean flag); summary.functional_axis_complete = evaluable == 10",
        "orf_type_handling": "n/a",
        "provisional": False,
    },
]


def _check_config_fields(cfg: ScoringConfig) -> None:
    """Fail loudly, and all at once, when a ``config_field`` no longer exists.

    ScoringConfig is the source of truth for every threshold here, and this table
    only names its fields — so a rename in config.py silently invalidates rows
    until something dereferences one. The CDLMPS rename (E1-E6 / F1-F7 -> C/D/L/
    M/P/S) did exactly that: ``e1_pident_min`` and friends lingered here long
    after the fields were gone. Raising per-field on first use meant fixing one
    name, re-running, and discovering the next; this reports the whole set.

    Covers ``secondary_config_fields`` too, so the extra levers of the
    multi-threshold criteria (M1, P2, P3, S2) are guarded rather than transcribed
    into the untestable prose of ``comparison``.
    """
    missing = []
    for entry in CRITERIA + ROLLUP:
        fields = [entry.get("config_field"), *entry.get("secondary_config_fields", [])]
        missing.extend(
            (entry["criterion_id"], field)
            for field in fields
            if field and not hasattr(cfg, field)
        )
    if missing:
        listed = "\n".join(f"  {cid}: {field}" for cid, field in missing)
        raise AttributeError(
            f"{len(missing)} config_field name(s) no longer exist on ScoringConfig "
            f"(src/swissisoform/config.py):\n{listed}\n"
            "Update this table to the current field names."
        )


def _match_series() -> dict[str, dict]:
    """Map each criterion id to the series holding its primary scored quantity.

    ``SERIES`` is per plottable quantity, so the multi-lever criteria have
    several. Pick the one whose ``config_field`` is the criterion's own primary
    threshold (M1 -> constraint enrichment, P2 -> shared RMSD, S2 -> GRAVY), and
    take the sole candidate where there is only one (S1, which has no config
    field at all). Criteria with no numeric series — L1, L2, P3 — are absent.
    """
    by_criterion: dict[str, list[dict]] = {}
    for spec in SERIES:
        by_criterion.setdefault(spec["criterion"], []).append(spec)

    matched: dict[str, dict] = {}
    for entry in CRITERIA:
        candidates = by_criterion.get(entry["criterion_id"], [])
        if len(candidates) == 1:
            matched[entry["criterion_id"]] = candidates[0]
        elif candidates:
            primary = next(
                (c for c in candidates if c.get("config_field") == entry.get("config_field")),
                None,
            )
            if primary is not None:
                matched[entry["criterion_id"]] = primary
    return matched


def distribution_summary() -> dict[str, dict[str, float]]:
    """Return the five-number summary per criterion from the genome-wide run.

    Empty when no run is on disk — the table is a config export first and must
    still write without one.
    """
    try:
        files = shard_files(DEFAULT_SHARDS)
    except SystemExit:
        print("no genome-wide run found — dist_* columns left empty")
        return {}

    matched = _match_series()
    columns = sorted({col for spec in matched.values() for col in spec["columns"]})
    print(f"reading {len(files)} shard(s) for the dist_* columns (takes a minute) ...")
    df = load_frame(files, columns)

    summary: dict[str, dict[str, float]] = {}
    for criterion, spec in matched.items():
        values = spec["transform"](df).dropna()
        if values.empty:
            continue
        q = values.quantile(list(_QUANTILES))
        summary[criterion] = {
            name: round(float(q[cut]), 4) for name, cut in zip(DIST_FIELDS, _QUANTILES)
        }
    print(f"summarized {len(summary)} of {len(CRITERIA)} criteria over {len(df):,} isoforms")
    return summary


def build_rows(cfg: ScoringConfig, dists: dict[str, dict[str, float]] | None = None) -> list[dict]:
    """Resolve each criterion's live threshold(s) from ``cfg`` and normalize rows.

    ``fires_when`` is formatted against every ScoringConfig field, so a renamed
    field raises ``KeyError`` here rather than silently shipping a stale number.
    """
    _check_config_fields(cfg)
    cfg_values = asdict(cfg)
    rows = []
    for entry in CRITERIA + ROLLUP:
        row = {k: entry.get(k, "") for k in FIELDNAMES}
        row["fires_when"] = entry["fires_when"].format(**cfg_values)
        field = entry.get("config_field")
        if field:
            row["threshold_value"] = getattr(cfg, field)
        elif "threshold_value" in entry:
            row["threshold_value"] = entry["threshold_value"]
        row["config_field"] = field or ""
        row["secondary_config_fields"] = "; ".join(
            f"{name}={getattr(cfg, name)}" for name in entry.get("secondary_config_fields", [])
        )
        row.update(dict.fromkeys(DIST_FIELDS, ""))
        row.update((dists or {}).get(entry["criterion_id"], {}))
        rows.append(row)
    return rows


def main() -> None:
    """Write the cutoff table, thresholds resolved against default ScoringConfig."""
    cfg = ScoringConfig()
    rows = build_rows(cfg, distribution_summary())

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    n_criteria = len(CRITERIA)
    n_rollup = len(ROLLUP)
    print(f"Wrote {len(rows)} rows ({n_criteria} criteria + {n_rollup} roll-up) -> {OUT_CSV}")
    # Spot-check a few live-resolved thresholds, including both multi-lever ones.
    print(f"  C3 c3_phylop_min      = {cfg.c3_phylop_min}")
    print(f"  P1 p1_plddt_threshold = {cfg.p1_plddt_threshold}")
    print(f"  P2 p2_rmsd_shared_min = {cfg.p2_rmsd_shared_min}")
    print(f"  P3 p3_min_sse_length  = {cfg.p3_min_sse_length} / {cfg.p3_min_sse_plddt} pLDDT")
    print(f"  S3 s3_top_delta_min   = {cfg.s3_top_delta_min}")


if __name__ == "__main__":
    main()
