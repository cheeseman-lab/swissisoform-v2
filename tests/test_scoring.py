"""Tests for EvidenceScoringModule — per-criterion + aggregate behaviour."""

from __future__ import annotations

from swissisoform.config import PipelineConfig, ScoringConfig
from swissisoform.models import (
    CellLineExpression,
    ORFType,
    TranslationInitiationSite,
)
from swissisoform.modules.scoring import (
    EXISTENCE_CRITERIA,
    FUNCTIONAL_CRITERIA,
    EvidenceScoringModule,
    _c1_primate_conservation,
    _c2_mammalian_conservation,
    _c3_phylop_coding_selection,
    _d1_multi_cell_line,
    _d2_initiation_efficiency,
    _d3_mass_spec,
    _p1_structured_extension,
    _l1_localization_change,
    _s1_domain_change,
    _l2_targeting_change,
    _m1_pathogenic_variant_enrichment,
    _m2_clinical_variant_overlap,
    _s2_biophysics,
    _s3_sae,
)


def _site() -> TranslationInitiationSite:
    return TranslationInitiationSite(
        tis_id="chr1:1000:+:ATG",
        gene_name="TEST",
        transcript_id="ENST_TEST",
        chrom="chr1",
        position=1000,
        strand="+",
        start_codon="ATG",
        orf_type=ORFType.EXTENDED,
        canonical_protein="MAAA",
        isoform_protein="MXXMAAA",
    )


# ---------------------------------------------------------------------------
# Existence — value paths
# ---------------------------------------------------------------------------


class TestE1PrimateConservation:
    def test_passes(self):
        site = _site()
        site.isoform_annotations["conservation_frame"] = {
            "primate_mean_pident": 0.9,
            "primate_frac_intact": 0.8,
            "summary": {"status": "ok"},
        }
        res = _c1_primate_conservation(site, ScoringConfig(e1_pident_min=0.8))
        assert res.value is True

    def test_fails_below_threshold(self):
        site = _site()
        site.isoform_annotations["conservation_frame"] = {
            "primate_mean_pident": 0.6,
            "primate_frac_intact": 0.2,
            "summary": {"status": "ok"},
        }
        res = _c1_primate_conservation(site, ScoringConfig(e1_pident_min=0.8))
        assert res.value is False

    def test_not_run(self):
        site = _site()
        site.isoform_annotations["conservation_frame"] = {
            "summary": {"status": "not_run"},
        }
        res = _c1_primate_conservation(site, ScoringConfig())
        assert res.value is None
        assert "not run" in res.reason

    def test_pident_missing(self):
        """conservation_frame ran but mean_pident unavailable → None."""
        site = _site()
        site.isoform_annotations["conservation_frame"] = {
            "primate_frac_intact": 0.8,
            "summary": {"status": "ok"},
        }
        res = _c1_primate_conservation(site, ScoringConfig())
        assert res.value is None
        assert "primate_mean_pident" in res.reason

    def test_missing_module(self):
        res = _c1_primate_conservation(_site(), ScoringConfig())
        assert res.value is None


class TestE2MammalianConservation:
    def test_passes(self):
        site = _site()
        site.isoform_annotations["conservation_frame"] = {
            "mammalian_mean_pident": 0.6,
            "mammalian_frac_intact": 0.4,
            "summary": {"status": "ok"},
        }
        res = _c2_mammalian_conservation(site, ScoringConfig(e2_pident_min=0.5))
        assert res.value is True

    def test_fails_below_threshold(self):
        site = _site()
        site.isoform_annotations["conservation_frame"] = {
            "mammalian_mean_pident": 0.3,
            "mammalian_frac_intact": 0.4,
            "summary": {"status": "ok"},
        }
        res = _c2_mammalian_conservation(site, ScoringConfig(e2_pident_min=0.5))
        assert res.value is False


class TestE3Phylop:
    def test_passes(self):
        site = _site()
        site.isoform_annotations["conservation"] = {
            "phylop_unique_region_mean": 2.5,
            "summary": {"region_status": "ok"},
        }
        # Default e3_phylop_min is 2.0 (absolute purifying-selection anchor).
        res = _c3_phylop_coding_selection(site, ScoringConfig())
        assert res.value is True

    def test_fails_below_threshold(self):
        site = _site()
        site.isoform_annotations["conservation"] = {
            "phylop_unique_region_mean": 1.5,
            "summary": {"region_status": "ok"},
        }
        res = _c3_phylop_coding_selection(site, ScoringConfig())
        assert res.value is False

    def test_region_not_ok(self):
        site = _site()
        site.isoform_annotations["conservation"] = {
            "summary": {"region_status": "no_skeleton"},
        }
        res = _c3_phylop_coding_selection(site, ScoringConfig())
        assert res.value is None


class TestE4MultiCellLine:
    def test_passes(self):
        site = _site()
        for cl in ("HeLa", "K562", "U2OS"):
            site.expression[cl] = CellLineExpression(raw_count=10, cpm=1.0, p_value=0.01)
        res = _d1_multi_cell_line(site, ScoringConfig(min_cell_lines=3))
        assert res.value is True

    def test_fails(self):
        site = _site()
        site.expression["HeLa"] = CellLineExpression(raw_count=10, cpm=1.0, p_value=0.01)
        res = _d1_multi_cell_line(site, ScoringConfig(min_cell_lines=3))
        assert res.value is False


class TestE5InitiationEfficiency:
    def test_none_available(self):
        site = _site()
        site.expression["HeLa"] = CellLineExpression(
            raw_count=10, cpm=1.0, p_value=0.01, initiation_efficiency=None
        )
        res = _d2_initiation_efficiency(site, ScoringConfig())
        assert res.value is None

    def test_passes(self):
        site = _site()
        site.expression["HeLa"] = CellLineExpression(
            raw_count=10, cpm=1.0, p_value=0.01, initiation_efficiency=0.05
        )
        res = _d2_initiation_efficiency(site, ScoringConfig(initiation_efficiency_min=0.01))
        assert res.value is True


class TestE6MassSpec:
    def test_passes_when_validated(self):
        site = _site()
        site.isoform_annotations["massspec"] = {
            "hits": [
                {"unique_to_isoform": True, "validated": True},
                {"unique_to_isoform": True, "validated": False},
            ],
            "summary": {"pepquery_run": True},
        }
        res = _d3_mass_spec(
            site, ScoringConfig(massspec_unique_peptides_min=1)
        )
        assert res.value is True

    def test_fails_when_no_validated_hits(self):
        site = _site()
        site.isoform_annotations["massspec"] = {
            "hits": [
                {"unique_to_isoform": True, "validated": False},
            ],
            "summary": {"pepquery_run": True},
        }
        res = _d3_mass_spec(
            site, ScoringConfig(massspec_unique_peptides_min=1)
        )
        assert res.value is False

    def test_none_when_pepquery_not_run(self):
        site = _site()
        site.isoform_annotations["massspec"] = {
            "hits": [{"unique_to_isoform": True, "validated": None}],
            "summary": {"pepquery_run": False},
        }
        res = _d3_mass_spec(site, ScoringConfig())
        assert res.value is None
        assert "pepquery" in res.reason.lower()


# ---------------------------------------------------------------------------
# Functional
# ---------------------------------------------------------------------------


def _distinct_biophysics() -> dict[str, float]:
    """Whole-protein biophysics deltas that clear a provisional S2 cutoff (gravy)."""
    return {"gravy_delta": 0.5, "fraction_charged_delta": 0.0, "disorder_delta": 0.0}


def _identical_biophysics() -> dict[str, float]:
    """Whole-protein biophysics deltas below every S2 cutoff."""
    return {"gravy_delta": 0.0, "fraction_charged_delta": 0.0, "disorder_delta": 0.0}


class TestP1StructuredExtension:
    """P1 is folding-only — diff-region pLDDT vs threshold (distinctness → S2)."""

    def test_no_data(self):
        """No structure annotation → None."""
        res = _p1_structured_extension(_site(), ScoringConfig())
        assert res.value is None

    def test_folded_true(self):
        """Folded diff region → True (no biophysics needed)."""
        site = _site()
        site.isoform_annotations["structure"] = {
            "plddt_diffregion_mean": 0.85,
            "status": "ok",
        }
        res = _p1_structured_extension(site, ScoringConfig())
        assert res.value is True

    def test_folded_true_regardless_of_biophysics(self):
        """P1 no longer couples to distinctness — folded is True even if identical."""
        site = _site()
        site.isoform_annotations["structure"] = {
            "plddt_diffregion_mean": 0.85,
            "status": "ok",
        }
        site.comparison["biophysics"] = _identical_biophysics()
        res = _p1_structured_extension(site, ScoringConfig())
        assert res.value is True

    def test_below_threshold(self):
        """Unfolded diff region → False."""
        site = _site()
        site.isoform_annotations["structure"] = {
            "plddt_diffregion_mean": 0.40,
            "status": "ok",
        }
        res = _p1_structured_extension(site, ScoringConfig())
        assert res.value is False

    def test_uniform_plddt_excluded(self):
        site = _site()
        site.isoform_annotations["structure"] = {
            "plddt_diffregion_mean": 0.85,
            "status": "uniform_plddt",
        }
        res = _p1_structured_extension(site, ScoringConfig())
        assert res.value is None

    def test_too_long_excluded(self):
        site = _site()
        site.isoform_annotations["structure"] = {
            "plddt_diffregion_mean": None,
            "status": "too_long",
        }
        res = _p1_structured_extension(site, ScoringConfig())
        assert res.value is None


class TestS2Biophysics:
    """S2 — whole-protein biophysical shift over gravy / fraction_charged / disorder."""

    def test_no_comparison_none(self):
        """No biophysics comparison → None."""
        res = _s2_biophysics(_site(), ScoringConfig())
        assert res.value is None

    def test_shifted_true(self):
        """A whole-protein delta clears its cutoff → True."""
        site = _site()
        site.comparison["biophysics"] = _distinct_biophysics()
        res = _s2_biophysics(site, ScoringConfig())
        assert res.value is True
        assert res.name == "S2_biophysics"

    def test_unshifted_false(self):
        """All whole-protein deltas below cutoff → False."""
        site = _site()
        site.comparison["biophysics"] = _identical_biophysics()
        res = _s2_biophysics(site, ScoringConfig())
        assert res.value is False

    def test_no_descriptors_none(self):
        """Comparison present but no *_delta lever numeric → None."""
        site = _site()
        site.comparison["biophysics"] = {"pI_delta": 2.0}
        res = _s2_biophysics(site, ScoringConfig())
        assert res.value is None


class TestS3Sae:
    """S3 — presence of differential (gained/lost) interpretable SAE features."""

    def test_no_annotation_none(self):
        res = _s3_sae(_site(), ScoringConfig())
        assert res.value is None

    def test_status_not_ok_none(self):
        site = _site()
        site.isoform_annotations["sae"] = {
            "status": "no_cache",
            "n_isoform_only": 3,
            "n_canonical_only": 1,
        }
        res = _s3_sae(site, ScoringConfig())
        assert res.value is None

    def test_deltas_unavailable_none(self):
        """No shared-feature deltas → nothing to measure magnitude on."""
        site = _site()
        site.isoform_annotations["sae"] = {
            "status": "ok",
            "n_isoform_only": 120,
            "n_canonical_only": 77,
            "top_gained_delta_max": None,
            "top_lost_delta_max": None,
        }
        res = _s3_sae(site, ScoringConfig())
        assert res.value is None

    def test_large_shift_true(self):
        """S3 scores on magnitude: the top shared-feature |delta| clears the gate."""
        cfg = ScoringConfig()
        site = _site()
        site.isoform_annotations["sae"] = {
            "status": "ok",
            "n_isoform_only": 2,
            "n_canonical_only": 0,
            "top_gained_delta_max": cfg.s3_top_delta_min + 2.5,
            "top_lost_delta_max": -1.0,
        }
        res = _s3_sae(site, cfg)
        assert res.value is True
        assert res.name == "S3_sae"

    def test_small_shift_false(self):
        """Hundreds of differential features do NOT make it True without magnitude.

        The old presence check (n_isoform_only + n_canonical_only > 0) was True for
        100% of the genome-wide run; this is the case that regression-guards it.
        """
        site = _site()
        site.isoform_annotations["sae"] = {
            "status": "ok",
            "n_isoform_only": 300,
            "n_canonical_only": 250,
            "top_gained_delta_max": 1.2,
            "top_lost_delta_max": -0.8,
        }
        res = _s3_sae(site, ScoringConfig())
        assert res.value is False

    def test_negative_shift_counts_by_magnitude(self):
        """A strongly *lost* feature is as scoreable as a gained one (abs value)."""
        cfg = ScoringConfig()
        site = _site()
        site.isoform_annotations["sae"] = {
            "status": "ok",
            "n_isoform_only": 1,
            "n_canonical_only": 1,
            "top_gained_delta_max": 0.5,
            "top_lost_delta_max": -(cfg.s3_top_delta_min + 5.0),
        }
        res = _s3_sae(site, cfg)
        assert res.value is True


class TestF2LocalizationChange:
    def test_changed(self):
        site = _site()
        site.comparison["localization"] = {
            "predicted_location_changed": True,
            "predicted_location_canonical": "Cytoplasm",
            "predicted_location_isoform": "Nucleus",
        }
        res = _l1_localization_change(site, ScoringConfig())
        assert res.value is True

    def test_unchanged(self):
        site = _site()
        site.comparison["localization"] = {
            "predicted_location_changed": False,
        }
        res = _l1_localization_change(site, ScoringConfig())
        assert res.value is False

    def test_missing(self):
        res = _l1_localization_change(_site(), ScoringConfig())
        assert res.value is None


class TestF3DomainChange:
    def test_no_ips_data(self):
        """No IPS comparison → None."""
        assert _s1_domain_change(_site(), ScoringConfig()).value is None

    def test_real_domain_changed_fires(self):
        """≥1 real InterPro domain gained/lost in the diff region → True."""
        site = _site()
        site.comparison["interproscan"] = {
            "n_real_domains_changed_in_diff_region": 1,
            "hits_canonical_status": "ok",
        }
        res = _s1_domain_change(site, ScoringConfig())
        assert res.value is True

    def test_no_real_domain_changed_false(self):
        """Zero real domains changed → False."""
        site = _site()
        site.comparison["interproscan"] = {
            "n_real_domains_changed_in_diff_region": 0,
            "hits_canonical_status": "ok",
        }
        res = _s1_domain_change(site, ScoringConfig())
        assert res.value is False

    def test_count_unavailable_none(self):
        """Comparator ran but did not emit the real-domain count → None."""
        site = _site()
        site.comparison["interproscan"] = {
            "hits_canonical_status": "ok",
        }
        res = _s1_domain_change(site, ScoringConfig())
        assert res.value is None
        assert "n_real_domains_changed_in_diff_region" in res.reason

    def test_canonical_status_not_ok_none(self):
        """Comparator present but canonical IPS run didn't complete → None."""
        site = _site()
        site.comparison["interproscan"] = {
            "n_real_domains_changed_in_diff_region": 1,
        }
        res = _s1_domain_change(site, ScoringConfig())
        assert res.value is None


class TestF4TargetingChange:
    def test_no_comparator_data(self):
        """No SignalP/TargetP comparison → None."""
        assert _l2_targeting_change(_site(), ScoringConfig()).value is None

    def test_signalp_change(self):
        site = _site()
        site.comparison["signalp"] = {"signalp_prediction_changed": True}
        res = _l2_targeting_change(site, ScoringConfig())
        assert res.value is True

    def test_targetp_change(self):
        site = _site()
        site.comparison["targetp"] = {"targetp_prediction_changed": True}
        res = _l2_targeting_change(site, ScoringConfig())
        assert res.value is True

    def test_no_change(self):
        site = _site()
        site.comparison["signalp"] = {"signalp_prediction_changed": False}
        site.comparison["targetp"] = {"targetp_prediction_changed": False}
        res = _l2_targeting_change(site, ScoringConfig())
        assert res.value is False


class TestF5GermlineToleranceConstraint:
    def test_no_data(self):
        """Empty site → None (neither plm_vep nor variant_intersection)."""
        res = _m1_pathogenic_variant_enrichment(_site(), ScoringConfig())
        assert res.value is None
        assert "constraint_enrichment" in res.reason

    def test_constraint_enrichment_true(self):
        """ESM-2 constraint_enrichment ≥ threshold → True (gnomAD absent)."""
        site = _site()
        site.isoform_annotations["plm_vep"] = {
            "constraint_enrichment": 2.5,
            "status": "ok",
        }
        res = _m1_pathogenic_variant_enrichment(site, ScoringConfig())
        assert res.value is True

    def test_gnomad_depletion_true(self):
        """gnomAD depletion ratio below threshold → True (plm absent)."""
        site = _site()
        site.isoform_annotations["variant_intersection"] = {
            "gnomad_depletion_ratio": 0.5,
            "summary": {"status": "ok"},
        }
        res = _m1_pathogenic_variant_enrichment(site, ScoringConfig())
        assert res.value is True

    def test_neither_branch_fires_false(self):
        """Both signals present but neither passes → False."""
        site = _site()
        site.isoform_annotations["plm_vep"] = {
            "constraint_enrichment": 1.0,
            "status": "ok",
        }
        site.isoform_annotations["variant_intersection"] = {
            "gnomad_depletion_ratio": 1.2,
            "summary": {"status": "ok"},
        }
        res = _m1_pathogenic_variant_enrichment(site, ScoringConfig())
        assert res.value is False

    def test_plm_not_ok_ignored(self):
        """plm_vep status not ok → constraint ignored; falls back to gnomAD."""
        site = _site()
        site.isoform_annotations["plm_vep"] = {
            "constraint_enrichment": 99.0,
            "status": "not_run",
        }
        site.isoform_annotations["variant_intersection"] = {
            "gnomad_depletion_ratio": 0.5,
            "summary": {"status": "ok"},
        }
        res = _m1_pathogenic_variant_enrichment(site, ScoringConfig())
        # plm ignored (not ok); depletion 0.5 < 0.80 → True
        assert res.value is True

    def test_threshold_respected(self):
        """Custom thresholds gate both branches."""
        site = _site()
        site.isoform_annotations["plm_vep"] = {
            "constraint_enrichment": 2.5,
            "status": "ok",
        }
        assert _m1_pathogenic_variant_enrichment(site, ScoringConfig()).value is True
        strict = ScoringConfig(f5_constraint_enrichment_min=3.0)
        assert _m1_pathogenic_variant_enrichment(site, strict).value is False


class TestF6ClinicalVariantOverlap:
    def test_positive(self):
        site = _site()
        site.isoform_annotations["variant_intersection"] = {
            "disease_enrichment_ratio": 2.0,
            "n_disease_in_unique_region": 2,
            "n_disease_in_shared_region": 1,
            "summary": {"status": "ok"},
        }
        res = _m2_clinical_variant_overlap(site, ScoringConfig())
        assert res.value is True
        assert "disease_enrichment_ratio=" in res.reason

    def test_below_threshold(self):
        site = _site()
        site.isoform_annotations["variant_intersection"] = {
            "disease_enrichment_ratio": 0.5,
            "n_disease_in_unique_region": 1,
            "n_disease_in_shared_region": 4,
            "summary": {"status": "ok"},
        }
        res = _m2_clinical_variant_overlap(site, ScoringConfig())
        assert res.value is False

    def test_ratio_missing_none(self):
        site = _site()
        site.isoform_annotations["variant_intersection"] = {
            "n_disease_in_unique_region": 1,
            "summary": {"status": "ok"},
        }
        res = _m2_clinical_variant_overlap(site, ScoringConfig())
        assert res.value is None
        assert "disease_enrichment_ratio" in res.reason


# ---------------------------------------------------------------------------
# Module integration
# ---------------------------------------------------------------------------


class TestModuleIntegration:
    def test_all_stubs_gives_mostly_none(self):
        """A TIS with no upstream annotations scores 0 evaluable per axis where stubs dominate."""
        mod = EvidenceScoringModule(PipelineConfig(scoring=ScoringConfig()))
        site = _site()  # no upstream data
        out = mod.annotate_site(site)
        # E4 always evaluates (reads site.expression), others None
        assert out["existence_evaluable"] == 1
        assert out["existence_score"] == 0
        # All functional criteria correctly return None on empty input:
        # F1 (structure), F3 (IPS), F5 (vi+plm) check status fields;
        # F2/F4/F6 need comparator/variant data not present here.
        assert out["functional_evaluable"] == 0

    def test_score_counts_only_true(self):
        cfg = ScoringConfig(
            e1_pident_min=0.8,
            e2_pident_min=0.5,
            min_cell_lines=1,
            existence_high_threshold=3,
            functional_high_threshold=1,
        )
        mod = EvidenceScoringModule(PipelineConfig(scoring=cfg))
        site = _site()
        site.isoform_annotations["conservation_frame"] = {
            "primate_mean_pident": 0.9,
            "mammalian_mean_pident": 0.6,
            "primate_frac_intact": 0.8,
            "mammalian_frac_intact": 0.4,
            "summary": {"status": "ok"},
        }
        site.isoform_annotations["variant_intersection"] = {
            "disease_enrichment_ratio": 2.0,
            "n_disease_in_unique_region": 2,
            "n_disease_in_shared_region": 1,
            "summary": {"status": "ok"},
        }
        site.expression["HeLa"] = CellLineExpression(raw_count=10, cpm=1.0, p_value=0.01)
        out = mod.annotate_site(site)

        # E1, E2, E4 True → existence_score = 3
        assert out["existence_score"] == 3
        assert out["existence_high_confidence"] is True
        # F5 (gnomAD depletion present? no — only disease ratio) and F6.
        # variant_intersection carries no gnomad_depletion_ratio and no plm_vep,
        # so F5 → None; F6 True → functional_score = 1.
        assert out["functional_score"] == 1
        assert out["functional_high_confidence"] is True
        assert out["criteria"]["C1_primate_conservation"] is True
        assert out["criteria"]["M2_clinical_variant_overlap"] is True

    def test_run_populates_annotations(self):
        mod = EvidenceScoringModule(PipelineConfig(scoring=ScoringConfig()))
        site = _site()
        mod.run([site])
        assert "scoring" in site.isoform_annotations
        assert "existence_score" in site.isoform_annotations["scoring"]


class TestMetadata:
    def test_counts(self):
        assert len(EXISTENCE_CRITERIA) == 6
        assert len(FUNCTIONAL_CRITERIA) == 10
        assert len(EXISTENCE_CRITERIA) + len(FUNCTIONAL_CRITERIA) == 16

    def test_output_columns(self):
        cols = EvidenceScoringModule.OUTPUT_COLUMNS
        assert "scoring_existence_score" in cols
        assert "scoring_functional_score" in cols
        assert "scoring_criteria" in cols
