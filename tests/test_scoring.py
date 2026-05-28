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
    _e1_primate_conservation,
    _e2_mammalian_conservation,
    _e3_phylop_coding_selection,
    _e4_multi_cell_line,
    _e5_initiation_efficiency,
    _e6_mass_spec,
    _f1_structured_extension,
    _f2_localization_change,
    _f3_domain_change,
    _f4_targeting_change,
    _f5_pathogenic_variant_enrichment,
    _f6_clinical_variant_overlap,
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
            "primate_frac_intact": 0.8,
            "summary": {"status": "ok"},
        }
        res = _e1_primate_conservation(site, ScoringConfig(primate_frac_intact_min=0.5))
        assert res.value is True

    def test_fails_below_threshold(self):
        site = _site()
        site.isoform_annotations["conservation_frame"] = {
            "primate_frac_intact": 0.2,
            "summary": {"status": "ok"},
        }
        res = _e1_primate_conservation(site, ScoringConfig(primate_frac_intact_min=0.5))
        assert res.value is False

    def test_not_run(self):
        site = _site()
        site.isoform_annotations["conservation_frame"] = {
            "summary": {"status": "not_run"},
        }
        res = _e1_primate_conservation(site, ScoringConfig())
        assert res.value is None
        assert "not run" in res.reason

    def test_missing_module(self):
        res = _e1_primate_conservation(_site(), ScoringConfig())
        assert res.value is None


class TestE2MammalianConservation:
    def test_passes(self):
        site = _site()
        site.isoform_annotations["conservation_frame"] = {
            "mammalian_frac_intact": 0.4,
            "summary": {"status": "ok"},
        }
        res = _e2_mammalian_conservation(
            site, ScoringConfig(mammalian_frac_intact_min=0.3)
        )
        assert res.value is True


class TestE3Phylop:
    def test_passes(self):
        site = _site()
        site.isoform_annotations["conservation"] = {
            "phylop_unique_region_mean": 2.5,
            "summary": {"region_status": "ok"},
        }
        res = _e3_phylop_coding_selection(site, ScoringConfig(phylop_coding_min=1.0))
        assert res.value is True

    def test_fails_below_threshold(self):
        site = _site()
        site.isoform_annotations["conservation"] = {
            "phylop_unique_region_mean": 0.2,
            "summary": {"region_status": "ok"},
        }
        res = _e3_phylop_coding_selection(site, ScoringConfig(phylop_coding_min=1.0))
        assert res.value is False

    def test_region_not_ok(self):
        site = _site()
        site.isoform_annotations["conservation"] = {
            "summary": {"region_status": "no_skeleton"},
        }
        res = _e3_phylop_coding_selection(site, ScoringConfig())
        assert res.value is None


class TestE4MultiCellLine:
    def test_passes(self):
        site = _site()
        for cl in ("HeLa", "K562", "U2OS"):
            site.expression[cl] = CellLineExpression(raw_count=10, cpm=1.0, p_value=0.01)
        res = _e4_multi_cell_line(site, ScoringConfig(min_cell_lines=3))
        assert res.value is True

    def test_fails(self):
        site = _site()
        site.expression["HeLa"] = CellLineExpression(raw_count=10, cpm=1.0, p_value=0.01)
        res = _e4_multi_cell_line(site, ScoringConfig(min_cell_lines=3))
        assert res.value is False


class TestE5InitiationEfficiency:
    def test_none_available(self):
        site = _site()
        site.expression["HeLa"] = CellLineExpression(
            raw_count=10, cpm=1.0, p_value=0.01, initiation_efficiency=None
        )
        res = _e5_initiation_efficiency(site, ScoringConfig())
        assert res.value is None

    def test_passes(self):
        site = _site()
        site.expression["HeLa"] = CellLineExpression(
            raw_count=10, cpm=1.0, p_value=0.01, initiation_efficiency=0.05
        )
        res = _e5_initiation_efficiency(site, ScoringConfig(initiation_efficiency_min=0.01))
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
        res = _e6_mass_spec(
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
        res = _e6_mass_spec(
            site, ScoringConfig(massspec_unique_peptides_min=1)
        )
        assert res.value is False

    def test_none_when_pepquery_not_run(self):
        site = _site()
        site.isoform_annotations["massspec"] = {
            "hits": [{"unique_to_isoform": True, "validated": None}],
            "summary": {"pepquery_run": False},
        }
        res = _e6_mass_spec(site, ScoringConfig())
        assert res.value is None
        assert "pepquery" in res.reason.lower()


# ---------------------------------------------------------------------------
# Functional
# ---------------------------------------------------------------------------


class TestF1StructuredExtension:
    def test_no_data(self):
        """No structure annotation → None."""
        res = _f1_structured_extension(_site(), ScoringConfig())
        assert res.value is None

    def test_above_threshold(self):
        site = _site()
        site.isoform_annotations["structure"] = {
            "plddt_diffregion_mean": 0.85,
            "status": "ok",
        }
        res = _f1_structured_extension(site, ScoringConfig())
        assert res.value is True

    def test_below_threshold(self):
        site = _site()
        site.isoform_annotations["structure"] = {
            "plddt_diffregion_mean": 0.40,
            "status": "ok",
        }
        res = _f1_structured_extension(site, ScoringConfig())
        assert res.value is False

    def test_uniform_plddt_excluded(self):
        site = _site()
        site.isoform_annotations["structure"] = {
            "plddt_diffregion_mean": 0.85,
            "status": "uniform_plddt",
        }
        res = _f1_structured_extension(site, ScoringConfig())
        assert res.value is None

    def test_too_long_excluded(self):
        site = _site()
        site.isoform_annotations["structure"] = {
            "plddt_diffregion_mean": None,
            "status": "too_long",
        }
        res = _f1_structured_extension(site, ScoringConfig())
        assert res.value is None


class TestF2LocalizationChange:
    def test_changed(self):
        site = _site()
        site.comparison["localization"] = {
            "predicted_location_changed": True,
            "predicted_location_canonical": "Cytoplasm",
            "predicted_location_isoform": "Nucleus",
        }
        res = _f2_localization_change(site, ScoringConfig())
        assert res.value is True

    def test_unchanged(self):
        site = _site()
        site.comparison["localization"] = {
            "predicted_location_changed": False,
        }
        res = _f2_localization_change(site, ScoringConfig())
        assert res.value is False

    def test_missing(self):
        res = _f2_localization_change(_site(), ScoringConfig())
        assert res.value is None


class TestF3DomainChange:
    def test_no_ips_data(self):
        """No IPS annotation → None (status=no_data)."""
        assert _f3_domain_change(_site(), ScoringConfig()).value is None

    def test_hit_starting_in_diff_fires(self):
        """A hit whose pos is INSIDE the diff window counts as a domain gain."""
        from swissisoform.models import DifferentialRegion
        site = _site()
        site.diff_region = DifferentialRegion(isoform_start=0, isoform_end=10, sequence="X" * 10)
        site.isoform_annotations["interproscan"] = {"summary": {"status": "ok"}}
        site.comparison["interproscan"] = {
            "hits_in_diff_region": [{"pos": 3, "end": 7, "name": "PF_test"}],
            "hits_source_pane": "isoform",
        }
        res = _f3_domain_change(site, ScoringConfig())
        assert res.value is True

    def test_boundary_spill_does_not_fire(self):
        """Body domain overlapping diff by 1 aa (pos OUTSIDE diff) → False, not True."""
        from swissisoform.models import DifferentialRegion
        site = _site()
        site.diff_region = DifferentialRegion(isoform_start=0, isoform_end=10, sequence="X" * 10)
        site.isoform_annotations["interproscan"] = {"summary": {"status": "ok"}}
        # Domain starts at 9 (still inside diff: 0..10) — should fire.
        # But a hit starting at 11 (outside) yet ending at 13 (overlapping by 0) would NOT fire.
        # Use a clear "body domain that overlaps boundary": pos=10, end=50 → starts AT/past end.
        site.comparison["interproscan"] = {
            "hits_in_diff_region": [{"pos": 10, "end": 50, "name": "PF_body_spill"}],
            "hits_source_pane": "isoform",
        }
        res = _f3_domain_change(site, ScoringConfig())
        assert res.value is False

    def test_no_hits_at_all(self):
        from swissisoform.models import DifferentialRegion
        site = _site()
        site.diff_region = DifferentialRegion(isoform_start=0, isoform_end=10, sequence="X" * 10)
        site.isoform_annotations["interproscan"] = {"summary": {"status": "ok"}}
        site.comparison["interproscan"] = {"hits_in_diff_region": [], "hits_source_pane": "isoform"}
        res = _f3_domain_change(site, ScoringConfig())
        assert res.value is False


class TestF4TargetingChange:
    def test_no_comparator_data(self):
        """No SignalP/TargetP comparison → None."""
        assert _f4_targeting_change(_site(), ScoringConfig()).value is None

    def test_signalp_change(self):
        site = _site()
        site.comparison["signalp"] = {"signalp_prediction_changed": True}
        res = _f4_targeting_change(site, ScoringConfig())
        assert res.value is True

    def test_targetp_change(self):
        site = _site()
        site.comparison["targetp"] = {"targetp_prediction_changed": True}
        res = _f4_targeting_change(site, ScoringConfig())
        assert res.value is True

    def test_no_change(self):
        site = _site()
        site.comparison["signalp"] = {"signalp_prediction_changed": False}
        site.comparison["targetp"] = {"targetp_prediction_changed": False}
        res = _f4_targeting_change(site, ScoringConfig())
        assert res.value is False


class TestF5PathogenicVariantEnrichment:
    def test_no_data(self):
        """Empty site → None (varianteffect not run)."""
        res = _f5_pathogenic_variant_enrichment(_site(), ScoringConfig())
        assert res.value is None
        assert "varianteffect not run" in res.reason

    def test_no_scorable_in_unique(self):
        """varianteffect ran but no scorable variant in the unique region → None."""
        site = _site()
        site.isoform_annotations["varianteffect"] = {
            "summary": {"status": "ok"},
            "n_scorable_in_unique": 0,
            "n_damaging_in_unique": 0,
        }
        res = _f5_pathogenic_variant_enrichment(site, ScoringConfig())
        assert res.value is None
        assert "no scorable" in res.reason

    def test_damaging_enrichment_true(self):
        """≥ f5_min damaging variants among scorable unique-region variants → True."""
        site = _site()
        site.isoform_annotations["varianteffect"] = {
            "summary": {"status": "ok"},
            "n_scorable_in_unique": 3,
            "n_damaging_in_unique": 2,
            "n_am_pathogenic_in_unique": 1,
            "mean_delta_llr_unique": -8.4,
        }
        res = _f5_pathogenic_variant_enrichment(site, ScoringConfig())
        assert res.value is True

    def test_scorable_but_not_damaging(self):
        """Scorable variants exist but none damaging → False."""
        site = _site()
        site.isoform_annotations["varianteffect"] = {
            "summary": {"status": "ok"},
            "n_scorable_in_unique": 4,
            "n_damaging_in_unique": 0,
            "n_am_pathogenic_in_unique": 0,
            "mean_delta_llr_unique": -1.2,
        }
        res = _f5_pathogenic_variant_enrichment(site, ScoringConfig())
        assert res.value is False

    def test_threshold_respected(self):
        """f5_min_pathogenic_in_unique gates how many damaging variants are needed."""
        site = _site()
        site.isoform_annotations["varianteffect"] = {
            "summary": {"status": "ok"},
            "n_scorable_in_unique": 3,
            "n_damaging_in_unique": 1,
            "n_am_pathogenic_in_unique": 1,
            "mean_delta_llr_unique": -7.9,
        }
        assert _f5_pathogenic_variant_enrichment(site, ScoringConfig()).value is True
        strict = ScoringConfig(f5_min_pathogenic_in_unique=2)
        assert _f5_pathogenic_variant_enrichment(site, strict).value is False

    def test_non_numeric_counts_none(self):
        site = _site()
        site.isoform_annotations["varianteffect"] = {
            "summary": {"status": "ok"},
            "n_scorable_in_unique": "oops",
            "n_damaging_in_unique": 1,
        }
        res = _f5_pathogenic_variant_enrichment(site, ScoringConfig())
        assert res.value is None
        assert "not numeric" in res.reason


class TestF6ClinicalVariantOverlap:
    def test_positive(self):
        site = _site()
        site.isoform_annotations["variant_intersection"] = {
            "n_in_unique_region": 1,
            "summary": {"status": "ok"},
        }
        res = _f6_clinical_variant_overlap(site, ScoringConfig())
        assert res.value is True


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
            primate_frac_intact_min=0.5,
            mammalian_frac_intact_min=0.3,
            min_cell_lines=1,
            existence_high_threshold=3,
            functional_high_threshold=1,
        )
        mod = EvidenceScoringModule(PipelineConfig(scoring=cfg))
        site = _site()
        site.isoform_annotations["conservation_frame"] = {
            "primate_frac_intact": 0.8,
            "mammalian_frac_intact": 0.4,
            "summary": {"status": "ok"},
        }
        site.isoform_annotations["variant_intersection"] = {
            "n_in_unique_region": 1,
            "summary": {"status": "ok"},
        }
        site.expression["HeLa"] = CellLineExpression(raw_count=10, cpm=1.0, p_value=0.01)
        out = mod.annotate_site(site)

        # E1, E2, E4 True → existence_score = 3
        assert out["existence_score"] == 3
        assert out["existence_high_confidence"] is True
        # F6 True only; F1/F3 None (no struct/ips annotations); F2/F4/F5 None
        # (no comparator or vi+plm data) → functional_score = 1
        assert out["functional_score"] == 1
        assert out["functional_high_confidence"] is True
        assert out["criteria"]["E1_primate_conservation"] is True
        assert out["criteria"]["F6_clinical_variant_overlap"] is True

    def test_run_populates_annotations(self):
        mod = EvidenceScoringModule(PipelineConfig(scoring=ScoringConfig()))
        site = _site()
        mod.run([site])
        assert "scoring" in site.isoform_annotations
        assert "existence_score" in site.isoform_annotations["scoring"]


class TestMetadata:
    def test_counts(self):
        assert len(EXISTENCE_CRITERIA) == 6
        assert len(FUNCTIONAL_CRITERIA) == 6

    def test_output_columns(self):
        cols = EvidenceScoringModule.OUTPUT_COLUMNS
        assert "scoring_existence_score" in cols
        assert "scoring_functional_score" in cols
        assert "scoring_criteria" in cols
