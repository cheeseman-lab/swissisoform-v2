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
    _e6_proteomics,
    _e7_mass_spec,
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


class TestE6Proteomics:
    def test_stubbed(self):
        res = _e6_proteomics(_site(), ScoringConfig())
        assert res.value is None
        assert "not wired" in res.reason


class TestE7MassSpec:
    def test_passes(self):
        site = _site()
        site.isoform_annotations["massspec"] = {
            "hits": [
                {"unique_to_isoform": True},
                {"unique_to_isoform": False},
            ],
        }
        res = _e7_mass_spec(site, ScoringConfig(massspec_unique_peptides_min=1))
        assert res.value is True

    def test_none_unique(self):
        site = _site()
        site.isoform_annotations["massspec"] = {"hits": []}
        res = _e7_mass_spec(site, ScoringConfig(massspec_unique_peptides_min=1))
        assert res.value is False


# ---------------------------------------------------------------------------
# Functional
# ---------------------------------------------------------------------------


class TestF1StructuredExtension:
    def test_stubbed(self):
        res = _f1_structured_extension(_site(), ScoringConfig())
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


class TestF3F4Stubs:
    def test_domain_stubbed(self):
        assert _f3_domain_change(_site(), ScoringConfig()).value is None

    def test_targeting_stubbed(self):
        assert _f4_targeting_change(_site(), ScoringConfig()).value is None


class TestF5PathogenicVariantEnrichment:
    def test_positive(self):
        site = _site()
        site.isoform_annotations["variant_intersection"] = {
            "n_pathogenic_in_unique_region": 2,
            "summary": {"status": "ok"},
        }
        res = _f5_pathogenic_variant_enrichment(site, ScoringConfig())
        assert res.value is True

    def test_zero(self):
        site = _site()
        site.isoform_annotations["variant_intersection"] = {
            "n_pathogenic_in_unique_region": 0,
            "summary": {"status": "ok"},
        }
        res = _f5_pathogenic_variant_enrichment(site, ScoringConfig())
        assert res.value is False

    def test_no_skeleton(self):
        site = _site()
        site.isoform_annotations["variant_intersection"] = {
            "n_pathogenic_in_unique_region": None,
            "summary": {"status": "no_skeleton"},
        }
        res = _f5_pathogenic_variant_enrichment(site, ScoringConfig())
        assert res.value is None


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
        # F1–F4 stubs, F5/F6 require variant_intersection — nothing evaluates
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
            "n_pathogenic_in_unique_region": 1,
            "n_in_unique_region": 1,
            "summary": {"status": "ok"},
        }
        site.expression["HeLa"] = CellLineExpression(raw_count=10, cpm=1.0, p_value=0.01)
        out = mod.annotate_site(site)

        # E1, E2, E4 True → existence_score = 3
        assert out["existence_score"] == 3
        assert out["existence_high_confidence"] is True
        # F5, F6 True → functional_score = 2
        assert out["functional_score"] == 2
        assert out["functional_high_confidence"] is True
        assert out["criteria"]["E1_primate_conservation"] is True
        assert out["criteria"]["F5_pathogenic_variant_enrichment"] is True

    def test_run_populates_annotations(self):
        mod = EvidenceScoringModule(PipelineConfig(scoring=ScoringConfig()))
        site = _site()
        mod.run([site])
        assert "scoring" in site.isoform_annotations
        assert "existence_score" in site.isoform_annotations["scoring"]


class TestMetadata:
    def test_counts(self):
        assert len(EXISTENCE_CRITERIA) == 7
        assert len(FUNCTIONAL_CRITERIA) == 6

    def test_output_columns(self):
        cols = EvidenceScoringModule.OUTPUT_COLUMNS
        assert "scoring_existence_score" in cols
        assert "scoring_functional_score" in cols
        assert "scoring_criteria" in cols
