"""Tests for domain model, config, and module protocol."""

from __future__ import annotations

from pathlib import Path

import pytest

from swissisoform.config import (
    ClinicalConfig,
    ConservationConfig,
    FilterConfig,
    PipelineConfig,
    ScoringConfig,
    StructureConfig,
)
from swissisoform.models import (
    CellLineExpression,
    Gene,
    ORFType,
    TranslationInitiationSite,
    VariantAnnotation,
    orf_type_from_ribotish,
)
from swissisoform.modules.base import validate_module_output

# --- ORFType enum ---


class TestORFType:
    def test_enum_values(self):
        assert ORFType.ANNOTATED.value == "annotated"
        assert ORFType.EXTENDED.value == "extended"
        assert ORFType.TRUNCATED.value == "truncated"
        assert ORFType.UORF.value == "uorf"
        assert ORFType.UOORF.value == "uoorf"
        assert ORFType.INTERNAL_OUT_OF_FRAME.value == "internal_oof"
        assert ORFType.THREE_UTR_ORF.value == "3utr_orf"
        assert ORFType.ALT_ORF.value == "alt_orf"

    def test_all_members(self):
        assert len(ORFType) == 8


# --- orf_type_from_ribotish ---


class TestORFTypeFromRibotish:
    """Test mapping of all 16 Ribo-TISH TisType strings."""

    @pytest.mark.parametrize(
        "tis_type,expected",
        [
            ("Annotated", ORFType.ANNOTATED),
            ("Annotated:Known", ORFType.ANNOTATED),
            ("Truncated", ORFType.TRUNCATED),
            ("Truncated:Known", ORFType.TRUNCATED),
            ("Extended", ORFType.EXTENDED),
            ("Extended:CDSFrameOverlap", ORFType.EXTENDED),
            ("Internal", ORFType.INTERNAL_OUT_OF_FRAME),
            ("Internal:CDSFrameOverlap", ORFType.INTERNAL_OUT_OF_FRAME),
            ("5'UTR", ORFType.UORF),
            ("5'UTR:Known", ORFType.UORF),
            ("5'UTR:CDSFrameOverlap", ORFType.UOORF),
            ("5'UTR:CDSFrameOverlap:Known", ORFType.UOORF),
            ("3'UTR", ORFType.THREE_UTR_ORF),
            ("3'UTR:Known", ORFType.THREE_UTR_ORF),
            ("Novel:CDSFrameOverlap", ORFType.UOORF),
            ("Novel", ORFType.ALT_ORF),
        ],
    )
    def test_mapping(self, tis_type: str, expected: ORFType):
        assert orf_type_from_ribotish(tis_type) == expected

    def test_unknown_defaults_to_alt_orf(self):
        assert orf_type_from_ribotish("SomethingUnknown") == ORFType.ALT_ORF


# --- CellLineExpression ---


class TestCellLineExpression:
    def test_creation(self):
        expr = CellLineExpression(raw_count=100, cpm=5.5, p_value=0.01)
        assert expr.raw_count == 100
        assert expr.cpm == 5.5
        assert expr.p_value == 0.01
        assert expr.initiation_efficiency is None

    def test_with_efficiency(self):
        expr = CellLineExpression(
            raw_count=200, cpm=10.0, p_value=0.001, initiation_efficiency=0.75
        )
        assert expr.initiation_efficiency == 0.75


# --- TranslationInitiationSite ---


class TestTranslationInitiationSite:
    def test_creation_minimal(self):
        site = TranslationInitiationSite(
            tis_id="chr1:100:+:ATG",
            gene_name="TP53",
            transcript_id="ENST00000269305",
            chrom="chr1",
            position=100,
            strand="+",
            start_codon="ATG",
            orf_type=ORFType.ANNOTATED,
        )
        assert site.tis_id == "chr1:100:+:ATG"
        assert site.gene_name == "TP53"
        assert site.gene_id == ""
        assert site.transcript_start == 0
        assert site.aa_len == 0
        assert site.tis_pvalue is None
        assert site.ribo_pvalue is None
        assert site.fisher_qvalue is None
        assert site.expression == {}
        assert site.canonical_protein == ""
        assert site.isoform_protein == ""
        assert site.diff_region is None
        assert site.kozak_context is None
        assert site.isoform_annotations == {}
        assert site.comparison == {}

    def test_annotations_dict_isolation(self):
        """Each site gets its own annotations dict."""
        site_a = TranslationInitiationSite(
            tis_id="a",
            gene_name="G",
            transcript_id="T",
            chrom="chr1",
            position=1,
            strand="+",
            start_codon="ATG",
            orf_type=ORFType.UORF,
        )
        site_b = TranslationInitiationSite(
            tis_id="b",
            gene_name="G",
            transcript_id="T",
            chrom="chr1",
            position=2,
            strand="+",
            start_codon="ATG",
            orf_type=ORFType.UORF,
        )
        site_a.isoform_annotations["mod1"] = {"score": 1.0}
        assert "mod1" not in site_b.isoform_annotations

    def test_statistical_fields(self):
        site = TranslationInitiationSite(
            tis_id="x",
            gene_name="G",
            transcript_id="T",
            chrom="chr1",
            position=1,
            strand="+",
            start_codon="ATG",
            orf_type=ORFType.EXTENDED,
            tis_pvalue=0.001,
            ribo_pvalue=0.01,
            fisher_qvalue=0.05,
        )
        assert site.tis_pvalue == 0.001
        assert site.ribo_pvalue == 0.01
        assert site.fisher_qvalue == 0.05


# --- Gene ---


class TestGene:
    def test_creation(self):
        gene = Gene(
            gene_name="TP53",
            gene_id="ENSG00000141510",
            canonical_transcript_id="ENST00000269305",
            canonical_protein="MEEPQ...",
        )
        assert gene.gene_name == "TP53"
        assert gene.tis_sites == []
        assert gene.canonical_annotations == {}
        assert gene.gene_annotations == {}


# --- VariantAnnotation ---


class TestVariantAnnotation:
    def test_creation(self):
        va = VariantAnnotation(
            tis_id="chr1:100:+:ATG",
            source="gnomAD",
            variant_id="rs12345",
            chrom="chr1",
            position=150,
            ref="A",
            alt="G",
            protein_pos=50,
        )
        assert va.source == "gnomAD"
        assert va.protein_pos == 50
        assert va.metadata == {}


# --- validate_module_output ---


def _make_site(tis_id: str = "site1") -> TranslationInitiationSite:
    return TranslationInitiationSite(
        tis_id=tis_id,
        gene_name="G",
        transcript_id="T",
        chrom="chr1",
        position=1,
        strand="+",
        start_codon="ATG",
        orf_type=ORFType.ANNOTATED,
    )


class TestValidateModuleOutput:
    def test_valid_output(self):
        site = _make_site()
        site.isoform_annotations["biophysics"] = {"molecular_weight": 50.0, "pi": 6.5}
        validate_module_output(
            input_sites=[_make_site()],
            output_sites=[site],
            module_name="biophysics",
            output_columns=["biophysics_molecular_weight", "biophysics_pi"],
        )

    def test_dropped_sites_error(self):
        with pytest.raises(ValueError, match="dropped"):
            validate_module_output(
                input_sites=[_make_site(), _make_site("site2")],
                output_sites=[_make_site()],
                module_name="biophysics",
                output_columns=["biophysics_mw"],
            )

    def test_missing_annotation_key(self):
        site = _make_site()
        # No annotations set for this module
        with pytest.raises(ValueError, match="missing annotations"):
            validate_module_output(
                input_sites=[_make_site()],
                output_sites=[site],
                module_name="biophysics",
                output_columns=["biophysics_mw"],
            )

    def test_missing_column(self):
        site = _make_site()
        site.isoform_annotations["biophysics"] = {"molecular_weight": 50.0}
        with pytest.raises(ValueError, match="missing.*column"):
            validate_module_output(
                input_sites=[_make_site()],
                output_sites=[site],
                module_name="biophysics",
                output_columns=["biophysics_molecular_weight", "biophysics_pi"],
            )


# --- Config defaults ---


class TestConfig:
    def test_filter_defaults(self):
        fc = FilterConfig()
        assert fc.transcript_support_levels == ["1", "2", "3"]
        assert fc.min_normalized_counts == 0.1
        assert fc.tis_enrichment_max_p == 0.01
        assert fc.frame_test_max_p == 0.01
        assert fc.combined_test_max_q == 0.05
        assert fc.tis_distance_buffer == 30

    def test_pipeline_defaults(self):
        pc = PipelineConfig()
        assert pc.cell_lines == ["HeLa", "K562", "U2OS", "RPE1_Async", "RPE1_Que", "RPE1_Sen"]
        assert pc.data_dir == Path("data/reference")
        assert pc.genome_fasta is None
        assert pc.gtf_path is None
        assert pc.protein_fasta is None
        assert pc.output_dir == Path("results")
        assert pc.filtering is not None
        assert pc.conservation is None
        assert pc.structure is None
        assert pc.scoring is None
        assert pc.clinical is None

    def test_conservation_config(self):
        cc = ConservationConfig()
        assert cc.diamond_db is None

    def test_structure_config(self):
        sc = StructureConfig()
        assert sc.method == "chai1"
        assert sc.device == "cuda"
        assert sc.batch_size == 4

    def test_scoring_config(self):
        sc = ScoringConfig()
        assert sc.min_cell_lines == 3
        assert sc.existence_high_threshold == 5
        assert sc.functional_high_threshold == 3
        assert sc.truncation_max_aa == 200

    def test_clinical_config(self):
        cc = ClinicalConfig()
        assert cc.gnomad_api_url == "https://gnomad.broadinstitute.org/api"
        assert cc.clinvar_email == ""
        assert cc.cosmic_db is None
