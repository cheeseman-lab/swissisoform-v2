"""Tests for the Clinical variant annotation module."""

from __future__ import annotations

import copy

import pytest

from swissisoform.config import PipelineConfig
from swissisoform.models import TranslationInitiationSite
from swissisoform.modules.clinical import ClinicalModule, parse_hgvsp_position

SAMPLE_VARIANTS: dict[str, list[dict]] = {
    "TESTGENE_POS": [
        {
            "source": "gnomAD",
            "variant_id": "1-1000-A-G",
            "chrom": "chr1",
            "genomic_pos": 1000,
            "ref": "A",
            "alt": "G",
            "consequence": "missense_variant",
            "protein_pos": 5,
            "hgvsp": "p.Arg6Gly",
            "allele_frequency": 0.001,
            "clinical_significance": None,
            "metadata": {},
        },
        {
            "source": "gnomAD",
            "variant_id": "1-1050-C-T",
            "chrom": "chr1",
            "genomic_pos": 1050,
            "ref": "C",
            "alt": "T",
            "consequence": "synonymous_variant",
            "protein_pos": 22,
            "hgvsp": "p.Ala23=",
            "allele_frequency": 0.05,
            "clinical_significance": None,
            "metadata": {},
        },
        {
            "source": "ClinVar",
            "variant_id": "ClinVar:12345",
            "chrom": "chr1",
            "genomic_pos": 1010,
            "ref": "G",
            "alt": "A",
            "consequence": "missense_variant",
            "protein_pos": 8,
            "hgvsp": "p.Gly9Asp",
            "allele_frequency": None,
            "clinical_significance": "Pathogenic",
            "metadata": {"review_status": "criteria_provided,_single_submitter"},
        },
        {
            "source": "gnomAD",
            "variant_id": "1-900-T-C",
            "chrom": "chr1",
            "genomic_pos": 900,
            "ref": "T",
            "alt": "C",
            "consequence": "intron_variant",
            "protein_pos": None,  # Can't map to protein
            "hgvsp": None,
            "allele_frequency": 0.1,
            "clinical_significance": None,
            "metadata": {},
        },
    ],
    "TESTGENE_NEG": [
        {
            "source": "COSMIC",
            "variant_id": "COSV12345",
            "chrom": "chr2",
            "genomic_pos": 5000,
            "ref": "G",
            "alt": "T",
            "consequence": "stop_gained",
            "protein_pos": 10,
            "hgvsp": "p.Glu11*",
            "allele_frequency": None,
            "clinical_significance": None,
            "metadata": {"cosmic_count": 5},
        },
    ],
}


@pytest.fixture()
def clinical_module(config: PipelineConfig) -> ClinicalModule:
    """ClinicalModule with sample variant cache."""
    return ClinicalModule(config, variant_cache=copy.deepcopy(SAMPLE_VARIANTS))


@pytest.fixture()
def clinical_module_no_cache(config: PipelineConfig) -> ClinicalModule:
    """ClinicalModule without variant cache."""
    return ClinicalModule(config)


class TestClinicalModule:
    """Tests for ClinicalModule."""

    def test_annotate_with_cache(self, clinical_module: ClinicalModule) -> None:
        """Variants with protein_pos are returned; intron variant filtered out."""
        result = clinical_module.annotate("FAKE_PROTEIN", "TESTGENE_POS")
        # 4 raw variants, but one has protein_pos=None -> 3 hits
        assert len(result["hits"]) == 3

    def test_annotate_no_cache_returns_empty(
        self, clinical_module_no_cache: ClinicalModule
    ) -> None:
        """No cache provided returns empty hits."""
        result = clinical_module_no_cache.annotate("FAKE", "TESTGENE_POS")
        assert result["hits"] == []
        assert result["summary"]["total_variants"] == 0

    def test_annotate_unknown_gene_returns_empty(
        self, clinical_module: ClinicalModule
    ) -> None:
        """Gene not in cache returns empty hits."""
        result = clinical_module.annotate("FAKE", "UNKNOWN_GENE")
        assert result["hits"] == []
        assert result["summary"]["total_variants"] == 0

    def test_summary_counts(self, clinical_module: ClinicalModule) -> None:
        """Verify total_variants, by_source, by_consequence counts."""
        result = clinical_module.annotate("", "TESTGENE_POS")
        summary = result["summary"]
        assert summary["total_variants"] == 3
        assert summary["by_source"] == {"gnomAD": 2, "ClinVar": 1}
        assert summary["by_consequence"] == {
            "missense_variant": 2,
            "synonymous_variant": 1,
        }

    def test_summary_pathogenic_count(self, clinical_module: ClinicalModule) -> None:
        """One ClinVar Pathogenic variant in TESTGENE_POS."""
        result = clinical_module.annotate("", "TESTGENE_POS")
        assert result["summary"]["pathogenic_count"] == 1

    def test_hits_have_required_keys(self, clinical_module: ClinicalModule) -> None:
        """Every hit has source, variant_id, protein_pos, consequence."""
        result = clinical_module.annotate("", "TESTGENE_POS")
        required_keys = {"source", "variant_id", "protein_pos", "consequence"}
        for hit in result["hits"]:
            assert required_keys.issubset(hit.keys())

    def test_intron_variant_filtered(self, clinical_module: ClinicalModule) -> None:
        """Variant with protein_pos=None is excluded from hits."""
        result = clinical_module.annotate("", "TESTGENE_POS")
        for hit in result["hits"]:
            assert hit["protein_pos"] is not None

    def test_annotate_by_key(self, clinical_module: ClinicalModule) -> None:
        """annotate_by_key returns same result as annotate with empty protein."""
        direct = clinical_module.annotate("", "TESTGENE_NEG")
        by_key = clinical_module.annotate_by_key("TESTGENE_NEG")
        assert direct == by_key
        assert len(by_key["hits"]) == 1
        assert by_key["hits"][0]["source"] == "COSMIC"

    def test_run_wrapper(
        self,
        synthetic_tis: list[TranslationInitiationSite],
        clinical_module: ClinicalModule,
    ) -> None:
        """run() populates isoform_annotations for each site."""
        result = clinical_module.run(synthetic_tis)
        for site in result:
            assert "clinical" in site.isoform_annotations
            ann = site.isoform_annotations["clinical"]
            assert "hits" in ann
            assert "summary" in ann

    def test_no_sites_lost(
        self,
        synthetic_tis: list[TranslationInitiationSite],
        clinical_module: ClinicalModule,
    ) -> None:
        """run() doesn't drop sites."""
        n_before = len(synthetic_tis)
        result = clinical_module.run(synthetic_tis)
        assert len(result) == n_before

    def test_module_scope(self, clinical_module: ClinicalModule) -> None:
        """SCOPE == 'C'."""
        assert clinical_module.SCOPE == "C"


class TestParseHgvspPosition:
    """Tests for parse_hgvsp_position static helper."""

    @pytest.mark.parametrize(
        ("hgvsp", "expected"),
        [
            ("p.Arg42Gly", 41),
            ("p.R42G", 41),
            ("p.Ter100Arg", 99),
            ("p.Met1?", 0),
            ("p.Glu11*", 10),
            ("p.Ala23=", 22),
        ],
    )
    def test_parse_hgvsp_position(self, hgvsp: str, expected: int) -> None:
        """Various HGVS strings parse to correct 0-indexed positions."""
        assert parse_hgvsp_position(hgvsp) == expected

    @pytest.mark.parametrize("value", [None, "", "p.?"])
    def test_parse_hgvsp_none(self, value: str | None) -> None:
        """None/empty/unparseable returns None."""
        assert parse_hgvsp_position(value) is None
