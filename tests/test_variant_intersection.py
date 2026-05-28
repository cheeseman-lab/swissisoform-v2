"""Tests for VariantIntersectionModule."""

from __future__ import annotations

from swissisoform.models import ORFType, TranslationInitiationSite
from swissisoform.modules.variant_intersection import VariantIntersectionModule


def _site(
    orf_exons: list[tuple[int, int]] | None = None,
    canonical_orf_exons: list[tuple[int, int]] | None = None,
    hits: list[dict] | None = None,
    strand: str = "+",
) -> TranslationInitiationSite:
    site = TranslationInitiationSite(
        tis_id="chr1:1000:+:ATG",
        gene_name="TEST",
        transcript_id="ENST_TEST",
        chrom="chr1",
        position=1000,
        strand=strand,
        start_codon="ATG",
        orf_type=ORFType.EXTENDED,
        canonical_protein="MAAA",
        isoform_protein="MXXMAAA",
    )
    site.orf_exons = orf_exons or []
    site.canonical_orf_exons = canonical_orf_exons or []
    site.isoform_annotations["clinical"] = {"hits": hits or [], "summary": {}}
    return site


def _hit(pos: int, *, sig: str | None = None, source: str = "gnomAD") -> dict:
    return {
        "source": source,
        "variant_id": f"x{pos}",
        "chrom": "chr1",
        "genomic_pos": pos,
        "ref": "A",
        "alt": "G",
        "consequence": "missense_variant",
        "protein_pos": 0,
        "hgvsp": None,
        "allele_frequency": None,
        "clinical_significance": sig,
        "metadata": {},
    }


class TestBasicIntersection:
    def test_extension_hit_in_unique_region(self):
        # isoform = [900,930) + [1000,1030); canonical = [1000,1030) → unique=[900,930)
        mod = VariantIntersectionModule()
        site = _site(
            orf_exons=[(900, 930), (1000, 1030)],
            canonical_orf_exons=[(1000, 1030)],
            hits=[_hit(910), _hit(1010)],
        )
        out = mod.annotate_site(site)

        assert out["n_total"] == 2
        assert out["n_in_unique_region"] == 1
        assert out["n_in_shared_region"] == 1
        assert out["hits"][0]["in_isoform_unique"] is True
        assert out["hits"][0]["in_isoform_shared"] is False
        assert out["hits"][1]["in_isoform_unique"] is False
        assert out["hits"][1]["in_isoform_shared"] is True
        assert out["summary"]["status"] == "ok"

    def test_hit_outside_isoform_is_dropped(self):
        """Variants outside both coding regions are dropped (n_dropped_outside_coding++)."""
        mod = VariantIntersectionModule()
        site = _site(
            orf_exons=[(1000, 1030)],
            canonical_orf_exons=[(1000, 1030)],
            hits=[_hit(2000)],
        )
        out = mod.annotate_site(site)
        assert out["n_total"] == 0
        assert out["n_in_unique_region"] == 0
        assert out["n_in_shared_region"] == 0
        assert out["n_dropped_outside_coding"] == 1
        assert out["hits"] == []

    def test_pathogenic_tallied_in_unique(self):
        mod = VariantIntersectionModule()
        site = _site(
            orf_exons=[(900, 930), (1000, 1030)],
            canonical_orf_exons=[(1000, 1030)],
            hits=[
                _hit(910, sig="Pathogenic"),
                _hit(920, sig="Benign"),
                _hit(1010, sig="Likely_pathogenic"),
            ],
        )
        out = mod.annotate_site(site)
        assert out["n_pathogenic_in_unique_region"] == 1
        assert out["n_pathogenic_in_shared_region"] == 1


class TestMissingInputs:
    def test_no_skeleton(self):
        mod = VariantIntersectionModule()
        site = _site(orf_exons=[], hits=[_hit(910)])
        out = mod.annotate_site(site)
        assert out["summary"]["status"] == "no_skeleton"
        assert out["hits"][0]["in_isoform"] is None
        assert out["n_in_unique_region"] is None

    def test_hit_without_genomic_pos(self):
        mod = VariantIntersectionModule()
        hit = _hit(910)
        hit.pop("genomic_pos")
        site = _site(
            orf_exons=[(900, 930)],
            canonical_orf_exons=[],
            hits=[hit],
        )
        out = mod.annotate_site(site)
        assert out["n_total"] == 1
        assert out["summary"]["n_scored"] == 0
        assert out["summary"]["n_unscored"] == 1
        assert out["hits"][0]["in_isoform_unique"] is None

    def test_no_clinical_data(self):
        mod = VariantIntersectionModule()
        site = _site(orf_exons=[(900, 930)], canonical_orf_exons=[])
        site.isoform_annotations.pop("clinical", None)
        out = mod.annotate_site(site)
        assert out["n_total"] == 0
        assert out["summary"]["status"] == "ok"


class TestClinicalHitsShareAugmented:
    def test_clinical_hits_now_carry_augmentation(self):
        """Per-TIS contract: clinical['hits'] points at the augmented list.

        The comparator's positional subset reads in_isoform_unique +
        isoform_protein_pos from there, so it doesn't need to know about
        variant_intersection's own annotation.
        """
        mod = VariantIntersectionModule()
        hit = _hit(910)
        site = _site(
            orf_exons=[(900, 930)],
            canonical_orf_exons=[],
            hits=[hit],
        )
        out = mod.annotate_site(site)
        augmented = site.isoform_annotations["clinical"]["hits"][0]
        assert augmented["in_isoform_unique"] is True
        # And the variant_intersection annotation references the same data.
        assert out["hits"][0]["in_isoform_unique"] is True


class TestMetadata:
    def test_module_name(self):
        assert VariantIntersectionModule.MODULE_NAME == "variant_intersection"

    def test_scope(self):
        assert VariantIntersectionModule.SCOPE == "C"

    def test_output_columns(self):
        cols = VariantIntersectionModule.OUTPUT_COLUMNS
        assert "variant_intersection_n_in_unique_region" in cols
        assert "variant_intersection_n_pathogenic_in_unique_region" in cols
        assert "variant_intersection_summary" in cols


class TestRunWrapper:
    def test_populates_isoform_annotations(self):
        mod = VariantIntersectionModule()
        site = _site(
            orf_exons=[(900, 930), (1000, 1030)],
            canonical_orf_exons=[(1000, 1030)],
            hits=[_hit(910)],
        )
        mod.run([site])
        out = site.isoform_annotations["variant_intersection"]
        assert out["n_in_unique_region"] == 1
