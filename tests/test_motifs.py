"""Tests for the motifs (SLiM scanning) annotation module."""

from __future__ import annotations

import pytest

from swissisoform.models import DifferentialRegion, ORFType, TranslationInitiationSite
from swissisoform.modules.motifs import MotifsModule


def _make_tis(seq: str) -> TranslationInitiationSite:
    """Create a minimal TIS with a known diff_region sequence."""
    return TranslationInitiationSite(
        tis_id="test:0:+:ATG",
        gene_name="TEST",
        transcript_id="ENST_TEST",
        chrom="chr1",
        position=0,
        strand="+",
        start_codon="ATG",
        orf_type=ORFType.EXTENDED,
        diff_region=DifferentialRegion(sequence=seq),
    )


class TestMotifsModule:
    """Tests for MotifsModule.annotate() and run()."""

    # ------------------------------------------------------------------
    # annotate() tests
    # ------------------------------------------------------------------

    def test_annotate_empty_string(self, config):
        """annotate('') returns hits=[] and all-zero summary."""
        mod = MotifsModule(config)
        result = mod.annotate("")
        assert result["hits"] == []
        for key, val in result["summary"].items():
            assert val == 0 or val == 0.0, f"Expected 0 for {key}, got {val}"

    def test_annotate_known_cdk_hit(self, config):
        """annotate('AAASPMMM') has 1 CDK hit at pos=3, match='SP'."""
        mod = MotifsModule(config)
        result = mod.annotate("AAASPMMM")
        cdk_hits = [h for h in result["hits"] if h["name"] == "CDK_Sites_SP_TP"]
        assert len(cdk_hits) == 1
        assert cdk_hits[0]["pos"] == 3
        assert cdk_hits[0]["end"] == 5
        assert cdk_hits[0]["match"] == "SP"

    def test_annotate_multiple_hits(self, config):
        """annotate('SPSPSP') has 3 CDK hits at positions 0, 2, 4."""
        mod = MotifsModule(config)
        result = mod.annotate("SPSPSP")
        cdk_hits = [h for h in result["hits"] if h["name"] == "CDK_Sites_SP_TP"]
        assert len(cdk_hits) == 3
        positions = [h["pos"] for h in cdk_hits]
        assert positions == [0, 2, 4]

    def test_annotate_1433_mode1(self, config):
        """annotate('RSASLPMMMM') has Mode1_1433 hit and total_1433 >= 1."""
        mod = MotifsModule(config)
        result = mod.annotate("RSASLPMMMM")
        mode1_hits = [h for h in result["hits"] if h["name"] == "Mode1_1433"]
        assert len(mode1_hits) >= 1
        assert result["summary"]["total_1433"] >= 1

    def test_annotate_rgg(self, config):
        """annotate('AAARGGA') has RGG hit at pos=3."""
        mod = MotifsModule(config)
        result = mod.annotate("AAARGGA")
        rgg_hits = [h for h in result["hits"] if h["name"] == "RGG_Sites"]
        assert len(rgg_hits) == 1
        assert rgg_hits[0]["pos"] == 3
        assert rgg_hits[0]["end"] == 6
        assert rgg_hits[0]["match"] == "RGG"

    def test_annotate_no_rgg_for_rg(self, config):
        """annotate('MRGSHHHHHGS') has no RGG hits (only RG, not RGG)."""
        mod = MotifsModule(config)
        result = mod.annotate("MRGSHHHHHGS")
        rgg_hits = [h for h in result["hits"] if h["name"] == "RGG_Sites"]
        assert len(rgg_hits) == 0

    def test_annotate_rgg_cluster(self, config):
        """annotate('RGGAARGG') has RGG_clusters == 1."""
        mod = MotifsModule(config)
        result = mod.annotate("RGGAARGG")
        assert result["summary"]["RGG_clusters"] >= 1

    def test_annotate_rg_rich_threshold(self, config):
        """RG_rich=1 when >= 3 RG; RG_rich=0 when < 3 RG."""
        mod = MotifsModule(config)

        result_rich = mod.annotate("RGAARGAARG")
        assert result_rich["summary"]["RG_rich"] == 1

        result_not_rich = mod.annotate("RGAARG")
        assert result_not_rich["summary"]["RG_rich"] == 0

    def test_annotate_hit_positions(self, config):
        """Each hit has name, pos, end, match keys."""
        mod = MotifsModule(config)
        result = mod.annotate("AAASPMMM")
        for hit in result["hits"]:
            assert "name" in hit
            assert "pos" in hit
            assert "end" in hit
            assert "match" in hit

    def test_heme_cp_hrm_not_dipeptide(self, config):
        """Heme_CP_HRM must not match bare CP dipeptides — the pattern
        requires C[^C].{2}C[^C]H context (ELM LIG_Heme_HRM_1-like).
        """
        mod = MotifsModule(config)
        # Bare CP in a proteome-like context: should NOT hit.
        bare = mod.annotate("MAACPDEFGHIK")
        assert [h for h in bare["hits"] if h["name"] == "Heme_CP_HRM"] == []
        # C[^C].{2}C[^C]H with non-C flanks: should hit once at pos 2 on "CADACAH".
        legit = mod.annotate("MACADACAHMMM")
        heme_hits = [h for h in legit["hits"] if h["name"] == "Heme_CP_HRM"]
        assert len(heme_hits) == 1
        assert heme_hits[0]["pos"] == 2
        assert heme_hits[0]["match"] == "CADACAH"

    def test_annotate_strip_stop_codon(self, config):
        """annotate('SP*') finds 1 CDK hit (trailing * stripped)."""
        mod = MotifsModule(config)
        result = mod.annotate("SP*")
        cdk_hits = [h for h in result["hits"] if h["name"] == "CDK_Sites_SP_TP"]
        assert len(cdk_hits) == 1
        # Density computed on length 2 (without *)
        assert result["summary"]["CDK_Sites_SP_TP_density_per100aa"] == pytest.approx(50.0)

    def test_annotate_density_calculation(self, config):
        """density == count * 100 / len(seq), rounded to 4 decimals."""
        mod = MotifsModule(config)
        result = mod.annotate("SPSPSP")
        expected_density = round(3 * 100 / 6, 4)
        assert result["summary"]["CDK_Sites_SP_TP_density_per100aa"] == pytest.approx(
            expected_density
        )

    def test_annotate_pure_function(self, config):
        """Two calls with same input produce identical results."""
        mod = MotifsModule(config)
        r1 = mod.annotate("AAASPMMM")
        r2 = mod.annotate("AAASPMMM")
        assert r1 == r2

    # ------------------------------------------------------------------
    # run() wrapper tests
    # ------------------------------------------------------------------

    def test_run_no_sites_lost(self, synthetic_tis, config):
        """len(output) == len(input)."""
        mod = MotifsModule(config)
        result = mod.run(synthetic_tis)
        assert len(result) == len(synthetic_tis)

    def test_run_populates_isoform_annotations(self, synthetic_tis, config):
        """Each site has isoform_annotations['motifs']."""
        mod = MotifsModule(config)
        mod.run(synthetic_tis)
        for site in synthetic_tis:
            assert "motifs" in site.isoform_annotations

    def test_run_output_has_hits_and_summary(self, synthetic_tis, config):
        """Each annotation has 'hits' and 'summary' keys."""
        mod = MotifsModule(config)
        mod.run(synthetic_tis)
        for site in synthetic_tis:
            ann = site.isoform_annotations["motifs"]
            assert "hits" in ann
            assert "summary" in ann
            assert isinstance(ann["hits"], list)
            assert isinstance(ann["summary"], dict)
