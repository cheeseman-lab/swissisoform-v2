"""Tests for the ConservationFrameModule SiteModule wrapper."""

from __future__ import annotations

from swissisoform.config import ConservationConfig, PipelineConfig
from swissisoform.models import ORFType, TranslationInitiationSite
from swissisoform.conservation_frame.module import (
    ConservationFrameModule,
    _revcomp_maf,
)


def _tis(strand: str = "+") -> TranslationInitiationSite:
    return TranslationInitiationSite(
        tis_id="test",
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


class TestNotRun:
    def test_no_config(self):
        cfg = PipelineConfig(conservation=None)
        mod = ConservationFrameModule(cfg)
        out = mod.annotate_site(_tis())
        assert out["summary"]["status"] == "not_run"
        assert out["primate_frac_intact"] is None

    def test_missing_hal(self, tmp_path):
        cfg = PipelineConfig(
            conservation=ConservationConfig(hal_path=tmp_path / "nope.hal"),
        )
        mod = ConservationFrameModule(cfg)
        out = mod.annotate_site(_tis())
        assert out["summary"]["status"] == "not_run"


class TestRequiresSkeleton:
    def test_no_skeleton(self, monkeypatch, tmp_path):
        # Force module to think it's available, then give a TIS with no
        # orf_exons so it short-circuits on skeleton-missing.
        fake_hal = tmp_path / "fake.hal"
        fake_hal.write_bytes(b"")
        cfg = PipelineConfig(conservation=ConservationConfig(hal_path=fake_hal))
        mod = ConservationFrameModule(cfg)
        monkeypatch.setattr(mod, "_available", True)

        out = mod.annotate_site(_tis())
        assert out["summary"]["status"] == "no_skeleton"

    def test_no_unique_region(self, monkeypatch, tmp_path):
        fake_hal = tmp_path / "fake.hal"
        fake_hal.write_bytes(b"")
        cfg = PipelineConfig(conservation=ConservationConfig(hal_path=fake_hal))
        mod = ConservationFrameModule(cfg)
        monkeypatch.setattr(mod, "_available", True)

        site = _tis()
        # Make iso == canonical so unique region is empty
        site.orf_exons = [(1000, 1030)]
        site.canonical_orf_exons = [(1000, 1030)]
        out = mod.annotate_site(site)
        assert out["summary"]["status"] == "no_unique_region"


class TestMetadata:
    def test_module_name(self):
        assert ConservationFrameModule.MODULE_NAME == "conservation_frame"

    def test_scope(self):
        assert ConservationFrameModule.SCOPE == "C"

    def test_output_columns(self):
        cols = ConservationFrameModule.OUTPUT_COLUMNS
        assert "conservation_primate_frac_intact" in cols
        assert "conservation_mammalian_frac_intact" in cols
        assert "conservation_frame_summary" in cols


class TestDeepestIntact:
    """End-to-end: tree supplied via config, alignment faked via monkeypatch."""

    def test_picks_deepest_intact_species(self, monkeypatch, tmp_path):
        newick = (
            "(((hg38,panTro6)hominid,"
            "(rheMac10,calJac4)simian)primate,"
            "mm10)euarchontoglires;"
        )
        fake_hal = tmp_path / "fake.hal"
        fake_hal.write_bytes(b"")
        cfg = PipelineConfig(
            conservation=ConservationConfig(
                hal_path=fake_hal,
                hal_tree_newick=newick,
                primate_species=["panTro6", "rheMac10", "calJac4"],
                mammalian_species=["panTro6", "rheMac10", "mm10"],
            ),
        )
        mod = ConservationFrameModule(cfg)
        monkeypatch.setattr(mod, "_available", True)

        # Fake alignment: every species has an intact ATG-anchored ORF.
        ref = "ATGAAACCCGGGTAA"

        def fake_fetch(chrom, intervals, strand):
            return ref, {sp: ref for sp in ("panTro6", "rheMac10", "calJac4", "mm10")}

        monkeypatch.setattr(mod, "_fetch_alignment", fake_fetch)

        site = _tis()
        site.orf_exons = [(1000, 1030)]
        site.canonical_orf_exons = [(1050, 1080)]  # disjoint → unique = site.orf_exons
        out = mod.annotate_site(site)

        # Primate list: deepest should be rheMac10 or calJac4 (both depth 2)
        assert out["primate_deepest_species"] in {"rheMac10", "calJac4"}
        assert out["primate_max_depth"] == 2
        # Mammalian list: mm10 is the outgroup (depth 3 in this tree)
        assert out["mammalian_deepest_species"] == "mm10"
        assert out["mammalian_max_depth"] == 3
        assert out["summary"]["tree_loaded"] is True

    def _mod_and_site(self, monkeypatch, tmp_path):
        fake_hal = tmp_path / "fake.hal"
        fake_hal.write_bytes(b"")
        cfg = PipelineConfig(
            conservation=ConservationConfig(
                hal_path=fake_hal,
                primate_species=["panTro6", "rheMac10"],
                mammalian_species=["mm10"],
            ),
        )
        mod = ConservationFrameModule(cfg)
        monkeypatch.setattr(mod, "_available", True)
        ref = "ATGAAACCCGGGTAA"
        monkeypatch.setattr(
            mod,
            "_fetch_alignment",
            lambda chrom, intervals, strand: (ref, {sp: ref for sp in ("panTro6", "rheMac10", "mm10")}),
        )
        site = _tis()
        site.orf_exons = [(1000, 1030)]
        return mod, site

    def test_canonical_orf_baseline_populated(self, monkeypatch, tmp_path):
        """Flag H: the canonical ORF is scored into {clade}_canonical_* twins."""
        mod, site = self._mod_and_site(monkeypatch, tmp_path)
        site.canonical_orf_exons = [(1050, 1080)]  # disjoint → unique = orf_exons
        out = mod.annotate_site(site)
        # Both the unique-region metric and its canonical twin are present.
        assert out["primate_frac_intact"] is not None
        assert out["primate_canonical_frac_intact"] is not None
        assert out["mammalian_canonical_frac_intact"] is not None
        assert out["summary"]["canonical_status"] == "ok"

    def test_canonical_twins_none_without_canonical_exons(self, monkeypatch, tmp_path):
        """No canonical skeleton → canonical twins are None, but unique still scores."""
        mod, site = self._mod_and_site(monkeypatch, tmp_path)
        # Truncation-style: unique = canonical \ isoform needs canonical exons, so
        # use an extension with a non-empty unique region but no canonical exons.
        site.canonical_orf_exons = []
        out = mod.annotate_site(site)
        assert out["primate_frac_intact"] is not None
        assert out["primate_canonical_frac_intact"] is None
        assert out["summary"]["canonical_status"] == "no_skeleton"

    def test_no_tree_leaves_deepest_none(self, monkeypatch, tmp_path):
        fake_hal = tmp_path / "fake.hal"
        fake_hal.write_bytes(b"")
        cfg = PipelineConfig(conservation=ConservationConfig(hal_path=fake_hal))
        mod = ConservationFrameModule(cfg)
        monkeypatch.setattr(mod, "_available", True)

        ref = "ATGAAATAA"
        monkeypatch.setattr(
            mod,
            "_fetch_alignment",
            lambda c, i, s: (ref, {"panTro6": ref}),
        )

        site = _tis()
        site.orf_exons = [(1000, 1009)]
        site.canonical_orf_exons = [(2000, 2009)]
        out = mod.annotate_site(site)
        assert out["primate_deepest_species"] is None
        assert out["primate_max_depth"] is None
        assert out["summary"]["tree_loaded"] is False


class TestRevcomp:
    def test_roundtrip(self):
        assert _revcomp_maf("ATGC") == "GCAT"

    def test_preserves_gaps(self):
        assert _revcomp_maf("AT-GC") == "GC-AT"

    def test_lowercase(self):
        assert _revcomp_maf("atGc") == "gCat"
