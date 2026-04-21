"""Tests for the Zoonomia-backed conservation SiteModule."""

from __future__ import annotations

from pathlib import Path

import pyBigWig
import pytest

from swissisoform.config import ConservationConfig, PipelineConfig
from swissisoform.models import ORFType, TranslationInitiationSite
from swissisoform.modules.conservation import (
    KOZAK_UPSTREAM_NT,
    START_CODON_LEN,
    ConservationModule,
)

# ---------------------------------------------------------------------------
# Synthetic BigWig helpers
# ---------------------------------------------------------------------------


def _write_bigwig(path: Path, chrom: str, length: int, value_fn) -> None:
    """Write a single-chrom BigWig with one value per base from *value_fn*.

    *value_fn* is called with the 0-based position (0..length-1) and must
    return a float.  Simple enough to let each test shape the track.
    """
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader([(chrom, length)])
    starts = list(range(length))
    ends = list(range(1, length + 1))
    values = [float(value_fn(i)) for i in starts]
    bw.addEntries([chrom] * length, starts, ends=ends, values=values)
    bw.close()


@pytest.fixture()
def ramp_bigwigs(tmp_path: Path) -> tuple[Path, Path]:
    """Two BigWigs on chr1 (length 2000).

    - PhyloP:    value == position (so interval means are known exactly)
    - PhastCons: value == position / 1000 (bounded 0..2)
    """
    phylop = tmp_path / "phylop.bw"
    phastcons = tmp_path / "phastcons.bw"
    _write_bigwig(phylop, "chr1", 2000, value_fn=lambda i: i)
    _write_bigwig(phastcons, "chr1", 2000, value_fn=lambda i: i / 1000.0)
    return phylop, phastcons


@pytest.fixture()
def config_with_bigwigs(ramp_bigwigs: tuple[Path, Path]) -> PipelineConfig:
    phylop, phastcons = ramp_bigwigs
    return PipelineConfig(
        conservation=ConservationConfig(
            phylop_bigwig=phylop,
            phastcons_bigwig=phastcons,
        ),
    )


def _tis(chrom: str, position: int, strand: str) -> TranslationInitiationSite:
    return TranslationInitiationSite(
        tis_id=f"{chrom}:{position}:{strand}:ATG",
        gene_name="TEST",
        transcript_id="ENST_TEST_001",
        chrom=chrom,
        position=position,
        strand=strand,
        start_codon="ATG",
        orf_type=ORFType.ANNOTATED,
        canonical_protein="M",
        isoform_protein="M",
    )


# ---------------------------------------------------------------------------
# Module metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_module_name(self):
        assert ConservationModule.MODULE_NAME == "conservation"

    def test_module_scope(self):
        assert ConservationModule.SCOPE == "C"

    def test_output_columns_include_new_keys(self):
        cols = ConservationModule.OUTPUT_COLUMNS
        assert "conservation_phylop_at_tis" in cols
        assert "conservation_phylop_kozak_mean" in cols
        assert "conservation_phastcons_at_tis" in cols
        assert "conservation_phastcons_kozak_mean" in cols
        assert "conservation_phylop_enrichment" in cols
        assert "conservation_summary" in cols


# ---------------------------------------------------------------------------
# Coordinate windows
# ---------------------------------------------------------------------------


class TestCoordinateWindows:
    def test_codon_window_plus_strand(self):
        start, end = ConservationModule._tis_codon_window(1000, "+")
        assert (start, end) == (1000, 1003)

    def test_codon_window_minus_strand(self):
        start, end = ConservationModule._tis_codon_window(1000, "-")
        # Minus strand: position is exclusive end, codon spans [pos-3, pos)
        assert (start, end) == (997, 1000)

    def test_kozak_window_plus_strand(self):
        start, end = ConservationModule._kozak_window(1000, "+")
        # 13 nt window: [1000-9, 1000+4) == [991, 1004)
        assert (start, end) == (991, 1004)
        assert end - start == 13

    def test_kozak_window_minus_strand(self):
        start, end = ConservationModule._kozak_window(1000, "-")
        # [1000-4, 1000+9) == [996, 1009)
        assert (start, end) == (996, 1009)
        assert end - start == 13

    def test_kozak_clamped_at_zero(self):
        start, end = ConservationModule._kozak_window(5, "+")
        assert start == 0
        assert end == 5 + 4

    def test_kozak_upstream_constant_matches_assembly(self):
        # The +strand upstream count is the 9 nt directly before the ATG.
        assert KOZAK_UPSTREAM_NT == 9
        assert START_CODON_LEN == 3


# ---------------------------------------------------------------------------
# BigWig lookups
# ---------------------------------------------------------------------------


class TestBigWigLookups:
    def test_phylop_at_tis_plus_strand(self, config_with_bigwigs):
        mod = ConservationModule(config_with_bigwigs)
        try:
            site = _tis("chr1", 1000, "+")
            out = mod.annotate_site(site)
        finally:
            mod.close()
        # values at 1000, 1001, 1002 → mean 1001.0
        assert out["phylop_at_tis"] == pytest.approx(1001.0)

    def test_phylop_kozak_mean_plus_strand(self, config_with_bigwigs):
        mod = ConservationModule(config_with_bigwigs)
        try:
            site = _tis("chr1", 1000, "+")
            out = mod.annotate_site(site)
        finally:
            mod.close()
        # window [991, 1004); mean of 991..1003 == 997.0
        assert out["phylop_kozak_mean"] == pytest.approx(997.0)

    def test_minus_strand_uses_different_window(self, config_with_bigwigs):
        mod = ConservationModule(config_with_bigwigs)
        try:
            site = _tis("chr1", 1000, "-")
            out = mod.annotate_site(site)
        finally:
            mod.close()
        # codon [997, 1000); mean 998.0
        assert out["phylop_at_tis"] == pytest.approx(998.0)
        # kozak [996, 1009); mean 1002.0
        assert out["phylop_kozak_mean"] == pytest.approx(1002.0)

    def test_phastcons_track_independent(self, config_with_bigwigs):
        mod = ConservationModule(config_with_bigwigs)
        try:
            site = _tis("chr1", 1000, "+")
            out = mod.annotate_site(site)
        finally:
            mod.close()
        # values divided by 1000
        assert out["phastcons_at_tis"] == pytest.approx(1.001)
        assert out["phastcons_kozak_mean"] == pytest.approx(0.997)

    def test_missing_chrom_returns_none(self, config_with_bigwigs):
        mod = ConservationModule(config_with_bigwigs)
        try:
            site = _tis("chrZ", 1000, "+")
            out = mod.annotate_site(site)
        finally:
            mod.close()
        assert out["phylop_at_tis"] is None
        assert out["phastcons_at_tis"] is None
        assert out["summary"]["phylop_status"] == "ok"

    def test_out_of_bounds_returns_none(self, config_with_bigwigs):
        mod = ConservationModule(config_with_bigwigs)
        try:
            site = _tis("chr1", 5000, "+")  # past length=2000
            out = mod.annotate_site(site)
        finally:
            mod.close()
        assert out["phylop_at_tis"] is None

    def test_accepts_bare_chrom_name(self, tmp_path):
        """BigWig stored as "1" (no chr prefix) is still found."""
        bw = tmp_path / "phylop.bw"
        _write_bigwig(bw, "1", 2000, value_fn=lambda i: float(i))
        cfg = PipelineConfig(conservation=ConservationConfig(phylop_bigwig=bw))
        mod = ConservationModule(cfg)
        try:
            site = _tis("chr1", 1000, "+")
            out = mod.annotate_site(site)
        finally:
            mod.close()
        assert out["phylop_at_tis"] == pytest.approx(1001.0)


# ---------------------------------------------------------------------------
# Region-mean stubs
# ---------------------------------------------------------------------------


class TestRegionStubs:
    def test_region_means_are_none(self, config_with_bigwigs):
        mod = ConservationModule(config_with_bigwigs)
        try:
            site = _tis("chr1", 1000, "+")
            out = mod.annotate_site(site)
        finally:
            mod.close()
        assert out["phylop_unique_region_mean"] is None
        assert out["phylop_shared_region_mean"] is None
        assert out["phylop_enrichment"] is None
        assert out["phastcons_unique_region_mean"] is None
        assert out["phastcons_shared_region_mean"] is None
        assert out["summary"]["region_status"] == "region_map_not_implemented"


# ---------------------------------------------------------------------------
# Configless / not-run
# ---------------------------------------------------------------------------


class TestNotRun:
    def test_no_conservation_config(self):
        cfg = PipelineConfig(conservation=None)
        mod = ConservationModule(cfg)
        site = _tis("chr1", 1000, "+")
        out = mod.annotate_site(site)
        assert out["phylop_at_tis"] is None
        assert out["phastcons_at_tis"] is None
        assert out["summary"]["phylop_status"] == "not_run"
        assert out["summary"]["phastcons_status"] == "not_run"

    def test_missing_bigwig_file(self, tmp_path):
        cfg = PipelineConfig(
            conservation=ConservationConfig(phylop_bigwig=tmp_path / "nope.bw"),
        )
        mod = ConservationModule(cfg)
        site = _tis("chr1", 1000, "+")
        out = mod.annotate_site(site)
        assert out["summary"]["phylop_status"] == "not_run"
        assert out["summary"]["phylop_bigwig"] is None


# ---------------------------------------------------------------------------
# run() preserves sites + wires through SiteModule surface
# ---------------------------------------------------------------------------


class TestRunWrapper:
    def test_run_preserves_all_sites(self, synthetic_tis, config):
        mod = ConservationModule(config)
        try:
            out = mod.run(synthetic_tis)
        finally:
            mod.close()
        assert len(out) == len(synthetic_tis)
        for site in out:
            assert "conservation" in site.isoform_annotations

    def test_run_writes_expected_keys(self, synthetic_tis, config_with_bigwigs):
        mod = ConservationModule(config_with_bigwigs)
        try:
            mod.run(synthetic_tis)
        finally:
            mod.close()
        for site in synthetic_tis:
            ann = site.isoform_annotations["conservation"]
            for key in (
                "phylop_at_tis",
                "phylop_kozak_mean",
                "phastcons_at_tis",
                "phastcons_kozak_mean",
                "summary",
            ):
                assert key in ann
