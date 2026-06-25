"""Tests for the PLM VEP module + the embed cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from swissisoform.plm.embed import (
    load_cache,
    precompute_plm,
    protein_hash,
)
from swissisoform.plm.module import PLMVEPModule


def _seed_cache(cache_dir: Path, seq: str, llr: np.ndarray) -> str:
    """Write a synthetic .npz cache entry mimicking the precompute output."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = protein_hash(seq)
    np.savez_compressed(cache_dir / f"{h}.npz", llr=llr.astype(np.float32))
    return h


class TestEmbedCache:
    """Round-trip the cache layer without touching torch/transformers."""

    def test_load_cache_missing_returns_none(self, tmp_path):
        assert load_cache("deadbeef", tmp_path) is None

    def test_load_cache_roundtrip(self, tmp_path):
        seq = "MAEPRSTV"
        llr = np.linspace(-2.0, -0.1, len(seq)).astype(np.float32)
        h = _seed_cache(tmp_path, seq, llr)
        out = load_cache(h, tmp_path)
        assert out is not None
        np.testing.assert_array_almost_equal(out["llr"], llr)

    def test_precompute_skip_missing_when_inline_false(self, tmp_path):
        """With inline=False, uncached proteins are skipped silently."""
        proteins = {"a": "MAEPRSTV", "b": "MGGGAA"}
        res = precompute_plm(proteins, cache_dir=tmp_path, inline=False)
        assert res == {}

    def test_precompute_returns_cached_only(self, tmp_path):
        seq = "MAEPRSTV"
        llr = np.array([-1.0] * len(seq), dtype=np.float32)
        h = _seed_cache(tmp_path, seq, llr)
        res = precompute_plm({"x": seq}, cache_dir=tmp_path, inline=False)
        assert h in res
        np.testing.assert_array_almost_equal(res[h]["llr"], llr)


class TestPLMVEPModule:
    """SiteModule contract + region slicing for cached LLR."""

    def test_no_cache_returns_no_cache_status(self, synthetic_tis, config, tmp_path):
        module = PLMVEPModule(config, cache_dir=tmp_path)
        ann = module.annotate_site(synthetic_tis[0])
        assert ann["status"] == "no_cache"
        assert ann["mean_llr_isoform"] is None
        assert ann["constraint_enrichment"] is None

    def test_extension_isoform_space(self, synthetic_tis, config, tmp_path):
        # Site 3: extension +strand; isoform is "LRRPPAGA" + canonical, diff 0:8
        site = next(s for s in synthetic_tis if s.tis_id == "chr1:940:+:CTG")
        iso_seq = site.isoform_protein.rstrip("*")
        # Build LLR: extension residues strongly constrained (-6.0), shared mild (-1.0).
        llr = np.full(len(iso_seq), -1.0, dtype=np.float32)
        llr[: site.diff_region.isoform_end] = -6.0
        _seed_cache(tmp_path, site.isoform_protein, llr)
        # canonical too (all -1.0)
        can_seq = site.canonical_protein.rstrip("*")
        _seed_cache(tmp_path, site.canonical_protein, np.full(len(can_seq), -1.0, dtype=np.float32))

        module = PLMVEPModule(config, cache_dir=tmp_path, constraint_threshold=-5.0)
        ann = module.annotate_site(site)
        assert ann["status"] == "ok"
        assert ann["mean_llr_unique_region"] == pytest.approx(-6.0)
        assert ann["mean_llr_shared_region"] == pytest.approx(-1.0)
        assert ann["constraint_enrichment"] == pytest.approx(6.0)
        assert ann["n_constrained_positions_unique"] == site.diff_region.isoform_end
        assert ann["n_constrained_positions_shared"] == 0

    def test_truncation_uses_canonical_space(self, synthetic_tis, config, tmp_path):
        # Site 5: truncation +strand; lost region in canonical 0:9
        site = next(s for s in synthetic_tis if s.tis_id == "chr1:1030:+:ATG")
        can_seq = site.canonical_protein.rstrip("*")
        llr = np.full(len(can_seq), -1.0, dtype=np.float32)
        llr[: site.diff_region.canonical_end] = -7.0
        _seed_cache(tmp_path, site.canonical_protein, llr)
        # isoform too
        iso_seq = site.isoform_protein.rstrip("*")
        _seed_cache(tmp_path, site.isoform_protein, np.full(len(iso_seq), -1.0, dtype=np.float32))

        module = PLMVEPModule(config, cache_dir=tmp_path, constraint_threshold=-5.0)
        ann = module.annotate_site(site)
        assert ann["status"] == "ok"
        assert ann["mean_llr_unique_region"] == pytest.approx(-7.0)
        assert ann["mean_llr_shared_region"] == pytest.approx(-1.0)
        assert ann["n_constrained_positions_unique"] == site.diff_region.canonical_end

    def test_run_does_not_drop_sites(self, synthetic_tis, config, tmp_path):
        module = PLMVEPModule(config, cache_dir=tmp_path)
        out = module.run(synthetic_tis)
        assert len(out) == len(synthetic_tis)
        for s in out:
            assert "plm_vep" in s.isoform_annotations
            keys = set(s.isoform_annotations["plm_vep"])
            for k in [
                "status",
                "mean_llr_isoform",
                "mean_llr_canonical",
                "mean_llr_unique_region",
                "mean_llr_shared_region",
                "constraint_enrichment",
                "n_constrained_positions_unique",
                "n_constrained_positions_shared",
            ]:
                assert k in keys
