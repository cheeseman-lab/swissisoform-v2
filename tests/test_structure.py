"""Tests for the structure prediction subpackage + SiteModule.

No GPU, no Boltz/Chai — we mock the cache and exercise the lookup +
comparison logic. Structural metrics (TM-score / RMSD / contacts) skip
gracefully when biotite / tmtools aren't installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from swissisoform.structure.compare import compare_confidence
from swissisoform.structure.fold import (
    cache_path,
    derive_metrics,
    load_cache,
    precompute_fold,
    protein_hash,
    write_cache,
)
from swissisoform.structure.module import StructureModule


def _seed_cache(
    cache_dir: Path,
    seq: str,
    *,
    backend: str = "boltz",
    plddt: list[float] | None = None,
    ptm: float | None = None,
    status: str = "ok",
    write_cif: bool = False,
) -> str:
    h = protein_hash(seq)
    metrics = derive_metrics(backend=backend, plddt=plddt, ptm=ptm, status=status)
    metrics["length"] = len(seq.rstrip("*"))
    confidence = (
        {"plddt": plddt, "ptm": ptm, "iptm": None} if plddt is not None else None
    )
    write_cache(
        h,
        cache_dir=cache_dir,
        backend=backend,
        confidence=confidence,
        metrics=metrics,
        cif_text=("# stub CIF\n" if write_cif else None),
    )
    return h


class TestCacheRoundtrip:
    """Cache I/O — write + load are symmetric."""

    def test_load_missing_returns_none(self, tmp_path):
        assert load_cache("deadbeef", tmp_path) is None

    def test_write_load_roundtrip(self, tmp_path):
        seq = "MAEPRSTV"
        h = _seed_cache(tmp_path, seq, plddt=[80.0] * len(seq), ptm=0.83)
        out = load_cache(h, tmp_path, "boltz")
        assert out is not None
        assert out["metrics"]["plddt_mean"] == pytest.approx(80.0)
        assert out["metrics"]["ptm"] == 0.83
        assert out["confidence"]["plddt"] == [80.0] * len(seq)

    def test_cache_path_layout(self, tmp_path):
        h = "abc123"
        p = cache_path(tmp_path, "boltz", h)
        assert p == tmp_path / "boltz" / h

    def test_pae_path_absent_by_default(self, tmp_path):
        seq = "MAEPRSTV"
        h = _seed_cache(tmp_path, seq, plddt=[80.0] * len(seq), ptm=0.83)
        out = load_cache(h, tmp_path, "boltz")
        assert out is not None
        # Entries folded before PAE capture carry no pae.npy.
        assert out["pae_path"] is None

    def test_pae_path_exposed_when_present(self, tmp_path):
        np = pytest.importorskip("numpy")
        seq = "MAEPRSTV"
        h = _seed_cache(tmp_path, seq, plddt=[80.0] * len(seq), ptm=0.83)
        base = cache_path(tmp_path, "boltz", h)
        np.save(base / "pae.npy", np.zeros((len(seq), len(seq)), dtype=np.float16))
        out = load_cache(h, tmp_path, "boltz")
        assert out is not None and out["pae_path"] == base / "pae.npy"
        assert out["pae_path"].exists()

    def test_precompute_inline_false_skips_missing(self, tmp_path):
        proteins = {"a": "MAEPRSTV", "b": "MGGGAA"}
        res = precompute_fold(proteins, cache_dir=tmp_path, inline=False)
        assert res == {}

    def test_precompute_returns_cached_only(self, tmp_path):
        seq = "MAEPRSTV"
        h = _seed_cache(tmp_path, seq, plddt=[70.0] * len(seq), ptm=0.5)
        res = precompute_fold({"x": seq}, cache_dir=tmp_path, backend="boltz", inline=False)
        assert h in res
        assert res[h]["metrics"]["plddt_mean"] == pytest.approx(70.0)

    def test_too_long_marks_status(self, tmp_path):
        # inline=True but no backend installed → too-long path still works
        # because length check happens before backend dispatch.
        long_seq = "A" * 200
        res = precompute_fold(
            {"x": long_seq},
            cache_dir=tmp_path,
            max_seq_len=50,
            inline=True,
        )
        h = protein_hash(long_seq)
        assert h in res
        assert res[h]["metrics"]["status"] == "too_long"
        assert res[h]["metrics"]["length"] == 200


class TestCompareConfidence:
    """pLDDT-derived comparison metrics — pure numpy."""

    def test_extension_diffregion_picks_isoform(self):
        # canonical len=10 (uniform 50), isoform = "EXT" + canonical (uniform 90 in EXT, 50 in body)
        ext_n = 3
        canonical_plddt = [50.0] * 10
        isoform_plddt = [90.0] * ext_n + [50.0] * 10
        out = compare_confidence(
            canonical_plddt,
            isoform_plddt,
            diff_isoform_start=0,
            diff_isoform_end=ext_n,
            diff_canonical_start=None,
            diff_canonical_end=None,
            orf_type="extended",
        )
        assert out["plddt_diffregion_mean"] == pytest.approx(90.0)
        assert out["plddt_canonical_mean"] == pytest.approx(50.0)
        # Shared region of isoform after ext is uniform 50 — delta vs canonical body is 0.
        assert out["plddt_delta_shared"] == pytest.approx(0.0)

    def test_truncation_diffregion_picks_canonical(self):
        # Truncated: lost first 4 residues from canonical
        canonical_plddt = [80.0] * 4 + [60.0] * 8
        isoform_plddt = [60.0] * 8
        out = compare_confidence(
            canonical_plddt,
            isoform_plddt,
            diff_isoform_start=None,
            diff_isoform_end=None,
            diff_canonical_start=0,
            diff_canonical_end=4,
            orf_type="truncated",
        )
        assert out["plddt_diffregion_mean"] == pytest.approx(80.0)
        assert out["plddt_delta_shared"] == pytest.approx(0.0)

    def test_no_inputs_returns_nones(self):
        out = compare_confidence(
            None,
            None,
            diff_isoform_start=0,
            diff_isoform_end=5,
            diff_canonical_start=None,
            diff_canonical_end=None,
        )
        assert out["plddt_diffregion_mean"] is None
        assert out["plddt_canonical_mean"] is None
        assert out["plddt_isoform_mean"] is None


class TestStructureModule:
    """Module contract — SiteModule semantics + status surfacing."""

    def test_no_cache_status(self, synthetic_tis, config, tmp_path):
        module = StructureModule(config, cache_dir=tmp_path, backend="boltz")
        ann = module.annotate_site(synthetic_tis[0])
        assert ann["status"] == "no_cache"
        assert ann["plddt_diffregion_mean"] is None

    def test_extension_metrics_from_cache(self, synthetic_tis, config, tmp_path):
        # site 3 — chr1:940:+:CTG, extension on +strand
        site = next(s for s in synthetic_tis if s.tis_id == "chr1:940:+:CTG")
        ext_n = site.diff_region.isoform_end
        iso_seq = site.isoform_protein.rstrip("*")
        can_seq = site.canonical_protein.rstrip("*")
        iso_plddt = [85.0] * ext_n + [60.0] * (len(iso_seq) - ext_n)
        can_plddt = [60.0] * len(can_seq)
        _seed_cache(tmp_path, site.isoform_protein, plddt=iso_plddt, ptm=0.7)
        _seed_cache(tmp_path, site.canonical_protein, plddt=can_plddt, ptm=0.7)

        module = StructureModule(config, cache_dir=tmp_path, backend="boltz")
        ann = module.annotate_site(site)
        assert ann["status"] == "ok"
        assert ann["backend"] == "boltz"
        assert ann["plddt_diffregion_mean"] == pytest.approx(85.0)
        assert ann["plddt_canonical_mean"] == pytest.approx(60.0)
        assert ann["plddt_isoform_mean"] == pytest.approx(
            (85.0 * ext_n + 60.0 * (len(iso_seq) - ext_n)) / len(iso_seq)
        )

    def test_truncation_uses_canonical(self, synthetic_tis, config, tmp_path):
        # site 5 — chr1:1030:+:ATG, truncation
        site = next(s for s in synthetic_tis if s.tis_id == "chr1:1030:+:ATG")
        can_seq = site.canonical_protein.rstrip("*")
        iso_seq = site.isoform_protein.rstrip("*")
        lost_n = site.diff_region.canonical_end
        can_plddt = [85.0] * lost_n + [60.0] * (len(can_seq) - lost_n)
        iso_plddt = [60.0] * len(iso_seq)
        _seed_cache(tmp_path, site.canonical_protein, plddt=can_plddt, ptm=0.7)
        _seed_cache(tmp_path, site.isoform_protein, plddt=iso_plddt, ptm=0.7)

        module = StructureModule(config, cache_dir=tmp_path, backend="boltz")
        ann = module.annotate_site(site)
        assert ann["status"] == "ok"
        assert ann["plddt_diffregion_mean"] == pytest.approx(85.0)

    def test_too_long_status_propagates(self, synthetic_tis, config, tmp_path):
        site = synthetic_tis[2]  # extension
        # canonical too_long, isoform ok
        h_can = _seed_cache(tmp_path, site.canonical_protein, status="too_long")
        iso_n = len(site.isoform_protein.rstrip("*"))
        _seed_cache(tmp_path, site.isoform_protein, plddt=[70.0] * iso_n, ptm=0.6)
        # check the metrics file says too_long
        with open(tmp_path / "boltz" / h_can / "metrics.json") as fh:
            assert json.load(fh)["status"] == "too_long"
        module = StructureModule(config, cache_dir=tmp_path, backend="boltz")
        ann = module.annotate_site(site)
        assert ann["status"] == "too_long"

    def test_run_does_not_drop_sites(self, synthetic_tis, config, tmp_path):
        module = StructureModule(config, cache_dir=tmp_path, backend="boltz")
        out = module.run(synthetic_tis)
        assert len(out) == len(synthetic_tis)
        for s in out:
            keys = set(s.isoform_annotations["structure"])
            for k in [
                "status",
                "backend",
                "plddt_canonical_mean",
                "plddt_isoform_mean",
                "plddt_diffregion_mean",
                "plddt_diffregion_std",
                "plddt_delta_shared",
                "plddt_shared_mean_isoform",
                "plddt_shared_mean_canonical",
                "tm_score",
                "rmsd_global",
                "extension_contacts",
                "rmsd_shared",
                "tm_score_shared",
                "shared_region_len",
                "rmsd_shared_status",
                "ptm_isoform",
                "ptm_canonical",
                "pae_diff_vs_diff",
                "pae_body_vs_body",
                "pae_diff_vs_body",
                "pae_status",
            ]:
                assert k in keys

    def test_ptm_surfaced_from_cache(self, synthetic_tis, config, tmp_path):
        """pTM is read straight from metrics.json for both sides."""
        site = next(s for s in synthetic_tis if s.tis_id == "chr1:940:+:CTG")
        iso_n = len(site.isoform_protein.rstrip("*"))
        can_n = len(site.canonical_protein.rstrip("*"))
        _seed_cache(tmp_path, site.isoform_protein, plddt=[70.0] * iso_n, ptm=0.81)
        _seed_cache(tmp_path, site.canonical_protein, plddt=[70.0] * can_n, ptm=0.42)
        module = StructureModule(config, cache_dir=tmp_path, backend="boltz")
        ann = module.annotate_site(site)
        assert ann["ptm_isoform"] == pytest.approx(0.81)
        assert ann["ptm_canonical"] == pytest.approx(0.42)

    def test_pae_blocks_on_extension(self, synthetic_tis, config, tmp_path):
        """PAE blocks compute on the isoform fold when pae.npy is present."""
        np = pytest.importorskip("numpy")
        site = next(s for s in synthetic_tis if s.tis_id == "chr1:940:+:CTG")
        iso_n = len(site.isoform_protein.rstrip("*"))
        can_n = len(site.canonical_protein.rstrip("*"))
        h_iso = _seed_cache(tmp_path, site.isoform_protein, plddt=[70.0] * iso_n, ptm=0.7)
        _seed_cache(tmp_path, site.canonical_protein, plddt=[70.0] * can_n, ptm=0.7)
        # Low intra-diff PAE, high diff↔body PAE (a "dangling" extension).
        pae = np.full((iso_n, iso_n), 20.0, dtype=np.float16)
        d = site.diff_region.isoform_end
        pae[:d, :d] = 2.0  # rigid extension internally
        np.save(cache_path(tmp_path, "boltz", h_iso) / "pae.npy", pae)
        module = StructureModule(config, cache_dir=tmp_path, backend="boltz")
        ann = module.annotate_site(site)
        assert ann["pae_status"] == "ok"
        assert ann["pae_diff_vs_diff"] == pytest.approx(2.0, abs=0.1)
        assert ann["pae_diff_vs_body"] > ann["pae_diff_vs_diff"]

    def test_pae_status_no_pae_when_absent(self, synthetic_tis, config, tmp_path):
        """No pae.npy → pae_status='no_pae', block means null."""
        site = next(s for s in synthetic_tis if s.tis_id == "chr1:940:+:CTG")
        iso_n = len(site.isoform_protein.rstrip("*"))
        can_n = len(site.canonical_protein.rstrip("*"))
        _seed_cache(tmp_path, site.isoform_protein, plddt=[70.0] * iso_n, ptm=0.7)
        _seed_cache(tmp_path, site.canonical_protein, plddt=[70.0] * can_n, ptm=0.7)
        module = StructureModule(config, cache_dir=tmp_path, backend="boltz")
        ann = module.annotate_site(site)
        assert ann["pae_status"] == "no_pae"
        assert ann["pae_diff_vs_diff"] is None


class TestStructuralMetricsLazy:
    """Structural metrics return None when biotite/tmtools missing or files absent."""

    def test_compare_structures_missing_files(self):
        from swissisoform.structure.compare import compare_structures

        out = compare_structures(None, None)
        assert out["tm_score"] is None
        assert out["rmsd_global"] is None
        assert out["extension_contacts"] is None


class TestPaeRegionBlocks:
    """PAE region-block partitioning — pure numpy arithmetic."""

    def test_none_matrix(self):
        from swissisoform.structure.compare import pae_region_blocks

        assert pae_region_blocks(None, 0, 5)["pae_status"] == "no_pae"

    def test_no_diff_region(self):
        np = pytest.importorskip("numpy")
        from swissisoform.structure.compare import pae_region_blocks

        m = np.zeros((10, 10), dtype=np.float16)
        assert pae_region_blocks(m, None, None)["pae_status"] == "no_diff_region"

    def test_whole_is_diff_no_body(self):
        np = pytest.importorskip("numpy")
        from swissisoform.structure.compare import pae_region_blocks

        m = np.full((8, 8), 5.0, dtype=np.float16)
        out = pae_region_blocks(m, 0, 8)
        assert out["pae_status"] == "no_body"
        assert out["pae_diff_vs_diff"] == pytest.approx(5.0)
        assert out["pae_body_vs_body"] is None

    def test_block_means_and_asymmetry(self):
        np = pytest.importorskip("numpy")
        from swissisoform.structure.compare import pae_region_blocks

        # diff = [0,2); body = [2,6). Distinct constant fill per block pair.
        m = np.zeros((6, 6), dtype=float)
        m[:2, :2] = 1.0  # diff-diff
        m[2:, 2:] = 9.0  # body-body
        m[:2, 2:] = 4.0  # diff→body
        m[2:, :2] = 6.0  # body→diff
        out = pae_region_blocks(m, 0, 2)
        assert out["pae_status"] == "ok"
        assert out["pae_diff_vs_diff"] == pytest.approx(1.0)
        assert out["pae_body_vs_body"] == pytest.approx(9.0)
        # inter-block averages BOTH off-diagonal rectangles: (4+6)/2 = 5.
        assert out["pae_diff_vs_body"] == pytest.approx(5.0)
