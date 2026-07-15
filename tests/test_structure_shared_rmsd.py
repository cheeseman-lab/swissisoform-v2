"""Tests for the F7 shared-region Cα RMSD metric + evidence criterion.

Three layers, no GPU / no real CIFs:

- ``_kabsch_rmsd`` — pure numpy; rigid transforms → 0, internal perturbation → >0.
- ``compare_structures`` shared-region block — the CIF parsers are monkeypatched
  to hand back controlled Cα coordinates so the pairing math + status logic is
  exercised directly.
- ``f7_shared_rmsd.score`` — the full tri-state matrix over a synthetic
  ``structure`` annotation dict.
"""

from __future__ import annotations

import numpy as np
import pytest

from swissisoform.config import ScoringConfig
from swissisoform.evidence import f7_shared_rmsd
from swissisoform.structure import compare as C

# ---------------------------------------------------------------------------
# _kabsch_rmsd
# ---------------------------------------------------------------------------


class TestKabsch:
    def test_identical_is_zero(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(30, 3))
        assert C._kabsch_rmsd(x, x) == pytest.approx(0.0, abs=1e-9)

    def test_rigid_transform_is_zero(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(25, 3))
        theta = 0.9
        rot = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        y = x @ rot.T + np.array([10.0, -4.0, 7.0])  # rotate + translate
        assert C._kabsch_rmsd(x, y) == pytest.approx(0.0, abs=1e-9)

    def test_internal_perturbation_is_positive(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(20, 3))
        y = x.copy()
        y[0] += np.array([4.0, 0.0, 0.0])  # move one residue — real change
        assert C._kabsch_rmsd(x, y) > 0.5

    def test_shape_mismatch_returns_none(self):
        x = np.zeros((10, 3))
        assert C._kabsch_rmsd(x, x[:5]) is None

    def test_empty_returns_none(self):
        assert C._kabsch_rmsd(np.zeros((0, 3)), np.zeros((0, 3))) is None


# ---------------------------------------------------------------------------
# compare_structures — shared-region block (monkeypatched CIF parsers)
# ---------------------------------------------------------------------------


def _patch_coords(monkeypatch, can_xyz, can_seq, iso_xyz, iso_seq):
    """Make the (lazy) CIF parsers return controlled coords keyed by filename."""

    def fake_load(path):
        # Match the basename, not the full path — pytest tmp dirs derived from a
        # test name like "…canonical_tail" would otherwise collide on "canon".
        from pathlib import Path as _P

        return "canon" if _P(str(path)).name.startswith("canon") else "iso"

    def fake_ca(struct):
        if struct == "canon":
            return np.asarray(can_xyz, dtype=float), can_seq
        return np.asarray(iso_xyz, dtype=float), iso_seq

    monkeypatch.setattr(C, "_try_load_structure", fake_load)
    monkeypatch.setattr(C, "_ca_coords_and_seq", fake_ca)


@pytest.fixture
def cif_paths(tmp_path):
    can = tmp_path / "canon.cif"
    iso = tmp_path / "iso.cif"
    can.write_text("# stub\n")
    iso.write_text("# stub\n")
    return can, iso


class TestCompareStructuresShared:
    def test_extension_identical_shared_is_zero(self, monkeypatch, cif_paths):
        rng = np.random.default_rng(3)
        can = rng.normal(size=(12, 3)) * 5.0
        ext = rng.normal(size=(3, 3)) * 5.0
        iso = np.vstack([ext, can])  # extension: iso = ext + canonical body
        can_seq = "ACDEFGHIKLMN"
        iso_seq = "PQR" + can_seq
        _patch_coords(monkeypatch, can, can_seq, iso, iso_seq)

        out = C.compare_structures(
            cif_paths[0],
            cif_paths[1],
            diff_isoform_start=0,
            diff_isoform_end=3,
            orf_type="extended",
            diff_region_confidence="exact",
        )
        assert out["rmsd_shared_status"] == "ok"
        assert out["rmsd_shared"] == pytest.approx(0.0, abs=1e-6)
        assert out["shared_region_len"] == 12
        assert out["tm_score_shared"] == pytest.approx(1.0, abs=1e-3)

    def test_extension_perturbed_shared_is_positive(self, monkeypatch, cif_paths):
        rng = np.random.default_rng(4)
        can = rng.normal(size=(14, 3)) * 5.0
        ext = rng.normal(size=(2, 3)) * 5.0
        can_pert = can.copy()
        can_pert[0] += np.array([6.0, 0.0, 0.0])  # refolded residue in the shared body
        iso = np.vstack([ext, can_pert])
        can_seq = "ACDEFGHIKLMNPQ"
        iso_seq = "RS" + can_seq
        _patch_coords(monkeypatch, can, can_seq, iso, iso_seq)

        out = C.compare_structures(
            cif_paths[0],
            cif_paths[1],
            diff_isoform_end=2,
            orf_type="extended",
            diff_region_confidence="tail_verified",
        )
        assert out["rmsd_shared_status"] == "ok"
        assert out["rmsd_shared"] > 0.5
        assert out["shared_region_len"] == 14

    def test_truncation_pairs_canonical_tail(self, monkeypatch, cif_paths):
        rng = np.random.default_rng(5)
        can = rng.normal(size=(15, 3)) * 5.0
        iso = can[4:].copy()  # truncation: iso = canonical[4:]
        can_seq = "ACDEFGHIKLMNPQR"
        iso_seq = can_seq[4:]
        _patch_coords(monkeypatch, can, can_seq, iso, iso_seq)

        out = C.compare_structures(
            cif_paths[0],
            cif_paths[1],
            diff_canonical_start=0,
            diff_canonical_end=4,
            orf_type="truncated",
            diff_region_confidence="exact",
        )
        assert out["rmsd_shared_status"] == "ok"
        assert out["rmsd_shared"] == pytest.approx(0.0, abs=1e-6)
        assert out["shared_region_len"] == 11

    def test_initiator_met_truncation_drops_leading_M(self, monkeypatch, cif_paths):
        # Near-cognate truncation: iso = installed-M + canonical[k:]. The leading
        # M is isoform-unique (a start-codon substitution); the shared suffix
        # iso[1:] == canonical[k:] is byte-identical. Suffix-alignment must drop
        # the M and superpose iso[1:] onto canonical[k:] → RMSD ~0. A prefix
        # slice would pair off-by-one and give a large RMSD.
        rng = np.random.default_rng(8)
        can = rng.normal(size=(15, 3)) * 5.0
        k = 4  # diff_canonical_end = len(can) - (len(iso) - 1) = 15 - 11 = 4
        m_coord = rng.normal(size=(1, 3)) * 5.0  # installed Met, arbitrary position
        iso = np.vstack([m_coord, can[k:]])  # iso = M + canonical[k:]  (12 residues)
        can_seq = "ACDEFGHIKLMNPQR"
        iso_seq = "M" + can_seq[k:]  # leading M substitutes; downstream identical
        _patch_coords(monkeypatch, can, can_seq, iso, iso_seq)

        out = C.compare_structures(
            cif_paths[0],
            cif_paths[1],
            diff_canonical_start=0,
            diff_canonical_end=k,
            orf_type="truncated",
            diff_region_confidence="initiator_met",  # verified tier
        )
        assert out["rmsd_shared_status"] == "ok"  # tier admitted
        assert out["rmsd_shared"] == pytest.approx(0.0, abs=1e-6)  # M dropped, not off-by-one
        assert out["shared_region_len"] == 11
        assert out["tm_score_shared"] == pytest.approx(1.0, abs=1e-3)

    def test_uorf_has_no_shared_region(self, monkeypatch, cif_paths):
        rng = np.random.default_rng(6)
        can = rng.normal(size=(10, 3))
        iso = rng.normal(size=(8, 3))
        _patch_coords(monkeypatch, can, "ACDEFGHIKL", iso, "ACDEFGHI")
        out = C.compare_structures(
            cif_paths[0],
            cif_paths[1],
            orf_type="uorf",
            diff_region_confidence="exact",
        )
        assert out["rmsd_shared_status"] == "no_shared_region"
        assert out["rmsd_shared"] is None

    def test_unverified_alignment_skipped(self, monkeypatch, cif_paths):
        rng = np.random.default_rng(7)
        can = rng.normal(size=(12, 3))
        iso = np.vstack([rng.normal(size=(3, 3)), can])
        _patch_coords(monkeypatch, can, "ACDEFGHIKLMN", iso, "PQRACDEFGHIKLMN")
        out = C.compare_structures(
            cif_paths[0],
            cif_paths[1],
            diff_isoform_end=3,
            orf_type="extended",
            diff_region_confidence="length_fallback",  # unverified tier
        )
        assert out["rmsd_shared_status"] == "unverified_alignment"
        assert out["rmsd_shared"] is None

    def test_too_short_shared_is_no_region(self, monkeypatch, cif_paths):
        # Only 2 shared residues — fewer than the 3 needed to superpose.
        can = np.zeros((2, 3))
        iso = np.vstack([np.ones((3, 3)), can])
        _patch_coords(monkeypatch, can, "AC", iso, "PQRAC")
        out = C.compare_structures(
            cif_paths[0],
            cif_paths[1],
            diff_isoform_end=3,
            orf_type="extended",
            diff_region_confidence="exact",
        )
        assert out["rmsd_shared_status"] == "no_shared_region"

    def test_missing_files_status_no_cache(self):
        out = C.compare_structures(None, None, orf_type="extended")
        assert out["rmsd_shared"] is None
        assert out["rmsd_shared_status"] == "no_cache"


# ---------------------------------------------------------------------------
# f7_shared_rmsd.score — tri-state matrix
# ---------------------------------------------------------------------------


class _FakeSite:
    def __init__(self, structure_ann):
        self.isoform_annotations = {"structure": structure_ann}


def _struct_ann(**over):
    base = {
        "status": "ok",
        "rmsd_shared_status": "ok",
        "rmsd_shared": 3.0,
        "tm_score_shared": 0.6,
        "shared_region_len": 120,
        "plddt_shared_mean_isoform": 0.85,
        "plddt_shared_mean_canonical": 0.82,
    }
    base.update(over)
    return base


class TestF7Score:
    cfg = ScoringConfig()

    def test_missing_annotation_is_none(self):
        site = _FakeSite(None)
        # isoform_annotations has {"structure": None} → _annotation returns None
        r = f7_shared_rmsd.score(site, self.cfg)
        assert r.value is None

    def test_bad_overall_status_is_none(self):
        for st in ("no_cache", "too_long", "failed", "uniform_plddt", "partial"):
            r = f7_shared_rmsd.score(_FakeSite(_struct_ann(status=st)), self.cfg)
            assert r.value is None, st

    def test_no_shared_region_is_none(self):
        r = f7_shared_rmsd.score(
            _FakeSite(_struct_ann(rmsd_shared_status="no_shared_region", rmsd_shared=None)),
            self.cfg,
        )
        assert r.value is None

    def test_unverified_alignment_is_none(self):
        r = f7_shared_rmsd.score(
            _FakeSite(_struct_ann(rmsd_shared_status="unverified_alignment", rmsd_shared=None)),
            self.cfg,
        )
        assert r.value is None

    def test_too_short_is_none(self):
        r = f7_shared_rmsd.score(_FakeSite(_struct_ann(shared_region_len=10)), self.cfg)
        assert r.value is None
        assert "too short" in r.reason

    def test_low_confidence_is_none(self):
        r = f7_shared_rmsd.score(
            _FakeSite(_struct_ann(plddt_shared_mean_isoform=0.55)), self.cfg
        )
        assert r.value is None
        assert "confidently folded" in r.reason

    def test_high_confidence_over_threshold_is_true(self):
        r = f7_shared_rmsd.score(_FakeSite(_struct_ann(rmsd_shared=3.0)), self.cfg)
        assert r.value is True
        assert "RMSD 3.00" in r.reason

    def test_high_confidence_under_threshold_is_false(self):
        r = f7_shared_rmsd.score(_FakeSite(_struct_ann(rmsd_shared=0.8)), self.cfg)
        assert r.value is False

    def test_name_is_stable(self):
        r = f7_shared_rmsd.score(_FakeSite(_struct_ann()), self.cfg)
        assert r.name == "F7_shared_structural_change"
