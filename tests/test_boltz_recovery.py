"""Per-residue pLDDT recovery from the CIF B-factor column.

Boltz-2 v2.2.x writes only the scalar ``complex_plddt`` to ``confidence.json``
(parsed per-residue list = ``None``), yet the real per-residue values live in
the CIF B-factor column. The recovery helper must prefer the CIF in that case
— not only when the JSON list is present-but-uniform — and the cache-repair
routine must lift stale ``uniform_plddt`` entries back to ``ok`` in place.
"""

from __future__ import annotations

import json

from swissisoform.structure.boltz import (
    _parse_plddt_from_cif,
    _recover_per_residue_plddt,
    repair_uniform_plddt_cache,
)
from swissisoform.structure.fold import derive_metrics, write_cache

# Minimal Boltz-shaped CIF: three CA atoms with varied B-factors (0–100 scale).
_CIF_VARIED = """data_test
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.B_iso_or_equiv
ATOM 1 CA ALA 1 0.0 0.0 0.0 55.0
ATOM 2 CA GLY 2 1.0 1.0 1.0 70.0
ATOM 3 CA SER 3 2.0 2.0 2.0 90.0
"""

# Same shape but every CA B-factor identical — CIF can't improve on a planted
# scalar, so recovery must decline and leave the fallback to the caller.
_CIF_FLAT = """data_test
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.B_iso_or_equiv
ATOM 1 CA ALA 1 0.0 0.0 0.0 85.0
ATOM 2 CA GLY 2 1.0 1.0 1.0 85.0
ATOM 3 CA SER 3 2.0 2.0 2.0 85.0
"""


def _write_cif(tmp_path, text):
    p = tmp_path / "model.cif"
    p.write_text(text)
    return p


class TestParsePlddtFromCif:
    def test_varied_b_factors_normalized(self, tmp_path):
        cif = _write_cif(tmp_path, _CIF_VARIED)
        assert _parse_plddt_from_cif(cif) == [0.55, 0.70, 0.90]

    def test_missing_file_returns_none(self, tmp_path):
        assert _parse_plddt_from_cif(tmp_path / "absent.cif") is None


class TestRecoverPerResiduePlddt:
    def test_none_json_recovers_from_cif(self, tmp_path):
        # The regression: JSON has no per-residue list, but the CIF does.
        cif = _write_cif(tmp_path, _CIF_VARIED)
        assert _recover_per_residue_plddt(None, cif) == [0.55, 0.70, 0.90]

    def test_uniform_json_recovers_from_cif(self, tmp_path):
        cif = _write_cif(tmp_path, _CIF_VARIED)
        assert _recover_per_residue_plddt([0.8, 0.8, 0.8], cif) == [0.55, 0.70, 0.90]

    def test_varied_json_passes_through(self, tmp_path):
        cif = _write_cif(tmp_path, _CIF_VARIED)
        good = [0.4, 0.6, 0.9]
        assert _recover_per_residue_plddt(good, cif) is good

    def test_none_json_no_cif_stays_none(self):
        assert _recover_per_residue_plddt(None, None) is None

    def test_none_json_flat_cif_stays_none(self, tmp_path):
        # CIF can't improve a planted scalar — caller's fallback still applies.
        cif = _write_cif(tmp_path, _CIF_FLAT)
        assert _recover_per_residue_plddt(None, cif) is None


class TestRepairUniformPlddtCache:
    def _seed_uniform_entry(self, cache_dir, h, cif_text):
        """A stale entry: uniform_plddt metrics + flat confidence + good CIF."""
        write_cache(
            h,
            cache_dir=cache_dir,
            backend="boltz",
            confidence={"plddt": [0.85, 0.85, 0.85], "ptm": 0.9, "iptm": None},
            metrics={
                "status": "uniform_plddt",
                "backend": "boltz",
                "length": 3,
                "plddt_mean": None,
                "plddt_std": None,
                "ptm": 0.9,
            },
            cif_text=cif_text,
        )

    def test_repairs_stale_entry_in_place(self, tmp_path):
        self._seed_uniform_entry(tmp_path, "hashvaried", _CIF_VARIED)
        repaired = repair_uniform_plddt_cache(cache_dir=tmp_path, backend="boltz")
        assert repaired == ["hashvaried"]

        base = tmp_path / "boltz" / "hashvaried"
        metrics = json.loads((base / "metrics.json").read_text())
        conf = json.loads((base / "confidence.json").read_text())
        assert metrics["status"] == "ok"
        assert metrics["plddt_mean"] == derive_metrics(
            backend="boltz", plddt=[0.55, 0.70, 0.90], ptm=0.9, status="ok"
        )["plddt_mean"]
        assert conf["plddt"] == [0.55, 0.70, 0.90]
        # CIF is left untouched.
        assert (base / "model.cif").read_text() == _CIF_VARIED

    def test_leaves_flat_cif_entry_alone(self, tmp_path):
        self._seed_uniform_entry(tmp_path, "hashflat", _CIF_FLAT)
        repaired = repair_uniform_plddt_cache(cache_dir=tmp_path, backend="boltz")
        assert repaired == []
        metrics = json.loads(
            (tmp_path / "boltz" / "hashflat" / "metrics.json").read_text()
        )
        assert metrics["status"] == "uniform_plddt"

    def test_ignores_healthy_entries(self, tmp_path):
        write_cache(
            "hashok",
            cache_dir=tmp_path,
            backend="boltz",
            confidence={"plddt": [0.4, 0.6, 0.9], "ptm": 0.8, "iptm": None},
            metrics=derive_metrics(
                backend="boltz", plddt=[0.4, 0.6, 0.9], ptm=0.8, status="ok"
            ),
            cif_text=_CIF_VARIED,
        )
        assert repair_uniform_plddt_cache(cache_dir=tmp_path, backend="boltz") == []
