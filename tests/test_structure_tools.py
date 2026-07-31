"""Tests for the P-category LLM reader tools over the fold cache.

No network, no GPU: the fold cache is seeded on disk from synthetic arrays, so
these exercise resolution, the coordinate-space selection, the aggregation rules
and the degradation paths. The tool loop itself is tested in
``test_run_llm_interpretation.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from swissisoform.site import structure_tools as st
from swissisoform.structure.fold import protein_hash, write_cache

CAN_SEQ = "M" + "A" * 99  # 100 aa canonical
ISO_SEQ = "M" + "C" * 129  # 130 aa isoform


def _raw(
    *,
    diff_space: str = "isoform",
    diff_start: int = 0,
    diff_end: int = 30,
    backend: str = "esmfold2",
    canonical_hash: str | None = None,
    isoform_hash: str | None = None,
) -> dict:
    """An evidence-record ``_raw`` mirror carrying just what the readers read."""
    return {
        "isoform_structure_canonical_hash": (
            protein_hash(CAN_SEQ) if canonical_hash is None else canonical_hash
        ),
        "isoform_structure_isoform_hash": (
            protein_hash(ISO_SEQ) if isoform_hash is None else isoform_hash
        ),
        "isoform_structure_backend": backend,
        "diff_space": diff_space,
        "diff_start": diff_start,
        "diff_end": diff_end,
    }


def _seed(cache_dir: Path, seq: str, plddt: list[float], *, pae=None, status="ok") -> str:
    """Write one fold-cache entry and return its hash."""
    h = protein_hash(seq)
    write_cache(
        h,
        cache_dir=cache_dir,
        backend="esmfold2",
        confidence={"plddt": plddt, "ptm": 0.5, "iptm": None},
        metrics={"status": status, "backend": "esmfold2", "length": len(plddt),
                 "plddt_mean": sum(plddt) / len(plddt) if plddt else None,
                 "plddt_std": 0.0, "ptm": 0.5},
    )
    if pae is not None:
        np.save(cache_dir / "esmfold2" / h / "pae.npy", np.asarray(pae, dtype="float16"))
    return h


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    """A seeded cache: isoform 130 aa (first 30 confident), canonical 100 aa flat."""
    st._ca_coords.cache_clear()
    iso_plddt = [0.90] * 30 + [0.60] * 100
    can_plddt = [0.70] * 100
    # PAE: cheap deterministic pattern — within the first 30 residues it is low,
    # across the diff/body boundary it is high. Asymmetric on purpose.
    n = 130
    pae = np.full((n, n), 20.0)
    pae[:30, :30] = 2.0
    pae[:30, 30:] = 25.0
    pae[30:, :30] = 15.0
    _seed(tmp_path, ISO_SEQ, iso_plddt, pae=pae)
    _seed(tmp_path, CAN_SEQ, can_plddt, pae=np.full((100, 100), 5.0))
    return tmp_path


# ── require_structure_hashes ──────────────────────────────────────────────


def test_require_structure_hashes_names_the_producing_module():
    with pytest.raises(st.StructureHashesMissing) as e:
        st.require_structure_hashes({})
    msg = str(e.value)
    assert "StructureModule" in msg
    assert "--no-tools" in msg


def test_require_structure_hashes_accepts_a_current_record():
    st.require_structure_hashes(_raw())  # does not raise


# ── Coordinate space: which structure carries the differential region ─────


def test_extension_diff_region_defaults_to_the_isoform(cache):
    """diff_space="isoform" → the added N-terminal segment lives in the isoform."""
    out = st.plddt_profile(_raw(diff_space="isoform"), side="isoform", cache_dir=cache)
    assert out["region"] == [1, 30]
    assert out["mean"] == pytest.approx(0.90)
    assert out["protein_length"] == 130


def test_truncation_diff_region_defaults_to_the_canonical(cache):
    """diff_space="canonical" → the removed segment exists only in the canonical."""
    raw = _raw(diff_space="canonical", diff_start=0, diff_end=20)
    out = st.plddt_profile(raw, side="canonical", cache_dir=cache)
    assert out["region"] == [1, 20]
    assert out["protein_length"] == 100


def test_diff_default_does_not_leak_across_sides(cache):
    """Asking for the side that does NOT hold the diff region profiles it whole.

    Silently applying the other space's coordinates would profile arbitrary
    residues of the wrong protein and report them as the differential region.
    """
    raw = _raw(diff_space="canonical", diff_start=0, diff_end=20)
    out = st.plddt_profile(raw, side="isoform", cache_dir=cache)
    assert out["region"] == [1, 130]  # whole isoform, not [1, 20]


def test_side_is_always_reported(cache):
    for side in ("isoform", "canonical"):
        assert st.plddt_profile(_raw(), side=side, cache_dir=cache)["side"] == side


def test_default_side_follows_diff_space(cache):
    """Regression: defaulting to "isoform" profiled the whole isoform on every
    truncation, silently answering a different question than the one asked.
    """
    ext = st.plddt_profile(_raw(diff_space="isoform"), cache_dir=cache)
    assert ext["side"] == "isoform"
    assert ext["region"] == [1, 30]

    trunc = st.plddt_profile(
        _raw(diff_space="canonical", diff_start=0, diff_end=20), cache_dir=cache
    )
    assert trunc["side"] == "canonical"
    assert trunc["region"] == [1, 20]  # NOT [1, 130] — the whole isoform
    assert trunc["protein_length"] == 100


def test_default_side_applies_to_every_reader(cache):
    raw = _raw(diff_space="canonical", diff_start=0, diff_end=20)
    assert st.plddt_profile(raw, cache_dir=cache)["side"] == "canonical"
    assert st.pae_block(raw, cache_dir=cache)["side"] == "canonical"
    assert st.contacts(raw, cache_dir=cache)["side"] == "canonical"


def test_default_side_falls_back_for_a_separate_orf(cache):
    """uORF/altORF: the whole isoform IS the differential region."""
    out = st.plddt_profile(_raw(diff_space="isoform_whole"), cache_dir=cache)
    assert out["side"] == "isoform"
    assert out["region"] == [1, 130]


def test_unknown_side_raises(cache):
    with pytest.raises(ValueError, match="side"):
        st.plddt_profile(_raw(), side="both", cache_dir=cache)


# ── plddt_profile ─────────────────────────────────────────────────────────


def test_plddt_profile_reports_the_scale(cache):
    """0-1, matching the aggregate columns and the F1 threshold — not 0-100."""
    out = st.plddt_profile(_raw(), side="isoform", cache_dir=cache)
    assert out["scale"] == "0-1"
    assert 0.0 <= out["max"] <= 1.0


def test_plddt_profile_exposes_shape_a_mean_would_hide(cache):
    """The whole point: a mediocre mean can be a confident block plus a floppy one."""
    out = st.plddt_profile(_raw(), side="isoform", start=1, end=130, cache_dir=cache)
    assert out["mean"] == pytest.approx((0.90 * 30 + 0.60 * 100) / 130, abs=1e-3)
    assert out["min"] == pytest.approx(0.60)
    assert out["max"] == pytest.approx(0.90)
    assert out["per_residue"][:3] == [0.9, 0.9, 0.9]
    assert out["per_residue"][-1] == 0.6


def test_plddt_profile_window_is_clamped(cache):
    out = st.plddt_profile(_raw(), side="isoform", start=-5, end=9999, cache_dir=cache)
    assert out["region"] == [1, 130]
    assert out["n_residues"] == 130


def test_plddt_profile_truncates_a_long_array_but_keeps_stats_exact(tmp_path):
    st._ca_coords.cache_clear()
    n = st.MAX_PROFILE_RESIDUES + 200
    _seed(tmp_path, ISO_SEQ, [0.5] * n)
    out = st.plddt_profile(_raw(), side="isoform", start=1, end=n, cache_dir=tmp_path)
    assert out["n_residues"] == n
    assert len(out["per_residue"]) == st.MAX_PROFILE_RESIDUES
    assert out["per_residue_truncated"] is True
    assert out["mean"] == pytest.approx(0.5)


# ── pae_block ─────────────────────────────────────────────────────────────


def test_pae_block_returns_aggregates_never_the_matrix(cache):
    """O(L^2) must collapse to O(1) — a raw block would dominate the conversation."""
    out = st.pae_block(_raw(), side="isoform", rows=[1, 30], cols=[31, 130], cache_dir=cache)
    assert set(out) >= {"mean", "min", "max", "n_pairs", "rows", "cols"}
    assert not any(isinstance(v, list) and len(v) > 2 for v in out.values())
    assert len(json.dumps(out)) < 250


def test_pae_block_averages_both_off_diagonal_rectangles(cache):
    """PAE is asymmetric: PAE[i,j] != PAE[j,i], so an inter-block mean uses both."""
    out = st.pae_block(_raw(), side="isoform", rows=[1, 30], cols=[31, 130], cache_dir=cache)
    # 25.0 one way, 15.0 the other, equal counts → 20.0.
    assert out["mean"] == pytest.approx(20.0)
    assert out["n_pairs"] == 2 * 30 * 100


def test_pae_block_diagonal_block_is_not_double_counted(cache):
    out = st.pae_block(_raw(), side="isoform", rows=[1, 30], cols=[1, 30], cache_dir=cache)
    assert out["mean"] == pytest.approx(2.0)
    assert out["n_pairs"] == 30 * 30


def test_pae_block_defaults_to_the_diff_region(cache):
    out = st.pae_block(_raw(diff_space="isoform"), side="isoform", cache_dir=cache)
    assert out["rows"] == [1, 30] and out["cols"] == [1, 30]


def test_pae_block_reports_units(cache):
    assert st.pae_block(_raw(), side="isoform", cache_dir=cache)["units"] == "angstrom"


def test_pae_block_missing_matrix_degrades(tmp_path):
    st._ca_coords.cache_clear()
    _seed(tmp_path, ISO_SEQ, [0.8] * 130)  # no pae.npy
    out = st.pae_block(_raw(), side="isoform", cache_dir=tmp_path)
    assert out["status"] == "no_pae"
    assert out["mean"] is None


# ── contacts ──────────────────────────────────────────────────────────────


def test_contacts_missing_structure_degrades(tmp_path):
    """No model.cif in the entry → absence, not a crash."""
    st._ca_coords.cache_clear()
    _seed(tmp_path, ISO_SEQ, [0.8] * 130)
    out = st.contacts(_raw(), side="isoform", cache_dir=tmp_path)
    assert out["status"] == "no_structure"
    assert out["n_contacts"] == 0


def test_contacts_reports_the_cutoff_it_used(tmp_path):
    st._ca_coords.cache_clear()
    _seed(tmp_path, ISO_SEQ, [0.8] * 130)
    out = st.contacts(_raw(), side="isoform", cutoff=6.0, cache_dir=tmp_path)
    assert out["cutoff_angstrom"] == 6.0


# ── Degradation ───────────────────────────────────────────────────────────


def test_too_long_entry_reports_not_folded(tmp_path):
    """A >1024 aa protein has a cache dir and metrics but was never folded."""
    st._ca_coords.cache_clear()
    h = protein_hash(ISO_SEQ)
    write_cache(h, cache_dir=tmp_path, backend="esmfold2",
                metrics={"status": "too_long", "backend": "esmfold2", "length": None})
    out = st.plddt_profile(_raw(), side="isoform", cache_dir=tmp_path)
    assert out["status"] == "not_folded"
    assert out["per_residue"] == []


def test_absent_entry_reports_no_cache(tmp_path):
    st._ca_coords.cache_clear()
    out = st.plddt_profile(_raw(), side="isoform", cache_dir=tmp_path)
    assert out["status"] == "no_cache"


def test_absent_hash_reports_no_hash(cache):
    raw = _raw(isoform_hash="")
    assert st.plddt_profile(raw, side="isoform", cache_dir=cache)["status"] == "no_hash"


def test_missing_diff_columns_fall_back_to_the_whole_protein(cache):
    raw = _raw()
    del raw["diff_start"], raw["diff_end"]
    out = st.plddt_profile(raw, side="isoform", cache_dir=cache)
    assert out["region"] == [1, 130]


# ── Dispatch ──────────────────────────────────────────────────────────────


def test_dispatch_routes_each_reader(cache):
    d = st.make_p_dispatch(_raw(), cache_dir=cache)
    assert d("plddt_profile", {})["status"] == "ok"
    assert d("pae_block", {})["status"] == "ok"
    assert d("contacts", {})["status"] in {"ok", "no_structure"}


def test_dispatch_uses_the_backend_from_the_record(cache):
    d = st.make_p_dispatch(_raw(backend="boltz"), cache_dir=cache)
    # The seeded entries are under esmfold2, so a boltz lookup finds nothing.
    assert d("plddt_profile", {})["status"] == "no_cache"


def test_dispatch_ignores_model_supplied_cache_dir_and_backend(cache):
    """Those describe the run, not a choice the model gets to make."""
    d = st.make_p_dispatch(_raw(), cache_dir=cache)
    out = d("plddt_profile", {"cache_dir": "/nonexistent", "backend": "boltz"})
    assert out["status"] == "ok"


def test_dispatch_unknown_tool_returns_error_not_exception(cache):
    d = st.make_p_dispatch(_raw(), cache_dir=cache)
    out = d("delete_everything", {})
    assert "error" in out and "Unknown tool" in out["error"]


def test_dispatch_bad_argument_returns_error_so_model_can_retry(cache):
    d = st.make_p_dispatch(_raw(), cache_dir=cache)
    assert "error" in d("plddt_profile", {"side": "sideways"})
    assert "error" in d("plddt_profile", {"nonexistent_kwarg": 1})


def test_dispatch_results_are_json_serialisable(cache):
    d = st.make_p_dispatch(_raw(), cache_dir=cache)
    for name in ("plddt_profile", "pae_block", "contacts"):
        json.dumps(d(name, {}))  # raises TypeError if a numpy scalar leaked


# ── Tool schemas ──────────────────────────────────────────────────────────


def test_every_tool_schema_is_well_formed():
    for tool in st.P_TOOLS:
        assert tool["name"] and tool["description"]
        assert tool["input_schema"]["type"] == "object"
        assert "properties" in tool["input_schema"]


def test_emit_verdict_is_terminal_and_excluded_from_data_tools():
    names = {t["name"] for t in st.P_TOOLS}
    assert st.EMIT_VERDICT in names
    assert st.DATA_TOOL_NAMES == names - {st.EMIT_VERDICT}
    assert st.DATA_TOOL_NAMES == {"plddt_profile", "pae_block", "contacts"}


def test_emit_verdict_requires_verdict_and_reasoning():
    emit = next(t for t in st.P_TOOLS if t["name"] == st.EMIT_VERDICT)
    assert set(emit["input_schema"]["required"]) == {"verdict", "reasoning"}
    assert emit["input_schema"]["properties"]["verdict"]["enum"] == [
        "interesting", "neutral", "not_interesting",
    ]
