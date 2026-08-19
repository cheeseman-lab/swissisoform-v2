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
    """Asking for the diff region on the side that does not hold it is an error.

    Applying the other space's coordinates would profile arbitrary residues of the
    wrong protein; falling back to the whole protein answered a different question
    under a status="ok" the model cannot tell from a real one. Both directions:
    a truncation's region lives in the canonical, an extension's in the isoform.
    """
    trunc = _raw(diff_space="canonical", diff_start=0, diff_end=20)
    with pytest.raises(ValueError, match="canonical numbering"):
        st.plddt_profile(trunc, side="isoform", cache_dir=cache)

    ext = _raw(diff_space="isoform", diff_start=0, diff_end=20)
    with pytest.raises(ValueError, match="isoform numbering"):
        st.plddt_profile(ext, side="canonical", cache_dir=cache)

    # The message carries the region's own coordinates and both ways out.
    with pytest.raises(ValueError, match=r"residues 1-20.*explicit start/end"):
        st._window(None, None, 130, trunc, "isoform")


def test_wrong_side_default_reaches_the_model_as_an_error(cache):
    """Not a plausible-looking answer: the loop gets {"error": ...} to correct."""
    dispatch = st.make_p_dispatch(
        _raw(diff_space="canonical", diff_start=0, diff_end=20), cache_dir=cache
    )
    for name in ("plddt_profile", "pae_block"):
        result = dispatch(name, {"side": "isoform"})
        assert "error" in result, f"{name} returned {result!r} instead of an error"
        assert "canonical" in result["error"] and "isoform" in result["error"]
        assert result.get("status") != "ok"


def test_explicit_bounds_still_read_the_other_side(cache):
    """The comparison _SIDE_PROP invites stays available — you just say the window."""
    raw = _raw(diff_space="canonical", diff_start=0, diff_end=20)
    out = st.plddt_profile(raw, side="isoform", start=1, end=20, cache_dir=cache)
    assert out["status"] == "ok"
    assert out["region"] == [1, 20]


def test_side_is_always_reported(cache):
    for side in ("isoform", "canonical"):
        out = st.plddt_profile(_raw(), side=side, start=1, end=20, cache_dir=cache)
        assert out["side"] == side


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
    """0-1, matching the aggregate columns and the P1 threshold — not 0-100."""
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


def test_window_past_the_c_terminus_is_an_error_not_an_empty_answer(cache):
    """``end`` was clamped and ``start`` was not, so a window past the protein
    became ``[start, start]`` — answered with ``status="ok"`` and nothing in it,
    or an IndexError.
    """
    dispatch = st.make_p_dispatch(_raw(), cache_dir=cache)  # isoform is 130 residues

    # contacts and secondary_structure need a CIF this fixture does not write, so
    # they stop at no_structure first. _window itself is pinned below.
    for name, kwargs in [
        ("plddt_profile", {"side": "isoform", "start": 200}),
        ("pae_block", {"side": "isoform", "rows": [200, 210]}),
    ]:
        result = dispatch(name, kwargs)
        assert "error" in result, f"{name} returned {result!r} instead of an error"
        assert "200" in result["error"] and "130" in result["error"], result["error"]
        # Never a plausible-looking negative.
        assert result.get("status") != "ok"


def test_window_rejects_a_range_entirely_outside_the_protein():
    """The shared helper behind all four readers."""
    raw = _raw()
    with pytest.raises(ValueError, match="past the end"):
        st._window(200, 210, 130, raw, "isoform")
    with pytest.raises(ValueError, match="before the start"):
        st._window(None, 0, 130, raw, "isoform")
    # Overlapping and default windows are unaffected.
    assert st._window(120, 9999, 130, raw, "isoform") == (120, 130)
    assert st._window(None, None, 130, raw, "isoform") == (1, 30)


def test_inverted_window_is_an_error_not_a_single_residue(cache):
    """`max(s, e)` collapsed end<start to [start, start] and answered status="ok" —
    a one-residue reading of a malformed question, the paper-over the out-of-bounds
    guards already reject.
    """
    with pytest.raises(ValueError, match=r"end=10 is before start=50"):
        st._window(50, 10, 130, _raw(), "isoform")

    dispatch = st.make_p_dispatch(_raw(), cache_dir=cache)
    for name, kwargs in [
        ("plddt_profile", {"side": "isoform", "start": 50, "end": 10}),
        ("pae_block", {"side": "isoform", "rows": [50, 10]}),
    ]:
        result = dispatch(name, kwargs)
        assert "error" in result, f"{name} returned {result!r} instead of an error"
        assert "50" in result["error"] and "10" in result["error"], result["error"]
        assert result.get("status") != "ok"


def test_window_ending_before_the_n_terminus_is_an_error(cache):
    dispatch = st.make_p_dispatch(_raw(), cache_dir=cache)
    result = dispatch("plddt_profile", {"side": "isoform", "end": 0})
    assert "error" in result and "end=0" in result["error"]


def test_a_window_that_merely_overhangs_is_still_clamped(cache):
    """Only a window ENTIRELY outside the protein is an error; partial overlap
    stays clamped, which is what makes `start=1, end=9999` usable as "whole".
    """
    out = st.plddt_profile(_raw(), side="isoform", start=120, end=9999, cache_dir=cache)
    assert out["status"] == "ok"
    assert out["region"] == [120, 130]


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
    assert st.DATA_TOOL_NAMES == {
        "plddt_profile",
        "secondary_structure",
        "pae_block",
        "contacts",
    }


def test_emit_verdict_requires_verdict_reasoning_and_evidence():
    emit = next(t for t in st.P_TOOLS if t["name"] == st.EMIT_VERDICT)
    assert set(emit["input_schema"]["required"]) == {"verdict", "reasoning", "evidence_used"}
    assert emit["input_schema"]["properties"]["verdict"]["enum"] == [
        "interesting", "neutral", "not_interesting",
    ]


def test_emit_verdict_is_strict_and_schema_stays_strict_compatible():
    from tests.test_site_tools import assert_strict_compatible

    emit = next(t for t in st.P_TOOLS if t["name"] == st.EMIT_VERDICT)
    assert emit["strict"] is True
    assert_strict_compatible(emit["input_schema"])


def test_only_emit_verdict_is_strict():
    """Only the persisted verdict is strict.

    pae_block's two-element range params are not strict-expressible (minItems:2),
    and a reader's bad argument is already answered with an error the model can
    recover from.
    """
    strict = {t["name"] for t in st.P_TOOLS if t.get("strict")}
    assert strict == {st.EMIT_VERDICT}


# ── secondary_structure: whole-protein scan, window selects ───────────────
#
# The reader used to scan only INSIDE its window, and sse_elements applies its
# length floors AFTER clipping — so a boundary-crossing element was truncated,
# and one whose remainder fell below the floor vanished. It contradicted the P3
# verdict on 5 of 18 cheeseman_test isoforms.


def _sse_raw(sse_iso: str, **kw) -> dict:
    """A raw mirror with a synthetic SSE string — the reader prefers the stored
    assignment over the CIF, so no coordinates are needed.
    """
    return {**_raw(**kw), "isoform_structure_sse_isoform": sse_iso}


def test_element_crossing_the_window_is_returned_at_full_extent(cache):
    """Clipping reported this 16 aa helix as the 6 aa that fit inside [1, 30]."""
    sse = "c" * 24 + "a" * 16 + "c" * 90  # helix 25-40
    out = st.secondary_structure(_sse_raw(sse), cache_dir=cache)

    assert out["region"] == [1, 30]  # default window is the differential region
    (helix,) = [e for e in out["elements"] if e["type"] == "helix"]
    assert (helix["start"], helix["end"], helix["length"]) == (25, 40, 16)
    assert helix["region"] == "spans"
    assert out["longest_helix"] == 16


def test_element_is_not_lost_to_the_length_floor_after_clipping(cache):
    """The TRIP13 / TRNT1 shape.

    Clipped to 3 aa it fell under MIN_LENGTH and disappeared, so the reader
    reported no structure where P3 scored an element.
    """
    sse = "c" * 27 + "a" * 5 + "c" * 98  # helix 28-32, clipped to 28-30 = 3 aa
    out = st.secondary_structure(_sse_raw(sse), cache_dir=cache)

    (helix,) = [e for e in out["elements"] if e["type"] == "helix"]
    assert (helix["start"], helix["end"], helix["length"]) == (28, 32, 5)
    assert helix["region"] == "spans"


def test_window_selects_rather_than_clips(cache):
    """The window still filters — it just stops truncating what it admits."""
    sse = "c" * 59 + "a" * 10 + "c" * 61  # helix 60-69, wholly outside [1, 30]
    raw = _sse_raw(sse)
    assert st.secondary_structure(raw, cache_dir=cache)["elements"] == []

    wide = st.secondary_structure(raw, start=1, end=130, cache_dir=cache)
    assert [(e["start"], e["end"]) for e in wide["elements"]] == [(60, 69)]
    assert wide["elements"][0]["region"] == "shared"


def test_region_tags_track_the_differential_region_not_the_window(cache):
    """A wide window must not turn every element `unique`.

    region is membership in the DIFFERENTIAL region, so it means the same here as
    in the parquet, on the P3 card and in the scorer.
    """
    sse = (
        "a" * 10 + "c" * 14  # helix 1-10   inside [1, 30]
        + "a" * 10 + "c" * 25  # helix 25-34  crosses 30
        + "a" * 10 + "c" * 61  # helix 60-69  beyond it
    )
    out = st.secondary_structure(_sse_raw(sse), start=1, end=130, cache_dir=cache)

    assert out["region_is_differential"] is True
    assert [(e["start"], e["region"]) for e in out["elements"]] == [
        (1, "unique"),
        (25, "spans"),
        (60, "shared"),
    ]


def test_untaggable_side_returns_elements_without_a_region(cache):
    """Asking for the side that does NOT hold the differential region.

    The bounds are in canonical numbering, so applying them would label the
    retained core as the removed segment — the tag-level twin of the hazard
    test_diff_default_does_not_leak_across_sides pins for coordinates.
    """
    raw = _sse_raw(
        "c" * 24 + "a" * 16 + "c" * 90, diff_space="canonical", diff_start=0, diff_end=20
    )
    # Explicit bounds: the default is an error on this side, and reading the whole
    # isoform is exactly the comparison you have to ask for by name.
    out = st.secondary_structure(raw, side="isoform", start=1, end=130, cache_dir=cache)

    assert out["region_is_differential"] is False
    assert out["region"] == [1, 130]
    assert out["elements"]
    assert all("region" not in e for e in out["elements"])


def test_longest_confident_applies_the_p3_threshold(cache):
    """Without min_plddt, summarise_elements treats every element as confident,
    so the field reported the longest element of ANY confidence.
    """
    from swissisoform.config import ScoringConfig

    # cache fixture: isoform pLDDT is 0.90 over residues 1-30, 0.60 beyond.
    sse = "c" * 4 + "a" * 10 + "c" * 26 + "a" * 20 + "c" * 70  # 5-14 conf; 41-60 not
    out = st.secondary_structure(_sse_raw(sse), start=1, end=130, cache_dir=cache)

    assert out["confident_min_plddt"] == ScoringConfig().p3_min_sse_plddt
    assert out["longest_helix"] == 20  # the low-confidence helix is the longest
    assert out["longest_confident"] == 10  # but only the confident one counts


def test_every_status_path_returns_the_same_keys(cache):
    """A degraded read must not be missing fields a successful one has."""
    ok = st.secondary_structure(_sse_raw("a" * 10 + "c" * 120), cache_dir=cache)
    missing = st.secondary_structure({})
    assert set(missing) == set(ok) - {"protein_length"}
    assert missing["region_is_differential"] is False
    assert missing["longest_confident"] == 0
