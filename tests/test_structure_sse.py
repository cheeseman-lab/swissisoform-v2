"""Tests for secondary-structure assignment and the P3 criterion.

The pure-logic tests use synthetic SSE strings. The UBE2M cases at the bottom run
against the real cached fold and pin the failure that motivated this work.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from swissisoform.config import ScoringConfig
from swissisoform.evidence.p3_secondary_structure import score as p3_score
from swissisoform.structure.sse import (
    MIN_LENGTH,
    annotate_sse,
    sse_elements,
    summarise_elements,
)

ROOT = Path(__file__).resolve().parent.parent


# ── sse_elements ──────────────────────────────────────────────────────────


def test_collapses_runs_into_elements():
    els = sse_elements("ccaaaaacbbbcc")
    assert [(e["type"], e["start"], e["end"], e["length"]) for e in els] == [
        ("helix", 3, 7, 5),
        ("strand", 9, 11, 3),
    ]


def test_coil_is_never_an_element():
    assert sse_elements("cccccc") == []


def test_bounds_are_one_based_inclusive():
    """So an element can be handed straight to plddt_profile / contacts.

    min_length=0 isolates the collapsing arithmetic from the biophysical floor.
    """
    (el,) = sse_elements("caaac", min_length=0)
    assert (el["start"], el["end"]) == (2, 4)
    assert el["length"] == 3


def test_element_carries_its_own_plddt():
    plddt = [0.1, 0.9, 0.9, 0.9, 0.1]
    (el,) = sse_elements("caaac", plddt, min_length=0)
    assert el["plddt_mean"] == pytest.approx(0.9)


def test_window_restricts_the_scan():
    els = sse_elements("aaaaaccccbbbbb", start=10, end=14)
    assert [e["type"] for e in els] == ["strand"]


def test_min_length_filters():
    assert sse_elements("caaac", min_length=4) == []
    assert len(sse_elements("caaac", min_length=3)) == 1


def test_helix_and_strand_have_different_floors():
    """A 2-residue "helix" is a labelling blip, not a helix: an alpha-helix is
    3.6 residues per turn. Strands are legitimately shorter, so a single shared
    cutoff would discard real ones — on cheeseman_test, 7 of 19.
    """
    assert MIN_LENGTH["a"] == 5 and MIN_LENGTH["b"] == 3
    assert sse_elements("caac") == []            # 2-res helix — dropped
    assert sse_elements("caaaac") == []          # 4-res helix — one turn, still dropped
    assert len(sse_elements("caaaaac")) == 1     # 5-res helix — kept
    assert sse_elements("cbbc") == []            # 2-res strand — dropped
    assert len(sse_elements("cbbbc")) == 1       # 3-res strand — kept


def test_floor_can_be_overridden():
    assert len(sse_elements("caac", min_length=0)) == 1
    assert len(sse_elements("caac", min_length={"a": 2, "b": 2})) == 1


def test_trailing_run_is_flushed():
    """A run ending at the last residue must not be dropped."""
    (el,) = sse_elements("ccaaa", min_length=0)
    assert (el["start"], el["end"]) == (3, 5)


def test_empty_and_none_are_safe():
    assert sse_elements(None) == []
    assert sse_elements("") == []
    assert sse_elements("aaa", start=10, end=5) == []


def test_summarise_separates_longest_from_longest_confident():
    """A caller must see both 'there is a long helix' and 'but it is not confident'."""
    els = [
        {"type": "helix", "start": 1, "end": 20, "length": 20, "plddt_mean": 0.45},
        {"type": "strand", "start": 30, "end": 36, "length": 7, "plddt_mean": 0.92},
    ]
    s = summarise_elements(els, min_plddt=0.70)
    assert s["longest_helix"] == 20
    assert s["longest_confident"] == 7  # the 20-mer fails the pLDDT floor


def test_annotate_sse_missing_file_returns_none():
    assert annotate_sse(None) is None
    assert annotate_sse("/nonexistent/model.cif") is None


# ── P3 scorer ─────────────────────────────────────────────────────────────


def _site(**structure):
    base = {"status": "ok", "sse_status": "ok", "sse_diff_elements": []}
    base.update(structure)
    return SimpleNamespace(isoform_annotations={"structure": base})


CFG = ScoringConfig()


def test_p3_fires_on_a_long_confident_element():
    r = p3_score(
        _site(sse_diff_elements=[
            {"type": "helix", "start": 16, "end": 32, "length": 17, "plddt_mean": 0.75}
        ]),
        CFG,
    )
    assert r.value is True
    assert "17 aa helix at 16-32" in r.reason


def test_p3_requires_confidence_not_just_length():
    """Derived from predicted coordinates — a clean helix through a disordered
    stretch is geometry fitted to a guess.
    """
    r = p3_score(
        _site(sse_diff_elements=[
            {"type": "helix", "start": 16, "end": 32, "length": 17, "plddt_mean": 0.55}
        ]),
        CFG,
    )
    assert r.value is False


def test_p3_requires_length_not_just_confidence():
    r = p3_score(
        _site(sse_diff_elements=[
            {"type": "strand", "start": 1, "end": 4, "length": 4, "plddt_mean": 0.98}
        ]),
        CFG,
    )
    assert r.value is False


def test_p3_false_when_region_is_all_coil():
    r = p3_score(_site(), CFG)
    assert r.value is False
    assert "no helix or strand" in r.reason


def test_p3_not_evaluable_when_fold_is_unusable():
    for status in ("no_cache", "too_long", "failed", "uniform_plddt"):
        assert p3_score(_site(status=status), CFG).value is None
    assert p3_score(_site(sse_status="no_structure"), CFG).value is None
    assert p3_score(SimpleNamespace(isoform_annotations={}), CFG).value is None


def test_p3_picks_the_longest_qualifying_element():
    r = p3_score(
        _site(sse_diff_elements=[
            {"type": "strand", "start": 2, "end": 9, "length": 8, "plddt_mean": 0.80},
            {"type": "helix", "start": 16, "end": 32, "length": 17, "plddt_mean": 0.75},
        ]),
        CFG,
    )
    assert "17 aa helix" in r.reason


def test_p3_is_registered_in_the_p_category():
    from swissisoform.evidence import CATEGORY_CRITERIA

    names = [f.__module__.split(".")[-1] for f in CATEGORY_CRITERIA["P"]]
    assert names == ["p1_structure", "p2_shared_rmsd", "p3_secondary_structure"]


# ── UBE2M regression — the case that motivated this ───────────────────────

CACHE = ROOT / "data" / "cache" / "structure" / "esmfold2"
PAIRED = ROOT / "data" / "output" / "cheeseman_test" / "all_paired.parquet"
_real = pytest.mark.skipif(
    not (CACHE.exists() and PAIRED.exists()), reason="fold cache / cheeseman_test absent"
)


@pytest.fixture
def ube2m():
    import pandas as pd

    df = pd.read_parquet(PAIRED)
    return df[df.gene_name == "UBE2M"].iloc[0].to_dict()


@_real
def test_ube2m_extension_contains_the_helix(ube2m):
    """The model picked residues 37-59 by pLDDT and missed the helix at 16-32.

    pLDDT is not a proxy for secondary structure: 37-59 is the highest-confidence
    stretch (0.91) and it is coil, while the real helix scores 0.75.
    """
    from swissisoform.site import structure_tools as st

    out = st.secondary_structure(ube2m, start=1, end=65)
    assert out["status"] == "ok"
    helices = [e for e in out["elements"] if e["type"] == "helix"]
    assert len(helices) == 1
    helix = helices[0]
    assert helix["start"] == 16 and helix["end"] == 32 and helix["length"] == 17
    assert 0.70 <= helix["plddt_mean"] < 0.80


@_real
def test_ube2m_helix_contacts_the_canonical_core(ube2m):
    """Querying the helix — rather than the pLDDT-confident window — finds the
    loop-back onto the core that the original run reported as absent.
    """
    from swissisoform.site import structure_tools as st

    out = st.contacts(ube2m, start=16, end=32)
    assert out["n_contacts"] == 20
    assert out["n_residues_in_contact"] == 10
    core = {p["residue"] for p in out["contact_partners"] if p["residue"] > 65}
    assert {178, 183, 192, 195, 196} <= core


@_real
def test_ube2m_pldd_confident_window_is_coil_and_has_no_core_contacts(ube2m):
    """Pins the trap itself: the window the model chose really is structureless
    and really does touch nothing, so the guard has to come from elsewhere.
    """
    from swissisoform.site import structure_tools as st

    sse = st.secondary_structure(ube2m, start=37, end=59)
    assert [e for e in sse["elements"] if e["type"] == "helix"] == []
    contacts = st.contacts(ube2m, start=37, end=59)
    assert {p["residue"] for p in contacts["contact_partners"] if p["residue"] > 65} == set()


# ── Tile headline must agree with the P3 verdict ──────────────────────────


def _headline_is_positive(headline: str | None) -> bool:
    """The tile reads as a finding (named element), not a near-miss or absence."""
    if not headline:
        return False
    return "below threshold" not in headline and "No helix or strand" not in headline


@_real
def test_headline_never_disagrees_with_the_score():
    """Guards a real duplication: site/evidence.py restates P3's thresholds
    because the tile renders without a ScoringConfig. If the two drift, a tile
    can name an element as a finding while the criterion scores False — which
    is exactly what a 2-residue "helix" at pLDDT 0.98 used to do.
    """
    import pandas as pd

    from swissisoform.site.evidence import slice_criterion

    df = pd.read_parquet(PAIRED)
    checked = 0
    for i in range(len(df)):
        row = df.iloc[i]
        if row["isoform_structure_sse_status"] != "ok":
            continue
        els = [
            dict(e)
            for e in list(row["isoform_structure_sse_diff_elements"])
            if e["length"] >= MIN_LENGTH["a" if e["type"] == "helix" else "b"]
        ]
        raw = row.to_dict()
        raw["isoform_structure_sse_diff_elements"] = els
        sl = slice_criterion(
            {"tis_id": row["tis_id"], "_raw": raw, "scoring": {"criteria": {}}},
            "P3_secondary_structure",
        )
        verdict = p3_score(
            SimpleNamespace(
                isoform_annotations={
                    "structure": {"status": "ok", "sse_status": "ok", "sse_diff_elements": els}
                }
            ),
            CFG,
        )
        assert _headline_is_positive(sl["headline"]) is bool(verdict.value), (
            f"{row['gene_name']} {row['orf_type']}: headline {sl['headline']!r} "
            f"disagrees with P3 value {verdict.value}"
        )
        checked += 1
    assert checked >= 15, f"only {checked} isoforms exercised"


@_real
def test_slice_criterion_survives_a_raw_numpy_row():
    """`raw.get(...) or []` raises on a numpy object array; the hits branch
    avoids that with an explicit None check and the headline must too.
    """
    import pandas as pd

    from swissisoform.site.evidence import slice_criterion

    df = pd.read_parquet(PAIRED)
    row = df[df["isoform_structure_sse_status"] == "ok"].iloc[0]
    sl = slice_criterion(
        {"tis_id": row["tis_id"], "_raw": row.to_dict(), "scoring": {"criteria": {}}},
        "P3_secondary_structure",
    )
    assert sl["headline"]
