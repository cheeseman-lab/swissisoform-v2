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
    base = {"status": "ok", "sse_status": "ok", "sse_all_elements": []}
    base.update(structure)
    return SimpleNamespace(isoform_annotations={"structure": base})


CFG = ScoringConfig()


def test_p3_fires_on_a_long_confident_element():
    r = p3_score(
        _site(sse_all_elements=[
            {
                "type": "helix", "start": 16, "end": 32, "length": 17,
                "region": "unique", "plddt_mean": 0.75,
            }
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
        _site(sse_all_elements=[
            {
                "type": "helix", "start": 16, "end": 32, "length": 17,
                "region": "unique", "plddt_mean": 0.55,
            }
        ]),
        CFG,
    )
    assert r.value is False


def test_p3_requires_length_not_just_confidence():
    r = p3_score(
        _site(sse_all_elements=[
            {
                "type": "strand", "start": 1, "end": 4, "length": 4,
                "region": "unique", "plddt_mean": 0.98,
            }
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
        _site(sse_all_elements=[
            {
                "type": "strand", "start": 2, "end": 9, "length": 8,
                "region": "unique", "plddt_mean": 0.80,
            },
            {
                "type": "helix", "start": 16, "end": 32, "length": 17,
                "region": "unique", "plddt_mean": 0.75,
            },
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


def test_p3_thresholds_have_one_source():
    """Every consumer of "does this SSE element qualify" reads ScoringConfig.

    The headline, the modal counts, the hit ranking and the scorer used to hold
    three separate copies of 6 / 0.70, so retuning moved only the verdict.

    Does NOT cover non-default configs: these constants are read from the
    DEFAULTS at import, so passing a custom config to the scorer still diverges.
    Threading one through slice_criterion and the viewer is a separate change.
    """
    from swissisoform.site import evidence as ev

    assert ev.P3_MIN_SSE_LENGTH == ScoringConfig().p3_min_sse_length
    assert ev.P3_MIN_SSE_PLDDT == ScoringConfig().p3_min_sse_plddt
    # The private aliases the headline and hit ranking use are the same objects.
    assert ev._P3_MIN_LEN is ev.P3_MIN_SSE_LENGTH
    assert ev._P3_MIN_PLDDT is ev.P3_MIN_SSE_PLDDT


def test_website_sse_qualifier_matches_the_scorer():
    """The modal's qualifier and the criterion agree element-for-element.

    ``_sse_qualifies`` is nested in the page builder, so this exercises the
    thresholds it closes over — the drift is in the numbers, not the comparison.
    """
    pytest.importorskip("swissisoform_site")
    from swissisoform_site import data as site_data

    assert site_data._P3_MIN_SSE_LENGTH == ScoringConfig().p3_min_sse_length
    assert site_data._P3_MIN_SSE_PLDDT == ScoringConfig().p3_min_sse_plddt

    cfg = ScoringConfig()
    def _element(length: int, plddt: float) -> dict:
        return {"type": "helix", "start": 1, "end": length, "length": length, "plddt_mean": plddt}

    borderline = [
        _element(cfg.p3_min_sse_length, cfg.p3_min_sse_plddt),  # exactly on both floors
        _element(cfg.p3_min_sse_length - 1, 0.99),  # confident but too short
        _element(30, cfg.p3_min_sse_plddt - 0.01),  # long but just under confidence
    ]
    for element in borderline:
        scored = p3_score(
            SimpleNamespace(
                isoform_annotations={
                    "structure": {
                        "status": "ok",
                        "sse_status": "ok",
                        "sse_all_elements": [{**element, "region": "unique"}],
                    }
                }
            ),
            cfg,
        )
        rendered = (element["length"] >= site_data._P3_MIN_SSE_LENGTH) and (
            element["plddt_mean"] >= site_data._P3_MIN_SSE_PLDDT
        )
        assert bool(scored.value) is rendered, element


def _headline_is_positive(headline: str | None) -> bool:
    """The tile reads as a finding (named element), not a near-miss or absence."""
    if not headline:
        return False
    return "No helix or strand" not in headline


@_real
def test_headline_never_disagrees_with_the_score():
    """The tile must never name an element as a finding the criterion scores
    False — which is exactly what a 2-residue "helix" at pLDDT 0.98 used to do.

    The thresholds now come from one place (see
    ``test_p3_thresholds_have_one_source``), so this guards the logic rather
    than the constants.
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
            for e in list(row["isoform_structure_sse_all_elements"])
            if e["length"] >= MIN_LENGTH["a" if e["type"] == "helix" else "b"]
        ]
        raw = row.to_dict()
        raw["isoform_structure_sse_all_elements"] = els
        sl = slice_criterion(
            {"tis_id": row["tis_id"], "_raw": raw, "scoring": {"criteria": {}}},
            "P3_secondary_structure",
        )
        verdict = p3_score(
            SimpleNamespace(
                isoform_annotations={
                    "structure": {"status": "ok", "sse_status": "ok", "sse_all_elements": els}
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


# ── Region classification (whole-protein scan for the P3 card) ────────────


def _els(*spans):
    return [{"type": "helix", "start": s, "end": e, "length": e - s + 1} for s, e in spans]


def test_classify_tags_unique_shared_and_spanning():
    from swissisoform.structure.sse import classify_elements

    # diff region 1-20; elements: inside, outside, crossing the boundary.
    out = classify_elements(_els((3, 10), (25, 40), (15, 30)), diff_start=1, diff_end=20)
    assert [e["region"] for e in out] == ["unique", "shared", "spans"]


def test_classify_boundary_exactly_flush_is_not_spanning():
    """An element ending exactly on the last diff residue is unique, not spanning."""
    from swissisoform.structure.sse import classify_elements

    out = classify_elements(_els((10, 20), (21, 30)), diff_start=1, diff_end=20)
    assert [e["region"] for e in out] == ["unique", "shared"]


def test_classify_counts_every_element_exactly_once():
    """The reason this is one tagged scan and not two window scans.

    Two separate scans clip a boundary-crossing element at each window edge, so
    it appears twice with two wrong lengths. Here the totals must reconcile and
    the true length must survive.
    """
    from swissisoform.structure.sse import classify_elements

    spans = [(3, 10), (15, 30), (25, 40), (50, 60)]
    out = classify_elements(_els(*spans), diff_start=1, diff_end=20)
    assert len(out) == len(spans)
    crossing = [e for e in out if e["region"] == "spans"]
    assert len(crossing) == 1
    assert crossing[0]["length"] == 16  # 15-30 inclusive, unclipped


def test_classify_leaves_other_fields_alone():
    from swissisoform.structure.sse import classify_elements

    els = [{"type": "strand", "start": 5, "end": 9, "length": 5, "plddt_mean": 0.91}]
    out = classify_elements(els, diff_start=1, diff_end=20)
    assert out[0]["plddt_mean"] == 0.91 and out[0]["type"] == "strand"


@_real
def test_module_emits_only_the_whole_protein_scan():
    """The window-clipped columns are gone; every element is region-tagged.

    The clipped scan measured only the portion of an element inside the
    differential region and, when that remainder fell below the per-type
    emission floor, dropped it entirely — so it is not a second opinion worth
    keeping alongside the true-length scan.
    """
    import pandas as pd
    import pytest

    df = pd.read_parquet(PAIRED)
    dropped = [
        "isoform_structure_sse_diff_elements",
        "isoform_structure_sse_longest_helix_diff",
        "isoform_structure_sse_longest_strand_diff",
        "isoform_structure_sse_max_confident_element_diff",
    ]
    present = [c for c in dropped if c in df.columns]
    if present:
        pytest.skip(f"parquet predates the column removal: {present}")
    checked = 0
    for _, row in df.iterrows():
        if row["isoform_structure_sse_status"] != "ok":
            continue
        els = [
            e
            for e in list(row["isoform_structure_sse_all_elements"])
            if isinstance(e, dict)
        ]
        assert els, "an ok sse_status with no elements at all"
        assert all(e.get("region") in {"unique", "shared", "spans"} for e in els)
        checked += 1
    assert checked >= 15, f"only {checked} isoforms exercised"


# ── P3 tile headline: a count, not a named element ────────────────────────


def _headline_for(*elements, status="ok"):
    from swissisoform.site.evidence import slice_criterion

    raw = {
        "isoform_structure_sse_status": status,
        "isoform_structure_sse_all_elements": list(elements),
    }
    return slice_criterion({"_raw": raw, "scoring": {"criteria": {}}}, "P3_secondary_structure")


def _e(kind, length, plddt=0.9, region="unique"):
    return {
        "type": kind, "start": 10, "end": 9 + length, "length": length,
        "plddt_mean": plddt, "region": region,
    }


def test_headline_is_a_count_of_confident_elements():
    assert (
        _headline_for(_e("helix", 17), _e("helix", 9))["headline"]
        == "2 secondary structures identified in unique region"
    )
    assert (
        _headline_for(_e("strand", 8))["headline"]
        == "1 secondary structure identified in unique region"
    )


def test_headline_does_not_name_element_types():
    """Deliberately type-agnostic: the modal's element listing names each one."""
    h = _headline_for(_e("helix", 15), _e("strand", 7))["headline"]
    assert h == "2 secondary structures identified in unique region"
    assert "helix" not in h and "strand" not in h


def test_headline_counts_only_elements_clearing_both_thresholds():
    """The count must match what P3 scores on, or the tile and the verdict
    disagree — a sub-threshold element is not a finding.
    """
    h = _headline_for(_e("helix", 17), _e("helix", 4), _e("helix", 12, plddt=0.5))["headline"]
    assert h == "1 secondary structure identified in unique region"


def test_headline_reads_as_empty_when_nothing_qualifies():
    """Sub-threshold elements count as zero, so the tile reads the same as
    genuinely-empty. The near-miss detail is in the modal's element listing.
    """
    assert _headline_for(_e("strand", 5))["headline"] == "No helix or strand in diff region"


def test_headline_distinguishes_empty_from_not_run():
    assert _headline_for()["headline"] == "No helix or strand in diff region"
    assert _headline_for(status="no_cache")["headline"] is None


def test_headline_segments_bold_only_the_count():
    segs = _headline_for(_e("helix", 17), _e("helix", 9))["headline_segments"]
    assert segs[0] == {"t": "2", "strong": True}
    assert segs[1]["strong"] is False
    assert segs[1]["t"].startswith(" secondary structures")


# ── The scored elements must survive shared-core filler ───────────────────


def test_p3_scored_elements_survive_shared_filler():
    """A >30-element protein must not lose the elements P3 scored on.

    ~1 SSE element per 18 residues, so a flat 30-element cap bites above ~545 aa
    (39% of full_catalog isoforms) and the survivors would be the first 30 by
    position. Splitting scored from shared removes the competition entirely.
    """
    from swissisoform.site.evidence import slice_criterion

    els = []
    # One qualifying unique element, buried past any cap by C-terminal filler.
    els.append({"type": "helix", "start": 500, "end": 520, "length": 21,
                "plddt_mean": 0.95, "region": "unique"})
    # A sub-threshold one in the same region.
    els.append({"type": "strand", "start": 530, "end": 532, "length": 3,
                "plddt_mean": 0.95, "region": "unique"})
    for i in range(60):  # shared-core filler, all qualifying
        s = 1 + i * 8
        els.append({"type": "strand", "start": s, "end": s + 6, "length": 7,
                    "plddt_mean": 0.9, "region": "shared"})
    raw = {"isoform_structure_sse_status": "ok", "orf_type": "extended",
           "diff_space": "isoform", "diff_start": 499, "diff_end": 540,
           "isoform_structure_sse_all_elements": els}
    sl = slice_criterion({"_raw": raw, "scoring": {"criteria": {}}}, "P3_secondary_structure")

    kept = sl["elements_gained"]
    assert [(k["start"], k["qualifies"]) for k in kept] == [(500, True), (530, False)]
    assert "hits" not in sl, "P3 ships the reframed lists, not the flat dump"

    # All 60 shared elements are counted, not just those that fit a cap.
    assert sl["shared_elements_summary"] == {"n_helix": 0, "n_strand": 60, "longest": 7}
    assert sl["n_hits_total"] == 62


def test_p3_shared_only_protein_reports_an_empty_scored_list():
    """CBX1's shape: nothing in the differential region. The count still lands."""
    from swissisoform.site.evidence import slice_criterion

    els = [{"type": "helix", "start": 10, "end": 30, "length": 21,
            "plddt_mean": 0.9, "region": "shared"} for _ in range(5)]
    raw = {"isoform_structure_sse_status": "ok", "orf_type": "truncated",
           "diff_space": "canonical", "diff_start": 0, "diff_end": 9,
           "isoform_structure_sse_all_elements": els}
    sl = slice_criterion({"_raw": raw, "scoring": {"criteria": {}}}, "P3_secondary_structure")
    assert sl["elements_lost"] == []
    assert sl["shared_elements_summary"]["n_helix"] == 5
    assert sl["n_hits_total"] == 5 and sl["n_hits_shown"] == 0


def test_p3_frames_each_orf_family():
    """lost / gained / separate-ORF. The third has no instance in cheeseman_test,
    so real data never exercises it.
    """
    from swissisoform.site.evidence import slice_criterion

    el = {"type": "helix", "start": 3, "end": 20, "length": 18,
          "plddt_mean": 0.9, "region": "unique"}

    def _slice(orf, space):
        raw = {"isoform_structure_sse_status": "ok", "orf_type": orf, "diff_space": space,
               "diff_start": 0, "diff_end": 30, "isoform_structure_sse_all_elements": [el]}
        return slice_criterion({"_raw": raw, "scoring": {"criteria": {}}},
                               "P3_secondary_structure")

    trunc = _slice("truncated", "canonical")
    assert trunc["residue_space"] == "canonical"
    assert "canonical residues 1-30" in trunc["frame"] and "LOST" in trunc["frame"]
    assert trunc["elements_lost"][0]["status"] == "lost"

    ext = _slice("extended", "isoform")
    assert ext["residue_space"] == "isoform"
    assert "isoform residues 1-30" in ext["frame"] and "GAINED" in ext["frame"]
    assert ext["elements_gained"][0]["status"] == "gained"

    for orf in ("uorf", "uoorf", "internal_oof", "3utr_orf", "alt_orf"):
        sep = _slice(orf, "isoform")
        assert "no shared region" in sep["frame"], orf
        assert sep["elements_gained"][0]["status"] == "gained", orf


def test_p3_keeps_region_alongside_status():
    """`spans` must stay visible — it is the vocabulary the card and scorer use,
    and it means the element continues into retained sequence.
    """
    from swissisoform.site.evidence import slice_criterion

    els = [{"type": "strand", "start": 28, "end": 33, "length": 6,
            "plddt_mean": 0.85, "region": "spans"}]
    raw = {"isoform_structure_sse_status": "ok", "orf_type": "truncated",
           "diff_space": "canonical", "diff_start": 0, "diff_end": 29,
           "isoform_structure_sse_all_elements": els}
    sl = slice_criterion({"_raw": raw, "scoring": {"criteria": {}}}, "P3_secondary_structure")
    (e,) = sl["elements_lost"]
    assert e["region"] == "spans" and e["status"] == "lost"


def test_p3_qualifies_uses_the_config_thresholds():
    from swissisoform.config import ScoringConfig
    from swissisoform.site.evidence import slice_criterion

    cfg = ScoringConfig()
    els = [
        {"type": "helix", "start": 1, "end": cfg.p3_min_sse_length, "region": "unique",
         "length": cfg.p3_min_sse_length, "plddt_mean": cfg.p3_min_sse_plddt},
        {"type": "helix", "start": 40, "end": 60, "length": 21, "region": "unique",
         "plddt_mean": cfg.p3_min_sse_plddt - 0.01},
    ]
    raw = {"isoform_structure_sse_status": "ok", "orf_type": "extended",
           "diff_space": "isoform", "diff_start": 0, "diff_end": 60,
           "isoform_structure_sse_all_elements": els}
    sl = slice_criterion({"_raw": raw, "scoring": {"criteria": {}}}, "P3_secondary_structure")
    assert [e["qualifies"] for e in sl["elements_gained"]] == [True, False]
    assert str(cfg.p3_min_sse_length) in sl["qualifies_when"]
