"""Tests for CRITERIA_FOR_PAGE + category_verdicts_for_isoform in data.py."""

import json
from pathlib import Path

import pytest

pytest.importorskip("swissisoform_site")  # optional website package; skip if not installed

from swissisoform_site.data import CARD_GROUPS, CRITERIA_FOR_PAGE, category_verdicts_for_isoform


def test_criteria_for_page_has_16_entries() -> None:
    # 13 flat criteria + S2 biophysics + S3 SAE + P3 secondary structure.
    assert len(CRITERIA_FOR_PAGE) == 16
    e_count = sum(1 for c in CRITERIA_FOR_PAGE if c["axis"] == "E")
    f_count = sum(1 for c in CRITERIA_FOR_PAGE if c["axis"] == "F")
    assert e_count == 6
    assert f_count == 10


def test_criteria_for_page_ids_match_ber() -> None:
    """Page config must align with CRITERIA in build_evidence_records."""
    from swissisoform.site.evidence import CRITERIA

    assert {c["id"] for c in CRITERIA_FOR_PAGE} == set(CRITERIA)


def test_card_groups_derived_from_backend_categories() -> None:
    """CARD_GROUPS mirrors the backend CATEGORIES (single source of truth)."""
    from swissisoform.site.evidence import CATEGORIES

    assert [g["letter"] for g in CARD_GROUPS] == [c["letter"] for c in CATEGORIES]
    assert [g["members"] for g in CARD_GROUPS] == [c["members"] for c in CATEGORIES]


def test_category_verdicts_returns_empty_when_missing(tmp_path: Path) -> None:
    out = category_verdicts_for_isoform(llm_dir=tmp_path, tis_slug="x")
    assert out == {}


def test_category_verdicts_reads_categories_json(tmp_path: Path) -> None:
    slug = "chr3-3129127-+-ATG-test"
    (tmp_path / slug).mkdir()
    (tmp_path / slug / "categories.json").write_text(
        json.dumps(
            {
                "Conservation": {
                    "verdict": "interesting",
                    "reasoning": "primate frame intact (frac_intact 0.96).",
                }
            }
        )
    )
    out = category_verdicts_for_isoform(llm_dir=tmp_path, tis_slug=slug)
    assert out["Conservation"]["verdict"] == "interesting"
    assert "frac_intact" in out["Conservation"]["reasoning"]


# ── _clean_nan must not collapse one-element arrays ───────────────────────


def test_clean_nan_keeps_a_one_element_array_a_list() -> None:
    """A size-1 numpy object array must stay a list, not become its element.

    ``.item()`` on a size-1 array returns the single element, so a one-hit
    positional column silently arrived as a bare dict. Consumers iterate it and
    get key strings instead of hits — which is how MAD2L1's real 19-residue
    helix (pLDDT 0.97) rendered as "No helix or strand in region". It hit 6
    columns (SSE elements, motifs, massspec, InterProScan), always and only when
    a protein had exactly one hit, which is why it survived: the multi-hit
    proteins everyone checked looked fine.
    """
    np = pytest.importorskip("numpy")
    from swissisoform_site.data import _clean_nan

    one = np.empty(1, dtype=object)
    one[0] = {"type": "helix", "start": 13, "end": 31, "length": 19}
    out = _clean_nan(one)
    assert isinstance(out, list) and len(out) == 1
    assert out[0]["type"] == "helix"

    two = np.empty(2, dtype=object)
    two[0], two[1] = {"a": 1}, {"a": 2}
    assert [d["a"] for d in _clean_nan(two)] == [1, 2]

    empty = np.empty(0, dtype=object)
    assert _clean_nan(empty) == []


def test_clean_nan_still_unwraps_numpy_scalars() -> None:
    """The ndim guard must not break the branch it sits in front of."""
    np = pytest.importorskip("numpy")
    from swissisoform_site.data import _clean_nan

    assert _clean_nan(np.float64(0.966)) == pytest.approx(0.966)
    assert _clean_nan(np.int64(19)) == 19
    assert isinstance(_clean_nan(np.int64(19)), int)
    assert _clean_nan(np.float64("nan")) is None
    assert _clean_nan("abc") == "abc"


# ── P3 evidence modal: two-column SSE listing ─────────────────────────────


class _Iso:
    def __init__(self, raw, orf_type="extended", diff_space="isoform"):
        self.raw, self.orf_type, self.diff_space = raw, orf_type, diff_space


def _sse_raw(*elements):
    return {
        "isoform_structure_sse_status": "ok",
        "isoform_structure_sse_all_elements": list(elements),
    }


def _el(kind, start, end, region, plddt=0.9):
    return {
        "type": kind, "start": start, "end": end,
        "length": end - start + 1, "plddt_mean": plddt, "region": region,
    }


def _p3(iso):
    from swissisoform_site.data import criterion_evidence_for

    return criterion_evidence_for(iso)["P3_secondary_structure"]


def test_p3_modal_splits_unique_and_shared() -> None:
    # All three qualify (>=6 aa, pLDDT >=0.70) so this test is about the
    # unique/shared split, not the threshold — that is covered separately.
    ce = _p3(_Iso(_sse_raw(
        _el("helix", 16, 32, "unique"),
        _el("strand", 40, 46, "shared"),
        _el("helix", 60, 75, "shared"),
    )))
    summary, listing = ce["sections"][0], ce["sections"][1]
    assert summary["cmp_headers"] == ["Metric", "Differential", "Shared"]
    # helices: 1 unique / 1 shared;  strands: 0 unique / 1 shared
    assert [r["cols"] for r in summary["compare_rows"]][:2] == [["1", "1"], ["0", "1"]]
    assert [r["label"] for r in summary["compare_rows"]][:2] == ["Alpha helices", "Beta strands"]
    # One row per element, padded to the longer side.
    assert len(listing["pairs"]) == 2
    assert listing["pairs"][0]["left"]["span"] == "16–32"
    assert listing["pairs"][1]["left"] is None  # padding


def test_p3_modal_relabels_the_unique_column_for_truncations() -> None:
    """On a truncation the region exists only in the canonical protein, so
    calling that column "isoform-unique" would name the one protein lacking it.
    """
    ce = _p3(_Iso(_sse_raw(_el("helix", 13, 31, "unique")),
                  orf_type="truncated", diff_space="canonical"))
    # The compare table follows the house Differential|Shared convention; the
    # ORF-aware wording is on the paired listing.
    assert ce["sections"][0]["cmp_headers"][1] == "Differential"
    assert ce["sections"][1]["pair_headers"][0] == "Removed (canonical)"


def test_p3_modal_counts_a_boundary_spanning_element_for_the_isoform() -> None:
    """A spanning element has secondary structure in the unique region, so it
    counts for the isoform and lists on the unique side at its full span — no
    separate annotation about where it ends.
    """
    ce = _p3(_Iso(_sse_raw(_el("helix", 13, 35, "spans"))))
    summary, listing = ce["sections"][0], ce["sections"][1]
    assert [r["cols"] for r in summary["compare_rows"]][0] == ["1", "0"]  # alpha helices
    cell = listing["pairs"][0]["left"]
    assert cell["span"] == "13–35"
    assert "note" not in cell
    assert listing["pairs"][0]["right"] is None


def test_p3_modal_absent_when_sse_did_not_run() -> None:
    from swissisoform_site.data import criterion_evidence_for

    ce = criterion_evidence_for(_Iso({"isoform_structure_sse_status": "no_cache"}))
    assert ce["P3_secondary_structure"]["sections"] == []


def test_p3_modal_survives_a_numpy_element_array() -> None:
    """The parquet hands back a numpy object array, not a list."""
    np = pytest.importorskip("numpy")
    arr = np.empty(1, dtype=object)
    arr[0] = _el("helix", 16, 32, "unique")
    ce = _p3(_Iso({
        "isoform_structure_sse_status": "ok",
        "isoform_structure_sse_all_elements": arr,
    }))
    assert ce["sections"][1]["pairs"][0]["left"]["span"] == "16–32"


def test_p3_modal_badges_name_the_structure_type() -> None:
    """The badge reads "alpha helix" / "beta strand", but `kind` stays the bare
    P-SEA name because it keys the CSS class (a space would break it).
    """
    ce = _p3(_Iso(_sse_raw(_el("helix", 16, 32, "unique"), _el("strand", 40, 47, "shared"))))
    left = ce["sections"][1]["pairs"][0]["left"]
    right = ce["sections"][1]["pairs"][0]["right"]
    assert (left["kind"], left["label"]) == ("helix", "alpha helix")
    assert (right["kind"], right["label"]) == ("strand", "beta strand")


def test_p3_modal_moves_sub_threshold_elements_to_their_own_section() -> None:
    """Every count on the card must agree, so the summary and the first listing
    hold only what P3 counts; the rest gets its own heading rather than being
    dropped. TRIP13 is the live case: a 14 aa helix at pLDDT 0.63 made the table
    read 2 elements while the headline read 1.
    """
    ce = _p3(_Iso(_sse_raw(
        _el("strand", 7, 12, "unique", plddt=0.78),       # qualifies
        _el("helix", 21, 34, "spans", plddt=0.63),        # long, but not confident
        _el("strand", 40, 42, "shared", plddt=0.95),      # confident, but short
    )))
    summary, good, weak = ce["sections"]
    # Summary counts the qualifying set only: 0 helices / 1 strand on the unique side.
    assert [r["cols"] for r in summary["compare_rows"]][:2] == [["0", "0"], ["1", "0"]]
    assert good["title"] == "Elements and coordinates"
    assert good["pairs"][0]["left"]["span"] == "7–12"
    assert weak["title"] == "Below threshold"
    assert weak["pairs"][0]["left"]["span"] == "21–34"       # low pLDDT
    assert weak["pairs"][0]["right"]["span"] == "40–42"      # too short


def test_p3_modal_omits_the_below_threshold_section_when_empty() -> None:
    ce = _p3(_Iso(_sse_raw(_el("helix", 16, 32, "unique"))))
    assert [s["title"] for s in ce["sections"]] == [
        "Secondary structure · differential vs shared region",
        "Elements and coordinates",
    ]
