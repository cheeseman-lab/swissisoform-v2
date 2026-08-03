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
