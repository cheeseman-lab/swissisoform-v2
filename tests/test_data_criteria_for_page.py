"""Tests for CRITERIA_FOR_PAGE + category_verdicts_for_isoform in data.py."""

import json
from pathlib import Path

import pytest

pytest.importorskip("swissisoform_site")  # optional website package; skip if not installed

from swissisoform_site.data import CARD_GROUPS, CRITERIA_FOR_PAGE, category_verdicts_for_isoform


def test_criteria_for_page_has_13_entries() -> None:
    assert len(CRITERIA_FOR_PAGE) == 13
    e_count = sum(1 for c in CRITERIA_FOR_PAGE if c["axis"] == "E")
    f_count = sum(1 for c in CRITERIA_FOR_PAGE if c["axis"] == "F")
    assert e_count == 6
    assert f_count == 7


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
