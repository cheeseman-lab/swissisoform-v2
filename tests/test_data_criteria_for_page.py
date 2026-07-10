"""Tests for CRITERIA_FOR_PAGE + llm_criterion_for_isoform in data.py."""

import json
from pathlib import Path

import pytest

pytest.importorskip("swissisoform_site")  # optional website package; skip if not installed

from swissisoform_site.data import CRITERIA_FOR_PAGE, llm_criterion_for_isoform


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


def test_llm_criterion_for_isoform_returns_none_when_missing(tmp_path: Path) -> None:
    out = llm_criterion_for_isoform(
        llm_dir=tmp_path, tis_slug="x", criterion_id="E1_primate_conservation"
    )
    assert out is None


def test_llm_criterion_for_isoform_reads_criteria_json(tmp_path: Path) -> None:
    slug = "chr3-3129127-+-ATG-test"
    (tmp_path / slug).mkdir()
    (tmp_path / slug / "criteria.json").write_text(
        json.dumps(
            {
                "E1_primate_conservation": {
                    "criterion_id": "E1_primate_conservation",
                    "axis": "E",
                    "label": "Primate conservation",
                    "headline": "frac_intact 0.96",
                    "summary": "preserved",
                    "confidence": "high",
                }
            }
        )
    )
    out = llm_criterion_for_isoform(
        llm_dir=tmp_path, tis_slug=slug, criterion_id="E1_primate_conservation"
    )
    assert out is not None
    assert out["label"] == "Primate conservation"
