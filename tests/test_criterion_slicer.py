"""Tests for slice_criterion — the per-criterion LLM/UI input builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from swissisoform.site import evidence as ber


def test_criteria_dict_has_exactly_13_entries() -> None:
    assert len(ber.CRITERIA) == 13
    # 6 E and 7 F
    e_count = sum(1 for c in ber.CRITERIA.values() if c["axis"] == "E")
    f_count = sum(1 for c in ber.CRITERIA.values() if c["axis"] == "F")
    assert e_count == 6
    assert f_count == 7


def test_criterion_ids_match_scoring_module() -> None:
    """The 13 criterion ids must match what src/swissisoform/modules/scoring.py emits."""
    expected = {
        "E1_primate_conservation",
        "E2_mammalian_conservation",
        "E3_phylop_coding_selection",
        "E4_multi_cell_line",
        "E5_initiation_efficiency",
        "E6_mass_spec",
        "F1_structured_extension",
        "F2_localization_change",
        "F3_domain_change",
        "F4_targeting_change",
        "F5_pathogenic_variant_enrichment",
        "F6_clinical_variant_overlap",
        "F7_shared_structural_change",
    }
    assert set(ber.CRITERIA) == expected


def test_unknown_criterion_raises_key_error() -> None:
    with pytest.raises(KeyError):
        ber.slice_criterion({}, "bogus_criterion")


def test_slice_carries_criterion_metadata() -> None:
    sl = ber.slice_criterion({"tis_id": "x"}, "E1_primate_conservation")
    assert sl["criterion_id"] == "E1_primate_conservation"
    assert sl["axis"] == "E"
    assert sl["label"] == "Primate conservation"
    assert "frame" in sl["interpretation_hint"].lower()


def test_slice_includes_isoform_identity() -> None:
    sl = ber.slice_criterion(
        {
            "tis_id": "T1",
            "gene": {"name": "G1"},
            "orf_type": "truncated",
            "diff_space": "canonical",
        },
        "E1_primate_conservation",
    )
    assert sl["isoform"]["tis_id"] == "T1"
    assert sl["isoform"]["gene_name"] == "G1"
    assert sl["isoform"]["orf_type"] == "truncated"


def test_slice_extracts_value_and_reason_from_scoring() -> None:
    sl = ber.slice_criterion(
        {
            "tis_id": "x",
            "scoring": {
                "criteria": {
                    "E1_primate_conservation": {"value": True, "reason": "0.96 >= 0.30"},
                },
            },
        },
        "E1_primate_conservation",
    )
    assert sl["value"] is True
    assert sl["reason"] == "0.96 >= 0.30"


def test_slice_extracts_headline_from_raw() -> None:
    # E1 headline is now mean_pident (frac_intact is context only).
    sl = ber.slice_criterion(
        {
            "tis_id": "x",
            "_raw": {"isoform_conservation_frame_primate_mean_pident": 0.96},
        },
        "E1_primate_conservation",
    )
    assert sl["headline"] == 0.96


def test_slice_extracts_all_evidence_cols() -> None:
    raw = {
        "isoform_conservation_frame_primate_mean_pident": 0.96,
        "isoform_conservation_frame_primate_frac_intact": 0.88,
        "isoform_conservation_frame_primate_n_species_aligned": 25,
        "isoform_conservation_frame_primate_deepest_species": "Callithrix_jacchus",
        # Should not be picked up:
        "isoform_conservation_frame_mammalian_frac_intact": 0.05,
    }
    sl = ber.slice_criterion({"_raw": raw}, "E1_primate_conservation")
    assert sl["evidence"]["isoform_conservation_frame_primate_mean_pident"] == 0.96
    assert sl["evidence"]["isoform_conservation_frame_primate_frac_intact"] == 0.88
    assert sl["evidence"]["isoform_conservation_frame_primate_n_species_aligned"] == 25
    # Cross-criterion col not present:
    assert "isoform_conservation_frame_mammalian_frac_intact" not in sl["evidence"]


def test_real_trnt1_record_e1_e2_e3() -> None:
    """End-to-end: refresh TRNT1 record, slice E1/E2/E3, sanity-check values."""
    p = Path("data/output/cheeseman_13gene/llm_evidence/TRNT1.json")
    if not p.exists():
        pytest.skip("real TRNT1 record not present")
    rec = json.loads(p.read_text())
    iso = rec["isoforms"][0]
    iso_with_gene = {**iso, "gene": {"name": "TRNT1"}}

    e1 = ber.slice_criterion(iso_with_gene, "E1_primate_conservation")
    assert e1["headline"] is not None
    assert 0 <= e1["headline"] <= 1

    e3 = ber.slice_criterion(iso_with_gene, "E3_phylop_coding_selection")
    assert "isoform_conservation_phylop_unique_region_mean" in e3["evidence"]


def test_f5_extracts_variants_as_hits() -> None:
    """F5 has evidence_hits_col — must populate hits list; headline is the
    gnomAD depletion ratio (germline tolerance/constraint), not a damaging count.
    """
    iso = {
        "_raw": {
            "isoform_variant_intersection_hits": [
                {"variant_id": "V1", "clinical_significance": "Pathogenic"},
                {"variant_id": "V2"},
            ],
            "isoform_variant_intersection_gnomad_depletion_ratio": 0.5,
        },
    }
    sl = ber.slice_criterion(iso, "F5_pathogenic_variant_enrichment")
    assert len(sl["hits"]) == 2
    assert sl["headline"] == 0.5
