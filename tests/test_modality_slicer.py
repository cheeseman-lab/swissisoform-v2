"""Tests for slice_modality — the modality-shaped LLM input builder."""

from __future__ import annotations

import pytest

from scripts.site import build_evidence_records as ber

_ISOFORM = {
    "tis_id": "chr3:3129127:+:ATG:ENST00000434583.5",
    "gene": {"name": "TRNT1"},
    "orf_type": "truncated",
    "differential_sequence": "MLRCLYHWHRPVLNRRWSRLCLPKQYLFT",
    "diff_space": "canonical",
    "isoform_length_aa": 405,
    "canonical_length_aa": 434,
    "scoring": {
        "criteria": {
            "E1_primate_frame_conservation": {"value": True, "reason": "0.96 ≥ 0.30"},
            "E5_clinical_variants_exist": {"value": True, "reason": "n=117"},
            "F5_pathogenic_variant_enrichment": {
                "value": True,
                "reason": "n_damaging=2/104; n_am_pathogenic=2",
            },
            "F1_pLDDT_diff_region": {"value": True, "reason": "0.46 ≤ 0.60"},
        },
    },
    "key_metrics": {
        "n_variants_in_unique": 117,
        "n_pathogenic_in_unique": 3,
        "mean_delta_llr_unique": -0.34,
    },
    "pathogenic_variants_in_unique": [
        {"id": "ClinVar:1068618", "hgvsp": "p.Leu13fs"},
    ],
    "_raw": {  # synthetic raw columns for the test
        "isoform_conservation_phylop_at_tis": 6.42,
        "isoform_conservation_phylop_unique_region_mean": 1.5,
        "isoform_conservation_frame_primate_frac_intact": 0.96,
        "isoform_structure_plddt_diffregion_mean": 0.46,
        "isoform_structure_tm_score": 0.953,
        "expr_HeLa_cpm": 2.1,
        "expr_HeLa_initiation_efficiency": 0.7,
        "isoform_variant_intersection_n_in_unique_region": 117,
    },
}


def test_slice_existence_for_variants_includes_e_criteria_only() -> None:
    sl = ber.slice_modality(_ISOFORM, "variants", axis="existence")
    cited = {c["name"] for c in sl["criteria"]}
    assert "E5_clinical_variants_exist" in cited
    assert "F5_pathogenic_variant_enrichment" not in cited


def test_slice_functional_for_variants_includes_f_criteria_only() -> None:
    sl = ber.slice_modality(_ISOFORM, "variants", axis="functional")
    cited = {c["name"] for c in sl["criteria"]}
    assert "F5_pathogenic_variant_enrichment" in cited
    assert "E5_clinical_variants_exist" not in cited


def test_slice_carries_modality_name_and_axis() -> None:
    sl = ber.slice_modality(_ISOFORM, "variants", axis="existence")
    assert sl["modality"] == "variants"
    assert sl["axis"] == "existence"


def test_slice_includes_isoform_identity_block() -> None:
    sl = ber.slice_modality(_ISOFORM, "variants", axis="existence")
    assert sl["isoform"]["tis_id"] == _ISOFORM["tis_id"]
    assert sl["isoform"]["orf_type"] == "truncated"
    assert sl["isoform"]["differential_sequence"].startswith("MLRC")


def test_slice_modality_with_no_criteria_returns_axis_unavailable() -> None:
    # Mass spec only has E6; functional axis is empty by design.
    sl = ber.slice_modality(_ISOFORM, "mass_spec", axis="functional")
    assert sl["criteria"] == []
    assert sl.get("axis_unavailable") is True


def test_slice_pulls_modality_specific_evidence_columns() -> None:
    sl = ber.slice_modality(_ISOFORM, "structure", axis="functional")
    ev = sl["evidence"]
    assert "isoform_structure_plddt_diffregion_mean" in ev
    assert ev["isoform_structure_plddt_diffregion_mean"] == pytest.approx(0.46)
    # Conservation columns should NOT appear in a structure slice:
    assert "isoform_conservation_phylop_at_tis" not in ev
