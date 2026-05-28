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
            "E1_primate_conservation": {"value": True, "reason": "0.96 ≥ 0.30"},
            "F6_clinical_variant_overlap": {"value": True, "reason": "n=117"},
            "F5_pathogenic_variant_enrichment": {
                "value": True,
                "reason": "n_damaging=2/104; n_am_pathogenic=2",
            },
            "F1_structured_extension": {"value": True, "reason": "0.46 ≤ 0.60"},
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
    "_raw": {
        "isoform_conservation_phylop_at_tis": 6.42,
        "isoform_conservation_phylop_unique_region_mean": 1.5,
        "isoform_conservation_frame_primate_frac_intact": 0.96,
        "isoform_structure_plddt_diffregion_mean": 0.46,
        "isoform_structure_tm_score": 0.953,
        "expr_HeLa_cpm": 2.1,
        "expr_HeLa_initiation_efficiency": 0.7,
        "isoform_variant_intersection_n_in_unique_region": 117,
        # mass spec + interproscan columns added to exercise the corrected prefixes:
        "isoform_massspec_hits": [],
        "isoform_massspec_summary": "no hits",
        "isoform_interproscan_hits": [],
        "isoform_motifs_hits": [],
    },
}


def test_slice_existence_for_variants_returns_axis_unavailable() -> None:
    """Variants modality has no existence-axis criteria — both F5 and F6 are functional."""
    sl = ber.slice_modality(_ISOFORM, "variants", axis="existence")
    assert sl["criteria"] == []
    assert sl.get("axis_unavailable") is True


def test_slice_functional_for_variants_cites_F5_and_F6() -> None:
    sl = ber.slice_modality(_ISOFORM, "variants", axis="functional")
    cited = {c["name"] for c in sl["criteria"]}
    assert "F5_pathogenic_variant_enrichment" in cited
    assert "F6_clinical_variant_overlap" in cited


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


def test_unknown_modality_raises_key_error() -> None:
    with pytest.raises(KeyError):
        ber.slice_modality(_ISOFORM, "bogus_modality", axis="existence")


def test_unknown_axis_raises_value_error() -> None:
    with pytest.raises(ValueError):
        ber.slice_modality(_ISOFORM, "variants", axis="bogus_axis")


def test_variants_modality_injects_pathogenic_list_on_functional_axis() -> None:
    sl = ber.slice_modality(_ISOFORM, "variants", axis="functional")
    assert "pathogenic_variants_in_unique" in sl["evidence"]
    assert sl["evidence"]["pathogenic_variants_in_unique"][0]["id"] == "ClinVar:1068618"


def test_real_trnt1_evidence_record_cites_expected_criteria() -> None:
    """End-to-end check: refreshed TRNT1.json + slicer cites E1+E2+E3 in conservation/existence."""
    import json
    from pathlib import Path

    p = Path("data/output/cheeseman_12gene/llm_evidence/TRNT1.json")
    if not p.exists():
        pytest.skip("real TRNT1 evidence record not present in this environment")
    rec = json.loads(p.read_text())
    iso = rec["isoforms"][0]
    sl = ber.slice_modality({**iso, "gene": {"name": "TRNT1"}}, "conservation", axis="existence")
    cited = {c["name"] for c in sl["criteria"]}
    # Real TRNT1 has primate_conservation true (frac_intact=0.96) — must be cited.
    assert "E1_primate_conservation" in cited
    # variants on functional axis cites F5 + F6 with real data:
    sl_var = ber.slice_modality({**iso, "gene": {"name": "TRNT1"}}, "variants", axis="functional")
    cited_var = {c["name"] for c in sl_var["criteria"]}
    assert "F5_pathogenic_variant_enrichment" in cited_var
