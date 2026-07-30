"""Tests for swissisoform.site.evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_PARQUET = REPO_ROOT / "data" / "output" / "cheeseman_13gene" / "all_paired.parquet"


def _load_module():
    from swissisoform.site import evidence

    return evidence


ber = _load_module()


REQUIRED_ISOFORM_KEYS = {
    "tis_id",
    "orf_type",
    "alt_start_codon",
    "isoform_length_aa",
    "canonical_length_aa",
    "differential_sequence",
    "diff_space",
    "kozak_context",
    "scoring",
    "key_metrics",
    "pathogenic_variants_in_unique",
}


@pytest.fixture(scope="module")
def real_outputs(tmp_path_factory):
    if not REAL_PARQUET.exists():
        pytest.skip(f"Real parquet not present at {REAL_PARQUET}")
    out_dir = tmp_path_factory.mktemp("llm_evidence")
    ber.write_evidence_records(REAL_PARQUET, out_dir)
    return out_dir


def test_all_13_gene_files_have_top_level_keys(real_outputs):
    files = sorted(real_outputs.glob("*.json"))
    assert len(files) == 13, f"Expected 13 gene files, got {len(files)}"
    for fp in files:
        rec = json.loads(fp.read_text(encoding="utf-8"))
        assert set(rec.keys()) == {"gene", "isoforms"}, f"{fp.name}: bad top-level keys"
        assert rec["gene"]["name"] == fp.stem
        assert isinstance(rec["isoforms"], list) and len(rec["isoforms"]) >= 1


def test_each_isoform_has_required_keys(real_outputs):
    for fp in sorted(real_outputs.glob("*.json")):
        rec = json.loads(fp.read_text(encoding="utf-8"))
        for iso in rec["isoforms"]:
            missing = REQUIRED_ISOFORM_KEYS - set(iso.keys())
            assert not missing, f"{fp.name}: isoform missing keys {missing}"


def test_diff_space_consistent_with_orf_type(real_outputs):
    """diff_space == 'canonical' iff orf_type == 'truncated' (lowercase)."""
    for fp in sorted(real_outputs.glob("*.json")):
        rec = json.loads(fp.read_text(encoding="utf-8"))
        for iso in rec["isoforms"]:
            orf_type = (iso.get("orf_type") or "").lower()
            diff_space = iso.get("diff_space")
            if orf_type == "truncated":
                assert diff_space == "canonical", (
                    f"{fp.name}/{iso['tis_id']}: truncated should be canonical-space, "
                    f"got {diff_space}"
                )
            else:
                assert diff_space == "isoform", (
                    f"{fp.name}/{iso['tis_id']}: {orf_type} should be isoform-space, "
                    f"got {diff_space}"
                )


def test_scoring_score_types_and_range(real_outputs):
    for fp in sorted(real_outputs.glob("*.json")):
        rec = json.loads(fp.read_text(encoding="utf-8"))
        for iso in rec["isoforms"]:
            scoring = iso["scoring"]
            for key in ("existence_score", "functional_score"):
                v = scoring.get(key)
                assert v is None or isinstance(v, int), (
                    f"{fp.name}/{iso['tis_id']} {key} should be int|None, got {type(v).__name__}"
                )
                if v is not None:
                    assert 0 <= v <= 6, f"{fp.name}/{iso['tis_id']} {key} out of range: {v}"


def test_nan_serialization_emits_nulls():
    """A synthetic all-NaN row must serialize to dict full of None, not 'nan'."""
    nan_row = {col: np.nan for col in _NAN_SCHEMA_COLUMNS}
    nan_row["gene_name"] = "FAKEGENE"
    nan_row["isoform_scoring_criteria"] = None
    nan_row["isoform_scoring_reasons"] = None
    nan_row["isoform_variant_intersection_hits"] = None
    nan_row["isoform_massspec_hits"] = None
    nan_row["orf_type"] = None
    df = pd.DataFrame([nan_row])
    record = ber.build_gene_record("FAKEGENE", df)
    iso = record["isoforms"][0]
    # Walk and check no string "nan" leaked in.
    payload = json.dumps(record)
    assert "NaN" not in payload, "Found literal NaN in JSON serialization"
    assert '"nan"' not in payload.lower().replace('"name"', ""), (
        "Found 'nan' string in JSON serialization"
    )
    # Most fields should be None.
    assert iso["tis_id"] is None
    assert iso["isoform_length_aa"] is None
    assert iso["scoring"]["existence_score"] is None
    assert iso["scoring"]["functional_score"] is None
    assert iso["key_metrics"]["primate_frac_intact"] is None
    assert iso["pathogenic_variants_in_unique"] == []
    # diff_space derives from orf_type=None → None
    assert iso["diff_space"] is None


_VARIANTS_LONG_NEW_COLS = (
    "allele_frequency",
    "aa_ref",
    "aa_alt",
    "isoform_aa_ref",
    "isoform_aa_alt",
    "isoform_consequence",
    "hgvsc",
    "rs_id",
    "cosmic_sample_count",
    "clinvar_title",
)


def test_variants_long_undrops_extra_fields(tmp_path):
    """write_variants_long must surface AF/AA-change/isoform-frame/metadata fields."""
    hit = {
        "variant_id": "ClinVar:12345",
        "source": "ClinVar",
        "chrom": "chr17",
        "genomic_pos": 48071438,
        "ref": "G",
        "alt": "C",
        "hgvsp": "ENSP00000225603.4:p.Asn185Lys",
        "consequence": "missense_variant",
        "clinical_significance": "Pathogenic",
        "protein_pos": 184,
        "isoform_protein_pos": 238,
        "in_isoform_unique": True,
        "in_isoform_shared": False,
        "allele_frequency": 6.8e-07,
        "aa_ref": "N",
        "aa_alt": "K",
        "isoform_aa_ref": "D",
        "isoform_aa_alt": "E",
        "isoform_consequence": "missense_variant",
        "metadata": {
            "hgvsc": "ENST00000225603.9:c.555C>G",
            "rs_id": "rs9999",
            "cosmic_sample_count": 7,
            "title": "NM_000.1(GENE):c.555C>G (p.Asn185Lys)",
        },
    }
    row = {
        "tis_id": "chr17:48101392:-:GTG:ENST00000225603.9",
        "gene_name": "CBX1",
        "isoform_variant_intersection_hits": [hit],
        "isoform_varianteffect_hits": None,
    }
    parquet = tmp_path / "all_paired.parquet"
    pd.DataFrame([row]).to_parquet(parquet, index=False)

    out = tmp_path / "variants_long.parquet"
    n = ber.write_variants_long(parquet, out)
    assert n == 1
    df = pd.read_parquet(out)

    for col in _VARIANTS_LONG_NEW_COLS:
        assert col in df.columns, f"missing column {col!r}"

    r = df.iloc[0]
    assert abs(float(r["allele_frequency"]) - 6.8e-07) < 1e-12
    assert r["aa_ref"] == "N"
    assert r["aa_alt"] == "K"
    assert r["isoform_aa_ref"] == "D"
    assert r["isoform_aa_alt"] == "E"
    assert r["isoform_consequence"] == "missense_variant"
    assert r["hgvsc"] == "ENST00000225603.9:c.555C>G"
    assert r["rs_id"] == "rs9999"
    assert int(r["cosmic_sample_count"]) == 7
    assert r["clinvar_title"] == "NM_000.1(GENE):c.555C>G (p.Asn185Lys)"


_NAN_SCHEMA_COLUMNS = (
    "tis_id",
    "orf_type",
    "start_codon",
    "isoform_len",
    "canonical_len",
    "differential_sequence",
    "diff_space",
    "kozak_context",
    "isoform_scoring_existence_score",
    "isoform_scoring_existence_evaluable",
    "isoform_scoring_existence_high_confidence",
    "isoform_scoring_functional_score",
    "isoform_scoring_functional_evaluable",
    "isoform_scoring_functional_high_confidence",
    "isoform_conservation_frame_primate_frac_intact",
    "isoform_conservation_frame_mammalian_frac_intact",
    "isoform_conservation_phylop_unique_region_mean",
    "isoform_conservation_phylop_shared_region_mean",
    "isoform_conservation_phylop_at_tis",
    "isoform_structure_plddt_diffregion_mean",
    "isoform_structure_tm_score",
    "isoform_structure_rmsd_global",
    "canonical_localization_deeploc_prediction",
    "isoform_localization_deeploc_prediction",
    "isoform_variant_intersection_n_in_unique_region",
    "isoform_variant_intersection_n_pathogenic_in_unique_region",
    "isoform_varianteffect_n_damaging_in_unique",
    "isoform_varianteffect_mean_delta_llr_unique",
    "isoform_varianteffect_mean_am_pathogenicity_unique",
    "generef_uniprot_id",
    "generef_function",
    "generef_subcellular_location",
)


# ── CDLMPS categories + slice_category ─────────────────────────────────────


def test_categories_cover_all_criteria_once():
    """Every scored criterion (incl. S2 biophysics + S3 SAE) appears in exactly one category."""
    members = [m for cat in ber.CATEGORIES for m in cat["members"]]
    # All members are first-class criteria now — no descriptive magic strings.
    assert all(m in ber.CRITERIA for m in members)
    # All 15 criteria covered exactly once.
    assert sorted(members) == sorted(ber.CRITERIA)
    assert len(members) == len(set(members)) == 15
    # S2/S3 live once each, in category S.
    assert members.count("S2_biophysics") == 1
    assert members.count("S3_sae") == 1
    assert not hasattr(ber, "LLM_EXCLUDED_CRITERIA")
    assert not hasattr(ber, "DESCRIPTIVE_MEMBERS")


def _S_CATEGORY() -> dict:
    return next(c for c in ber.CATEGORIES if c["letter"] == "S")


def test_slice_category_bundles_scored_members():
    """Category S = S1 domain + S2 biophysics + S3 sae — all first-class criteria."""
    iso_record = {
        "tis_id": "chr1:100:+:ATG:ENST_A",
        "gene": {"name": "GENE_A"},
        "orf_type": "extended",
        "differential_sequence": "MABC",
        "diff_space": "isoform",
        "isoform_length_aa": 60,
        "canonical_length_aa": 50,
        "scoring": {
            "criteria": {
                "S1_domain_change": {"value": True, "reason": "domain gained"},
                "S2_biophysics": {"value": True, "reason": "gravy distinct"},
                "S3_sae": {"value": False, "reason": "n_unique_region_features=0"},
            }
        },
        "_raw": {
            "isoform_interproscan_summary": {"n_domains": 2},
            "cmp_interproscan_n_hits_in_diff_region": 1,
            "cmp_interproscan_n_real_domains_changed_in_diff_region": 1,
            "cmp_biophysics_gravy_unique": 0.4,
            "cmp_biophysics_gravy_shared": 0.1,
            "cmp_biophysics_gravy_ratio": 4.0,
            "cmp_biophysics_gravy_delta": 0.35,
            "isoform_sae_status": "ok",
            "isoform_sae_n_isoform_only": 3,
            "isoform_sae_n_canonical_only": 1,
            "isoform_sae_n_shared": 20,
            "isoform_sae_top_gained_feature_index": 42,
            "isoform_sae_top_gained_feature_label": "beta strand",
            "isoform_sae_top_gained_delta_max": 1.2,
            "isoform_sae_unique_region_top_features": [
                {"feature_index": 42, "label": "beta strand", "max": 1.2, "prevalence": 4},
            ],
        },
    }
    out = ber.slice_category(iso_record, _S_CATEGORY())
    assert out["category"] == "S"
    assert out["name"] == "Structural Characteristics"
    assert out["isoform"]["tis_id"] == "chr1:100:+:ATG:ENST_A"

    by_member = {m["criterion_id"]: m for m in out["members"]}
    assert set(by_member) == {"S1_domain_change", "S2_biophysics", "S3_sae"}
    # Every member is a first-class criterion — uniform kind + criterion_id.
    assert all(m["kind"] == "criterion" for m in out["members"])
    assert by_member["S1_domain_change"]["value"] is True
    # S2/S3 carry the scored value + reason via the criterion path, plus their
    # rich nested evidence built by the evidence_builder hook.
    assert by_member["S2_biophysics"]["value"] is True
    assert by_member["S2_biophysics"]["reason"] == "gravy distinct"
    assert "gravy_delta" in by_member["S2_biophysics"]["evidence"]
    assert by_member["S3_sae"]["value"] is False
    assert by_member["S3_sae"]["evidence"]["counts"]["isoform_only"] == 3
    assert by_member["S3_sae"]["evidence"]["top_gained"]["feature_index"] == 42


def test_slice_category_omits_empty_builder_members_without_data():
    """S2/S3 (omit_if_empty) drop out when their columns are absent; S1 survives."""
    iso_record = {
        "tis_id": "chr1:100:+:ATG:ENST_A",
        "gene": {"name": "GENE_A"},
        "orf_type": "extended",
        "scoring": {"criteria": {}},
        "_raw": {},  # no biophysics cols, sae status missing
    }
    out = ber.slice_category(iso_record, _S_CATEGORY())
    members = [m["criterion_id"] for m in out["members"]]
    # S1 (flat criterion) survives even with no data; S2/S3 are omitted (omit_if_empty).
    assert members == ["S1_domain_change"]
