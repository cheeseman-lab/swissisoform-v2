"""Tests for the long-form variants parquet writer."""

from pathlib import Path

import pandas as pd
import pytest

from scripts.site import build_evidence_records as ber


def _two_row_parquet(tmp_path: Path) -> Path:
    """Synthetic 2-row parquet with one isoform carrying two variants, one carrying zero."""
    df = pd.DataFrame(
        [
            {
                "tis_id": "chr1:100:+:ATG:ENST_A",
                "gene_name": "GENE_A",
                "isoform_variant_intersection_hits": [
                    {
                        "variant_id": "ClinVar:1",
                        "source": "ClinVar",
                        "chrom": "chr1",
                        "pos": 110,
                        "ref": "G",
                        "alt": "A",
                        "hgvsp": "p.Leu1fs",
                        "consequence": "frameshift_variant",
                        "clinical_significance": "Pathogenic",
                        "protein_pos": 10,
                        "isoform_protein_pos": None,
                        "in_isoform_unique": True,
                        "in_isoform_shared": False,
                    },
                    {
                        "variant_id": "gnomAD:chr1-115-G-T",
                        "source": "gnomAD",
                        "chrom": "chr1",
                        "pos": 115,
                        "ref": "G",
                        "alt": "T",
                        "hgvsp": None,
                        "consequence": "missense_variant",
                        "clinical_significance": None,
                        "protein_pos": 12,
                        "isoform_protein_pos": None,
                        "in_isoform_unique": True,
                        "in_isoform_shared": False,
                    },
                ],
                "isoform_varianteffect_hits": [
                    {
                        "variant_id": "ClinVar:1",
                        "am_class": "likely_pathogenic",
                        "am_pathogenicity": 0.91,
                        "plm_delta_llr": -5.4,
                        "plm_llr_wt": 0.0,
                        "plm_llr_alt": -5.4,
                        "effect_damaging": True,
                    }
                ],
            },
            {
                "tis_id": "chr2:200:+:CTG:ENST_B",
                "gene_name": "GENE_B",
                "isoform_variant_intersection_hits": [],
                "isoform_varianteffect_hits": [],
            },
        ]
    )
    p = tmp_path / "synthetic_paired.parquet"
    df.to_parquet(p, index=False)
    return p


def test_variants_long_one_row_per_tis_variant_pair(tmp_path: Path) -> None:
    src = _two_row_parquet(tmp_path)
    out = tmp_path / "variants_long.parquet"
    n = ber.write_variants_long(src, out)
    df = pd.read_parquet(out)
    assert n == 2
    assert len(df) == 2
    assert set(df["variant_id"]) == {"ClinVar:1", "gnomAD:chr1-115-G-T"}


def test_variants_long_carries_effect_columns_joined_on_variant_id(tmp_path: Path) -> None:
    src = _two_row_parquet(tmp_path)
    out = tmp_path / "variants_long.parquet"
    ber.write_variants_long(src, out)
    df = pd.read_parquet(out)
    cv = df[df["variant_id"] == "ClinVar:1"].iloc[0]
    assert cv["am_class"] == "likely_pathogenic"
    assert cv["am_pathogenicity"] == pytest.approx(0.91)
    assert cv["plm_delta_llr"] == pytest.approx(-5.4)
    assert cv["effect_damaging"] is True or cv["effect_damaging"] == 1
    # A variant with no effect entry has nulls in those columns:
    gnom = df[df["variant_id"] == "gnomAD:chr1-115-G-T"].iloc[0]
    assert pd.isna(gnom["am_pathogenicity"])


def test_variants_long_schema_columns(tmp_path: Path) -> None:
    src = _two_row_parquet(tmp_path)
    out = tmp_path / "variants_long.parquet"
    ber.write_variants_long(src, out)
    df = pd.read_parquet(out)
    expected = {
        "tis_id",
        "gene_name",
        "variant_id",
        "source",
        "chrom",
        "pos",
        "ref",
        "alt",
        "hgvsp",
        "consequence",
        "clinical_significance",
        "protein_pos",
        "isoform_protein_pos",
        "in_isoform_unique",
        "in_isoform_shared",
        "am_class",
        "am_pathogenicity",
        "plm_delta_llr",
        "plm_llr_wt",
        "plm_llr_alt",
        "effect_damaging",
    }
    assert expected.issubset(set(df.columns))


def test_variants_long_isoform_with_no_variants_yields_zero_rows(tmp_path: Path) -> None:
    src = _two_row_parquet(tmp_path)
    out = tmp_path / "variants_long.parquet"
    ber.write_variants_long(src, out)
    df = pd.read_parquet(out)
    # GENE_B / chr2 had no variants — must not appear.
    assert (df["tis_id"] == "chr2:200:+:CTG:ENST_B").sum() == 0
