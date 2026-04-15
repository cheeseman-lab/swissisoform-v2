"""Tests for Parquet serialization round-trip."""

from __future__ import annotations

import pandas as pd

from swissisoform.io.parquet import dataframe_to_tis, genes_to_dataframe, tis_to_dataframe
from swissisoform.models import (
    CellLineExpression,
    DifferentialRegion,
    Gene,
    ORFType,
    TranslationInitiationSite,
)


def _make_tis() -> TranslationInitiationSite:
    """Create a minimal TIS for testing."""
    return TranslationInitiationSite(
        tis_id="chr1:100:+:ATG",
        gene_name="BRCA1",
        transcript_id="ENST00000001",
        chrom="chr1",
        position=100,
        strand="+",
        start_codon="ATG",
        orf_type=ORFType.ANNOTATED,
        canonical_protein="MVLSPADKTNVKAAWGKVGAHAGEYGAEALERM",
        isoform_protein="MVLSPADKTNVK",
        diff_region=DifferentialRegion(sequence="AAWGKVGAHAGEYGAEALERM"),
        kozak_context="GCCACCATGG",
        expression={
            "HeLa": CellLineExpression(
                raw_count=500,
                cpm=12.5,
                p_value=0.001,
                initiation_efficiency=0.45,
            ),
            "HEK293": CellLineExpression(
                raw_count=300,
                cpm=8.2,
                p_value=0.01,
                initiation_efficiency=None,
            ),
        },
        isoform_annotations={
            "testmod": {"score": 0.95, "label": "high"},
            "biophysics": {"disorder_fraction": 0.3},
        },
    )


class TestTisToDataframe:
    """Tests for tis_to_dataframe."""

    def test_returns_dataframe_with_correct_length(self) -> None:
        sites = [_make_tis(), _make_tis()]
        df = tis_to_dataframe(sites)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_has_identity_columns(self) -> None:
        df = tis_to_dataframe([_make_tis()])
        for col in [
            "tis_id",
            "gene_name",
            "transcript_id",
            "chrom",
            "position",
            "strand",
            "start_codon",
            "orf_type",
        ]:
            assert col in df.columns, f"Missing identity column: {col}"

    def test_orf_type_serialized_as_string(self) -> None:
        df = tis_to_dataframe([_make_tis()])
        assert df["orf_type"].iloc[0] == "Annotated"

    def test_has_expression_columns(self) -> None:
        df = tis_to_dataframe([_make_tis()])
        assert "expr_HeLa_cpm" in df.columns
        assert "expr_HeLa_raw_count" in df.columns
        assert "expr_HEK293_p_value" in df.columns
        assert "expr_HeLa_initiation_efficiency" in df.columns

    def test_has_annotation_columns(self) -> None:
        df = tis_to_dataframe([_make_tis()])
        assert "testmod_score" in df.columns
        assert "testmod_label" in df.columns
        assert "biophysics_disorder_fraction" in df.columns

    def test_protein_columns(self) -> None:
        df = tis_to_dataframe([_make_tis()])
        assert df["canonical_protein"].iloc[0] == "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERM"
        assert df["isoform_protein"].iloc[0] == "MVLSPADKTNVK"


class TestRoundTrip:
    """Tests for tis_to_dataframe -> dataframe_to_tis round-trip."""

    def test_identity_fields_preserved(self) -> None:
        original = _make_tis()
        df = tis_to_dataframe([original])
        restored = dataframe_to_tis(df)
        assert len(restored) == 1
        r = restored[0]
        assert r.tis_id == original.tis_id
        assert r.gene_name == original.gene_name
        assert r.orf_type == ORFType.ANNOTATED

    def test_protein_fields_preserved(self) -> None:
        original = _make_tis()
        df = tis_to_dataframe([original])
        restored = dataframe_to_tis(df)[0]
        assert restored.canonical_protein == original.canonical_protein
        assert restored.isoform_protein == original.isoform_protein

    def test_expression_round_trip(self) -> None:
        original = _make_tis()
        df = tis_to_dataframe([original])
        restored = dataframe_to_tis(df)[0]
        assert "HeLa" in restored.expression
        assert restored.expression["HeLa"].raw_count == 500
        assert restored.expression["HeLa"].cpm == 12.5
        assert restored.expression["HEK293"].raw_count == 300

    def test_annotation_round_trip(self) -> None:
        original = _make_tis()
        df = tis_to_dataframe([original])
        restored = dataframe_to_tis(df)[0]
        assert "testmod" in restored.isoform_annotations
        assert restored.isoform_annotations["testmod"]["score"] == 0.95
        assert restored.isoform_annotations["testmod"]["label"] == "high"
        assert restored.isoform_annotations["biophysics"]["disorder_fraction"] == 0.3

    def test_none_initiation_efficiency_round_trip(self) -> None:
        original = _make_tis()
        df = tis_to_dataframe([original])
        restored = dataframe_to_tis(df)[0]
        assert restored.expression["HEK293"].initiation_efficiency is None


class TestGenesToDataframe:
    """Tests for genes_to_dataframe."""

    def test_basic_gene_serialization(self) -> None:
        gene = Gene(
            gene_name="BRCA1",
            gene_id="ENSG00000001",
            canonical_transcript_id="ENST00000001",
            canonical_protein="MVLSPADKTNVK",
            gene_annotations={"summary": {"score": 0.8}},
        )
        df = genes_to_dataframe([gene])
        assert len(df) == 1
        assert df["gene_name"].iloc[0] == "BRCA1"
        assert df["gene_id"].iloc[0] == "ENSG00000001"
        assert df["canonical_transcript_id"].iloc[0] == "ENST00000001"
        assert df["canonical_protein"].iloc[0] == "MVLSPADKTNVK"
        assert "summary_score" in df.columns
        assert df["summary_score"].iloc[0] == 0.8
