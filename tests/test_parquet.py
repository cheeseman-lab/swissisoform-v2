"""Tests for Parquet serialization round-trip."""

from __future__ import annotations

import pandas as pd

from swissisoform.io.parquet import (
    dataframe_to_tis,
    genes_to_dataframe,
    paired_tis_dataframe,
    tis_to_dataframe,
)
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
        assert df["orf_type"].iloc[0] == "annotated"

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


# ─────────────────────────────────────────────────────────────────────────────
# paired_tis_dataframe — genomic ORF intervals for downstream GLM handoff
# ─────────────────────────────────────────────────────────────────────────────


def _gene_with_orf_exons(
    *,
    orf_type: ORFType,
    orf_exons: list[tuple[int, int]],
    canonical_orf_exons: list[tuple[int, int]],
    strand: str = "+",
    iso_protein: str = "MVLSPADKTNVK",
    canonical_protein: str = "MVLSPADKTNVKAAWGKVGAHAGEYG",
) -> Gene:
    """Synthetic Gene with a single TIS carrying explicit ORF exon intervals."""
    site = TranslationInitiationSite(
        tis_id=f"chr1:100:{strand}:ATG:ENST_TEST.1",
        gene_name="GENE",
        transcript_id="ENST_TEST.1",
        chrom="chr1",
        position=100,
        strand=strand,
        start_codon="ATG",
        orf_type=orf_type,
        aa_len=len(iso_protein),
        canonical_protein=canonical_protein,
        isoform_protein=iso_protein,
        diff_region=DifferentialRegion(sequence="X"),
        orf_exons=orf_exons,
        canonical_orf_exons=canonical_orf_exons,
        expression={
            "HeLa": CellLineExpression(raw_count=1, cpm=1.0, p_value=1.0),
        },
    )
    return Gene(
        gene_name="GENE",
        gene_id="ENSG_TEST",
        canonical_transcript_id="ENST_TEST.1",
        canonical_protein=canonical_protein,
        tis_sites=[site],
    )


class TestPairedDataframeGenomicIntervals:
    """Verify the GLM-handoff fields land in the paired dataframe correctly."""

    EXPECTED_COLS = (
        "orf_exons",
        "canonical_orf_exons",
        "unique_genomic_intervals",
        "shared_genomic_intervals",
    )

    def test_columns_present(self) -> None:
        gene = _gene_with_orf_exons(
            orf_type=ORFType.EXTENDED,
            orf_exons=[(100, 200), (300, 400)],
            canonical_orf_exons=[(150, 200), (300, 400)],
        )
        df = paired_tis_dataframe([gene])
        for c in self.EXPECTED_COLS:
            assert c in df.columns, f"missing column: {c}"

    def test_extension_unique_is_isoform_minus_canonical(self) -> None:
        r"""Extension: unique = isoform_orf \ canonical_orf (the added 5' region)."""
        gene = _gene_with_orf_exons(
            orf_type=ORFType.EXTENDED,
            orf_exons=[(100, 200), (300, 400)],  # isoform adds [100,150)
            canonical_orf_exons=[(150, 200), (300, 400)],
        )
        df = paired_tis_dataframe([gene])
        row = df.iloc[0]
        assert list(map(tuple, row["orf_exons"])) == [(100, 200), (300, 400)]
        assert list(map(tuple, row["canonical_orf_exons"])) == [(150, 200), (300, 400)]
        # Unique = the new extension exon segment [100, 150).
        assert list(map(tuple, row["unique_genomic_intervals"])) == [(100, 150)]
        # Shared = the body the two ORFs overlap on.
        assert list(map(tuple, row["shared_genomic_intervals"])) == [
            (150, 200),
            (300, 400),
        ]

    def test_truncation_unique_is_canonical_minus_isoform(self) -> None:
        r"""Truncation: unique = canonical_orf \ isoform_orf (the LOST N-terminus)."""
        gene = _gene_with_orf_exons(
            orf_type=ORFType.TRUNCATED,
            orf_exons=[(150, 200), (300, 400)],  # isoform starts later
            canonical_orf_exons=[(100, 200), (300, 400)],
        )
        df = paired_tis_dataframe([gene])
        row = df.iloc[0]
        # Unique = the LOST canonical N-terminus [100, 150).
        assert list(map(tuple, row["unique_genomic_intervals"])) == [(100, 150)]
        # Shared = everything the truncated isoform retains.
        assert list(map(tuple, row["shared_genomic_intervals"])) == [
            (150, 200),
            (300, 400),
        ]

    def test_minus_strand_intervals_emitted_in_plus_coords(self) -> None:
        """orf_exons are documented as 0-based half-open *plus-strand* — verify
        we don't accidentally strand-flip the coordinates on serialization.
        """
        gene = _gene_with_orf_exons(
            orf_type=ORFType.EXTENDED,
            orf_exons=[(1000, 1200), (2000, 2300)],
            canonical_orf_exons=[(1100, 1200), (2000, 2300)],
            strand="-",
        )
        df = paired_tis_dataframe([gene])
        row = df.iloc[0]
        assert row["strand"] == "-"
        # Coords ascend even though the mRNA is read 3'→5' on the plus strand.
        assert list(map(tuple, row["orf_exons"])) == [(1000, 1200), (2000, 2300)]
        # Unique = the minus-strand-mRNA-leading extension exon segment.
        assert list(map(tuple, row["unique_genomic_intervals"])) == [(1000, 1100)]

    def test_missing_skeleton_yields_empty_lists(self) -> None:
        """If either orf_exons or canonical_orf_exons is empty (no skeleton)
        we don't synthesize fake intervals — both derived sets are empty.
        """
        gene = _gene_with_orf_exons(
            orf_type=ORFType.EXTENDED,
            orf_exons=[],
            canonical_orf_exons=[],
        )
        df = paired_tis_dataframe([gene])
        row = df.iloc[0]
        assert list(row["orf_exons"]) == []
        assert list(row["canonical_orf_exons"]) == []
        assert list(row["unique_genomic_intervals"]) == []
        assert list(row["shared_genomic_intervals"]) == []

    def test_parquet_round_trip_preserves_interval_lists(self, tmp_path) -> None:
        """Write to parquet + read back — interval lists survive the trip."""
        gene = _gene_with_orf_exons(
            orf_type=ORFType.EXTENDED,
            orf_exons=[(100, 200), (300, 400)],
            canonical_orf_exons=[(150, 200), (300, 400)],
        )
        df = paired_tis_dataframe([gene])
        path = tmp_path / "paired.parquet"
        df.to_parquet(path, index=False)
        back = pd.read_parquet(path)
        for c in self.EXPECTED_COLS:
            assert c in back.columns
        # The round-tripped values should match the originals coordinate-wise.
        for c in self.EXPECTED_COLS:
            orig = [tuple(x) for x in df[c].iloc[0]]
            roundtripped = [tuple(x) for x in back[c].iloc[0]]
            assert orig == roundtripped, f"round-trip failed for {c}"
