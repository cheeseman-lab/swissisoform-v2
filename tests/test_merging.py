"""Tests for cross-cell-line merging and canonical pairing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swissisoform.merging import combine_cell_lines, pair_canonical_isoforms

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample_df(sample: str, n_rows: int = 3) -> pd.DataFrame:
    """Build a minimal TIS DataFrame for one sample."""
    return pd.DataFrame(
        {
            "Sample": [sample] * n_rows,
            "Tid": [f"ENST0000000000{i}.1" for i in range(1, n_rows + 1)],
            "Start": [100 + i * 10 for i in range(n_rows)],
            "StartCodon": ["ATG"] * n_rows,
            "TISCounts": [50 + i * 5 for i in range(n_rows)],
            "NormTISCounts": [1.0 + i * 0.5 for i in range(n_rows)],
            "RecatTISType": ["Annotated", "Extended", "Truncated"][:n_rows],
        }
    )


def _make_paired_df() -> pd.DataFrame:
    """Build a DataFrame with both Annotated and alternative TIS on the same transcript."""
    return pd.DataFrame(
        {
            "Sample": ["HeLa"] * 4,
            "Tid": ["TX1", "TX1", "TX2", "TX2"],
            "Start": [100, 80, 200, 250],
            "StartCodon": ["ATG", "CTG", "ATG", "ATG"],
            "TISCounts": [100, 30, 200, 10],
            "NormTISCounts": [5.0, 1.5, 10.0, 0.5],
            "RecatTISType": ["Annotated", "Extended", "Annotated", "Truncated"],
        }
    )


# ---------------------------------------------------------------------------
# combine_cell_lines tests
# ---------------------------------------------------------------------------


class TestCombineCellLines:
    """Tests for combine_cell_lines."""

    def test_combines_two_samples(self) -> None:
        df1 = _make_sample_df("HeLa")
        df2 = _make_sample_df("K562")
        result = combine_cell_lines([df1, df2])
        assert len(result) == 6
        assert set(result["Sample"].unique()) == {"HeLa", "K562"}

    def test_preserves_sample_column(self) -> None:
        df1 = _make_sample_df("HeLa", n_rows=2)
        result = combine_cell_lines([df1])
        assert "Sample" in result.columns
        assert list(result["Sample"]) == ["HeLa", "HeLa"]

    def test_single_dataframe(self) -> None:
        df1 = _make_sample_df("U2OS")
        result = combine_cell_lines([df1])
        assert len(result) == 3

    def test_empty_list(self) -> None:
        result = combine_cell_lines([])
        assert len(result) == 0


# ---------------------------------------------------------------------------
# pair_canonical_isoforms tests
# ---------------------------------------------------------------------------


class TestPairCanonicalIsoforms:
    """Tests for pair_canonical_isoforms."""

    def test_only_returns_non_annotated_rows(self) -> None:
        df = _make_paired_df()
        result = pair_canonical_isoforms(df)
        assert all(result["RecatTISType"] != "Annotated")

    def test_has_canonical_columns(self) -> None:
        df = _make_paired_df()
        result = pair_canonical_isoforms(df)
        assert "CanonicalStart" in result.columns
        assert "CanonicalTISCounts" in result.columns
        assert "CanonicalNormTISCounts" in result.columns

    def test_computes_log_fold_change(self) -> None:
        df = _make_paired_df()
        result = pair_canonical_isoforms(df)
        assert "LogFoldChangeFromCanonical" in result.columns
        assert "FoldChangeFromCanonical" in result.columns
        assert "StartRelativeToCanonical" in result.columns
        assert "DistanceToCanonical" in result.columns
        # All values should be finite
        assert result["LogFoldChangeFromCanonical"].notna().all()

    def test_fold_change_values(self) -> None:
        """Verify fold-change computation for a known case."""
        df = _make_paired_df()
        result = pair_canonical_isoforms(df)
        # TX1 alternative: TISCounts=30, CanonicalTISCounts=100
        tx1_row = result[result["Tid"] == "TX1"].iloc[0]
        expected_fc = (30 + 1) / (100 + 1)
        assert tx1_row["FoldChangeFromCanonical"] == pytest.approx(expected_fc)
        assert tx1_row["LogFoldChangeFromCanonical"] == pytest.approx(np.log2(expected_fc))

    def test_start_relative_to_canonical(self) -> None:
        df = _make_paired_df()
        result = pair_canonical_isoforms(df)
        # TX1: CanonicalStart=100, AlternativeStart=80 => 100-80=20
        tx1_row = result[result["Tid"] == "TX1"].iloc[0]
        assert tx1_row["StartRelativeToCanonical"] == 20
        assert tx1_row["DistanceToCanonical"] == 20

    def test_missing_canonical_drops_alternative(self) -> None:
        """Alternatives without a matching canonical should not appear in output."""
        df = pd.DataFrame(
            {
                "Sample": ["HeLa", "HeLa"],
                "Tid": ["TX1", "TX2"],
                "Start": [100, 200],
                "StartCodon": ["ATG", "CTG"],
                "TISCounts": [50, 30],
                "NormTISCounts": [2.0, 1.0],
                "RecatTISType": ["Annotated", "Extended"],
            }
        )
        result = pair_canonical_isoforms(df)
        # TX2 alternative has no canonical match, so it should be dropped
        assert len(result) == 0

    def test_filters_id_columns_to_existing(self) -> None:
        """If Sample column is missing, id_columns should adapt."""
        df = pd.DataFrame(
            {
                "Tid": ["TX1", "TX1"],
                "Start": [100, 80],
                "StartCodon": ["ATG", "CTG"],
                "TISCounts": [100, 30],
                "NormTISCounts": [5.0, 1.5],
                "RecatTISType": ["Annotated", "Extended"],
            }
        )
        result = pair_canonical_isoforms(df)
        assert len(result) == 1
        assert result.iloc[0]["CanonicalStart"] == 100
