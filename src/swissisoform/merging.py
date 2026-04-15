"""Cross-cell-line merging and canonical pairing for TIS sites.

Provides utilities to combine filtered TIS DataFrames from multiple cell lines
and pair alternative TIS sites with their canonical counterparts for
downstream comparison.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def combine_cell_lines(filtered_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate filtered TIS DataFrames from multiple cell lines.

    Each input DataFrame should contain a ``Sample`` column identifying
    the cell line of origin.

    Args:
        filtered_dfs: List of per-cell-line filtered DataFrames.

    Returns:
        Combined DataFrame with all rows, index reset.
    """
    if not filtered_dfs:
        logger.warning("No DataFrames provided to combine_cell_lines")
        return pd.DataFrame()

    combined = pd.concat(filtered_dfs, ignore_index=True)
    samples = combined["Sample"].nunique() if "Sample" in combined.columns else "unknown"
    logger.info(
        "Combined %d cell lines: %d total TIS sites",
        samples if isinstance(samples, int) else 0,
        len(combined),
    )
    return combined


def pair_canonical_isoforms(
    df: pd.DataFrame,
    id_columns: list[str] | None = None,
    canonical_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Pair alternative TIS sites with their canonical counterparts.

    Separates Annotated (canonical) from non-Annotated (alternative) TIS sites,
    then merges so each alternative row carries its canonical counterpart's
    values for comparison. Computes start-position delta and fold-change metrics.

    Args:
        df: TIS DataFrame with a ``RecatTISType`` column. Must contain at least
            ``Start`` and ``TISCounts`` for downstream calculations.
        id_columns: Columns used to match alternatives to their canonical TIS.
            Defaults to ``["Sample", "Tid"]``, filtered to columns present in *df*.
        canonical_columns: Columns to carry over from the canonical row, prefixed
            with ``Canonical``. Defaults to
            ``["Start", "StartCodon", "TISCounts", "NormTISCounts"]``, filtered to
            columns present in *df*.

    Returns:
        DataFrame containing only alternative (non-Annotated) rows, enriched with
        ``Canonical*`` columns and computed metrics:
        ``StartRelativeToCanonical``, ``DistanceToCanonical``,
        ``FoldChangeFromCanonical``, ``LogFoldChangeFromCanonical``.
    """
    if id_columns is None:
        id_columns = ["Sample", "Tid"]
    if canonical_columns is None:
        canonical_columns = ["Start", "StartCodon", "TISCounts", "NormTISCounts"]

    df = df.copy()

    # Filter id_columns and canonical_columns to those actually present
    id_columns = [col for col in id_columns if col in df.columns]
    canonical_columns = [col for col in canonical_columns if col in df.columns]

    # Separate canonical (Annotated) from alternative TIS
    alternative_subset = df[~df["RecatTISType"].isin(["Annotated"])]
    canonical_subset = (
        df[df["RecatTISType"].isin(["Annotated"])]
        .drop_duplicates(subset=id_columns)
        .set_index(id_columns)[canonical_columns]
        .add_prefix("Canonical")
    )

    logger.info(
        "Pairing %d alternative TIS with %d canonical entries",
        len(alternative_subset),
        len(canonical_subset),
    )

    # Merge alternatives with their canonical match
    paired_df = alternative_subset.merge(
        canonical_subset, left_on=id_columns, right_index=True
    )

    # Compute downstream metrics
    if "CanonicalStart" in paired_df.columns and "Start" in paired_df.columns:
        paired_df = paired_df.assign(
            StartRelativeToCanonical=lambda x: x["CanonicalStart"] - x["Start"],
            DistanceToCanonical=lambda x: x["StartRelativeToCanonical"].abs(),
        )

    if "CanonicalTISCounts" in paired_df.columns and "TISCounts" in paired_df.columns:
        paired_df = paired_df.assign(
            FoldChangeFromCanonical=lambda x: (x["TISCounts"] + 1)
            / (x["CanonicalTISCounts"] + 1),
            LogFoldChangeFromCanonical=lambda x: np.log2(x["FoldChangeFromCanonical"]),
        )

    logger.info("Paired result: %d rows", len(paired_df))
    return paired_df.reset_index(drop=True)
