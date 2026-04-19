"""TIS filtering pipeline for Ribo-TISH predictions.

Provides RPM normalization, reference transcript identification, and a
multi-step filtering pipeline that produces high-confidence TIS sites.
Thresholds and ordering match the upstream reference filter this port is
audited against.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _append_tag(
    series: pd.Series,
    tag: str,
    target_mask: pd.Series,
    separator: str = "|",
) -> pd.Series:
    """Append a drop-reason tag to a Series, pipe-separating multiple reasons.

    Args:
        series: The DropReason series (may contain None values).
        tag: The tag string to append.
        target_mask: Boolean mask indicating which rows to tag.
        separator: Separator between multiple tags.

    Returns:
        Updated Series with tags appended where target_mask is True.
    """
    series = series.copy()
    null_target = target_mask & series.isnull()
    string_target = target_mask & ~series.isnull()
    series.loc[null_target] = tag
    series.loc[string_target] = series.loc[string_target] + separator + tag
    return series


def normalize_tis_counts(
    df: pd.DataFrame,
    total_reads: float | None = None,
    count_col: str = "TISCounts",
    output_col: str = "NormTISCounts",
    scale_factor: float = 1e6,
) -> pd.DataFrame:
    """Normalize TIS counts using RPM (reads per million) transformation.

    Args:
        df: DataFrame with a count column to normalize.
        total_reads: Total read count denominator. If None, uses the sum
            of unique TIS counts (deduplicated by Chromosome+Locus+Strand).
        count_col: Name of the raw count column.
        output_col: Name of the output normalized column.
        scale_factor: Multiplication factor (default 1e6 for RPM).

    Returns:
        Copy of the DataFrame with the normalized column added.
    """
    df = df.copy()
    if total_reads is None:
        total_reads = df.drop_duplicates(["Chromosome", "Locus", "Strand"])[count_col].sum()
    df[output_col] = (df[count_col] / total_reads) * scale_factor
    return df


def identify_reference_transcripts(
    df: pd.DataFrame,
    transcript_support_levels: list[str] | None = None,
    min_tis_counts: int = 0,
    count_col: str = "TISCounts",
    tis_enrichment_max_p: float = 1.0,
    frame_test_max_p: float = 1.0,
    combined_test_max_q: float = 1.0,
) -> list[str]:
    """Identify reference transcripts for downstream TIS filtering.

    Selects transcripts that are MANE_Select OR have a transcript support level
    in the allowed list. Among those, filters by minimum counts and statistical
    significance thresholds.

    Args:
        df: TIS DataFrame with MANE_Select, transcript_support_level, Tid,
            TISPvalue, RiboPvalue, FisherQvalue columns.
        transcript_support_levels: Allowed TSL values (e.g. ["1", "2", "3"]).
            Defaults to ["1", "2", "3"].
        min_tis_counts: Minimum count value for a TIS to qualify its transcript.
        count_col: Column name for count-based filtering.
        tis_enrichment_max_p: Maximum TISPvalue threshold.
        frame_test_max_p: Maximum RiboPvalue threshold.
        combined_test_max_q: Maximum FisherQvalue threshold.

    Returns:
        List of unique Tid values for reference transcripts.
    """
    if transcript_support_levels is None:
        transcript_support_levels = ["1", "2", "3"]

    logger.info("Selecting reference transcripts from %d TIS sites", len(df))

    # Step 1: MANE_Select or allowed TSL
    mane_mask = df["MANE_Select"] == True  # noqa: E712
    tsl_mask = df["transcript_support_level"].isin(transcript_support_levels)
    tis_to_keep = df[mane_mask | tsl_mask]

    logger.info(
        "Kept %d TIS from MANE_Select or TSL %s transcripts",
        len(tis_to_keep),
        "/".join(transcript_support_levels),
    )

    # Step 2: Filter by count support
    support_mask = tis_to_keep[count_col] >= min_tis_counts
    tis_to_keep = tis_to_keep[support_mask]

    # Step 3: Filter by significance
    enrichment_mask = tis_to_keep["TISPvalue"] <= tis_enrichment_max_p
    frame_mask = tis_to_keep["RiboPvalue"] <= frame_test_max_p
    combined_q_mask = tis_to_keep["FisherQvalue"] <= combined_test_max_q
    significance_mask = enrichment_mask & frame_mask & combined_q_mask
    tis_to_keep = tis_to_keep[significance_mask]

    tids = tis_to_keep["Tid"].unique().tolist()
    logger.info("Identified %d reference transcript IDs", len(tids))
    return tids


def filter_tis(
    df: pd.DataFrame,
    transcript_support_levels: list[str] | None = None,
    min_normalized_counts: float = 0.1,
    count_col: str = "NormTISCounts",
    tis_enrichment_max_p: float = 0.01,
    frame_test_max_p: float = 0.01,
    combined_test_max_q: float = 0.05,
    tis_distance_buffer: int = 30,
    exempt_annotated: bool = True,
    return_dropped: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Filter TIS sites through a 5-step pipeline.

    Steps:
        1. Select reference transcripts (MANE_Select or TSL 1-3).
        2. Filter by normalized read counts.
        3. Filter by statistical significance.
        4. Keep annotated start sites for surviving transcripts.
        5. Distance deduplication within each transcript.

    Each dropped TIS receives a DropReason annotation. Multiple reasons are
    pipe-separated (e.g. "LowReadcounts|NotSignificant").

    Args:
        df: TIS DataFrame, already normalized (must have count_col column).
        transcript_support_levels: Allowed TSL values for reference transcripts.
        min_normalized_counts: Minimum normalized count threshold.
        count_col: Column with normalized counts for filtering.
        tis_enrichment_max_p: Maximum TISPvalue.
        frame_test_max_p: Maximum RiboPvalue.
        combined_test_max_q: Maximum FisherQvalue.
        tis_distance_buffer: Minimum distance (nt) between kept TIS on the
            same transcript.
        exempt_annotated: If True (default), Annotated TIS from reference
            transcripts skip the count and significance drops — they still
            have to come from a reference transcript but are retained
            regardless of re-detection p-values.  This preserves canonical
            reference sequences downstream.  Set False for upstream-
            reference-faithful behavior.
        return_dropped: If True, return (filtered, dropped) tuple.

    Returns:
        Filtered DataFrame, or (filtered, dropped) tuple if return_dropped=True.
    """
    if transcript_support_levels is None:
        transcript_support_levels = ["1", "2", "3"]

    # Step 1: Identify reference transcripts with permissive thresholds
    reference_tids = identify_reference_transcripts(
        df,
        transcript_support_levels=transcript_support_levels,
        min_tis_counts=0,
        count_col="TISCounts",
        tis_enrichment_max_p=1.0,
        frame_test_max_p=1.0,
        combined_test_max_q=1.0,
    )

    tis_df = df.copy()

    # Add GenomeStart column
    def _genome_start(genome_pos: str) -> str:
        strand = genome_pos.split(":")[-1]
        if strand == "+":
            return genome_pos.split("-")[0]
        else:
            chrom = genome_pos.split(":")[0]
            end = genome_pos.split(":")[1].split("-")[1]
            return f"{chrom}:{end}"

    tis_df["GenomeStart"] = tis_df["GenomePos"].apply(_genome_start)
    tis_df["DropReason"] = None

    logger.info("Identified %d reference transcript IDs", len(reference_tids))

    # Step 2: Mark non-reference transcripts
    reference_mask = tis_df["Tid"].isin(reference_tids)
    tis_df["DropReason"] = _append_tag(
        tis_df["DropReason"], "NotReferenceTranscript", target_mask=~reference_mask
    )

    # When exempt_annotated, Annotated rows from reference transcripts are
    # protected from count and significance filters — they are reference
    # material, not hypotheses we test.  They must still pass Step 2
    # (reference transcript).
    if exempt_annotated:
        exempt_mask = reference_mask & tis_df["TisType"].astype(str).str.startswith("Annotated")
    else:
        exempt_mask = pd.Series(False, index=tis_df.index)

    # Step 3: Mark low-count TIS (exempt Annotated when configured)
    support_mask = tis_df[count_col] >= min_normalized_counts
    tis_df["DropReason"] = _append_tag(
        tis_df["DropReason"], "LowReadcounts", target_mask=~support_mask & ~exempt_mask
    )

    # Step 4: Mark non-significant TIS (exempt Annotated when configured)
    enrichment_mask = tis_df["TISPvalue"] <= tis_enrichment_max_p
    frame_mask = tis_df["RiboPvalue"] <= frame_test_max_p
    combined_q_mask = tis_df["FisherQvalue"] <= combined_test_max_q
    significance_mask = enrichment_mask & frame_mask & combined_q_mask
    tis_df["DropReason"] = _append_tag(
        tis_df["DropReason"],
        "NotSignificant",
        target_mask=~significance_mask & ~exempt_mask,
    )

    # Step 5: Distance deduplication on surviving TIS
    surviving = tis_df[tis_df["DropReason"].isnull()]
    tis_idx_to_keep: list[int] = []

    logger.info(
        "Distance deduplication on %d surviving TIS (buffer=%d nt)",
        len(surviving),
        tis_distance_buffer,
    )

    for tid, idxs in surviving.groupby("Tid").groups.items():
        subset = surviving.loc[idxs].sort_values(count_col, ascending=False)
        while len(subset) > 0:
            # Prefer annotated TIS first
            annotated_mask = subset["TisType"].str.contains("Annotated")
            if annotated_mask.sum() > 0:
                selected_idx = subset[annotated_mask].index[0]
            else:
                selected_idx = subset.index[0]

            # Identify TIS within distance buffer downstream of selected
            selected_start = subset.loc[selected_idx, "Start"]
            downstream_mask = (subset["Start"] >= selected_start) & (
                subset["Start"] <= selected_start + tis_distance_buffer
            )

            tis_idx_to_keep.append(selected_idx)
            subset = subset[~downstream_mask]

    # Mark excluded surviving TIS as UpstreamTIS
    downstream_exclusion_idxs = surviving[~surviving.index.isin(tis_idx_to_keep)].index.tolist()
    tis_df["DropReason"] = _append_tag(
        tis_df["DropReason"],
        "UpstreamTIS",
        target_mask=tis_df.index.isin(downstream_exclusion_idxs),
    )

    # Split into filtered and dropped
    inclusion_mask = tis_df["DropReason"].isnull()
    filtered_df = tis_df[inclusion_mask].reset_index(drop=True)
    dropped_df = tis_df[~inclusion_mask].reset_index(drop=True)

    logger.info("Kept %d TIS, dropped %d TIS", len(filtered_df), len(dropped_df))

    if return_dropped:
        return filtered_df, dropped_df
    return filtered_df
