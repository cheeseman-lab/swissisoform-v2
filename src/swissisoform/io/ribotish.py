"""Ribo-TISH TSV reader and TIS annotation utilities.

Reads ``predict_all.txt`` output from Ribo-TISH, annotates genomic locus
information, and provides TIS-type recategorization.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def parse_genome_pos(genome_pos: str) -> tuple[str, int, int, str]:
    """Parse a Ribo-TISH GenomePos string into components.

    Args:
        genome_pos: Genome position string in the format
            ``"chr1:944693-959240:-"``.

    Returns:
        Tuple of (chromosome, start, end, strand).

    Raises:
        ValueError: If the string cannot be parsed.
    """
    try:
        chrom, coords, strand = genome_pos.rsplit(":", maxsplit=2)
        start_str, end_str = coords.split("-")
        return chrom, int(start_str), int(end_str), strand
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"Cannot parse GenomePos '{genome_pos}'; "
            "expected format 'chr:start-end:strand'"
        ) from exc


def load_ribotish_predictions(
    filepath: str | Path,
    gtf_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load a Ribo-TISH predict_all.txt TSV and annotate locus information.

    Reads the 21-column TSV, adds Chromosome, Strand, and Locus columns
    derived from the GenomePos field. For forward-strand sites the locus is
    the start of the genomic range; for reverse-strand sites it is the end.

    If *gtf_path* is provided, transcript annotations (MANE_Select,
    transcript_support_level) are merged via a left join on gene_id and
    transcript_id.

    Args:
        filepath: Path to a Ribo-TISH ``predict_all.txt`` file.
        gtf_path: Optional path to a GENCODE GTF for annotation merging.

    Returns:
        DataFrame with original columns plus Chromosome, Strand, Locus
        (and GTF columns if *gtf_path* was provided).
    """
    filepath = Path(filepath)
    logger.info("Loading Ribo-TISH predictions from %s", filepath)

    tis_df = pd.read_csv(filepath, sep="\t")

    if tis_df.empty:
        tis_df["Chromosome"] = pd.Series(dtype="str")
        tis_df["Strand"] = pd.Series(dtype="str")
        tis_df["Locus"] = pd.Series(dtype="int64")
        return tis_df

    # Annotate locus from GenomePos
    tis_df = _annotate_tis_locus(tis_df)

    # Optionally merge GTF annotations
    if gtf_path is not None:
        tis_df = _merge_gtf_annotations(tis_df, str(gtf_path))

    logger.info("Loaded %d TIS predictions", len(tis_df))
    return tis_df


def recategorize_tis_type(
    df: pd.DataFrame,
    original_column: str = "TisType",
    output_column: str = "RecatTISType",
) -> pd.DataFrame:
    """Map Ribo-TISH TisType to simplified categories.

    Categories:
        - **Annotated**: canonical annotated start site
        - **Truncated**: N-terminally truncated isoform
        - **Extended**: N-terminally extended isoform
        - **uORF**: upstream open reading frame (5'UTR)
        - **Other**: all remaining types (Novel, Internal, 3'UTR, etc.)

    Args:
        df: DataFrame with a TisType column.
        original_column: Name of the column containing raw TisType values.
        output_column: Name of the new recategorized column.

    Returns:
        Copy of the DataFrame with the new *output_column* added.
    """
    df = df.copy()
    df[output_column] = df[original_column].apply(_simplify_tis_type)
    return df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _simplify_tis_type(tis_type: str) -> str:
    """Reduce a compound TisType string to one of five categories."""
    if "Annotated" in tis_type:
        return "Annotated"
    if "Truncated" in tis_type:
        return "Truncated"
    if "5'UTR" in tis_type:
        return "uORF"
    if "Extended" in tis_type:
        return "Extended"
    return "Other"


def _annotate_tis_locus(
    tis_df: pd.DataFrame,
    genome_position_column: str = "GenomePos",
) -> pd.DataFrame:
    """Add Chromosome, Strand, and Locus columns from GenomePos.

    For forward-strand genes the TIS locus is the start of the genomic range;
    for reverse-strand genes it is the end.
    """
    tis_df = tis_df.copy()

    tis_df["Chromosome"] = tis_df[genome_position_column].str.split(":").str[0]
    tis_df["Strand"] = tis_df[genome_position_column].str.get(-1)

    forward_mask = tis_df["Strand"] == "+"
    reverse_mask = tis_df["Strand"] == "-"

    # Extract start (forward) or end (reverse) as integer locus
    coords = tis_df[genome_position_column].str.split(":").str[1]
    tis_df.loc[forward_mask, "Locus"] = (
        coords[forward_mask].str.split("-").str[0].astype(int)
    )
    tis_df.loc[reverse_mask, "Locus"] = (
        coords[reverse_mask].str.split("-").str[1].astype(int)
    )
    tis_df["Locus"] = tis_df["Locus"].astype(int)

    return tis_df


def _merge_gtf_annotations(tis_df: pd.DataFrame, gtf_path: str) -> pd.DataFrame:
    """Merge GTF transcript annotations onto a TIS DataFrame."""
    from swissisoform.io.gtf import load_transcript_annotations

    gtf_df = load_transcript_annotations(gtf_path)
    if gtf_df.empty:
        return tis_df

    tis_columns = tis_df.columns.tolist()

    tis_df = tis_df.merge(
        gtf_df,
        how="left",
        left_on=["Gid", "Tid"],
        right_on=["gene_id", "transcript_id"],
    )

    # Keep original columns plus selected GTF columns
    gtf_merge_cols = [
        c for c in ["MANE_Select", "transcript_support_level", "gene_type",
                     "transcript_type"]
        if c in tis_df.columns
    ]
    tis_df = tis_df[tis_columns + gtf_merge_cols]

    return tis_df
