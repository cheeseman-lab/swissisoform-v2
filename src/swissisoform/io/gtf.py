"""GTF annotation loader for GENCODE gene annotations.

Parses a GENCODE GTF file and extracts transcript-level annotations including
gene/transcript IDs, biotypes, support levels, and MANE_Select status.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def load_transcript_annotations(
    gtf_path: str,
    feature_type: str = "transcript",
) -> pd.DataFrame:
    """Load and parse GTF annotations into a DataFrame.

    Reads a GENCODE-format GTF file line by line, filtering for the specified
    feature type. Extracts key attributes from the attributes column using
    regex matching.

    Args:
        gtf_path: Path to the genome annotation GTF file.
        feature_type: GTF feature type to retain (default: ``"transcript"``).

    Returns:
        DataFrame with columns: chromosome, source, feature_type, start, end,
        strand, gene_id, gene_type, transcript_id, transcript_type,
        transcript_support_level, MANE_Select.
    """
    features_list: list[dict] = []
    feature_tab = f"\t{feature_type}\t"

    logger.info("Loading GTF annotations from %s (feature_type=%s)", gtf_path, feature_type)

    with open(gtf_path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            if feature_tab not in line:
                continue

            fields = line.strip().split("\t")
            if len(fields) != 9:
                continue

            features_list.append(
                {
                    "chromosome": fields[0],
                    "source": fields[1],
                    "feature_type": fields[2],
                    "start": int(fields[3]),
                    "end": int(fields[4]),
                    "strand": fields[6],
                    "attributes": fields[8],
                }
            )

    if not features_list:
        logger.warning("No features of type '%s' found in %s", feature_type, gtf_path)
        return pd.DataFrame(
            columns=[
                "chromosome",
                "source",
                "feature_type",
                "start",
                "end",
                "strand",
                "gene_id",
                "gene_type",
                "transcript_id",
                "transcript_type",
                "transcript_support_level",
                "MANE_Select",
            ]
        )

    annotations = pd.DataFrame(features_list)

    # Extract quoted attribute values
    extracted_columns = [
        "gene_id",
        "gene_type",
        "transcript_id",
        "transcript_type",
        "transcript_support_level",
    ]
    for col in extracted_columns:
        annotations[col] = annotations["attributes"].str.extract(f'{col} "([^"]*)"')

    # MANE_Select is a tag (present or absent)
    annotations["MANE_Select"] = annotations["attributes"].str.contains("MANE_Select")

    # Select and order output columns
    output_columns = (
        [
            "chromosome",
            "source",
            "feature_type",
            "start",
            "end",
            "strand",
        ]
        + extracted_columns
        + ["MANE_Select"]
    )

    logger.info("Loaded %d %s annotations", len(annotations), feature_type)
    return annotations[output_columns].reset_index(drop=True)


def load_cds_features(gtf_path: str) -> pd.DataFrame:
    """Load CDS features from a GTF file.

    Parses CDS and start_codon features for building genomic-to-coding
    position maps used by the variant consequence validator.

    Args:
        gtf_path: Path to a GENCODE-format GTF file.

    Returns:
        DataFrame with columns: chromosome, start, end, strand, gene_id,
        transcript_id, feature_type.
    """
    output_columns = [
        "chromosome",
        "start",
        "end",
        "strand",
        "gene_id",
        "transcript_id",
        "feature_type",
    ]
    features_list: list[dict] = []

    logger.info("Loading CDS features from %s", gtf_path)

    with open(gtf_path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue

            fields = line.strip().split("\t")
            if len(fields) != 9:
                continue

            feat_type = fields[2]
            if feat_type not in ("CDS", "start_codon"):
                continue

            attrs = fields[8]

            # Extract gene_id and transcript_id from attributes
            gene_id_match = _extract_attr(attrs, "gene_id")
            transcript_id_match = _extract_attr(attrs, "transcript_id")

            features_list.append(
                {
                    "chromosome": fields[0],
                    "start": int(fields[3]),
                    "end": int(fields[4]),
                    "strand": fields[6],
                    "gene_id": gene_id_match,
                    "transcript_id": transcript_id_match,
                    "feature_type": feat_type,
                }
            )

    if not features_list:
        logger.warning("No CDS/start_codon features found in %s", gtf_path)
        return pd.DataFrame(columns=output_columns)

    df = pd.DataFrame(features_list)
    logger.info("Loaded %d CDS/start_codon features", len(df))
    return df[output_columns].reset_index(drop=True)


def _extract_attr(attributes: str, key: str) -> str:
    """Extract a quoted attribute value from a GTF attributes string.

    Args:
        attributes: The 9th column of a GTF line.
        key: Attribute name (e.g. ``"gene_id"``).

    Returns:
        The extracted value, or empty string if not found.
    """
    import re

    match = re.search(rf'{key} "([^"]*)"', attributes)
    return match.group(1) if match else ""
