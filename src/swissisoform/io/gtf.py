"""GTF annotation loader for GENCODE gene annotations.

Parses a GENCODE GTF file and extracts transcript-level annotations including
gene/transcript IDs, biotypes, support levels, and MANE_Select status.
"""

from __future__ import annotations

import logging

import pandas as pd

from swissisoform.models import TranscriptCoordinates

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


def load_exon_skeletons(gtf_path: str) -> dict[str, TranscriptCoordinates]:
    """Load per-transcript exon skeletons from a GTF file.

    Scans exon + CDS + start_codon features in a single pass and builds one
    :class:`TranscriptCoordinates` per transcript_id. Exons include the full
    transcript structure (5'UTR + CDS + 3'UTR) so that downstream walkers can
    handle ORFs that initiate upstream of the canonical ATG (N-terminal
    extensions, uORFs).

    GTF coordinates are 1-based inclusive; all output coordinates are
    converted to 0-based half-open.

    Args:
        gtf_path: Path to a GENCODE-format GTF file.

    Returns:
        Dict mapping ``transcript_id`` → :class:`TranscriptCoordinates`. Exons
        are sorted ascending by genomic start (transcript order for ``+``
        strand, reverse transcript order for ``-`` strand).
    """
    skeletons: dict[str, dict] = {}

    logger.info("Loading exon skeletons from %s", gtf_path)

    with open(gtf_path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue

            fields = line.strip().split("\t")
            if len(fields) != 9:
                continue

            feat_type = fields[2]
            if feat_type not in ("exon", "CDS", "start_codon", "stop_codon"):
                continue

            chrom = fields[0]
            start_1b = int(fields[3])
            end_1b = int(fields[4])
            strand = fields[6]
            start_0b = start_1b - 1
            end_0b = end_1b  # GTF inclusive → half-open exclusive

            tid = _extract_attr(fields[8], "transcript_id")
            if not tid:
                continue

            entry = skeletons.setdefault(
                tid,
                {
                    "chrom": chrom,
                    "strand": strand,
                    "exons": [],
                    "start_codon_positions": [],
                    "stop_codon_positions": [],
                    "cds_min": None,
                    "cds_max": None,
                },
            )

            if feat_type == "exon":
                entry["exons"].append((start_0b, end_0b))
            elif feat_type == "start_codon":
                entry["start_codon_positions"].append((start_0b, end_0b))
            elif feat_type == "stop_codon":
                entry["stop_codon_positions"].append((start_0b, end_0b))
            elif feat_type == "CDS":
                if entry["cds_min"] is None or start_0b < entry["cds_min"]:
                    entry["cds_min"] = start_0b
                if entry["cds_max"] is None or end_0b > entry["cds_max"]:
                    entry["cds_max"] = end_0b

    coords_by_tid: dict[str, TranscriptCoordinates] = {}
    for tid, entry in skeletons.items():
        exons = sorted(entry["exons"])
        strand = entry["strand"]

        cds_start = _resolve_cds_start(
            strand=strand,
            start_codons=entry["start_codon_positions"],
            cds_min=entry["cds_min"],
            cds_max=entry["cds_max"],
        )
        cds_end = _resolve_cds_end(
            strand=strand,
            stop_codons=entry["stop_codon_positions"],
            cds_min=entry["cds_min"],
            cds_max=entry["cds_max"],
        )

        coords_by_tid[tid] = TranscriptCoordinates(
            transcript_id=tid,
            chrom=entry["chrom"],
            strand=strand,
            exons=exons,
            cds_start=cds_start,
            cds_end=cds_end,
        )

    logger.info("Loaded %d transcript skeletons", len(coords_by_tid))
    return coords_by_tid


def _resolve_cds_start(
    *,
    strand: str,
    start_codons: list[tuple[int, int]],
    cds_min: int | None,
    cds_max: int | None,
) -> int | None:
    """Return plus-strand 0-based position of the canonical ATG's first nt."""
    if start_codons:
        positions = [pos[0] for pos in start_codons]
        if strand == "-":
            # A of ATG on mRNA corresponds to the higher plus-strand coord
            return max(pos[1] for pos in start_codons)
        return min(positions)
    # Fall back to CDS extremes
    if cds_min is None or cds_max is None:
        return None
    return cds_max if strand == "-" else cds_min


def _resolve_cds_end(
    *,
    strand: str,
    stop_codons: list[tuple[int, int]],
    cds_min: int | None,
    cds_max: int | None,
) -> int | None:
    """Return plus-strand 0-based position delimiting the stop codon."""
    if stop_codons:
        if strand == "-":
            return min(pos[0] for pos in stop_codons)
        return max(pos[1] for pos in stop_codons)
    if cds_min is None or cds_max is None:
        return None
    return cds_min if strand == "-" else cds_max


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
