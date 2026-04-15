"""I/O utilities for reading Ribo-TISH predictions, GTF annotations, and Parquet serialization."""

from swissisoform.io.gtf import load_transcript_annotations
from swissisoform.io.parquet import dataframe_to_tis, genes_to_dataframe, tis_to_dataframe
from swissisoform.io.ribotish import (
    load_ribotish_predictions,
    parse_genome_pos,
    recategorize_tis_type,
)

__all__ = [
    "dataframe_to_tis",
    "genes_to_dataframe",
    "load_transcript_annotations",
    "load_ribotish_predictions",
    "parse_genome_pos",
    "recategorize_tis_type",
    "tis_to_dataframe",
]
