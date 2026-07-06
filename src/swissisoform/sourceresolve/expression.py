"""Determine which transcripts are expressed (present) in a sample from long-read.

Long-read IsoQuant per-transcript counts collapse to one output — a set of present
transcript IDs (and a ``{transcript_id: count}`` map for abundance ranking). A
transcript "exists" in the sample if its abundance meets a threshold (long-read
default ``counts >= 3``). The resulting ID set is the first narrowing step of the
source-resolution cascade (drop unexpressed candidate transcripts from each TIS's
candidate set). See ``docs/plans/source_transcript_resolution.md``.

Short-read (salmon) support was removed: the unified cascade uses long-read only,
for both presence and abundance.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _bare_tid(name: str) -> str:
    """Reduce a GENCODE/IsoQuant target id to a bare versioned ENST id.

    GENCODE FASTA headers look like ``ENST00000123.4|ENSG...|...|GENE-201|...``;
    our ``Tid`` is the leading ``ENST00000123.4``. IsoQuant ``feature_id`` values
    are usually already bare but are normalized here so versioned/unversioned ids
    match ``filtered_df.Tid``.
    """
    return name.split("|")[0]


def load_isoquant_abundance(
    table_tsv: str | Path,
    id_col: str = "feature_id",
    value_col: str | None = None,
) -> dict[str, float]:
    """Read an IsoQuant per-transcript table into ``{transcript_id: abundance}``.

    Args:
        table_tsv: IsoQuant transcript counts/TPM TSV (e.g.
            ``*.transcript_counts.tsv`` / ``*.transcript_tpm.tsv``).
        id_col: Column holding the transcript id (IsoQuant uses ``feature_id``).
        value_col: Column holding the abundance value. When ``None``, the
            abundance is the **sum across all non-id columns** (IsoQuant grouped
            tables carry one column per replicate/sample), so a multi-replicate
            table is aggregated rather than silently reading only the first
            column.

    Returns:
        Mapping from bare transcript id to abundance.
    """
    df = pd.read_csv(table_tsv, sep="\t")
    ids = df[id_col].astype(str).map(_bare_tid)
    if value_col is not None:
        values = df[value_col].astype(float)
    else:
        value_cols = [c for c in df.columns if c != id_col]
        if not value_cols:
            raise ValueError(f"IsoQuant table {table_tsv} has no abundance columns")
        values = df[value_cols].astype(float).sum(axis=1)
    return dict(zip(ids, values))


def expressed_transcripts(abundance: dict[str, float], min_value: float = 1.0) -> set[str]:
    """Transcripts whose abundance meets the threshold = 'present in the sample'.

    Args:
        abundance: ``{transcript_id: value}`` from :func:`load_isoquant_abundance`.
        min_value: Inclusive presence threshold (long-read counts typically 3).

    Returns:
        The set of present transcript ids.
    """
    return {t for t, v in abundance.items() if v >= min_value}
