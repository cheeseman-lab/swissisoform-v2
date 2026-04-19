"""Combine per-sample filtered TIS tables into one deduplicated table.

A TIS is uniquely identified by ``(Symbol, Tid, GenomePos, StartCodon)`` —
same transcript + same genomic position + same start codon is the same
ORF regardless of which cell line called it.  This module collapses the
per-sample long-form output of :func:`swissisoform.pipeline.run_sample`
into a wide, deduplicated table where:

- **Shared fields** (Symbol, Gid, Tid, GenomePos, StartCodon, TisType,
  RecatTISType, AASeq, AALen, Start) appear once.
- **Per-sample metrics** (TISCounts, NormTISCounts, TISPvalue,
  RiboPvalue, FisherQvalue, Imputed) become wide columns named
  ``{sample}_{metric}``.
- **Inclusion flags** ``present_{sample}`` (bool) plus a ``samples`` list
  column record which samples called each TIS.

This lets the downstream annotation pipeline run **once per unique TIS**
instead of once per (TIS, sample) pair, which matters for modules that
are expensive (clinical API fetch, DIAMOND alignment, PepQuery lookup).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import pandas as pd

logger = logging.getLogger(__name__)

# Fields that identify a TIS uniquely across samples
DEDUP_KEY: tuple[str, ...] = ("Symbol", "Tid", "GenomePos", "StartCodon")

# Fields that should be identical across samples for a given dedup key.
# We copy these once from the first sample's row; conflicts raise an error.
SHARED_FIELDS: tuple[str, ...] = (
    "Symbol",
    "Gid",
    "Tid",
    "GenomePos",
    "StartCodon",
    "TisType",
    "RecatTISType",
    "AASeq",
    "AALen",
    "Start",
)

# Per-sample metrics pivoted into {sample}_{metric} wide columns
PER_SAMPLE_METRICS: tuple[str, ...] = (
    "TISCounts",
    "NormTISCounts",
    "TISPvalue",
    "RiboPvalue",
    "FisherQvalue",
    "Imputed",
    "GeneRNASeqCounts",
    "TotalRNASeqCounts",
)


def combine_filtered_samples(
    per_sample: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Dedupe TIS across samples, carrying per-sample metrics as wide columns.

    Args:
        per_sample: Mapping from sample name (cell line) to that sample's
            filtered DataFrame (output of :func:`run_sample`).  Each frame
            must carry the ``DEDUP_KEY`` and ``SHARED_FIELDS`` columns.

    Returns:
        One row per unique ``(Symbol, Tid, GenomePos, StartCodon)``.
        ``SHARED_FIELDS`` appear once; per-sample metrics are pivoted to
        ``{sample}_{metric}`` columns (NaN/None where the TIS wasn't called
        in that sample).  Bool ``present_{sample}`` flags indicate
        inclusion; a ``samples`` list column lists which samples called
        each TIS.

    Raises:
        ValueError: If shared fields (AASeq, TisType, etc.) disagree
            across samples for the same dedup key — indicates an upstream
            invariant violation.
    """
    if not per_sample:
        raise ValueError("per_sample is empty — need at least one sample")

    frames = []
    for sample, df in per_sample.items():
        if df.empty:
            logger.warning("Sample %s has no rows — skipping", sample)
            continue
        missing = [c for c in DEDUP_KEY if c not in df.columns]
        if missing:
            raise ValueError(f"Sample {sample!r} missing dedup key columns: {missing}")
        df = df.copy()
        df["_sample"] = sample
        frames.append(df)

    if not frames:
        raise ValueError("All samples were empty")

    long = pd.concat(frames, ignore_index=True)
    samples = [s for s in per_sample if s in long["_sample"].unique()]

    # Sanity-check: shared fields must agree across samples for each key
    _verify_shared_fields(long)

    # Take shared fields from the first row per key
    shared = long.drop_duplicates(subset=list(DEDUP_KEY), keep="first")[
        list(SHARED_FIELDS)
    ].reset_index(drop=True)

    # Pivot per-sample metrics to wide columns
    metric_cols = [m for m in PER_SAMPLE_METRICS if m in long.columns]
    wide = long.pivot_table(
        index=list(DEDUP_KEY),
        columns="_sample",
        values=metric_cols,
        aggfunc="first",
    )
    # Flatten MultiIndex columns: ("TISCounts", "HeLa") → "HeLa_TISCounts"
    wide.columns = [f"{sample}_{metric}" for metric, sample in wide.columns]
    wide = wide.reset_index()

    combined = shared.merge(wide, on=list(DEDUP_KEY), how="left")

    # Presence flags + samples list
    for sample in samples:
        col = f"{sample}_TISCounts"
        combined[f"present_{sample}"] = combined[col].notna() if col in combined.columns else False

    present_cols = [f"present_{s}" for s in samples]
    combined["samples"] = combined[present_cols].apply(
        lambda row: [s for s in samples if row[f"present_{s}"]],
        axis=1,
    )
    combined["n_samples"] = combined[present_cols].sum(axis=1).astype(int)

    logger.info(
        "combine_filtered_samples: %d samples → %d rows (long) → %d unique TIS (wide)",
        len(samples),
        len(long),
        len(combined),
    )
    return combined


def _verify_shared_fields(long: pd.DataFrame) -> None:
    """Raise if any dedup-key group has inconsistent shared fields.

    AASeq, TisType, AALen etc. must match across samples for the same
    ``(Symbol, Tid, GenomePos, StartCodon)`` — they are functions of the
    genome, not the sample.  A mismatch indicates upstream corruption
    (e.g. different GTFs, different imputation rules) that would silently
    break annotation.
    """
    check_fields = [
        f
        for f in ("AASeq", "AALen", "TisType", "RecatTISType", "Start", "Gid")
        if f in long.columns
    ]
    if not check_fields:
        return
    nunique = long.groupby(list(DEDUP_KEY))[check_fields].nunique()
    bad = nunique[(nunique > 1).any(axis=1)]
    if not bad.empty:
        examples = bad.head(3).to_dict(orient="index")
        raise ValueError(
            f"{len(bad)} TIS have inconsistent shared fields across samples. Examples: {examples}"
        )
