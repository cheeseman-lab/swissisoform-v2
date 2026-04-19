"""RNA-seq count-file parsing for TIS normalization.

Reads HTSeq-count output files (``ENSG<version>\\tcount``), drops QC
indices (``__no_feature`` etc.), and provides helpers to aggregate
replicate counts per sample, matching the upstream reference filter.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_QC_INDICES = {
    "__no_feature",
    "__ambiguous",
    "__too_low_aQual",
    "__not_aligned",
    "__alignment_not_unique",
}


def read_rnaseq_counts(path: str | Path) -> pd.Series:
    """Load an HTSeq-count file as a ``Gid → counts`` Series.

    QC-index rows (``__no_feature`` etc.) are dropped.
    """
    df = pd.read_csv(path, sep="\t", header=None, names=["Gid", "counts"])
    counts = df.set_index("Gid")["counts"]
    return counts[~counts.index.isin(_QC_INDICES)]


def sum_replicate_counts(paths: list[str | Path]) -> pd.Series:
    """Sum per-gene counts across replicate files for one sample."""
    if not paths:
        raise ValueError("At least one count file is required")
    return (
        pd.concat([read_rnaseq_counts(p) for p in paths], axis=1)
        .fillna(0)
        .sum(axis=1)
        .rename("GeneRNASeqCounts")
    )


def load_sample_manifest(
    sample_manifest: str | Path,
    replicate_manifest: str | Path,
    condition: str = "TIS",
) -> pd.DataFrame:
    """Merge sample + replicate manifests into a one-row-per-sample frame.

    For the chosen *condition* (default ``"TIS"``), gathers every
    replicate's ``rnaseq_count_file`` into a list on the sample row.
    """
    samples = pd.read_csv(sample_manifest)
    reps = pd.read_csv(replicate_manifest)
    tis_reps = reps[reps["condition"] == condition]
    files_per_sample = (
        tis_reps.groupby("sample")["rnaseq_count_file"].apply(list).rename(
            "rnaseq_count_files"
        )
    )
    merged = samples.merge(
        files_per_sample, left_on="sample", right_index=True, how="left"
    )
    missing = merged[merged["rnaseq_count_files"].isnull()]
    if not missing.empty:
        raise ValueError(
            f"Samples without {condition} replicates in manifest: "
            f"{missing['sample'].tolist()}"
        )
    return merged
