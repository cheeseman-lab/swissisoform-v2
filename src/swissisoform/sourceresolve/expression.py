"""Determine which transcripts are expressed (present) in a sample from RNA-seq.

Two input modalities collapse to one output — a set of present transcript IDs:

- **short-read** → salmon per-transcript TPM (``quant.sf``)
- **long-read**  → IsoQuant per-transcript counts/TPM

A transcript "exists" in the sample if its abundance meets a threshold
(short-read default ``TPM >= 1``; long-read default ``counts >= 3``). The
resulting ID set is used to drop unexpressed candidate transcripts from each
TIS's ``all_transcripts`` (the first narrowing step). See
``docs/plans/source_transcript_resolution.md``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _bare_tid(name: str) -> str:
    """Reduce a salmon/GENCODE target id to a bare versioned ENST id.

    GENCODE FASTA headers look like ``ENST00000123.4|ENSG...|...|GENE-201|...``;
    our ``Tid`` is the leading ``ENST00000123.4``.
    """
    return name.split("|")[0]


def load_salmon_tpm(quant_sf: str | Path) -> dict[str, float]:
    """Read one salmon ``quant.sf`` into ``{transcript_id: TPM}``.

    Args:
        quant_sf: Path to a salmon ``quant.sf`` (TSV with columns
            ``Name, Length, EffectiveLength, TPM, NumReads``).

    Returns:
        Mapping from bare ``ENST.version`` id to its TPM.
    """
    df = pd.read_csv(quant_sf, sep="\t")
    df["Name"] = df["Name"].map(_bare_tid)
    return dict(zip(df["Name"], df["TPM"]))


def load_salmon_replicates(quant_sfs: list[str | Path]) -> dict[str, float]:
    """Average TPM across replicate ``quant.sf`` files.

    Args:
        quant_sfs: One ``quant.sf`` path per replicate (HeLa has two).

    Returns:
        ``{transcript_id: mean TPM}`` over the union of transcripts; a
        transcript absent from a replicate counts as 0 in that replicate.
    """
    reps = [load_salmon_tpm(p) for p in quant_sfs]
    tids = set().union(*reps) if reps else set()
    return {t: sum(r.get(t, 0.0) for r in reps) / len(reps) for t in tids}


def load_isoquant_abundance(
    table_tsv: str | Path,
    id_col: str = "feature_id",
    value_col: str | None = None,
) -> dict[str, float]:
    """Read an IsoQuant per-transcript table into ``{transcript_id: abundance}``.

    Args:
        table_tsv: IsoQuant transcript counts/TPM TSV (e.g.
            ``*.transcript_counts.tsv`` / ``*.transcript_tpm.tsv``).
        id_col: Column holding the transcript id (IsoQuant uses
            ``feature_id``).
        value_col: Column holding the abundance value; defaults to the first
            non-id column.

    Returns:
        Mapping from transcript id to abundance.
    """
    df = pd.read_csv(table_tsv, sep="\t")
    if value_col is None:
        value_col = next(c for c in df.columns if c != id_col)
    return dict(zip(df[id_col].astype(str), df[value_col].astype(float)))


def expressed_transcripts(abundance: dict[str, float], min_value: float = 1.0) -> set[str]:
    """Transcripts whose abundance meets the threshold = 'present in the sample'.

    Args:
        abundance: ``{transcript_id: value}`` from any loader above.
        min_value: Inclusive presence threshold (short-read TPM default 1.0;
            long-read counts typically 3).

    Returns:
        The set of present transcript ids.
    """
    return {t for t, v in abundance.items() if v >= min_value}


def expressed_in_replicates(
    quant_sfs: list[str | Path],
    min_tpm: float = 1.0,
    require: str = "all",
) -> set[str]:
    """Present transcripts by a per-replicate salmon TPM rule.

    Unlike averaging, this evaluates each replicate independently. For HeLa
    (two reps) the default ``require="all"`` is the stringent presence rule:
    a transcript counts as present only if it clears ``min_tpm`` in **every**
    replicate.

    Args:
        quant_sfs: One salmon ``quant.sf`` per replicate.
        min_tpm: Per-replicate TPM threshold.
        require: ``"all"`` → TPM ≥ ``min_tpm`` in every replicate (default);
            ``"any"`` → in at least one replicate.

    Returns:
        The set of present transcript ids.
    """
    per_rep = [expressed_transcripts(load_salmon_tpm(p), min_tpm) for p in quant_sfs]
    if not per_rep:
        return set()
    if require == "all":
        return set.intersection(*per_rep)
    if require == "any":
        return set.union(*per_rep)
    raise ValueError(f"require must be 'all' or 'any', got {require!r}")
