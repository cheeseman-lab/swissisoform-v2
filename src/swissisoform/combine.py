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

import numpy as np
import pandas as pd

from swissisoform.plm.embed import protein_hash

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

# Per-sample metrics pivoted into {sample}_{metric} wide columns.  The
# source-resolution verdict (resolved / window_status / source_transcript /
# source_evidence / tie_initiation_efficiency) is per cell line — it depends on
# that sample's RNA-seq — so it pivots wide alongside the count metrics, present
# only for samples the stage ran on (HeLa today).
PER_SAMPLE_METRICS: tuple[str, ...] = (
    "TISCounts",
    "NormTISCounts",
    "TISPvalue",
    "RiboPvalue",
    "FisherQvalue",
    "Imputed",
    "GeneRNASeqCounts",
    "TotalRNASeqCounts",
    "resolved",
    "window_status",
    "source_transcript",
    "source_evidence",
    "tie_initiation_efficiency",
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

    # Pivot per-sample metrics to wide columns. NB: pandas pivot_table with
    # dropna=False builds MultiIndex.from_product over the index levels — here the
    # cartesian product of the four DEDUP_KEY levels (Symbol × Tid × GenomePos ×
    # StartCodon), which explodes to trillions of cells on the full catalog. Take
    # one row per (key, sample) and unstack instead: same wide layout, NaN for
    # absent samples, no cartesian blow-up.
    metric_cols = [m for m in PER_SAMPLE_METRICS if m in long.columns]
    one_per = long.drop_duplicates(subset=list(DEDUP_KEY) + ["_sample"], keep="first")
    wide = one_per.set_index(list(DEDUP_KEY) + ["_sample"])[metric_cols].unstack("_sample")
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


def _join_unique(s: pd.Series) -> str:
    """Comma-join the sorted unique string values of a Series (drops NaN)."""
    return ",".join(sorted(set(s.dropna().astype(str))))


def genome_pos_components(genome_pos: str) -> tuple[str, int, str]:
    """``(chrom, start_codon_genomic_pos, strand)`` from a ``chrom:lo-hi:strand`` span.

    ``GenomePos`` is ``chrom:lo-hi:strand`` (the full ORF span); the start codon
    sits at the 5' end — ``lo`` on the plus strand, ``hi`` on the minus. Splits
    from the right so any contig name is handled (incl. non-``chr`` scaffolds).
    """
    chrom, coords, strand = genome_pos.rsplit(":", 2)
    lo, hi = coords.split("-")
    return chrom, int(lo if strand == "+" else hi), strand


def init_site_from_genome_pos(genome_pos: str, start_codon: str) -> str:
    """Canonical init-site key ``chrom:gstart:strand:codon`` for one ORF span.

    The single source of truth for the init-site grouping key used across the
    combine / source-transcript layers (``benchmark``, ``window_purity``, etc.).
    """
    chrom, gstart, strand = genome_pos_components(genome_pos)
    return f"{chrom}:{gstart}:{strand}:{start_codon}"


def _init_site_from_genome_pos(genome_pos: pd.Series, start_codon: pd.Series) -> pd.Series:
    """Vectorized :func:`init_site_from_genome_pos` over two aligned Series."""
    keys = [
        init_site_from_genome_pos(g, str(c)) for g, c in zip(genome_pos.astype(str), start_codon)
    ]
    return pd.Series(keys, index=genome_pos.index)


def dedupe_unique_proteins(combined: pd.DataFrame) -> pd.DataFrame:
    """Collapse the combined per-TIS table to one row per unique protein.

    The combined table carries one row per ``(transcript × start)``; many of
    those translate to an identical protein — one genomic start shared by
    several transcripts with the same downstream ORF, or paralogous genes
    (histone / protocadherin clusters) encoding the same sequence. Structure
    folding and PLM embedding only need each distinct sequence once, so this
    collapses on the stop-stripped amino-acid sequence, keyed by
    :func:`swissisoform.plm.embed.protein_hash` — the same key the structure /
    PLM on-disk caches use.

    A deterministic representative TIS is kept per protein (most reproducible →
    most-supported → transcript id); every contributing transcript, start site,
    and gene is preserved as provenance so the table expands back to isoforms.

    Args:
        combined: Output of :func:`combine_filtered_samples`.

    Returns:
        One row per unique protein: ``protein_hash``, the representative TIS's
        metadata (``gene`` / ``representative_transcript`` / ``init_site`` /
        ``start_codon`` / ``genome_pos`` / ``orf_type`` / ``is_canonical`` /
        ``length_aa``), reproducibility (``n_cell_lines`` / ``max_cell_lines``,
        per-cell ``max_norm_{sample}``), and dedup provenance (``n_source_rows``
        / ``n_transcripts`` / ``n_init_sites`` / ``n_genes`` / ``all_genes`` /
        ``all_transcripts`` / ``orf_types_all``), plus the ``sequence``.
    """
    if combined.empty:
        return combined.head(0)

    df = combined.copy()
    df["protein_hash"] = df["AASeq"].astype(str).map(protein_hash)
    df["_init_site"] = _init_site_from_genome_pos(df["GenomePos"], df["StartCodon"])
    df["_is_canonical"] = df["TisType"].astype(str).str.startswith("Annotated")
    df["_n_samples"] = df["n_samples"] if "n_samples" in df.columns else 0

    # Representative = most reproducible, best-supported TIS for each protein.
    tis_cols = [c for c in df.columns if c.endswith("_TISCounts")]
    df["_support"] = df[tis_cols].fillna(0).sum(axis=1) if tis_cols else 0.0
    df = df.sort_values(
        ["protein_hash", "_n_samples", "_support", "Tid"],
        ascending=[True, False, False, True],
    )
    rep = df.drop_duplicates("protein_hash", keep="first").set_index("protein_hash")
    g = df.groupby("protein_hash")
    stripped = rep["AASeq"].astype(str).str.rstrip("*").str.upper()

    out = pd.DataFrame(
        {
            "protein_hash": rep.index,
            "gene": rep["Symbol"],
            "representative_transcript": rep["Tid"],
            "init_site": rep["_init_site"],
            "start_codon": rep["StartCodon"],
            "genome_pos": rep["GenomePos"],
            "orf_type": rep["RecatTISType"],
            "is_canonical": rep["_is_canonical"],
            "length_aa": stripped.str.len(),
            "n_cell_lines": rep["_n_samples"],
            "max_cell_lines": g["_n_samples"].max(),
            "n_source_rows": g.size(),
            "n_transcripts": g["Tid"].nunique(),
            "n_init_sites": g["_init_site"].nunique(),
            "n_genes": g["Symbol"].nunique(),
            "all_genes": g["Symbol"].apply(_join_unique),
            "all_transcripts": g["Tid"].apply(_join_unique),
            "orf_types_all": g["RecatTISType"].apply(_join_unique),
            "sequence": stripped,
        }
    ).reset_index(drop=True)

    for col in [c for c in combined.columns if c.endswith("_NormTISCounts")]:
        sample = col[: -len("_NormTISCounts")]
        out[f"max_norm_{sample}"] = g[col].max().reindex(out["protein_hash"]).values

    out = out.sort_values(
        ["is_canonical", "max_cell_lines", "n_source_rows"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    logger.info(
        "dedupe_unique_proteins: %d combined TIS rows → %d unique proteins "
        "(%d canonical, %d alternative)",
        len(combined),
        len(out),
        int(out["is_canonical"].sum()),
        int((~out["is_canonical"]).sum()),
    )
    return out


def dedupe_unique_init_sites(combined: pd.DataFrame) -> pd.DataFrame:
    """Collapse the combined per-TIS table to one row per genomic initiation site.

    The natural unit for a DNA / genome-language model is the genomic initiation
    site (``chrom:gstart:strand:codon``) — where the ribosome physically starts —
    not the protein.  One site can yield several proteins (distal-splice variants
    sharing an N-terminus), which a DNA model sees as a single genomic window;
    conversely paralogous loci encoding the same protein are distinct windows, so
    they stay separate here (they collapse in :func:`dedupe_unique_proteins`).

    Emits the maximally informative per-site record for downstream embedding: the
    genomic anchor, per-condition usage labels (``max_norm_{sample}`` peak RPM +
    ``present_{sample}``), reproducibility, full ORF/protein/gene provenance,
    per-condition transcription context (``gene_rnaseq_{sample}``), and best
    significance.  No DNA sequence is emitted — window extraction is the consuming
    repo's concern.

    A deterministic representative TIS is kept per site (most reproducible →
    most-supported → transcript id), mirroring :func:`dedupe_unique_proteins`.

    Args:
        combined: Output of :func:`combine_filtered_samples`.

    Returns:
        One row per ``init_site``.
    """
    if combined.empty:
        return combined.head(0)

    df = combined.copy()
    df["protein_hash"] = df["AASeq"].astype(str).map(protein_hash)
    df["_init_site"] = _init_site_from_genome_pos(df["GenomePos"], df["StartCodon"])
    bad = df["_init_site"].isna()
    if bad.any():
        logger.warning(
            "dedupe_unique_init_sites: dropping %d/%d rows with unparseable GenomePos",
            int(bad.sum()),
            len(df),
        )
        df = df[~bad].copy()
    df["_is_canonical"] = df["TisType"].astype(str).str.startswith("Annotated")
    df["_n_samples"] = df["n_samples"] if "n_samples" in df.columns else 0
    df["_len_aa"] = df["AASeq"].astype(str).str.rstrip("*").str.len()

    tis_cols = [c for c in df.columns if c.endswith("_TISCounts")]
    df["_support"] = df[tis_cols].fillna(0).sum(axis=1) if tis_cols else 0.0

    def _row_min(suffix: str) -> pd.Series:
        cols = [c for c in df.columns if c.endswith(suffix)]
        return df[cols].min(axis=1) if cols else pd.Series(np.nan, index=df.index)

    df["_min_tis_pvalue"] = _row_min("_TISPvalue")
    df["_min_ribo_pvalue"] = _row_min("_RiboPvalue")
    df["_min_fisher_qvalue"] = _row_min("_FisherQvalue")

    imp_cols = [c for c in df.columns if c.endswith("_Imputed")]
    df["_any_imputed"] = (
        df[imp_cols].astype(str).isin(["True", "true", "1", "1.0"]).any(axis=1)
        if imp_cols
        else False
    )

    # Representative = most reproducible, best-supported TIS for each site.
    df = df.sort_values(
        ["_init_site", "_n_samples", "_support", "Tid"],
        ascending=[True, False, False, True],
    )
    rep = df.drop_duplicates("_init_site", keep="first").set_index("_init_site")
    g = df.groupby("_init_site")

    site_parts = rep.index.to_series().str.extract(r"(chr[\w]+):(\d+):([+-]):(.+)")
    site_parts.columns = ["_chrom", "_gstart", "_strand", "_codon"]

    out = pd.DataFrame(
        {
            "init_site": rep.index,
            "chrom": site_parts["_chrom"],
            "gstart": site_parts["_gstart"].astype("int64"),
            "strand": site_parts["_strand"],
            "start_codon": rep["StartCodon"],
            "gene": rep["Symbol"],
            "representative_transcript": rep["Tid"],
            "protein_hash": rep["protein_hash"],
            "genome_pos": rep["GenomePos"],
            "orf_type": rep["RecatTISType"],
            "is_canonical": rep["_is_canonical"],
            "is_imputed": g["_any_imputed"].any(),
            "length_aa": rep["_len_aa"],
            "length_aa_min": g["_len_aa"].min(),
            "length_aa_max": g["_len_aa"].max(),
            "n_cell_lines": rep["_n_samples"],
            "max_cell_lines": g["_n_samples"].max(),
            "n_source_rows": g.size(),
            "n_transcripts": g["Tid"].nunique(),
            "n_proteins": g["protein_hash"].nunique(),
            "n_genes": g["Symbol"].nunique(),
            "n_genome_pos": g["GenomePos"].nunique(),
            "all_genes": g["Symbol"].apply(_join_unique),
            "all_transcripts": g["Tid"].apply(_join_unique),
            "all_protein_hashes": g["protein_hash"].apply(_join_unique),
            "orf_types_all": g["RecatTISType"].apply(_join_unique),
            "min_tis_pvalue": g["_min_tis_pvalue"].min(),
            "min_ribo_pvalue": g["_min_ribo_pvalue"].min(),
            "min_fisher_qvalue": g["_min_fisher_qvalue"].min(),
        }
    ).reset_index(drop=True)

    # Per-condition usage labels (y), presence, and transcription context.
    for col in [c for c in combined.columns if c.endswith("_NormTISCounts")]:
        sample = col[: -len("_NormTISCounts")]
        out[f"max_norm_{sample}"] = g[col].max().reindex(out["init_site"]).values
        pres = f"present_{sample}"
        if pres in combined.columns:
            out[pres] = g[pres].any().reindex(out["init_site"]).values
        grc = f"{sample}_GeneRNASeqCounts"
        if grc in combined.columns:
            out[f"gene_rnaseq_{sample}"] = rep[grc].reindex(out["init_site"]).values

    out = out.sort_values(
        ["is_canonical", "max_cell_lines", "n_source_rows"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    logger.info(
        "dedupe_unique_init_sites: %d combined TIS rows → %d unique init sites "
        "(%d canonical, %d alternative)",
        len(combined),
        len(out),
        int(out["is_canonical"].sum()),
        int((~out["is_canonical"]).sum()),
    )
    return out


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
