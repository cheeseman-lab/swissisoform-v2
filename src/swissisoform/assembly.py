"""Assembly: build Gene objects from Ribo-TISH DataFrames.

Converts a raw or filtered Ribo-TISH DataFrame into rich ``Gene`` and
``TranslationInitiationSite`` domain objects ready for the annotation
pipeline.  Handles canonical transcript selection, ORF type mapping,
``DifferentialRegion`` computation via protein-sequence comparison, and
optional Kozak context extraction from a reference genome FASTA.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from swissisoform.models import (
    CellLineExpression,
    DifferentialRegion,
    Gene,
    ORFType,
    TranslationInitiationSite,
    orf_type_from_ribotish,
)

logger = logging.getLogger(__name__)

# Nucleotide complements for reverse-strand Kozak extraction.
_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def _revcomp(seq: str) -> str:
    """Return the reverse-complement of a DNA sequence (upper-cased)."""
    return seq.upper().translate(_COMPLEMENT)[::-1]


def extract_kozak_context(
    fasta: Any,
    chrom: str,
    position: int,
    strand: str,
) -> str | None:
    """Extract a 13-nt Kozak context window around a TIS position.

    The Kozak window spans mRNA positions -9..-1 (9 upstream bases) + the
    ATG start codon (mRNA +1, +2, +3) + one downstream base (mRNA +4), for
    a total of 13 characters reading 5' → 3' in mRNA orientation.  This
    matches the ``KOZAK_CONSENSUS`` layout in ``initiation_context.py`` with
    the ATG at indices 9, 10, 11.

    **Coordinate convention:** Ribo-TISH ``GenomePos`` uses 0-based half-open
    coordinates.  ``position`` here is ``TIS.position`` as set by the assembly
    layer — for plus strand it is the 0-based position of the A of ATG; for
    minus strand it is the EXCLUSIVE end of the ORF range, so the A of ATG
    on mRNA is at plus-strand 0-based ``position - 1``.

    Args:
        fasta: A ``pysam.FastaFile`` (or compatible ``.fetch(chrom, start, end)``).
        chrom: Chromosome name (must match the FASTA contig names).
        position: 0-based genomic position from ``TIS.position``.
        strand: ``"+"`` or ``"-"``.

    Returns:
        13-character Kozak context string, or ``None`` if the fetch fails
        (out of bounds, contig missing, or short sequence).
    """
    try:
        if strand == "+":
            # A of ATG at 0-based position P. Window covers mRNA -9..+4:
            # plus-strand 0-based [P-9, P+4) = 13 bases with ATG at indices 9,10,11.
            start = position - 9
            end = position + 4
            if start < 0:
                return None
            seq = fasta.fetch(chrom, start, end).upper()
        else:
            # Minus strand: position is the exclusive end of the ORF range,
            # so the A of ATG on mRNA corresponds to 0-based plus-strand P-1.
            # Plus-strand window spanning mRNA -9..+4 is 0-based [P-4, P+9),
            # reverse-complemented to read 5'→3' in mRNA orientation.
            start = position - 4
            end = position + 9
            if start < 0:
                return None
            genomic = fasta.fetch(chrom, start, end).upper()
            seq = _revcomp(genomic)

        if len(seq) != 13:
            return None
        return seq
    except (ValueError, KeyError, OSError) as exc:
        logger.debug("Kozak fetch failed for %s:%d:%s — %s", chrom, position, strand, exc)
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_stop(seq: str) -> str:
    """Remove trailing '*' from a protein sequence."""
    return seq.rstrip("*")


# Minimum aligned tail length required after a prefix match to trust the offset.
# A 20-residue prefix match alone is not proof of identity (paralogs, conserved
# exons); we verify the rest of the sequence aligns after the claimed offset.
_MIN_TAIL_VERIFY = 50

# Fraction of aligned positions that must match when verifying the tail.
# 1.0 means exact identity; allow a small margin for cross-transcript C-terminal
# differences (same gene, different splice forms).
_TAIL_MATCH_FRACTION = 0.95


def _tail_matches(seq_a: str, seq_b: str) -> bool:
    """Return True if *seq_a* and *seq_b* align to at least _TAIL_MATCH_FRACTION
    over the shorter of (remaining length, ``_MIN_TAIL_VERIFY``).

    Returns True if either sequence is shorter than ``_MIN_TAIL_VERIFY`` AND
    they match exactly — that's the only sound way to accept very short tails.
    """
    if not seq_a or not seq_b:
        return False

    cmp_len = min(len(seq_a), len(seq_b))
    if cmp_len < _MIN_TAIL_VERIFY:
        # Too short to trust partial match; require exact equality
        return seq_a[:cmp_len] == seq_b[:cmp_len]

    cmp_len = min(cmp_len, _MIN_TAIL_VERIFY * 4)  # cap comparison for speed
    matches = sum(1 for a, b in zip(seq_a[:cmp_len], seq_b[:cmp_len]) if a == b)
    return (matches / cmp_len) >= _TAIL_MATCH_FRACTION


def _find_canonical_offset(isoform: str, canonical: str) -> int | None:
    """Find the 0-based offset in *isoform* where *canonical* begins.

    Tries exact suffix match first.  Falls back to a prefix search verified by
    tail alignment: after finding where canonical[:20] appears in isoform,
    verifies that isoform[idx+20:] aligns to canonical[20:] over a meaningful
    length.  Returns ``None`` if no reliable match is found.
    """
    iso = _strip_stop(isoform)
    can = _strip_stop(canonical)

    if not can:
        return None

    # Exact suffix match: isoform = extension + canonical
    if len(iso) > len(can) and iso.endswith(can):
        return len(iso) - len(can)

    # Prefix search + tail verification
    check_len = min(20, len(can))
    if check_len < 3:
        return None
    target = can[:check_len]
    idx = iso.find(target)
    if idx < 0:
        return None

    # Verify the tail aligns — prefix match alone is not enough
    iso_tail = iso[idx + check_len:]
    can_tail = can[check_len:]
    if _tail_matches(iso_tail, can_tail):
        return idx

    return None


def _find_truncation_offset(isoform: str, canonical: str) -> int | None:
    """Find the 0-based offset in *canonical* where *isoform* begins.

    Uses exact suffix match + tail-verified prefix search, same as
    ``_find_canonical_offset``.  Returns ``None`` if no reliable match is found.
    """
    iso = _strip_stop(isoform)
    can = _strip_stop(canonical)

    if not iso:
        return None

    # Exact suffix match: canonical = lost_prefix + isoform
    if len(can) > len(iso) and can.endswith(iso):
        return len(can) - len(iso)

    # Prefix search + tail verification
    check_len = min(20, len(iso))
    if check_len < 3:
        return None
    target = iso[:check_len]
    idx = can.find(target)
    if idx < 0:
        return None

    can_tail = can[idx + check_len:]
    iso_tail = iso[check_len:]
    if _tail_matches(can_tail, iso_tail):
        return idx

    return None


def compute_diff_region(
    orf_type: ORFType,
    isoform_protein: str,
    canonical_protein: str,
    context: str = "",
) -> DifferentialRegion:
    """Compute the differential region between isoform and canonical proteins.

    For extensions: the N-terminal prefix unique to the isoform.
    For truncations: the N-terminal prefix lost from canonical.
    For uORFs/altORFs: the entire isoform sequence.

    Note: Ribo-TISH classifies ORF type relative to the *transcript's*
    annotated start, but the canonical protein here is per-*gene*.  When
    a "Truncated" TIS is actually from a longer transcript, the isoform
    may be longer than canonical — making it effectively an extension.
    This function detects such cross-transcript mismatches by checking
    actual sequence relationships.

    When sequence heuristics fail (neither exact suffix nor tail-verified
    prefix match), the function logs a WARN with *context* and falls back
    to the least-bad default (length-based truncation or whole-isoform).
    Downstream code should treat such diff regions as low-confidence.

    Args:
        orf_type: Classified ORF type from Ribo-TISH.
        isoform_protein: Predicted isoform protein sequence.
        canonical_protein: Canonical protein sequence for this gene.
        context: Optional identifier (e.g. ``"TP53 chr17:...:+:ATG"``) for
            warning messages. Empty by default.

    Returns:
        DifferentialRegion with coordinates and sequence.
    """
    iso = _strip_stop(isoform_protein)
    can = _strip_stop(canonical_protein)

    if orf_type == ORFType.ANNOTATED:
        return DifferentialRegion(sequence="")

    # For non-unique ORF types (in-frame with canonical), determine the
    # actual relationship by sequence comparison, not just the nominal label.
    # Ribo-TISH labels are per-transcript; our canonical is per-gene.
    if orf_type in (
        ORFType.EXTENDED, ORFType.TRUNCATED, ORFType.UOORF,
    ):
        ext_offset = _find_canonical_offset(isoform_protein, canonical_protein)
        trunc_offset = _find_truncation_offset(isoform_protein, canonical_protein)

        # If canonical is found as a suffix of isoform → extension
        if ext_offset is not None and ext_offset > 0:
            return DifferentialRegion(
                isoform_start=0,
                isoform_end=ext_offset,
                sequence=iso[:ext_offset],
            )

        # If isoform is found as a suffix of canonical → truncation
        if trunc_offset is not None and trunc_offset > 0:
            return DifferentialRegion(
                canonical_start=0,
                canonical_end=trunc_offset,
                sequence=can[:trunc_offset],
            )

        # Length-based fallback for truncations (low confidence)
        if orf_type == ORFType.TRUNCATED:
            delta = len(can) - len(iso)
            if delta > 0:
                logger.warning(
                    "diff_region fallback (length-based) for %s: orf_type=%s, "
                    "iso_len=%d, can_len=%d — no sequence match found",
                    context or "unknown",
                    orf_type.value,
                    len(iso),
                    len(can),
                )
                return DifferentialRegion(
                    canonical_start=0,
                    canonical_end=delta,
                    sequence=can[:delta],
                )

        # Degraded fallback: treat entire isoform as differential
        logger.warning(
            "diff_region fallback (whole-isoform) for %s: orf_type=%s, "
            "iso_len=%d, can_len=%d — sequences are structurally unrelated; "
            "downstream positional analysis will be unreliable",
            context or "unknown",
            orf_type.value,
            len(iso),
            len(can),
        )
        return DifferentialRegion(
            isoform_start=0,
            isoform_end=len(iso),
            sequence=iso,
        )

    # uORF, altORF, internal OOF, 3'UTR ORF: entire isoform is unique
    return DifferentialRegion(
        isoform_start=0,
        isoform_end=len(iso),
        sequence=iso,
    )


# ---------------------------------------------------------------------------
# Canonical selection
# ---------------------------------------------------------------------------


def _select_canonical(gene_rows: pd.DataFrame) -> tuple[str, str, str]:
    """Select canonical transcript and protein for a gene.

    Picks the Annotated TIS with the longest AASeq.  If no Annotated TIS
    exists, falls back to the longest protein from any row.

    Args:
        gene_rows: DataFrame of all TIS rows for a single gene.

    Returns:
        (canonical_transcript_id, canonical_protein, gene_id)
    """
    annotated = gene_rows[gene_rows["TisType"].str.startswith("Annotated")]

    if annotated.empty:
        best = gene_rows.loc[gene_rows["AALen"].idxmax()]
        return str(best["Tid"]), str(best["AASeq"]), str(best["Gid"])

    best = annotated.loc[annotated["AALen"].idxmax()]
    return str(best["Tid"]), str(best["AASeq"]), str(best["Gid"])


def _build_canonical_by_tid(gene_rows: pd.DataFrame) -> dict[str, str]:
    """Map each transcript ID to its Annotated protein sequence.

    Ribo-TISH classifies ORF types relative to each transcript's own CDS,
    so the correct canonical to compare against is the Annotated row for
    that same Tid — not the gene-level longest.  This builds that lookup.

    If a Tid has multiple Annotated rows (rare; would imply duplicate
    records), the longest AASeq wins.

    Args:
        gene_rows: DataFrame of all TIS rows for a single gene.

    Returns:
        Dict from transcript_id to canonical protein sequence.  Empty if
        the gene has no Annotated rows at all.
    """
    annotated = gene_rows[gene_rows["TisType"].str.startswith("Annotated")]
    by_tid: dict[str, str] = {}
    for _, row in annotated.iterrows():
        tid = str(row["Tid"])
        aaseq = str(row["AASeq"])
        existing = by_tid.get(tid)
        if existing is None or len(aaseq) > len(existing):
            if existing is not None:
                logger.debug(
                    "Multiple Annotated rows for transcript %s; keeping longest", tid
                )
            by_tid[tid] = aaseq
    return by_tid


# ---------------------------------------------------------------------------
# Row → TIS
# ---------------------------------------------------------------------------


def _expression_from_row(
    row: pd.Series, samples: list[str],
) -> dict[str, CellLineExpression]:
    """Build ``{cell_line: CellLineExpression}`` from wide sample columns.

    Reads ``{sample}_TISCounts`` / ``{sample}_NormTISCounts`` /
    ``{sample}_FisherQvalue`` out of *row* for each sample the TIS is
    marked present in (``present_{sample}`` True).  Samples where the
    TIS wasn't called are skipped — absence is meaningful.
    """
    expression: dict[str, CellLineExpression] = {}
    for sample in samples:
        present_col = f"present_{sample}"
        if not bool(row.get(present_col, False)):
            continue
        raw = row.get(f"{sample}_TISCounts")
        cpm = row.get(f"{sample}_NormTISCounts")
        pval = row.get(f"{sample}_FisherQvalue")
        if pd.isna(raw) or pd.isna(cpm) or pd.isna(pval):
            continue
        expression[sample] = CellLineExpression(
            raw_count=int(raw), cpm=float(cpm), p_value=float(pval),
        )
    return expression


def _row_to_tis(
    row: pd.Series,
    canonical_protein: str,
    fasta: Any | None = None,
    samples: list[str] | None = None,
) -> TranslationInitiationSite:
    """Convert a single DataFrame row to a TranslationInitiationSite.

    If *fasta* is provided, the Kozak context (13-nt window -9..+4 around
    the start codon, strand-aware) is populated from the genome FASTA.
    Otherwise ``kozak_context`` is left as ``None``.

    If *samples* is provided (combined-table mode), per-sample metrics are
    read from ``{sample}_{metric}`` wide columns into
    ``TIS.expression``, and the top-level ``tis_pvalue`` /
    ``ribo_pvalue`` / ``fisher_qvalue`` are left ``None`` (sample-
    specific — look them up in ``expression``).  Otherwise (legacy per-
    sample mode), those fields come from ``TISPvalue`` / ``RiboPvalue`` /
    ``FisherQvalue`` directly and ``expression`` is empty.
    """
    genome_pos = str(row["GenomePos"])
    chrom, coords_str, strand = genome_pos.rsplit(":", maxsplit=2)
    start_str, end_str = coords_str.split("-")

    position = int(end_str) if strand == "-" else int(start_str)
    start_codon = str(row["StartCodon"])
    orf_type = orf_type_from_ribotish(str(row["TisType"]))
    isoform_protein = str(row["AASeq"])
    gene_name = str(row["Symbol"])
    tis_id = f"{chrom}:{position}:{strand}:{start_codon}"

    diff_region = compute_diff_region(
        orf_type, isoform_protein, canonical_protein,
        context=f"{gene_name} {tis_id}",
    )

    kozak_context: str | None = None
    if fasta is not None:
        kozak_context = extract_kozak_context(fasta, chrom, position, strand)

    if samples is not None:
        expression = _expression_from_row(row, samples)
        tis_pvalue = ribo_pvalue = fisher_qvalue = None
    else:
        expression = {}
        tis_pvalue = float(row["TISPvalue"]) if pd.notna(row["TISPvalue"]) else None
        ribo_pvalue = float(row["RiboPvalue"]) if pd.notna(row["RiboPvalue"]) else None
        fisher_qvalue = (
            float(row["FisherQvalue"]) if pd.notna(row["FisherQvalue"]) else None
        )

    return TranslationInitiationSite(
        tis_id=tis_id,
        gene_name=gene_name,
        transcript_id=str(row["Tid"]),
        chrom=chrom,
        position=position,
        strand=strand,
        start_codon=start_codon,
        orf_type=orf_type,
        gene_id=str(row["Gid"]),
        transcript_start=int(row["Start"]),
        aa_len=int(row["AALen"]),
        tis_pvalue=tis_pvalue,
        ribo_pvalue=ribo_pvalue,
        fisher_qvalue=fisher_qvalue,
        canonical_protein=canonical_protein,
        isoform_protein=isoform_protein,
        diff_region=diff_region,
        kozak_context=kozak_context,
        expression=expression,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_SHARED_REQUIRED: tuple[str, ...] = (
    "Symbol", "Gid", "Tid", "TisType", "GenomePos",
    "StartCodon", "Start", "AALen", "AASeq",
)
_PER_SAMPLE_REQUIRED: tuple[str, ...] = ("TISPvalue", "RiboPvalue", "FisherQvalue")


def _detect_samples(df: pd.DataFrame) -> list[str] | None:
    """Return sample list if *df* is a combined (wide) frame, else None.

    Combined frames are produced by ``combine_filtered_samples`` and carry
    ``present_{sample}`` bool columns.  Per-sample frames carry the raw
    ``TISPvalue`` / ``RiboPvalue`` / ``FisherQvalue`` columns directly.
    """
    present = [c for c in df.columns if c.startswith("present_")]
    if not present:
        return None
    return sorted(c[len("present_"):] for c in present)


def _validate_columns(df: pd.DataFrame, samples: list[str] | None) -> None:
    """Raise ValueError if any required column is missing from *df*."""
    missing = [c for c in _SHARED_REQUIRED if c not in df.columns]
    if samples is None:
        missing += [c for c in _PER_SAMPLE_REQUIRED if c not in df.columns]
    else:
        for sample in samples:
            for metric in ("TISCounts", "NormTISCounts", "FisherQvalue"):
                col = f"{sample}_{metric}"
                if col not in df.columns:
                    missing.append(col)
    if missing:
        raise ValueError(
            f"assemble_genes: DataFrame missing required columns: {missing}. "
            f"Got columns: {list(df.columns)}"
        )


def assemble_genes(
    df: pd.DataFrame,
    gene_names: list[str] | None = None,
    include_annotated: bool = False,
    genome_fasta: str | Path | None = None,
) -> list[Gene]:
    """Build Gene objects from an upstream filtered+imputed DataFrame.

    Expects the single-table output of :func:`swissisoform.pipeline.run_sample`
    (post-imputation, per cell line).  Every transcript in *df* is
    guaranteed to have at least one Annotated row (native Ribo-TISH or
    synthesized by imputation), so canonical selection is unambiguous.

    Groups rows by gene symbol, picks ``Gene.canonical_protein`` as the
    longest Annotated AASeq, builds a per-Tid canonical map for TIS-level
    comparison, and converts non-Annotated rows to
    :class:`TranslationInitiationSite` objects with
    :class:`DifferentialRegion` coordinates.

    If ``genome_fasta`` is provided, 13-nt Kozak context (-9..+4) is
    extracted for each TIS.  Without it, ``kozak_context`` is ``None``.

    Args:
        df: Upstream output DataFrame.  Required columns: Symbol, Gid,
            Tid, TisType, GenomePos, StartCodon, Start, AALen, AASeq,
            TISPvalue, RiboPvalue, FisherQvalue.
        gene_names: Optional subset of gene symbols to process.
        include_annotated: If ``True``, include Annotated rows as TIS
            sites (with empty diff_region).  Default ``False``.
        genome_fasta: Optional path to an indexed reference genome FASTA
            for Kozak extraction.

    Returns:
        List of Gene objects with ``tis_sites`` populated.

    Raises:
        ValueError: If any required column is missing, or if any Tid
            present in the alt-TIS rows lacks an Annotated row (indicates
            upstream did not run ``impute_missing_canonical_starts`` or
            the DataFrame was mis-filtered).
    """
    samples = _detect_samples(df)
    _validate_columns(df, samples)

    if gene_names is not None:
        df = df[df["Symbol"].isin(gene_names)]

    if df.empty:
        logger.warning("No TIS rows to assemble")
        return []

    fasta = None
    if genome_fasta is not None:
        try:
            import pysam
            fasta = pysam.FastaFile(str(genome_fasta))
            logger.info("Loaded genome FASTA %s for Kozak extraction", genome_fasta)
        except Exception as exc:
            logger.warning(
                "Failed to open genome FASTA %s — Kozak context will be None: %s",
                genome_fasta, exc,
            )
            fasta = None

    genes: list[Gene] = []

    for symbol, gene_df in df.groupby("Symbol"):
        canonical_tid, canonical_protein, gene_id = _select_canonical(gene_df)
        canonical_by_tid = _build_canonical_by_tid(gene_df)

        if include_annotated:
            rows = gene_df
        else:
            rows = gene_df[~gene_df["TisType"].str.startswith("Annotated")]

        missing = {
            str(r["Tid"])
            for _, r in rows.iterrows()
            if str(r["Tid"]) not in canonical_by_tid
        }
        if missing:
            raise ValueError(
                f"Gene {symbol}: {len(missing)} transcript(s) have alt TIS but "
                f"no Annotated row — upstream imputation didn't run or the "
                f"DataFrame was mis-filtered.  Orphan Tids: {sorted(missing)[:3]}"
            )

        tis_sites: list[TranslationInitiationSite] = [
            _row_to_tis(
                row, canonical_by_tid[str(row["Tid"])],
                fasta=fasta, samples=samples,
            )
            for _, row in rows.iterrows()
        ]

        if not tis_sites:
            logger.info("Gene %s has no alternative TIS — skipping", symbol)
            continue

        gene = Gene(
            gene_name=str(symbol),
            gene_id=gene_id,
            canonical_transcript_id=canonical_tid,
            canonical_protein=canonical_protein,
            tis_sites=tis_sites,
        )
        genes.append(gene)

        logger.info(
            "Assembled %s: %d TIS, canonical %s (%d aa)",
            symbol,
            len(tis_sites),
            canonical_tid,
            len(_strip_stop(canonical_protein)),
        )

    total_tis = sum(len(g.tis_sites) for g in genes)
    logger.info("Assembled %d genes, %d total TIS", len(genes), total_tis)
    return genes
