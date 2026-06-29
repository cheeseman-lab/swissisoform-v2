"""Per-sample source-transcript resolution: pin one source mRNA per TIS.

A single linear cascade (replacing the earlier three-arm union), run inside the
filtering cascade (``pipeline.run_sample``). It depends on the sample's own
long-read RNA-seq, so it is intrinsically per cell line. Per genomic initiation
site (``init_site``):

1. **Long-read filter** — keep only candidate transcripts present in the sample's
   IsoQuant table (count ≥ ``isoquant_min_count``). If none survive, the site is
   unresolved.
2. **±W window-purity** — run :func:`purity_decision` on the survivors. If their
   local sequence within ±W of the start codon agrees (pure / single candidate),
   the window is unambiguous regardless of which isoform a footprint came from.
3. **Abundance labeling**:
   - **pure / single** → source = most-abundant survivor (by long-read count).
   - **divergent** → an abundance threshold decides which survivor to keep
     (dominance fraction and/or absolute count); if it fails, the site is
     unresolved; with no threshold set, falls back to most-abundant-wins.

**Tag-only:** every TIS is annotated with its verdict and kept — nothing is
dropped (subsetting on resolution is a deliberate follow-up).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from swissisoform.combine import _init_site_from_genome_pos, genome_pos_components
from swissisoform.models import TranscriptCoordinates
from swissisoform.sourceresolve.expression import (
    expressed_transcripts,
    load_isoquant_abundance,
)
from swissisoform.sourceresolve.mrna import TisWindow, extract_tis_window
from swissisoform.sourceresolve.purity import purity_decision

logger = logging.getLogger(__name__)

# Columns the stage attaches to every row of the filtered table.
VERDICT_COLUMNS: tuple[str, ...] = (
    "resolved",
    "window_status",
    "source_transcript",
    "source_evidence",
    "tie_initiation_efficiency",
)


@dataclass
class Resolution:
    """The cascade verdict for a single initiation site.

    Attributes:
        resolved: ``True`` if the cascade pins a source transcript.
        window_status: ``"single"`` (one survivor), ``"pure"`` (survivors agree
            in-window), ``"divergent"`` (survivors disagree), or ``"none"`` (no
            survivor after the long-read filter).
        source_transcript: The chosen source mRNA, or ``None`` if unresolved.
        source_evidence: ``"window_pure"`` | ``"divergent_abundance"`` |
            ``"unresolved"``.
        tie_initiation_efficiency: The source transcript's ``NormTISCounts``
            divided by its long-read abundance, or ``None``.
    """

    resolved: bool
    window_status: str
    source_transcript: str | None
    source_evidence: str
    tie_initiation_efficiency: float | None


def _candidate_windows(
    cand_tids: list[str],
    gstart: int,
    start_codon: str,
    exon_skeletons: Mapping[str, TranscriptCoordinates],
    genome: Any,
    w: int,
) -> dict[str, TisWindow]:
    """Extract the ±W window for each candidate that contains the start codon.

    Candidates without an exon skeleton, whose start is intronic, or whose window
    does not open on the called start codon (a coordinate-error guard) are
    dropped. The start-codon comparison is case-insensitive so a soft-masked
    genome does not spuriously drop every candidate. ``down=w + 1`` so radius
    ``w`` (the w-th base 3' of the A) is reachable by the purity test.
    """
    expected = start_codon.upper()
    windows: dict[str, TisWindow] = {}
    for tid in cand_tids:
        coords = exon_skeletons.get(tid)
        if coords is None:
            continue
        win = extract_tis_window(coords, genome, gstart, up=w, down=w + 1)
        if win is None or win.downstream[:3].upper() != expected:
            continue
        windows[tid] = win
    return windows


def _most_abundant(tids: list[str], abundance: Mapping[str, float]) -> str | None:
    """Pick the highest-abundance tid, breaking ties on the smallest tid.

    Deterministic: sorts by ``(-abundance, tid)`` so equal-abundance candidates
    resolve to the lexicographically smallest transcript id rather than an
    arbitrary one.
    """
    if not tids:
        return None
    return min(tids, key=lambda t: (-float(abundance.get(t, 0.0)), t))


def _divergent_decision(
    survivors: list[str],
    abundance: Mapping[str, float],
    *,
    dominance_frac: float | None,
    min_count: float | None,
) -> str | None:
    """Pick a source among divergent survivors subject to abundance thresholds.

    The top survivor (by long-read abundance) must clear every configured
    threshold to be returned; otherwise the site is unresolved (``None``). With
    no threshold configured this is most-abundant-wins.

    Args:
        survivors: Divergent candidate tids (≥ 2).
        abundance: ``{tid: long-read count}``.
        dominance_frac: Required fraction of total abundance for the top
            survivor, or ``None`` to skip the check.
        min_count: Required absolute abundance for the top survivor, or ``None``
            to skip the check.

    Returns:
        The chosen source tid, or ``None`` when a threshold is not met.
    """
    top = _most_abundant(survivors, abundance)
    if top is None:
        return None
    top_count = float(abundance.get(top, 0.0))
    if min_count is not None and top_count < min_count:
        return None
    if dominance_frac is not None:
        total = sum(float(abundance.get(t, 0.0)) for t in survivors)
        if total <= 0 or (top_count / total) < dominance_frac:
            return None
    return top


def _resolve_site(
    windows: Mapping[str, TisWindow],
    iso_present: set[str],
    iso_abundance: Mapping[str, float],
    norm_tis_by_tid: Mapping[str, float],
    *,
    window: int,
    dominance_frac: float | None,
    min_count: float | None,
) -> Resolution:
    """Run the cascade for a single site and return its verdict."""
    survivors = [t for t in windows if t in iso_present]
    if not survivors:
        return Resolution(False, "none", None, "unresolved", None)

    pur = purity_decision([windows[t] for t in survivors], window)
    if pur.reason in ("single_candidate", "pure_window"):
        status = "single" if pur.reason == "single_candidate" else "pure"
        source = _most_abundant(survivors, iso_abundance)
        evidence = "window_pure"
    else:  # ambiguous_window
        status = "divergent"
        source = _divergent_decision(
            survivors, iso_abundance, dominance_frac=dominance_frac, min_count=min_count
        )
        evidence = "divergent_abundance" if source is not None else "unresolved"

    if source is None:
        return Resolution(False, status, None, "unresolved", None)

    denom = float(iso_abundance.get(source, 0.0))
    norm = norm_tis_by_tid.get(source)
    tie = (norm / denom) if (norm is not None and denom > 0) else None
    return Resolution(True, status, source, evidence, tie)


def resolve_sources(
    filtered_df: pd.DataFrame,
    *,
    exon_skeletons: Mapping[str, TranscriptCoordinates],
    genome: Any,
    isoquant_table: str | Path,
    window: int = 100,
    isoquant_min_count: float = 3.0,
    divergence_dominance_frac: float | None = None,
    divergence_min_count: float | None = None,
) -> pd.DataFrame:
    """Resolve the source transcript for each TIS in one sample's filtered table.

    Groups the table by initiation site (``init_site`` derived from ``GenomePos``
    + ``StartCodon``); the candidate transcripts at a site are the ``Tid`` values
    in that group. Runs the long-read → window-purity → abundance cascade and
    attaches the verdict (:data:`VERDICT_COLUMNS`) onto every row, broadcast by
    initiation site. **Tag-only — no row is dropped.**

    Args:
        filtered_df: One sample's filtered TIS table (output of
            ``pipeline.run_sample``); must carry ``Tid`` / ``GenomePos`` /
            ``StartCodon`` and, for the efficiency metric, ``NormTISCounts``.
        exon_skeletons: ``{transcript_id: TranscriptCoordinates}``
            (``UpstreamReference.exon_skeletons``).
        genome: Object with ``.fetch(chrom, start, end) -> str`` (e.g.
            ``pysam.FastaFile``).
        isoquant_table: IsoQuant transcript-abundance TSV for this sample.
        window: Window radius W in nt.
        isoquant_min_count: IsoQuant presence threshold (long-read filter).
        divergence_dominance_frac: Dominance-fraction threshold for divergent
            sites, or ``None``.
        divergence_min_count: Absolute-count threshold for divergent sites, or
            ``None``.

    Returns:
        ``filtered_df`` with :data:`VERDICT_COLUMNS` appended (same row count).
    """
    if filtered_df.empty:
        out = filtered_df.copy()
        for col in VERDICT_COLUMNS:
            out[col] = pd.Series(dtype="object")
        return out

    iso_abundance = load_isoquant_abundance(isoquant_table)
    iso_present = expressed_transcripts(iso_abundance, isoquant_min_count)

    df = filtered_df.copy()
    init_site = _init_site_from_genome_pos(df["GenomePos"], df["StartCodon"])
    has_norm = "NormTISCounts" in df.columns

    verdicts: dict[str, Resolution] = {}
    for site, idx in init_site.groupby(init_site).groups.items():
        grp = df.loc[idx]
        _, gstart, _ = genome_pos_components(str(grp["GenomePos"].iloc[0]))
        start_codon = str(grp["StartCodon"].iloc[0])
        cand_tids = list(dict.fromkeys(grp["Tid"].astype(str)))

        windows = _candidate_windows(cand_tids, gstart, start_codon, exon_skeletons, genome, window)

        # Per-transcript TIS signal for the efficiency numerator (max over the
        # transcript's own rows at this site).
        norm_tis_by_tid: dict[str, float] = {}
        if has_norm:
            norm_tis_by_tid = (
                grp.groupby(grp["Tid"].astype(str))["NormTISCounts"].max().astype(float).to_dict()
            )

        verdicts[site] = _resolve_site(
            windows,
            iso_present,
            iso_abundance,
            norm_tis_by_tid,
            window=window,
            dominance_frac=divergence_dominance_frac,
            min_count=divergence_min_count,
        )

    for col in VERDICT_COLUMNS:
        df[col] = init_site.map(lambda s, c=col: getattr(verdicts[s], c)).values

    n_sites = len(verdicts)
    n_resolved = sum(v.resolved for v in verdicts.values())
    logger.info(
        "resolve_sources: %d init sites → %d resolved (%d unresolved); window_status %s",
        n_sites,
        n_resolved,
        n_sites - n_resolved,
        {
            s: sum(v.window_status == s for v in verdicts.values())
            for s in ("single", "pure", "divergent", "none")
        },
    )
    return df
