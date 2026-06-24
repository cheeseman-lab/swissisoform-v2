"""Per-sample source-transcript resolution: pick one source mRNA per TIS.

Consolidates the three driver steps of Elizabeth's source-transcript workstream
(``candidate_mrna_divergence`` → ``disambiguate_expression`` → ``build_union3``)
into a single per-sample function, :func:`resolve_sources`, that runs inside the
filtering cascade (``pipeline.run_sample``). It depends on the sample's own
RNA-seq, so it is intrinsically per cell line.

Three arms, unioned per genomic initiation site (``init_site``):

- **S — sequence purity** (Tier 1): do the site's candidate transcripts share the
  same local sequence within ±W of the start codon? If so the window is
  unambiguous regardless of which isoform a footprint came from.
- **A — short-read expression** (salmon): keep candidates expressed in the
  sample, re-test purity on the survivors.
- **B — long-read expression** (IsoQuant): same, with long-read counts.

``resolved = S or A or B``; ``tier = 1 if S else 2 if B else 3`` (``0`` =
unresolved); source-transcript precedence ``long_read > short_read > sequence``.
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
    expressed_in_replicates,
    expressed_transcripts,
    load_isoquant_abundance,
    load_salmon_replicates,
)
from swissisoform.sourceresolve.mrna import TisWindow, extract_tis_window
from swissisoform.sourceresolve.purity import purity_decision

logger = logging.getLogger(__name__)

# Columns the stage attaches to every row of the filtered table.
VERDICT_COLUMNS: tuple[str, ...] = (
    "resolved",
    "agreement_tier",
    "source_transcript",
    "source_evidence",
    "tie_initiation_efficiency",
)


@dataclass
class ArmResult:
    """Outcome of one expression arm (salmon or IsoQuant) for a single TIS.

    Attributes:
        keep: Whether the expressed survivors pass the window-purity re-test.
        source_tid: The max-abundance surviving transcript, or ``None``.
        abundance_sum: Summed abundance over agreeing survivors (the TIE
            denominator), or ``None`` when no candidate is expressed.
    """

    keep: bool
    source_tid: str | None
    abundance_sum: float | None


@dataclass
class Resolution:
    """The three-way union verdict for a single initiation site.

    Attributes:
        resolved: ``True`` if any arm resolves the site.
        agreement_tier: ``1`` sequence-pure, ``2`` long-read, ``3`` salmon-only,
            ``0`` unresolved.
        source_transcript: The chosen source mRNA, or ``None`` if unresolved.
        source_evidence: ``"long_read"`` | ``"short_read"`` | ``"sequence"`` |
            ``"unresolved"``.
        tie_initiation_efficiency: ``NormTISCounts`` divided by the source
            abundance denominator (salmon-TPM units), or ``None``.
    """

    resolved: bool
    agreement_tier: int
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
    dropped. ``down=w + 1`` so radius ``w`` (the w-th base 3' of the A) is
    reachable by the purity test.
    """
    windows: dict[str, TisWindow] = {}
    for tid in cand_tids:
        coords = exon_skeletons.get(tid)
        if coords is None:
            continue
        win = extract_tis_window(coords, genome, gstart, up=w, down=w + 1)
        if win is None or win.downstream[:3] != start_codon:
            continue
        windows[tid] = win
    return windows


def _expression_arm(
    windows: Mapping[str, TisWindow],
    abundance: Mapping[str, float],
    present: set[str],
    w: int,
) -> ArmResult:
    """Filter windowed candidates to expressed ones, re-test purity, pick source.

    The source is the max-abundance surviving transcript; ``abundance_sum`` is the
    total over survivors (the efficiency denominator when several agree in-window).
    """
    kept = {t: wn for t, wn in windows.items() if t in present}
    pur = purity_decision(list(kept.values()), w)
    if not kept:
        return ArmResult(keep=pur.keep, source_tid=None, abundance_sum=None)
    survivors = [(t, float(abundance.get(t, 0.0))) for t in kept]
    source_tid = max(survivors, key=lambda x: x[1])[0]
    abundance_sum = sum(v for _, v in survivors)
    return ArmResult(keep=pur.keep, source_tid=source_tid, abundance_sum=abundance_sum)


def _union(
    *,
    seq_pure: bool,
    windowed_tids: list[str],
    salmon_arm: ArmResult | None,
    iso_arm: ArmResult | None,
    salmon_abundance: Mapping[str, float],
    norm_tis: float | None,
) -> Resolution:
    """Combine the three arms into one verdict (see module docstring)."""
    S = seq_pure
    A = bool(salmon_arm and salmon_arm.keep and salmon_arm.source_tid)
    B = bool(iso_arm and iso_arm.keep and iso_arm.source_tid)

    if not (S or A or B):
        return Resolution(False, 0, None, "unresolved", None)

    tier = 1 if S else (2 if B else 3)
    if B:
        source, evidence = iso_arm.source_tid, "long_read"
    elif A:
        source, evidence = salmon_arm.source_tid, "short_read"
    else:
        # sequence-pure: the source is the most-abundant windowed candidate.
        source = (
            max(windowed_tids, key=lambda t: salmon_abundance.get(t, 0.0))
            if windowed_tids
            else None
        )
        evidence = "sequence"

    # TIE denominator: summed salmon TPM over agreeing survivors when salmon
    # resolved (keeps the quotient in salmon-TPM units), else the source TPM.
    src_tpm = float(salmon_abundance.get(source, 0.0)) if source else 0.0
    denom = salmon_arm.abundance_sum if (A and salmon_arm.abundance_sum) else src_tpm
    tie = (norm_tis / denom) if (norm_tis is not None and denom and denom > 0) else None

    return Resolution(True, tier, source, evidence, tie)


def resolve_sources(
    filtered_df: pd.DataFrame,
    *,
    exon_skeletons: Mapping[str, TranscriptCoordinates],
    genome: Any,
    salmon_quant: list[str | Path] | None = None,
    isoquant_table: str | Path | None = None,
    window: int = 100,
    salmon_min_tpm: float = 0.1,
    isoquant_min_count: float = 3.0,
) -> pd.DataFrame:
    """Resolve the source transcript for each TIS in one sample's filtered table.

    Groups the table by initiation site (``init_site`` derived from ``GenomePos``
    + ``StartCodon``); the candidate transcripts at a site are the ``Tid`` values
    in that group. Runs the sequence-purity + salmon + IsoQuant arms and attaches
    the union verdict (:data:`VERDICT_COLUMNS`) onto every row, broadcast by
    initiation site. **Tag-only — no row is dropped.**

    Args:
        filtered_df: One sample's filtered TIS table (output of
            ``pipeline.run_sample``); must carry ``Tid`` / ``GenomePos`` /
            ``StartCodon`` and, for TIE, ``NormTISCounts``.
        exon_skeletons: ``{transcript_id: TranscriptCoordinates}``
            (``UpstreamReference.exon_skeletons``).
        genome: Object with ``.fetch(chrom, start, end) -> str`` (e.g.
            ``pysam.FastaFile``).
        salmon_quant: Per-replicate ``quant.sf`` paths for this sample, or
            ``None`` to skip the short-read arm.
        isoquant_table: IsoQuant transcript-abundance TSV for this sample, or
            ``None`` to skip the long-read arm.
        window: Window radius W in nt.
        salmon_min_tpm: Per-replicate salmon presence threshold.
        isoquant_min_count: IsoQuant presence threshold.

    Returns:
        ``filtered_df`` with :data:`VERDICT_COLUMNS` appended (same row count).
    """
    if filtered_df.empty:
        out = filtered_df.copy()
        for col in VERDICT_COLUMNS:
            out[col] = pd.Series(dtype="object")
        return out

    salmon_present: set[str] = set()
    salmon_abundance: dict[str, float] = {}
    if salmon_quant:
        salmon_present = expressed_in_replicates(
            salmon_quant, min_tpm=salmon_min_tpm, require="all"
        )
        salmon_abundance = load_salmon_replicates(salmon_quant)

    iso_present: set[str] = set()
    iso_abundance: dict[str, float] = {}
    if isoquant_table:
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
        seq_pur = purity_decision(list(windows.values()), window)
        salmon_arm = (
            _expression_arm(windows, salmon_abundance, salmon_present, window)
            if salmon_quant
            else None
        )
        iso_arm = (
            _expression_arm(windows, iso_abundance, iso_present, window) if isoquant_table else None
        )
        norm_tis = float(grp["NormTISCounts"].max()) if has_norm else None

        verdicts[site] = _union(
            seq_pure=seq_pur.keep,
            windowed_tids=list(windows),
            salmon_arm=salmon_arm,
            iso_arm=iso_arm,
            salmon_abundance=salmon_abundance,
            norm_tis=norm_tis,
        )

    df["resolved"] = init_site.map(lambda s: verdicts[s].resolved).values
    df["agreement_tier"] = init_site.map(lambda s: verdicts[s].agreement_tier).values
    df["source_transcript"] = init_site.map(lambda s: verdicts[s].source_transcript).values
    df["source_evidence"] = init_site.map(lambda s: verdicts[s].source_evidence).values
    df["tie_initiation_efficiency"] = init_site.map(
        lambda s: verdicts[s].tie_initiation_efficiency
    ).values

    n_sites = len(verdicts)
    n_resolved = sum(v.resolved for v in verdicts.values())
    logger.info(
        "resolve_sources: %d init sites → %d resolved (%d unresolved); tiers %s",
        n_sites,
        n_resolved,
        n_sites - n_resolved,
        {t: sum(v.agreement_tier == t for v in verdicts.values()) for t in (1, 2, 3)},
    )
    return df
