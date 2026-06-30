"""Read-distribution diagnostics for divergent source-resolution sites.

For choosing the divergent-case dominance threshold
(``SourceResolutionConfig.divergence_dominance_frac``) empirically: re-run the
front of the cascade (long-read filter → window-purity) and, for every
initiation site whose surviving candidates **diverge in-window**, record how the
sample's long-read reads split across those candidate transcripts. Backs
``scripts/export/export_source_divergence_distribution.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from swissisoform.combine import _init_site_from_genome_pos, genome_pos_components
from swissisoform.models import TranscriptCoordinates
from swissisoform.sourceresolve.expression import (
    expressed_transcripts,
    load_isoquant_abundance,
)
from swissisoform.sourceresolve.purity import purity_decision
from swissisoform.sourceresolve.resolve import _candidate_windows

# Per-site columns of the diagnostic table.
DISTRIBUTION_COLUMNS: tuple[str, ...] = (
    "init_site",
    "n_candidates",
    "transcripts",
    "abundances",
    "fractions",
    "total_abundance",
    "top_fraction",
    "top_to_runnerup_ratio",
)


def divergent_site_distribution(
    filtered_df: pd.DataFrame,
    *,
    exon_skeletons: Mapping[str, TranscriptCoordinates],
    genome: Any,
    isoquant_table: str | Path,
    window_upstream: int = 100,
    window_downstream: int = 100,
    isoquant_min_count: float = 3.0,
) -> pd.DataFrame:
    """Per-site long-read abundance distribution for divergent-window sites.

    Mirrors the front of :func:`resolve_sources` (long-read filter →
    ``purity_decision``) but, instead of picking a source, records the
    abundance split across the divergent survivors at each ambiguous site.

    Args:
        filtered_df: One sample's filtered TIS table (``Tid`` / ``GenomePos`` /
            ``StartCodon``).
        exon_skeletons: ``{transcript_id: TranscriptCoordinates}``.
        genome: Object with ``.fetch(chrom, start, end) -> str``.
        isoquant_table: IsoQuant transcript-abundance TSV for this sample.
        window_upstream: nt 5' of the start codon in the purity window.
        window_downstream: nt 3' of the start codon in the purity window.
        isoquant_min_count: IsoQuant presence threshold (long-read filter).

    Returns:
        One row per divergent site, columns :data:`DISTRIBUTION_COLUMNS`.
        ``transcripts`` / ``abundances`` / ``fractions`` are lists sorted by
        descending abundance. Empty (zero-row) frame when no site diverges.
    """
    iso_abundance = load_isoquant_abundance(isoquant_table)
    iso_present = expressed_transcripts(iso_abundance, isoquant_min_count)

    if filtered_df.empty:
        return pd.DataFrame(columns=list(DISTRIBUTION_COLUMNS))

    init_site = _init_site_from_genome_pos(filtered_df["GenomePos"], filtered_df["StartCodon"])

    rows: list[dict[str, Any]] = []
    for site, idx in init_site.groupby(init_site).groups.items():
        grp = filtered_df.loc[idx]
        _, gstart, _ = genome_pos_components(str(grp["GenomePos"].iloc[0]))
        start_codon = str(grp["StartCodon"].iloc[0])
        cand_tids = list(dict.fromkeys(grp["Tid"].astype(str)))

        windows = _candidate_windows(
            cand_tids, gstart, start_codon, exon_skeletons, genome,
            window_upstream, window_downstream,
        )
        survivors = [t for t in windows if t in iso_present]
        if len(survivors) < 2:
            continue
        pur = purity_decision(
            [windows[t] for t in survivors], window_upstream, window_downstream
        )
        if pur.reason != "ambiguous_window":
            continue

        ranked = sorted(survivors, key=lambda t: (-float(iso_abundance.get(t, 0.0)), t))
        abundances = [float(iso_abundance.get(t, 0.0)) for t in ranked]
        total = sum(abundances)
        fractions = [a / total for a in abundances] if total > 0 else [0.0] * len(abundances)
        top_frac = fractions[0] if fractions else 0.0
        has_runnerup = len(abundances) > 1 and abundances[1] > 0
        ratio = (abundances[0] / abundances[1]) if has_runnerup else None

        rows.append(
            {
                "init_site": site,
                "n_candidates": len(ranked),
                "transcripts": ranked,
                "abundances": abundances,
                "fractions": fractions,
                "total_abundance": total,
                "top_fraction": top_frac,
                "top_to_runnerup_ratio": ratio,
            }
        )

    return pd.DataFrame(rows, columns=list(DISTRIBUTION_COLUMNS))
