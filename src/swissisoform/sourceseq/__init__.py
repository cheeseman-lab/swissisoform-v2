"""Source-transcript resolution: pin each TIS to a high-confidence source mRNA.

Phase 1 (data-independent core): reconstruct a transcript's spliced mRNA from
its exon skeleton + genome, extract the sequence window around a TIS start
codon, and decide whether a TIS's surviving candidate transcripts agree within
that window (the window-purity test). Later phases add expression/CAGE
narrowing and the dataset export.

See ``docs/plans/source_transcript_resolution.md``.
"""

from __future__ import annotations

from swissisoform.sourceseq.mrna import (
    TisWindow,
    build_transcript_mrna,
    extract_tis_window,
    start_codon_index,
)
from swissisoform.sourceseq.purity import (
    PurityResult,
    divergence_radius,
    purity_decision,
)

__all__ = [
    "TisWindow",
    "build_transcript_mrna",
    "extract_tis_window",
    "start_codon_index",
    "PurityResult",
    "divergence_radius",
    "purity_decision",
]
