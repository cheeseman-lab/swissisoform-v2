"""Source-transcript resolution subpackage.

Pins each Ribo-TISH TIS to one high-confidence source mRNA, as a per-sample
filtering step driven by that sample's own long-read RNA-seq. A single linear
cascade:

- **long-read filter** (``expression``): narrow the candidate set to transcripts
  present in the sample's IsoQuant table.
- **window-purity** (``mrna`` / ``purity``): do the surviving candidates share the
  same local sequence around the start codon, so the window is unambiguous
  regardless of which isoform a footprint came from?
- **abundance labeling** (``resolve``): pick the source by long-read abundance —
  most-abundant survivor when the window is pure, or a threshold decision when the
  survivors diverge. :func:`resolve_sources` is the per-sample orchestrator and
  tags every TIS with its verdict (tag-only — no row is dropped).

The read alignment / quantification that feeds the long-read filter lives outside
the repo (the gitignored ``sourceseq/`` tool); only the disambiguation science
lives here, tested, in the filtering cascade.
"""

from swissisoform.sourceresolve.expression import (
    expressed_transcripts,
    load_isoquant_abundance,
)
from swissisoform.sourceresolve.mrna import (
    TisWindow,
    build_transcript_mrna,
    extract_tis_window,
    start_codon_index,
)
from swissisoform.sourceresolve.purity import (
    PurityResult,
    divergence_radius,
    purity_decision,
)
from swissisoform.sourceresolve.resolve import (
    VERDICT_COLUMNS,
    Resolution,
    resolve_sources,
)

__all__ = [
    "VERDICT_COLUMNS",
    "PurityResult",
    "Resolution",
    "TisWindow",
    "build_transcript_mrna",
    "divergence_radius",
    "expressed_transcripts",
    "extract_tis_window",
    "load_isoquant_abundance",
    "purity_decision",
    "resolve_sources",
    "start_codon_index",
]
