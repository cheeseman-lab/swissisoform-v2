"""Source-transcript resolution subpackage.

Pins each Ribo-TISH TIS to one high-confidence source mRNA, as a per-sample
filtering step driven by that sample's own RNA-seq. Three arms, unioned:

- ``mrna`` / ``purity``: sequence window-purity (Tier 1) — do a TIS's candidate
  transcripts share the same local sequence around the start codon, so the
  window is unambiguous regardless of which isoform a footprint came from?
- ``expression``: salmon (short-read) / IsoQuant (long-read) loaders that narrow
  the candidate set to transcripts expressed in the sample.
- ``resolve``: the per-sample orchestrator (:func:`resolve_sources`) that runs
  the arms, picks the source transcript by precedence, and tags every TIS with
  its resolution verdict (tag-only — no row is dropped).

The read alignment / quantification that feeds the expression arms lives outside
the repo (the gitignored ``sourceseq/`` tool); only the disambiguation science
lives here, tested, in the filtering cascade.
"""

from swissisoform.sourceresolve.expression import (
    expressed_in_replicates,
    expressed_transcripts,
    load_isoquant_abundance,
    load_salmon_replicates,
    load_salmon_tpm,
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

__all__ = [
    "PurityResult",
    "TisWindow",
    "build_transcript_mrna",
    "divergence_radius",
    "expressed_in_replicates",
    "expressed_transcripts",
    "extract_tis_window",
    "load_isoquant_abundance",
    "load_salmon_replicates",
    "load_salmon_tpm",
    "purity_decision",
    "start_codon_index",
]
