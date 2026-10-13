"""Resolve VCF variants against annotated alternative-ORF coordinates.

Answers one question, fast: does this variant land inside an annotated ORF, and
if so which isoform, which residue, and is that residue in the isoform-unique or
the shared region?

No Flask, no web server, and no genome: everything here runs off
``orf_index.parquet``, which is what lets the website vendor this package into its
Railway image (the seam ``website/prepare_deploy.sh`` already uses for
``swissisoform.site.evidence``) and keeps it unit-testable without a run. The
pyarrow read that materialises an index lives in :mod:`load`, not here.

Consequence classification is **not** implemented here — it is
:mod:`swissisoform.variantclass`, the pipeline's own classifier (which
``ConsequenceValidator`` delegates to), called with the coding sequence the index
carries. That module needs only the standard library, and so does everything this
package imports: the scan runs without pandas, Biopython or pyarrow, which is what
lets it run in the browser under Pyodide (issue #33). Keep it that way — an import
of either here would pull tens of MB into every browser that scans. What stays
local is what the pipeline has no counterpart for — resolving a bare coordinate
to an ORF, and the sequence-free fallback for an index built without a genome.

Coordinate conventions, inherited from :mod:`swissisoform.coords` and
``io/parquet.py``:

* ORF intervals are **0-based half-open, plus-strand, ascending** regardless of
  the transcript's strand.
* VCF positions are **1-based**, so membership is ``start < pos <= end`` — the
  same predicate as ``modules/variant_intersection.py``.
"""

from __future__ import annotations

from swissisoform.variantquery.index import OrfIndex, OrfRecord
from swissisoform.variantquery.scan import ScanResult, VariantHit, scan
from swissisoform.variantquery.spec import Rejection, VariantSpec, normalize_chrom, parse_line
from swissisoform.variantquery.vcf import (
    VcfLimitExceeded,
    iter_data_lines,
    iter_lines,
    open_vcf,
)

__all__ = [
    "OrfIndex",
    "OrfRecord",
    "Rejection",
    "ScanResult",
    "VariantHit",
    "VariantSpec",
    "VcfLimitExceeded",
    "iter_data_lines",
    "iter_lines",
    "normalize_chrom",
    "open_vcf",
    "parse_line",
    "scan",
]
