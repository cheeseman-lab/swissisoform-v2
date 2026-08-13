"""Resolve VCF variants against annotated alternative-ORF coordinates.

Answers one question, fast: does this variant land inside an annotated ORF, and
if so which isoform, which residue, and is that residue in the isoform-unique or
the shared region?

Deliberately **stdlib-only** — no pandas, no pysam, no Flask. That is what lets
the website vendor this package into its Railway image (the seam
``website/prepare_deploy.sh`` already uses for ``swissisoform.site.evidence``)
and what keeps it unit-testable without a run or a web server. The pyarrow read
that materialises an index lives at the call sites, not here.

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
from swissisoform.variantquery.vcf import iter_data_lines, open_vcf

__all__ = [
    "OrfIndex",
    "OrfRecord",
    "Rejection",
    "ScanResult",
    "VariantHit",
    "VariantSpec",
    "iter_data_lines",
    "normalize_chrom",
    "open_vcf",
    "parse_line",
    "scan",
]
