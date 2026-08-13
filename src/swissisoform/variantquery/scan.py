"""Walk a VCF against an :class:`OrfIndex` and emit hits plus a funnel.

The funnel is not decoration. Measured against the full catalogue, 28 of 34,706
PASS variants in a real somatic VCF land inside an annotated ORF (~0.08%), so
"no hits" is the *normal* outcome and has to be explainable — how many records
were read, how many passed the filter, how many were on contigs the catalogue
covers at all. A bare zero is indistinguishable from a broken scan.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from swissisoform.variantquery.frame import region_for, resolve_residue
from swissisoform.variantquery.index import OrfIndex
from swissisoform.variantquery.spec import Rejection, VariantSpec, parse_line
from swissisoform.variantquery.vcf import iter_data_lines

#: Hard ceiling on stored hits, so a germline whole-exome VCF cannot fill the
#: disk. Exceeding it sets ``truncated`` rather than failing the scan.
DEFAULT_MAX_HITS = 20_000


@dataclass(frozen=True, slots=True)
class VariantHit:
    """One (variant, ORF) pair. A position in N isoforms yields N hits.

    ``residue`` is **0-based**, matching ``protein_pos`` elsewhere in the
    pipeline — so TP53 p.R248 reports ``residue=247``.
    """

    line_no: int
    chrom: str
    pos: int
    ref: str
    alt: str
    vclass: str
    gene: str
    tis_id: str
    transcript_id: str
    orf_type: str
    frame: str
    residue: int | None
    region: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form for the scan digest."""
        return asdict(self)


@dataclass
class ScanCounts:
    """Funnel from raw lines down to hits."""

    lines: int = 0
    alleles: int = 0
    skipped_non_pass: int = 0
    off_catalog_contig: int = 0
    no_orf: int = 0
    hits: int = 0
    genes_hit: int = 0
    truncated: bool = False
    rejected: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        """Tally one rejected allele under ``reason``."""
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form for the scan digest."""
        return asdict(self)


@dataclass
class GeneSummary:
    """Per-gene rollup for the results list.

    Two counts, because they differ and the difference is the interesting part: a
    single variant inside three isoforms of one gene is **1 variant, 3 hits**.
    ``n_unique``/``n_shared`` are hit-record counts too — the same nucleotide can be
    isoform-unique in one ORF and shared in another, which is a real observation
    rather than a bookkeeping artifact.
    """

    gene: str
    n_hits: int = 0
    n_unique: int = 0
    n_shared: int = 0
    tis_ids: set[str] = field(default_factory=set)
    #: (line_no, alt) — identifies one allele occurrence in the source file, so the
    #: same position on two VCF lines counts twice while one line in N isoforms
    #: counts once.
    variant_keys: set[tuple[int, str]] = field(default_factory=set)

    @property
    def n_variants(self) -> int:
        """Distinct alleles from the VCF that hit this gene."""
        return len(self.variant_keys)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form; the two sets collapse to counts."""
        return {
            "gene": self.gene,
            "n_variants": self.n_variants,
            "n_hits": self.n_hits,
            "n_unique": self.n_unique,
            "n_shared": self.n_shared,
            "n_isoforms": len(self.tis_ids),
        }


@dataclass
class ScanResult:
    """Everything one scan produced: hits, per-gene rollup, and the funnel."""

    counts: ScanCounts
    hits: list[VariantHit]
    genes: list[GeneSummary]
    index_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form for the scan digest."""
        return {
            "counts": self.counts.to_dict(),
            "genes": [g.to_dict() for g in self.genes],
            "hits": [h.to_dict() for h in self.hits],
        }


def scan(
    path: str | Path,
    index: OrfIndex,
    *,
    pass_only: bool = True,
    max_hits: int = DEFAULT_MAX_HITS,
) -> ScanResult:
    """Resolve every variant in ``path`` against ``index``.

    Args:
        path: VCF, plain or gzipped (detected by magic bytes).
        index: The ORF interval index to resolve against.
        pass_only: Drop records whose FILTER is neither ``PASS`` nor ``.``.
        max_hits: Stop *recording* hits past this many; counting continues.

    Returns:
        A :class:`ScanResult`. Distinct negatives are kept apart in the counts:
        ``off_catalog_contig`` means the chromosome is absent from the index
        (usually a naming mismatch or a scaffold), ``no_orf`` means the contig is
        covered but the position is intronic, UTR or intergenic.
    """
    counts = ScanCounts()
    hits: list[VariantHit] = []
    genes: dict[str, GeneSummary] = {}

    for line_no, line in iter_data_lines(path):
        counts.lines += 1
        for parsed in parse_line(line):
            if isinstance(parsed, Rejection):
                counts.reject(parsed.reason)
                continue
            counts.alleles += 1
            spec: VariantSpec = parsed
            if pass_only and not spec.is_pass:
                counts.skipped_non_pass += 1
                continue
            if not index.has_chrom(spec.chrom):
                counts.off_catalog_contig += 1
                continue

            start, end = spec.span
            records = index.lookup_span(spec.chrom, start, end)
            if not records:
                counts.no_orf += 1
                continue

            for record in records:
                residue, frame, _gpos = resolve_residue(record, start, end)
                region = region_for(record, start, end)
                counts.hits += 1
                summary = genes.setdefault(record.gene_name, GeneSummary(record.gene_name))
                summary.n_hits += 1
                summary.tis_ids.add(record.tis_id)
                summary.variant_keys.add((line_no, spec.alt))
                if region == "unique":
                    summary.n_unique += 1
                elif region == "shared":
                    summary.n_shared += 1
                if len(hits) < max_hits:
                    hits.append(
                        VariantHit(
                            line_no=line_no,
                            chrom=spec.chrom,
                            pos=spec.pos,
                            ref=spec.ref,
                            alt=spec.alt,
                            vclass=spec.vclass,
                            gene=record.gene_name,
                            tis_id=record.tis_id,
                            transcript_id=record.transcript_id,
                            orf_type=record.orf_type,
                            frame=frame,
                            residue=residue,
                            region=region,
                        )
                    )
                else:
                    counts.truncated = True

    counts.genes_hit = len(genes)
    # Rank by distinct variants first — that is what "most affected gene" means —
    # then by hit records, then alphabetically for a stable order.
    ordered = sorted(genes.values(), key=lambda g: (-g.n_variants, -g.n_hits, g.gene))
    return ScanResult(counts=counts, hits=hits, genes=ordered, index_version=index.version)
