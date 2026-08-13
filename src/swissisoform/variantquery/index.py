"""Sorted genomic-interval index over annotated ORFs.

A VCF scan asks one question tens of thousands of times — *does this position
fall inside any ORF?* — so the interval set is flattened into per-chromosome
sorted arrays and queried by bisection. A linear scan would be ~54,000 interval
comparisons per variant.

Both the isoform ORF (``orf_exons``) and the per-transcript canonical ORF
(``canonical_orf_exons``) are indexed. That matters for **truncations**: the lost
N-terminal region is in the canonical ORF but *not* in the isoform's, and a
variant there is exactly the interesting case. Indexing only ``orf_exons`` would
miss it.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

Interval = tuple[int, int]

#: Columns projected out of ``all_paired.parquet`` by ``build_orf_index.py``.
INDEX_COLUMNS = (
    "gene_name",
    "tis_id",
    "transcript_id",
    "chrom",
    "strand",
    "orf_exons",
    "canonical_orf_exons",
    "unique_genomic_intervals",
    "shared_genomic_intervals",
    "canonical_len",
    "isoform_len",
    "diff_space",
    "orf_type",
    "start_codon",
    "canonical_start_codon",
)

#: Coding sequences the builder *derives* from the genome rather than projecting.
#: Optional so an index built before they existed still loads — consequence
#: classification then degrades to length-based classes only.
CDS_COLUMNS = ("orf_cds", "canonical_cds")


def _as_intervals(raw: Any) -> tuple[Interval, ...]:
    """Coerce a nested list/array of pairs into a tuple of int 2-tuples."""
    if raw is None:
        return ()
    out: list[Interval] = []
    for item in raw:
        if item is None:
            continue
        pair = tuple(item)
        if len(pair) != 2:
            raise ValueError(f"interval must have 2 elements, got {pair!r}")
        out.append((int(pair[0]), int(pair[1])))
    return tuple(out)


def _int_or_none(raw: Any) -> int | None:
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value


@dataclass(frozen=True, slots=True)
class OrfRecord:
    """One annotated ORF: its identity, its exons, and its region partition.

    All interval sets are 0-based half-open plus-strand, ascending by genomic
    coordinate irrespective of ``strand`` (see :mod:`swissisoform.coords`).
    """

    gene_name: str
    tis_id: str
    transcript_id: str
    chrom: str
    strand: str
    orf_exons: tuple[Interval, ...]
    canonical_orf_exons: tuple[Interval, ...]
    unique_intervals: tuple[Interval, ...]
    shared_intervals: tuple[Interval, ...]
    canonical_len: int | None
    isoform_len: int | None
    diff_space: str
    orf_type: str
    start_codon: str = ""
    canonical_start_codon: str = ""
    #: Coding sequence in mRNA order, no trailing stop. Empty when the index was
    #: built without a genome — consequence then falls back to length-based classes.
    orf_cds: str = ""
    canonical_cds: str = ""

    def cds_for(self, frame: str) -> str:
        """The CDS matching a frame from ``resolve_residue`` (isoform/canonical)."""
        return self.canonical_cds if frame == "canonical" else self.orf_cds

    def exons_for(self, frame: str) -> tuple[Interval, ...]:
        """The exons matching a frame, so callers cannot pair them mismatched."""
        return self.canonical_orf_exons if frame == "canonical" else self.orf_exons

    def start_codon_for(self, frame: str) -> str:
        """The start codon matching a frame."""
        return self.canonical_start_codon if frame == "canonical" else self.start_codon

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> OrfRecord:
        """Build from a parquet row (dict-like) with :data:`INDEX_COLUMNS`."""
        return cls(
            gene_name=str(row.get("gene_name") or ""),
            tis_id=str(row.get("tis_id") or ""),
            transcript_id=str(row.get("transcript_id") or ""),
            chrom=str(row.get("chrom") or ""),
            strand=str(row.get("strand") or ""),
            orf_exons=_as_intervals(row.get("orf_exons")),
            canonical_orf_exons=_as_intervals(row.get("canonical_orf_exons")),
            unique_intervals=_as_intervals(row.get("unique_genomic_intervals")),
            shared_intervals=_as_intervals(row.get("shared_genomic_intervals")),
            canonical_len=_int_or_none(row.get("canonical_len")),
            isoform_len=_int_or_none(row.get("isoform_len")),
            diff_space=str(row.get("diff_space") or ""),
            orf_type=str(row.get("orf_type") or ""),
            start_codon=str(row.get("start_codon") or ""),
            canonical_start_codon=str(row.get("canonical_start_codon") or ""),
            orf_cds=str(row.get("orf_cds") or ""),
            canonical_cds=str(row.get("canonical_cds") or ""),
        )


@dataclass
class _ChromIndex:
    """Exon intervals on one chromosome, sorted by start.

    ``max_len`` bounds how far back a query must walk: any interval containing
    position ``p`` satisfies ``start >= p - max_len``, because
    ``start >= end - max_len`` and ``end >= p``.
    """

    starts: list[int] = field(default_factory=list)
    ends: list[int] = field(default_factory=list)
    owners: list[int] = field(default_factory=list)
    max_len: int = 0


class OrfIndex:
    """Point/span membership over every annotated ORF exon.

    Build with :meth:`from_records`; query with :meth:`lookup` (a single 1-based
    position) or :meth:`lookup_span` (an inclusive 1-based range, for indels).
    """

    __slots__ = ("_by_chrom", "_records", "_tis_index", "version")

    def __init__(
        self,
        records: Sequence[OrfRecord],
        by_chrom: dict[str, _ChromIndex],
        version: str = "",
    ) -> None:
        """Store the records and their prebuilt per-chromosome interval arrays."""
        self._records = list(records)
        self._by_chrom = by_chrom
        self._tis_index: dict[str, OrfRecord] | None = None
        self.version = version

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_records(
        cls,
        rows: Iterable[Mapping[str, Any] | OrfRecord],
        *,
        version: str = "",
    ) -> OrfIndex:
        """Build the index from parquet rows or ready-made :class:`OrfRecord`s.

        Takes plain mappings rather than a DataFrame so this module stays free of
        pandas — the pyarrow read lives at the call site.
        """
        records: list[OrfRecord] = []
        by_chrom: dict[str, _ChromIndex] = {}
        # (start, end, owner) triples per chrom, sorted once at the end — cheaper
        # than keeping three parallel lists ordered during insertion.
        staging: dict[str, list[tuple[int, int, int]]] = {}

        for row in rows:
            record = row if isinstance(row, OrfRecord) else OrfRecord.from_mapping(row)
            if not record.chrom:
                continue
            owner = len(records)
            records.append(record)

            # Union of the two ORFs, deduplicated: an extension's canonical exons
            # are mostly a subset of its own, so most pairs collapse.
            seen: set[Interval] = set()
            for start, end in (*record.orf_exons, *record.canonical_orf_exons):
                if end <= start or (start, end) in seen:
                    continue
                seen.add((start, end))
                staging.setdefault(record.chrom, []).append((start, end, owner))

        for chrom, triples in staging.items():
            triples.sort()
            chrom_index = _ChromIndex(
                starts=[t[0] for t in triples],
                ends=[t[1] for t in triples],
                owners=[t[2] for t in triples],
                max_len=max(t[1] - t[0] for t in triples),
            )
            by_chrom[chrom] = chrom_index

        return cls(records, by_chrom, version=version)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def lookup(self, chrom: str, pos: int) -> list[OrfRecord]:
        """ORFs whose exons contain 1-based ``pos`` (``start < pos <= end``)."""
        return self.lookup_span(chrom, pos, pos)

    def lookup_span(self, chrom: str, start: int, end: int) -> list[OrfRecord]:
        """ORFs with an exon overlapping the inclusive 1-based range.

        An exon ``[s, e)`` covers 1-based positions ``s+1 .. e``, so it overlaps
        ``[start, end]`` when ``s < end and e >= start``.
        """
        chrom_index = self._by_chrom.get(chrom)
        if chrom_index is None:
            return []

        starts = chrom_index.starts
        # Everything at or after this index has s >= end, so cannot overlap.
        cutoff = bisect_left(starts, end)
        floor = start - chrom_index.max_len

        owners: list[int] = []
        seen: set[int] = set()
        for i in range(cutoff - 1, -1, -1):
            if starts[i] < floor:
                # Sorted, so no earlier interval can reach `start` either.
                break
            if chrom_index.ends[i] >= start:
                owner = chrom_index.owners[i]
                if owner not in seen:
                    seen.add(owner)
                    owners.append(owner)

        # Stable, run-independent order so digests are reproducible.
        return sorted((self._records[o] for o in owners), key=lambda r: r.tis_id)

    def by_tis_id(self, tis_id: str) -> OrfRecord | None:
        """The record for one ``tis_id``, or None.

        Built lazily on first use: the gene figure needs each hit's ORF to convert a
        residue into a plot coordinate (it needs ``canonical_len``/``isoform_len``),
        and scanning ``records`` per hit would be quadratic on a busy gene.
        """
        if self._tis_index is None:
            self._tis_index = {r.tis_id: r for r in self._records}
        return self._tis_index.get(tis_id)

    def has_chrom(self, chrom: str) -> bool:
        """True when the index carries any ORF on ``chrom``.

        Lets the scan tell "off-catalog contig" apart from "no ORF here", which
        is the difference between a naming mismatch and a real negative.
        """
        return chrom in self._by_chrom

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def records(self) -> Sequence[OrfRecord]:
        """Every indexed ORF, in insertion order."""
        return self._records

    @property
    def n_isoforms(self) -> int:
        """Number of indexed ORFs."""
        return len(self._records)

    @property
    def n_genes(self) -> int:
        """Number of distinct genes represented."""
        return len({r.gene_name for r in self._records if r.gene_name})

    @property
    def n_intervals(self) -> int:
        """Number of exon intervals across all chromosomes."""
        return sum(len(c.starts) for c in self._by_chrom.values())

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        """Compact size summary for logs."""
        return (
            f"OrfIndex(isoforms={self.n_isoforms}, genes={self.n_genes}, "
            f"intervals={self.n_intervals}, chroms={len(self._by_chrom)}, "
            f"version={self.version!r})"
        )
