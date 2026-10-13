"""Sorted genomic-interval index over annotated ORFs.

A VCF scan asks one question tens of thousands of times — *does this position
fall inside any ORF?* — so the interval set is flattened into per-chromosome
sorted arrays and queried by bisection. A linear scan would be ~54,000 interval
comparisons per variant.

The index is **packed**: every ORF field is one column — a list of strings, or a
stdlib ``array('q')`` of 8-byte integers, with each interval field flattened into
one bounds array plus per-ORF offsets (:class:`OrfColumns`). An :class:`OrfRecord`
is built only when a query returns that ORF. As one Python object per value, the
genome-wide catalogue cost ~280 MB per gunicorn worker against ~90 MB of actual
data, and the browser scan (Pyodide) holds the same index in a tab. ``array``
rather than numpy because the browser must not download numpy (issue #33, item 3).

Both the isoform ORF (``orf_exons``) and the per-transcript canonical ORF
(``canonical_orf_exons``) are indexed. That matters for **truncations**: the lost
N-terminal region is in the canonical ORF but *not* in the isoform's, and a
variant there is exactly the interesting case. Indexing only ``orf_exons`` would
miss it.
"""

from __future__ import annotations

from array import array
from bisect import bisect_left
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
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

#: Likewise derived, from the GTF: the figure's isoform-to-canonical x shift.
#: Optional for the same reason — without it the figure right-aligns, which is
#: correct for extensions and truncations and undefined for uORFs / altORFs.
DERIVED_COLUMNS = ("canonical_x_offset_nt",)

#: Projected like :data:`INDEX_COLUMNS`, but optional so an index built before the
#: column existed still loads. ``canonical_per_tid_length`` is the length of the
#: per-transcript canonical that ``canonical_orf_exons`` describes, which the figure
#: needs to place a canonical-frame residue on the gene-level bar it draws.
OPTIONAL_COLUMNS = ("canonical_per_tid_length",)


#: :class:`OrfRecord` field → parquet column, where the two names differ. The one
#: copy: :meth:`OrfRecord.from_mapping` and the column loader (``load``) both read
#: it, so the row path and the fast path cannot disagree on a column's name.
PARQUET_COLUMN = {
    "unique_intervals": "unique_genomic_intervals",
    "shared_intervals": "shared_genomic_intervals",
}


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


def int_or_none(raw: Any) -> int | None:
    """An integer field's value as the index stores it: ``int(raw)``, or ``None``.

    ``None`` for a null and for anything ``int`` refuses (NaN, ``"x"``); floats
    truncate. The row path and the column loader both coerce through this.
    """
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
    #: The figure's isoform→canonical x shift, in mRNA nucleotides: the distance
    #: between the two start codons, expressed against the gene-level canonical the
    #: figure draws. Divide by 3 for residues; negative means the ORF initiates
    #: upstream. ``None`` when the index predates the column or the transcript
    #: skeleton was unavailable, in which case callers right-align instead.
    canonical_x_offset_nt: int | None = None
    #: Residue count of the **per-transcript** canonical — the protein
    #: ``canonical_orf_exons`` describes, and the one a canonical-frame residue is
    #: numbered against. Differs from ``canonical_len`` (the gene-level
    #: representative the figure draws) for 1,670 of 6,462 ORFs in ``full_catalog``.
    #: ``None`` when the index predates the column; a canonical-frame residue is
    #: then left unshifted, which is right only where the two canonicals agree.
    canonical_per_tid_length: int | None = None

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
            unique_intervals=_as_intervals(row.get(PARQUET_COLUMN["unique_intervals"])),
            shared_intervals=_as_intervals(row.get(PARQUET_COLUMN["shared_intervals"])),
            canonical_len=int_or_none(row.get("canonical_len")),
            isoform_len=int_or_none(row.get("isoform_len")),
            diff_space=str(row.get("diff_space") or ""),
            orf_type=str(row.get("orf_type") or ""),
            start_codon=str(row.get("start_codon") or ""),
            canonical_start_codon=str(row.get("canonical_start_codon") or ""),
            orf_cds=str(row.get("orf_cds") or ""),
            canonical_cds=str(row.get("canonical_cds") or ""),
            canonical_x_offset_nt=int_or_none(row.get("canonical_x_offset_nt")),
            canonical_per_tid_length=int_or_none(row.get("canonical_per_tid_length")),
        )


#: Stands for ``None`` in the nullable integer columns.
NULL_INT = -(2**63)

#: :class:`OrfRecord` fields, grouped by how :class:`OrfColumns` stores them.
STRING_FIELDS = (
    "gene_name",
    "tis_id",
    "transcript_id",
    "chrom",
    "strand",
    "diff_space",
    "orf_type",
    "start_codon",
    "canonical_start_codon",
)
INT_FIELDS = ("canonical_len", "isoform_len", "canonical_x_offset_nt", "canonical_per_tid_length")
INTERVAL_FIELDS = ("orf_exons", "canonical_orf_exons", "unique_intervals", "shared_intervals")
CDS_FIELDS = ("orf_cds", "canonical_cds")


def _int_array(values: Iterable[int] = ()) -> array:
    return array("q", values)


def interned(values: Iterable[str | None]) -> list[str]:
    """One string column: ``None`` as ``""``, repeated values sharing one object.

    Most string columns repeat heavily (strand, ORF type, start codons, a gene's
    name per isoform), so sharing the objects is most of their memory saving.
    """
    pool: dict[str, str] = {}
    return [pool.setdefault(v or "", v or "") for v in values]


@dataclass
class OrfColumns:
    """Every ORF field as one column, row ``i`` being ORF ``i``.

    * ``strings[f]`` — one ``str`` per ORF; repeated values share one object.
    * ``ints[f]`` — ``array('q')``, :data:`NULL_INT` for ``None``.
    * ``bounds[f]`` / ``offsets[f]`` — an interval field flattened: ORF ``i``'s
      intervals are ``(bounds[2k], bounds[2k+1])`` for ``k`` in
      ``offsets[i] .. offsets[i+1]-1``; ``offsets`` has ``n + 1`` entries.
    * ``cds_pool`` / ``cds_bounds`` / ``cds_ids[f]`` — every *distinct* coding
      sequence once, concatenated; entry ``j`` is
      ``cds_pool[cds_bounds[j]:cds_bounds[j+1]]`` and ORF ``i``'s sequence for
      field ``f`` is entry ``cds_ids[f][i]``. Sequences dominate the index, and a
      third of them repeat (isoforms of one transcript share its canonical). The
      pool is built in ORF order, each ORF's own then canonical sequence, so the
      two — which largely overlap — sit side by side and gzip well.
    """

    strings: dict[str, list[str]]
    ints: dict[str, array]
    bounds: dict[str, array]
    offsets: dict[str, array]
    cds_pool: str
    cds_bounds: array
    cds_ids: dict[str, array]

    @property
    def n(self) -> int:
        """Number of ORFs."""
        return len(self.strings["tis_id"])

    @classmethod
    def from_records(cls, records: Sequence[OrfRecord]) -> OrfColumns:
        """Pack ready-made records (the slow, general path; loaders build columns)."""
        strings = {f: interned(getattr(r, f) for r in records) for f in STRING_FIELDS}
        ints = {
            f: _int_array(NULL_INT if getattr(r, f) is None else getattr(r, f) for r in records)
            for f in INT_FIELDS
        }
        bounds: dict[str, array] = {}
        offsets: dict[str, array] = {}
        for f in INTERVAL_FIELDS:
            flat, offs = _int_array(), _int_array([0])
            for r in records:
                for start, end in getattr(r, f):
                    flat.append(start)
                    flat.append(end)
                offs.append(len(flat) // 2)
            bounds[f], offsets[f] = flat, offs
        pool, cds_bounds, cds_ids = pool_sequences(
            {f: [getattr(r, f) for r in records] for f in CDS_FIELDS}
        )
        return cls(strings, ints, bounds, offsets, pool, cds_bounds, cds_ids)

    def check(self) -> None:
        """Raise ``ValueError`` unless the columns describe the same ``n`` ORFs.

        For columns read from outside (``portable``): a file whose parts disagree
        would otherwise load, then fail with an ``IndexError`` on whichever query
        first reaches a short column. Linear passes using ``min``/``max``, so cheap
        enough to run on every load, including in the browser.
        """
        n = self.n

        def need(ok: bool, what: str) -> None:
            if not ok:
                raise ValueError(f"inconsistent ORF columns: {what}")

        for f in STRING_FIELDS:
            need(len(self.strings[f]) == n, f"{f} has {len(self.strings[f])} of {n} ORFs")
        for f in INT_FIELDS:
            need(len(self.ints[f]) == n, f"{f} has {len(self.ints[f])} of {n} ORFs")
        for f in INTERVAL_FIELDS:
            offs, flat = self.offsets[f], self.bounds[f]
            need(len(offs) == n + 1 and offs[0] == 0, f"{f} offsets do not cover {n} ORFs")
            need(_non_decreasing(offs), f"{f} offsets decrease")
            need(offs[-1] * 2 == len(flat), f"{f} offsets do not match its bounds")
        bounds = self.cds_bounds
        need(len(bounds) >= 1 and bounds[0] == 0, "the sequence pool bounds do not start at 0")
        need(_non_decreasing(bounds), "the sequence pool bounds decrease")
        need(bounds[-1] == len(self.cds_pool), "the sequence pool bounds overrun the pool")
        for f in CDS_FIELDS:
            ids = self.cds_ids[f]
            need(len(ids) == n, f"{f} has {len(ids)} of {n} ORFs")
            need(not ids or (min(ids) >= 0 and max(ids) < len(bounds) - 1), f"{f} ids out of range")

    def intervals(self, f: str, i: int) -> tuple[Interval, ...]:
        """ORF ``i``'s intervals for field ``f``, as the record carries them."""
        flat, offs = self.bounds[f], self.offsets[f]
        return tuple((flat[2 * k], flat[2 * k + 1]) for k in range(offs[i], offs[i + 1]))

    def record(self, i: int) -> OrfRecord:
        """Build ORF ``i``'s :class:`OrfRecord`."""
        fields: dict[str, Any] = {f: self.strings[f][i] for f in STRING_FIELDS}
        for f in INT_FIELDS:
            value = self.ints[f][i]
            fields[f] = None if value == NULL_INT else value
        for f in INTERVAL_FIELDS:
            fields[f] = self.intervals(f, i)
        for f in CDS_FIELDS:
            j = self.cds_ids[f][i]
            fields[f] = self.cds_pool[self.cds_bounds[j] : self.cds_bounds[j + 1]]
        return OrfRecord(**fields)


def _non_decreasing(values: array) -> bool:
    return all(a <= b for a, b in zip(values, values[1:]))


def pool_sequences(per_field: Mapping[str, Sequence[str]]) -> tuple[str, array, dict[str, array]]:
    """``{field: [sequence per ORF]}`` → ``(pool, bounds, ids)`` for :class:`OrfColumns`.

    Distinct sequences in first-seen order, walking ORFs and, within each, the
    fields in :data:`CDS_FIELDS` order.
    """
    entries: dict[str, int] = {}
    ids = {f: _int_array() for f in CDS_FIELDS}
    n = len(per_field[CDS_FIELDS[0]])
    for i in range(n):
        for f in CDS_FIELDS:
            seq = per_field[f][i]
            j = entries.get(seq)
            if j is None:
                j = entries[seq] = len(entries)
            ids[f].append(j)
    bounds = _int_array([0])
    for seq in entries:
        bounds.append(bounds[-1] + len(seq))
    return "".join(entries), bounds, ids


@dataclass
class _ChromIndex:
    """Exon intervals on one chromosome, sorted by start.

    ``max_len`` bounds how far back a query must walk: any interval containing
    position ``p`` satisfies ``start >= p - max_len``, because
    ``start >= end - max_len`` and ``end >= p``.
    """

    starts: array
    ends: array
    owners: array
    max_len: int = 0


def _chrom_index(columns: OrfColumns) -> dict[str, _ChromIndex]:
    """Per-chromosome sorted interval arrays over both ORFs of every record."""
    staging: dict[str, list[tuple[int, int, int]]] = {}
    chroms = columns.strings["chrom"]
    for owner in range(columns.n):
        # Union of the two ORFs, deduplicated: an extension's canonical exons
        # are mostly a subset of its own, so most pairs collapse.
        seen: set[Interval] = set()
        for f in ("orf_exons", "canonical_orf_exons"):
            for start, end in columns.intervals(f, owner):
                if end <= start or (start, end) in seen:
                    continue
                seen.add((start, end))
                staging.setdefault(chroms[owner], []).append((start, end, owner))
    by_chrom: dict[str, _ChromIndex] = {}
    for chrom, triples in staging.items():
        # Sorted once at the end — cheaper than keeping three parallel arrays
        # ordered during insertion.
        triples.sort()
        by_chrom[chrom] = _ChromIndex(
            starts=_int_array(t[0] for t in triples),
            ends=_int_array(t[1] for t in triples),
            owners=_int_array(t[2] for t in triples),
            max_len=max(t[1] - t[0] for t in triples),
        )
    return by_chrom


class _Records(Sequence[OrfRecord]):
    """``OrfIndex.records`` — every ORF, built one at a time as it is read."""

    def __init__(self, index: OrfIndex) -> None:
        self._index = index

    def __len__(self) -> int:
        return self._index.n_isoforms

    # Built straight from the columns, not through the query cache: one pass over
    # every ORF would otherwise evict the scan's hot records and leave thousands
    # of cold ones alive behind it.
    def __getitem__(self, i: Any) -> Any:
        columns = self._index.columns
        if isinstance(i, slice):
            return [columns.record(j) for j in range(*i.indices(len(self)))]
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(i)
        return columns.record(i)

    def __iter__(self) -> Iterator[OrfRecord]:
        columns = self._index.columns
        return (columns.record(i) for i in range(len(self)))


class OrfIndex:
    """Point/span membership over every annotated ORF exon.

    Build with :meth:`from_records` (rows or records) or straight from packed
    :class:`OrfColumns` (what the loaders produce); query with :meth:`lookup` (a single 1-based
    position) or :meth:`lookup_span` (an inclusive 1-based range, for indels).
    """

    __slots__ = ("_by_chrom", "_cache", "_columns", "_tis_index", "version")

    #: Records kept built between queries, least recently used out. A scan returns
    #: the same few ORFs over and over (every variant in a gene), so a small cache
    #: avoids rebuilding them without growing back into one object per ORF.
    CACHE_SIZE = 4096

    def __init__(self, columns: OrfColumns, version: str = "") -> None:
        """Index packed columns, as the loaders build them (or :meth:`from_records`)."""
        self._columns = columns
        self._by_chrom = _chrom_index(columns)
        self._tis_index: dict[str, int] | None = None
        self._cache: OrderedDict[int, OrfRecord] = OrderedDict()
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
        """Build the index from parquet-style rows or ready-made :class:`OrfRecord`s.

        Takes plain mappings rather than a DataFrame so this module stays free of
        pandas. Rows without a chromosome are dropped. The loaders pack columns
        directly and call the constructor; this is the general path (tests, small
        builds).
        """
        records = [
            row if isinstance(row, OrfRecord) else OrfRecord.from_mapping(row) for row in rows
        ]
        return cls(OrfColumns.from_records([r for r in records if r.chrom]), version=version)

    @property
    def columns(self) -> OrfColumns:
        """The packed columns, for serialisers (``portable``)."""
        return self._columns

    def record(self, i: int) -> OrfRecord:
        """ORF ``i``'s record, built on demand and kept in a small LRU cache."""
        cached = self._cache.get(i)
        if cached is not None:
            self._cache.move_to_end(i)
            return cached
        cached = self._cache[i] = self._columns.record(i)
        if len(self._cache) > self.CACHE_SIZE:
            self._cache.popitem(last=False)
        return cached

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
        return sorted((self.record(o) for o in owners), key=lambda r: r.tis_id)

    def by_tis_id(self, tis_id: str) -> OrfRecord | None:
        """The record for one ``tis_id``, or None.

        The ``tis_id`` → row map is built lazily on first use: the gene figure
        needs each hit's ORF to convert a residue into a plot coordinate.
        """
        if self._tis_index is None:
            self._tis_index = {t: i for i, t in enumerate(self._columns.strings["tis_id"])}
        i = self._tis_index.get(tis_id)
        return None if i is None else self.record(i)

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
        """Every indexed ORF, in insertion order (built as it is read)."""
        return _Records(self)

    @property
    def n_isoforms(self) -> int:
        """Number of indexed ORFs."""
        return self._columns.n

    @property
    def n_genes(self) -> int:
        """Number of distinct genes represented."""
        return len({g for g in self._columns.strings["gene_name"] if g})

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
