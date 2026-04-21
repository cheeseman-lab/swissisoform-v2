"""Minimal MAF (Multiple Alignment Format) parser.

Reads the output of ``hal2maf`` (or any MAF file) into per-block dicts keyed
by species. Only ``s`` lines are consumed; ``a`` lines start a new block;
``i`` / ``e`` / ``q`` / comment lines are skipped.

``s`` line layout::

    s <species.chrom> <start> <size> <strand> <srcSize> <text>

The species token is the prefix before the first ``.`` (e.g. ``hg38``).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MafRow:
    """One aligned sequence row within a MAF block.

    Attributes:
        species: Leading token of the MAF source name (``hg38.chr1`` → ``hg38``).
        src: Full source name as written in the MAF.
        start: 0-based start of the sequence on its source.
        size: Number of non-gap bases covered.
        strand: ``'+'`` or ``'-'``.
        src_size: Total length of the source sequence.
        seq: Aligned text (may contain ``-`` gap characters).
    """

    species: str
    src: str
    start: int
    size: int
    strand: str
    src_size: int
    seq: str


@dataclass
class MafBlock:
    """A single alignment block as ``{species: MafRow}``."""

    rows: dict[str, MafRow] = field(default_factory=dict)

    def species(self) -> list[str]:
        """Species present in this block, in insertion order."""
        return list(self.rows)

    def get(self, species: str) -> MafRow | None:
        """Return the row for *species*, or ``None`` if absent."""
        return self.rows.get(species)


def parse_maf(text: str) -> list[MafBlock]:
    """Parse MAF *text* into a list of :class:`MafBlock` objects.

    Lines that start with ``a`` open a new block; subsequent ``s`` lines are
    appended to it. Blank lines, comments, and ``i`` / ``e`` / ``q`` lines
    are ignored. Malformed ``s`` lines are skipped silently — MAF is
    user-tool output, not a machine contract.

    Args:
        text: The full MAF file contents as a single string.

    Returns:
        List of :class:`MafBlock` in file order. Empty list if no alignment
        blocks are found.
    """
    blocks: list[MafBlock] = []
    current: MafBlock | None = None

    for raw in text.splitlines():
        if not raw or raw.startswith("#"):
            continue
        tag = raw[0]
        if tag == "a":
            current = MafBlock()
            blocks.append(current)
            continue
        if tag != "s":
            continue
        if current is None:
            current = MafBlock()
            blocks.append(current)

        parts = raw.split()
        if len(parts) < 7:
            continue
        _, src, start, size, strand, src_size, seq = parts[:7]
        species = src.split(".", 1)[0]
        try:
            row = MafRow(
                species=species,
                src=src,
                start=int(start),
                size=int(size),
                strand=strand,
                src_size=int(src_size),
                seq=seq,
            )
        except ValueError:
            continue
        current.rows[species] = row

    return [b for b in blocks if b.rows]


def concat_species_rows(
    blocks: list[MafBlock],
    species: str,
    reference: str,
) -> tuple[str, str] | None:
    """Return ``(reference_seq, species_seq)`` across *blocks*, both MAF-aligned.

    Missing species rows in a block are filled with ``-`` characters to
    match the reference block width; if the reference itself is missing from
    a block, that block is skipped (we have no frame for it). Returns
    ``None`` if the reference is absent from every block.

    Args:
        blocks: Parsed MAF blocks, in order.
        species: Species name to extract (e.g. ``'panTro6'``).
        reference: Reference species name (e.g. ``'hg38'``).

    Returns:
        Tuple of (ref_seq, species_seq) — equal length, same column ordering.
        Both may contain ``-``. Returns ``None`` when the reference never
        appears.
    """
    ref_parts: list[str] = []
    sp_parts: list[str] = []
    for block in blocks:
        ref_row = block.get(reference)
        if ref_row is None:
            continue
        sp_row = block.get(species)
        ref_parts.append(ref_row.seq)
        if sp_row is None:
            sp_parts.append("-" * len(ref_row.seq))
        else:
            sp_parts.append(sp_row.seq)

    if not ref_parts:
        return None
    return "".join(ref_parts), "".join(sp_parts)
