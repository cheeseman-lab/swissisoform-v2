"""One VCF data line → normalised variant specs, or a typed rejection.

Nothing here touches the filesystem or the ORF index; it is pure string parsing
so the awkward cases (multi-allelic, breakends, spanning deletions) can be
pinned by unit tests without a run.
"""

from __future__ import annotations

from dataclasses import dataclass

# Rejection reasons. Kept as a closed set so the scan funnel can tally them and
# the UI can say something specific rather than "malformed".
SV_BREAKEND = "sv_breakend"
UNSUPPORTED_ALT = "unsupported_alt"
MALFORMED = "malformed"

#: Characters that mark a symbolic / structural ALT: GRIDSS breakends
#: (``[19:531848[T``), symbolic alleles (``<DEL>``, ``<DUP>``). These are not
#: base substitutions and cannot be placed on a protein.
_SYMBOLIC_ALT_CHARS = frozenset("[]<>")

_VALID_BASES = frozenset("ACGTNacgtn")


@dataclass(frozen=True, slots=True)
class Rejection:
    """A line (or one ALT of a line) that cannot be resolved to a variant."""

    reason: str
    detail: str


@dataclass(frozen=True, slots=True)
class VariantSpec:
    """A single ALT allele of a single VCF record, normalised.

    Attributes:
        chrom: Chromosome with a ``chr`` prefix, matching pipeline coordinates.
        pos: 1-based VCF POS.
        ref: REF allele, uppercased.
        alt: One ALT allele, uppercased (multi-allelic lines yield one spec each).
        vclass: ``snv`` | ``mnv`` | ``insertion`` | ``deletion``.
        filter_field: Raw FILTER column, verbatim (``PASS``, ``.``, or a
            caller-specific tag like ``QSS_ref``).
        variant_id: The ID column, or ``""`` when it is ``.``.
    """

    chrom: str
    pos: int
    ref: str
    alt: str
    vclass: str
    filter_field: str
    variant_id: str

    @property
    def span(self) -> tuple[int, int]:
        """1-based inclusive genomic span the REF allele occupies."""
        return (self.pos, self.pos + len(self.ref) - 1)

    @property
    def is_pass(self) -> bool:
        """True when FILTER is ``PASS`` or empty (``.``).

        A ``.`` means the caller did not filter, which is not the same as
        failing — treat it as passing rather than silently dropping every
        record from an unfiltered VCF.
        """
        return self.filter_field in ("PASS", ".", "")


def normalize_chrom(raw: str) -> str:
    """Normalise a VCF CHROM to the pipeline's UCSC-style name.

    The real somatic VCFs use bare names (``17``) even on GRCh38, while our
    coordinates are ``chr17`` — so this runs on every line.
    """
    name = raw.strip()
    if not name.startswith("chr"):
        name = f"chr{name}"
    # Ensembl/1000G call the mitochondrion MT; UCSC calls it M.
    if name in ("chrMT", "chrmt", "chrMt"):
        return "chrM"
    return name


def _classify(ref: str, alt: str) -> str:
    if len(ref) == len(alt):
        return "snv" if len(ref) == 1 else "mnv"
    return "insertion" if len(alt) > len(ref) else "deletion"


def parse_line(line: str) -> list[VariantSpec | Rejection]:
    """Parse one VCF data line into one entry per ALT allele.

    Multi-allelic records are **split**, not rejected — one spec per ALT — so a
    ``G>A,T`` line never silently resolves as if only the first allele existed.
    A line whose fixed fields are unusable yields a single :class:`Rejection`.

    Returns:
        One entry per ALT allele: a :class:`VariantSpec` when resolvable, a
        :class:`Rejection` when that particular allele is not.
    """
    fields = line.rstrip("\r\n").split("\t")
    if len(fields) < 5:
        return [Rejection(MALFORMED, f"expected >=5 tab-separated fields, got {len(fields)}")]

    chrom_raw, pos_raw, id_raw, ref_raw = fields[0], fields[1], fields[2], fields[3]
    alt_raw = fields[4]
    filter_field = fields[6] if len(fields) > 6 else "."

    try:
        pos = int(pos_raw)
    except ValueError:
        return [Rejection(MALFORMED, f"POS is not an integer: {pos_raw!r}")]
    if pos < 1:
        return [Rejection(MALFORMED, f"POS must be >= 1, got {pos}")]

    ref = ref_raw.strip().upper()
    if not ref or not set(ref) <= _VALID_BASES:
        return [Rejection(MALFORMED, f"REF is not a DNA string: {ref_raw!r}")]

    chrom = normalize_chrom(chrom_raw)
    variant_id = "" if id_raw.strip() == "." else id_raw.strip()

    out: list[VariantSpec | Rejection] = []
    for alt_token in alt_raw.split(","):
        alt = alt_token.strip().upper()
        if not alt or alt == ".":
            out.append(Rejection(UNSUPPORTED_ALT, "empty or missing ALT"))
        elif _SYMBOLIC_ALT_CHARS & set(alt):
            out.append(Rejection(SV_BREAKEND, f"symbolic/breakend ALT: {alt_token!r}"))
        elif alt == "*":
            # Spanning deletion placeholder — the allele is absent because an
            # overlapping deletion removed it; there is nothing to place.
            out.append(Rejection(UNSUPPORTED_ALT, "spanning-deletion ALT (*)"))
        elif not set(alt) <= _VALID_BASES:
            out.append(Rejection(UNSUPPORTED_ALT, f"ALT is not a DNA string: {alt_token!r}"))
        else:
            out.append(
                VariantSpec(
                    chrom=chrom,
                    pos=pos,
                    ref=ref,
                    alt=alt,
                    vclass=_classify(ref, alt),
                    filter_field=filter_field,
                    variant_id=variant_id,
                )
            )
    return out
