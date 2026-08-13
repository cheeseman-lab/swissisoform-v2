"""Protein consequence of a variant against one ORF's coding sequence.

The gene figure groups variants into rows by consequence, so a hit needs one of a
fixed vocabulary of terms — and for substitutions that is only decidable from the
**codon**, which is why the ORF index carries its CDS.

One uniform mechanism rather than per-class codon arithmetic: splice the ALT into
the CDS at the right mRNA offset, translate both, and diff. That covers
substitutions, MNVs, in-frame indels and frameshifts through the same code path.

Stdlib only — a 64-entry codon table rather than biopython, so this runs inside the
website image. Cross-checked against ``ConsequenceValidator`` in the test suite,
which is the authority on the same arithmetic elsewhere in the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from swissisoform.variantquery.frame import coding_offset
from swissisoform.variantquery.index import Interval

# Terms the gene figure knows how to draw (plots/protein.py). Anything outside this
# set lands in a generic "other" row, so new terms must be added to both.
FRAMESHIFT = "frameshift_variant"
STOP_GAINED = "stop_gained"
STOP_LOST = "stop_lost"
START_LOST = "start_lost"
MISSENSE = "missense_variant"
SYNONYMOUS = "synonymous_variant"
INFRAME_DELETION = "inframe_deletion"
INFRAME_INSERTION = "inframe_insertion"
OTHER = "other"

_COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}

# Standard genetic code. Verified against Bio.Data.CodonTable in the tests rather
# than trusted — a single wrong entry here would silently mislabel one amino acid.
_CODONS = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}


@dataclass(frozen=True, slots=True)
class Consequence:
    """What a variant does to one ORF's protein.

    ``hgvsp`` is numbered against **this ORF**, not a transcript — the same
    nucleotide is a different residue in every ORF that contains it, because each
    starts at a different TIS. Callers must keep the frame alongside it.
    """

    term: str
    aa_ref: str = ""
    aa_alt: str = ""
    #: 1-based residue of the first affected codon; None when unmappable.
    aa_pos: int | None = None
    hgvsp: str = ""
    #: Why a notation is absent, when it is (e.g. an indel spanning an intron).
    note: str = ""


def revcomp(seq: str) -> str:
    """Reverse complement, for turning plus-strand REF/ALT into mRNA sense."""
    return "".join(_COMPLEMENT.get(b, "N") for b in reversed(seq.upper()))


def translate(cds: str, *, stop_at_first: bool = False) -> str:
    """Translate a CDS. Trailing partial codons are dropped.

    Args:
        cds: Coding sequence in mRNA order.
        stop_at_first: Truncate at the first stop (used for the mutant protein,
            where everything past a premature stop is not translated).
    """
    out: list[str] = []
    for i in range(0, len(cds) - len(cds) % 3, 3):
        aa = _CODONS.get(cds[i : i + 3].upper(), "X")
        if aa == "*" and stop_at_first:
            out.append(aa)
            return "".join(out)
        out.append(aa)
    return "".join(out)


def _trim_anchor(ref: str, alt: str, pos: int) -> tuple[str, str, int]:
    """Strip the shared leading base VCF indels carry.

    ``CGT>C`` means GT is deleted *after* ``pos`` — the base at ``pos`` is
    unchanged. Without trimming, the change appears to start one base early, which
    moves the residue whenever the anchor and the first changed base straddle a
    codon boundary.
    """
    while len(ref) > 1 and len(alt) > 1 and ref[0] == alt[0]:
        ref, alt, pos = ref[1:], alt[1:], pos + 1
    if len(ref) != len(alt) and ref and alt and ref[0] == alt[0]:
        # Pure indel: the anchor stays in REF/ALT but is not itself altered.
        ref, alt, pos = ref[1:], alt[1:], pos + 1
    return ref, alt, pos


def classify_without_sequence(ref: str, alt: str) -> Consequence:
    """Class from the length delta alone, for an index built without a genome.

    Frameshift vs in-frame is exact without any sequence. Substitutions cannot be
    resolved — missense, synonymous and stop_gained are indistinguishable from
    REF/ALT alone — so they come back as ``other`` with a note rather than a guess.
    """
    ref, alt, _pos = _trim_anchor(ref.upper(), alt.upper(), 0)
    delta = len(alt) - len(ref)
    if delta % 3 != 0:
        return Consequence(FRAMESHIFT, note="no coding sequence in the index")
    if delta < 0:
        return Consequence(INFRAME_DELETION, note="no coding sequence in the index")
    if delta > 0:
        return Consequence(INFRAME_INSERTION, note="no coding sequence in the index")
    return Consequence(OTHER, note="substitution needs the codon; index was built without a genome")


def classify(
    *,
    exons: tuple[Interval, ...],
    strand: str,
    cds: str,
    pos: int,
    ref: str,
    alt: str,
    start_codon: str = "",
) -> Consequence:
    """Consequence of a 1-based genomic variant on the protein of one ORF.

    Args:
        exons: The ORF's exons, 0-based half-open plus-strand ascending.
        strand: ``"+"`` or ``"-"``.
        cds: That ORF's coding sequence in mRNA order (no trailing stop).
        pos: 1-based genomic position of the VCF record.
        ref: REF allele, plus-strand.
        alt: One ALT allele, plus-strand.
        start_codon: The ORF's start codon, when known. Only used to note
            non-AUG starts; ``start_lost`` is decided by position.

    Returns:
        A :class:`Consequence`. The ``term`` is always set; the notation fields may
        be blank with ``note`` explaining why.
    """
    ref, alt, pos = _trim_anchor(ref.upper(), alt.upper(), pos)
    delta = len(alt) - len(ref)

    # Class first: the length delta alone decides frameshift vs in-frame, and needs
    # no sequence at all. So a variant we cannot place still gets the right row.
    if delta % 3 != 0:
        term = FRAMESHIFT
    elif delta < 0:
        term = INFRAME_DELETION
    elif delta > 0:
        term = INFRAME_INSERTION
    else:
        term = ""  # substitution — decided below, from the codon

    span_len = max(len(ref), 1)
    span = (pos, pos + span_len - 1)

    # Every changed base must sit in coding sequence; otherwise splicing the ALT
    # into the CDS would silently include intronic bases.
    offsets = [coding_offset(exons, strand, p) for p in range(span[0], span[1] + 1)]
    if not offsets or any(o is None for o in offsets):
        return Consequence(
            term=term or OTHER,
            note="span is not entirely within the ORF's coding sequence",
        )

    # On the minus strand mRNA order runs against genomic order, so the span's
    # first *translated* base is the one with the lowest offset.
    start_off = min(offsets)  # type: ignore[arg-type]
    if strand == "-":
        ref_m, alt_m = revcomp(ref), revcomp(alt)
    else:
        ref_m, alt_m = ref, alt

    if cds[start_off : start_off + len(ref_m)] != ref_m:
        return Consequence(
            term=term or OTHER,
            aa_pos=start_off // 3 + 1,
            note="REF does not match the ORF's reference sequence",
        )

    mutant = cds[:start_off] + alt_m + cds[start_off + len(ref_m) :]
    codon_i = start_off // 3
    ref_prot = translate(cds)
    alt_prot = translate(mutant, stop_at_first=True)
    aa_pos = codon_i + 1
    aa_ref = ref_prot[codon_i] if codon_i < len(ref_prot) else ""

    # A substitution inside the first codon removes the initiator.
    if term == "" and start_off < 3:
        aa_alt = alt_prot[codon_i] if codon_i < len(alt_prot) else ""
        note = f"non-AUG start ({start_codon})" if start_codon and start_codon != "ATG" else ""
        return Consequence(START_LOST, aa_ref, aa_alt, aa_pos, f"p.{aa_ref}{aa_pos}?", note)

    if term == "":
        # A multi-base substitution can straddle a codon boundary, changing two
        # residues. Reporting only the first would silently drop the second — the
        # fixture's minus-strand MNV spans codons 224-225 (F->S and E->K).
        last_codon = (start_off + len(ref_m) - 1) // 3
        ref_aas = ref_prot[codon_i : last_codon + 1]
        alt_aas = alt_prot[codon_i : last_codon + 1]

        if "*" in alt_aas:
            term = STOP_GAINED
        elif "*" in ref_aas and "*" not in alt_aas:
            term = STOP_LOST
        elif alt_aas == ref_aas:
            term = SYNONYMOUS
        else:
            term = MISSENSE

        if last_codon == codon_i:
            hgvsp = f"p.{ref_aas}{aa_pos}{alt_aas}"
        else:
            hgvsp = f"p.{ref_aas[0]}{aa_pos}_{ref_aas[-1]}{last_codon + 1}delins{alt_aas}"
        return Consequence(term, ref_aas, alt_aas, aa_pos, hgvsp)

    if term == FRAMESHIFT:
        # The new reading frame usually runs past the original stop into the 3'UTR,
        # which the CDS does not contain — so only append the stop distance when a
        # premature stop genuinely falls inside what we have.
        hgvsp = f"p.{aa_ref}{aa_pos}fs"
        if alt_prot.endswith("*"):
            hgvsp += f"*{len(alt_prot) - codon_i}"
        return Consequence(FRAMESHIFT, aa_ref, "", aa_pos, hgvsp)

    # In-frame indel: name the residues gained or lost by diffing the two proteins.
    n = abs(delta) // 3
    if term == INFRAME_DELETION:
        last = ref_prot[codon_i + n - 1] if codon_i + n - 1 < len(ref_prot) else ""
        hgvsp = (
            f"p.{aa_ref}{aa_pos}del" if n == 1 else f"p.{aa_ref}{aa_pos}_{last}{aa_pos + n - 1}del"
        )
        return Consequence(INFRAME_DELETION, aa_ref, "", aa_pos, hgvsp)

    inserted = alt_prot[codon_i : codon_i + n]
    nxt = ref_prot[codon_i] if codon_i < len(ref_prot) else ""
    prev = ref_prot[codon_i - 1] if codon_i > 0 else ""
    hgvsp = f"p.{prev}{aa_pos - 1}_{nxt}{aa_pos}ins{inserted}"
    return Consequence(INFRAME_INSERTION, "", inserted, aa_pos, hgvsp)
