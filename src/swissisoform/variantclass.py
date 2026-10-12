"""The variant consequence classifier, in the standard library alone.

Everything that decides what a variant does to an ORF — position mapping, the
codon walk, splice-edge and start-codon rules, indel translation — given the
ORF's exons, strand and coding sequence. Lifted out of ``clinical/validate.py``
(issue #33, item 2) so the VCF scan can run where pandas, Biopython and pysam do
not: the website image without them, and the browser under Pyodide.
``ConsequenceValidator`` is now the pipeline's caching adapter that acquires the
sequence (GTF + genome FASTA) and delegates here; :class:`OrfClassifier` is the
same for callers that already hold the sequence.

Translation uses the standard genetic code (NCBI table 1) with Biopython's
handling of IUPAC ambiguity codes, pinned against Biopython by
``tests/test_variantclass.py``.
"""

from __future__ import annotations

import functools
import itertools
import logging
from typing import Any

from swissisoform import coords
from swissisoform.contract import start_codon_effect

logger = logging.getLogger(__name__)

_revcomp = coords.revcomp

_BASES = "TCAG"
_AMINO = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
#: Standard genetic code (NCBI translation table 1).
CODON_TABLE: dict[str, str] = {
    "".join(codon): aa
    for codon, aa in zip(itertools.product(_BASES, repeat=3), _AMINO, strict=True)
}

_IUPAC = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    "U": "T",
    "R": "AG",
    "Y": "CT",
    "S": "CG",
    "W": "AT",
    "K": "GT",
    "M": "AC",
    "B": "CGT",
    "D": "AGT",
    "H": "ACT",
    "V": "ACG",
    "N": "ACGT",
}
#: One-letter → three-letter amino-acid codes, the 20 standard residues
#: (``Bio.Data.IUPACData.protein_letters_1to3``).
PROTEIN_LETTERS_1TO3: dict[str, str] = {
    "A": "Ala", "C": "Cys", "D": "Asp", "E": "Glu", "F": "Phe", "G": "Gly", "H": "His",
    "I": "Ile", "K": "Lys", "L": "Leu", "M": "Met", "N": "Asn", "P": "Pro", "Q": "Gln",
    "R": "Arg", "S": "Ser", "T": "Thr", "V": "Val", "W": "Trp", "Y": "Tyr",
}  # fmt: skip

#: Biopython's two-residue ambiguity letters.
_AMBIGUOUS_AA = {frozenset("DN"): "B", frozenset("EQ"): "Z", frozenset("IL"): "J"}


@functools.lru_cache(maxsize=4096)
def _translate_codon(codon: str) -> str:
    aa = CODON_TABLE.get(codon)
    if aa is not None:
        return aa
    try:
        options = {CODON_TABLE["".join(c)] for c in itertools.product(*(_IUPAC[b] for b in codon))}
    except KeyError:
        return "X"
    if len(options) == 1:
        return options.pop()
    return _AMBIGUOUS_AA.get(frozenset(options), "X")


def translate(seq: str) -> str:
    """Translate a nucleotide sequence; a trailing partial codon is dropped.

    Matches ``str(Bio.Seq.Seq(seq).translate())`` on the standard table, including
    IUPAC ambiguity codes (``GCN`` → ``A``, ``TAN`` → ``X``, ``RAY`` → ``B``).
    """
    seq = seq.upper()
    return "".join(_translate_codon(seq[i : i + 3]) for i in range(0, len(seq) - 2, 3))


def position_map(orf_exons: Any, strand: str) -> dict[int, int]:
    """1-based genomic position → 0-based coding offset for an ORF's exons.

    Args:
        orf_exons: 0-based half-open plus-strand intervals in ascending genomic
            order (as produced by :func:`swissisoform.coords.orf_exons_from_skeleton`).
        strand: ``"+"`` or ``"-"``; the minus strand is walked high to low.
    """
    return dict(coords.iter_coding_positions(orf_exons, strand))


class OrfClassifier:
    """Classify variants against ORFs whose coding sequence the caller holds.

    Caches one position map per ORF, so a scan that lands many variants in the
    same ORF builds its map once. The pipeline's ``ConsequenceValidator`` keeps
    the same cache shape; this is the dependency-free half of it.
    """

    def __init__(self) -> None:
        """Start with an empty position-map cache."""
        self._position_maps: dict[Any, dict[int, int]] = {}

    def build_position_map_from_orf(
        self, orf_exons: Any, strand: str, *, orf_key: Any = None
    ) -> dict[int, int]:
        """:func:`position_map`, cached under ``orf_key`` (or the exons + strand)."""
        if not orf_exons:
            return {}
        key = orf_key
        if key is None:
            key = (tuple((int(s), int(e)) for s, e in orf_exons), strand, "")
        cached = self._position_maps.get(key)
        if cached is None:
            cached = self._position_maps[key] = position_map(orf_exons, strand)
        return cached

    def classify_against_orf(
        self,
        *,
        orf_exons: Any,
        strand: str,
        cds: str,
        genomic_pos: int,
        ref: str,
        alt: str,
        orf_key: Any = None,
        context: str = "",
    ) -> dict[str, Any]:
        """Classify one variant against one ORF; see :func:`analyze_variant`.

        ``cds`` must be the sequence *these* exons describe — pairing one ORF's
        exons with another's sequence shifts every residue silently.
        """
        return analyze_variant(
            pos_map=self.build_position_map_from_orf(orf_exons, strand, orf_key=orf_key),
            coding_seq=cds,
            strand=strand,
            genomic_pos=genomic_pos,
            ref=ref,
            alt=alt,
            context=context,
        )


def analyze_variant(
    *,
    pos_map: dict[int, int],
    coding_seq: str,
    strand: str,
    genomic_pos: int,
    ref: str,
    alt: str,
    context: str = "",
) -> dict[str, Any]:
    """Classify a variant given a precomputed position map + coding sequence.

    The one classifier. ``ConsequenceValidator`` (canonical-CDS and ORF-frame
    entry points, pipeline) and :class:`OrfClassifier` (the VCF scan, website and
    browser) both end here, so consequence semantics are identical everywhere.

    Args:
        pos_map: 1-based genomic position → 0-based coding offset, as built by
            :func:`position_map`.
        coding_seq: Coding sequence in mRNA order, no trailing stop; empty when
            unknown, which leaves substitutions unvalidated.
        strand: ``"+"`` or ``"-"``.
        genomic_pos: 1-based VCF ``POS``.
        ref: Reference allele, plus-strand.
        alt: Alternate allele, plus-strand.
        context: Label used in warning messages.

    Returns:
        ``{consequence, protein_pos, aa_ref, aa_alt, codon_ref, codon_alt, note,
        validated}``.
    """
    if not pos_map:
        return null_result(validated=False)

    # One walk for every class. VCF pads an indel with the base *before* the
    # event, and that base is unchanged — HGVS numbers the first changed one, so
    # reading POS as the position reports a codon the variant never touches.
    # ``changed_bases`` strips the padding; for a pure insertion nothing in the
    # reference changes, so the span is the base after the insertion point.
    span_start, ref_changed, alt_changed = coords.changed_bases(genomic_pos, ref, alt)
    span_len = max(len(ref_changed), 1)

    # A multi-base change can straddle a codon boundary and alter two residues,
    # so the walk covers every codon the span touches rather than one.
    #
    # On the minus strand mRNA order runs against genomic order, so the span's
    # first *translated* base is the one with the lowest coding offset — POS is
    # its last. Taking the minimum is a no-op for an SNV and for the plus strand,
    # and it is why the padding base is trimmed into a span rather than into a
    # new anchor: "advance past the anchor" is a plus-strand-only reading.
    # The class comes from the length delta and needs no sequence at all, so an
    # indel stays classified even where the amino acids cannot be worked out.
    # ``None`` for a substitution, whose term is decided by translation below.
    indel_term = _indel_term(ref_changed, alt_changed)

    if ref_changed:
        span_offsets = [pos_map.get(p) for p in range(span_start, span_start + span_len)]
    else:
        # A pure insertion changes no reference base, so there is no span to walk
        # — it lands *between* two adjacent mRNA positions. Which two depends on
        # the strand: VCF inserts after the anchor in plus-strand terms, and on
        # the minus strand that puts the new material before the anchor in mRNA
        # order. Taking the lower of the two neighbouring offsets and stepping
        # one past it is correct on both strands, where mapping the genomic
        # successor alone lands a codon early on the minus strand.
        #
        # The neighbours are read either side of ``span_start``, the first base
        # after the insertion point once *all* the padding is trimmed — not of
        # POS, which is only right when VCF pads with exactly one base. Both must
        # map: with one side outside the coding sequence the insertion sits at an
        # exon edge or just outside the ORF, where it adds no coding bases.
        left, right = pos_map.get(span_start - 1), pos_map.get(span_start)
        # One exception: the ORF ends at its last *sense* base and excludes the
        # stop codon, so an insertion between the two has only one coding
        # neighbour and yet adds bases to the reading frame — c.15_16insT shifts
        # it through the stop. The 3' neighbour in mRNA order is the higher
        # genomic base on the plus strand and the lower one on the minus strand.
        # (pos_map alone cannot tell this from a CDS that ends exactly at an exon
        # end with the stop in the next exon; that rare layout is read the same
        # way, as it was before the both-neighbours rule.)
        last = max(pos_map.values())
        five_prime, three_prime = (left, right) if strand == "+" else (right, left)
        if five_prime == last and three_prime is None and indel_term:
            return {
                "consequence": indel_term,
                "protein_pos": (last + 1) // 3,
                "aa_ref": None,
                "aa_alt": None,
                "codon_ref": None,
                "codon_alt": None,
                "note": (
                    "inserted between the last sense codon and the stop codon; "
                    "residues past the CDS cannot be named"
                ),
                "validated": True,
            }
        span_offsets = [min(left, right) + 1] if left is not None and right is not None else [None]

    mapped = [offset for offset in span_offsets if offset is not None]
    if not mapped:
        # Nothing the variant changes is coding. Decided on the changed bases, not
        # on POS: VCF's padding base can sit in the intron while the bases it
        # deletes are the first of the next exon, and reading POS alone called
        # that frameshift intronic.
        return {
            "consequence": "intronic",
            "protein_pos": None,
            "aa_ref": None,
            "aa_alt": None,
            "codon_ref": None,
            "codon_alt": None,
            "note": "",
            "validated": True,
        }

    if len(mapped) < len(span_offsets):
        # Part of the span lies outside this ORF's coding sequence, so splicing
        # the ALT in would translate intronic bases.
        #
        # An indel keeps its class here where a substitution cannot: the length
        # delta needs no sequence, so "frameshift" stays true even though the
        # residues cannot be named. Previously this case was invisible — the
        # anchor-only mapping never looked at the rest of the span — so the note
        # is the whole point: the overrun is now reported rather than silently
        # producing a call that looks fully resolved.
        #
        # Where the span leaves decides what it means, and two exits are answered
        # before the length delta is: an exon edge inside the ORF (a splice site)
        # and the start codon. The length delta counts every base in REF/ALT,
        # intronic and UTR ones included, so it is the wrong question there.
        edge = _span_edge_effect(
            pos_map=pos_map,
            coding_seq=coding_seq,
            strand=strand,
            span_start=span_start,
            span_offsets=span_offsets,
        )
        if edge is not None:
            return edge

        # For a substitution the class *depends* on translation, so a span that
        # leaves the coding sequence at the 3' end leaves nothing to say: it is
        # intronic. That is the second of the two ways a variant is called
        # intronic — the first (above) is a span that maps nowhere and carries no
        # residue.
        return {
            "consequence": indel_term or "intronic",
            "protein_pos": min(mapped) // 3,
            "aa_ref": None,
            "aa_alt": None,
            "codon_ref": None,
            "codon_alt": None,
            "note": ("span leaves the ORF's coding sequence, so the residues cannot be named")
            if indel_term
            else "",
            "validated": True,
        }
    coding_pos = min(mapped)

    if not coding_seq:
        return {
            "consequence": indel_term,
            "protein_pos": coding_pos // 3,
            "aa_ref": None,
            "aa_alt": None,
            "codon_ref": None,
            "codon_alt": None,
            "note": "no coding sequence available to name the residues" if indel_term else "",
            # An indel's class is a real answer; a substitution's is not, and
            # only the substitution path needs the sequence it did not get.
            "validated": bool(indel_term),
        }

    first_codon = coding_pos // 3
    last_codon = (coding_pos + max(len(ref_changed), 1) - 1) // 3
    codon_start = first_codon * 3
    codon_end = (last_codon + 1) * 3
    offset = coding_pos - codon_start

    if codon_end > len(coding_seq):
        logger.warning(
            "Codon extends beyond coding sequence for %s at gpos %d",
            context or "<unknown-orf>",
            genomic_pos,
        )
        return {
            "consequence": indel_term,
            "protein_pos": first_codon,
            "aa_ref": None,
            "aa_alt": None,
            "codon_ref": None,
            "codon_alt": None,
            "note": "codon runs past the end of the coding sequence" if indel_term else "",
            "validated": False,
        }

    codon_ref = coding_seq[codon_start:codon_end]

    # Whole-string reverse complement, not per base: a multi-base allele is
    # written on the plus strand in genomic order, so it needs reversing as well
    # as complementing to read in mRNA sense. Identical to a per-base complement
    # when the allele is one base.
    ref_bases = ref_changed
    alt_bases = alt_changed
    if strand == "-":
        ref_bases = _revcomp(ref_bases)
        alt_bases = _revcomp(alt_bases)

    # Reached by every class now. An indel used to return before this, so a REF
    # that disagreed with the reference was reported as an ordinary frameshift.
    if codon_ref[offset : offset + len(ref_bases)] != ref_bases:
        logger.warning(
            "Reference mismatch at %s gpos=%d — expected %s at offset %d of codon %s, got %s",
            context or "<unknown-orf>",
            genomic_pos,
            ref_bases,
            offset,
            codon_ref,
            codon_ref[offset : offset + len(ref_bases)],
        )
        return {
            "consequence": "reference_mismatch",
            "protein_pos": first_codon,
            "aa_ref": None,
            "aa_alt": None,
            "codon_ref": codon_ref,
            "codon_alt": None,
            "note": "",
            "validated": False,
        }

    if len(ref_bases) != len(alt_bases):
        # The REF check above covers the *changed* bases; the padding base is
        # trimmed off before it and would otherwise never be verified, so a
        # record whose anchor disagrees with the reference would read as a clean
        # indel. Check the whole supplied REF when all of it maps — it does not
        # when the padding base sits outside the ORF, which is legitimate for an
        # indel at the very start.
        full_offsets = [pos_map.get(p) for p in range(genomic_pos, genomic_pos + len(ref))]
        if all(o is not None for o in full_offsets):
            low = min(full_offsets)  # type: ignore[type-var]
            supplied = ref.upper() if strand != "-" else _revcomp(ref.upper())
            found = coding_seq[low : low + len(supplied)]
            if found != supplied:
                logger.warning(
                    "Reference mismatch at %s gpos=%d — indel REF %s, found %s",
                    context or "<unknown-orf>",
                    genomic_pos,
                    supplied,
                    found,
                )
                return {
                    "consequence": "reference_mismatch",
                    "protein_pos": first_codon,
                    "aa_ref": None,
                    "aa_alt": None,
                    "codon_ref": codon_ref,
                    "codon_alt": None,
                    "note": "",
                    "validated": False,
                }

        # Indels need the whole coding sequence, not a codon window: splicing a
        # length-changing allele into three bases cannot express a frameshift,
        # and the residues gained or lost are only visible against the full
        # translation. The codon-local path below stays exact for substitutions.
        return _indel_result(
            coding_seq=coding_seq,
            coding_pos=coding_pos,
            first_codon=first_codon,
            codon_ref=codon_ref,
            ref_bases=ref_bases,
            alt_bases=alt_bases,
        )

    codon_alt = codon_ref[:offset] + alt_bases + codon_ref[offset + len(ref_bases) :]
    aa_ref = translate(codon_ref)
    aa_alt = translate(codon_alt)

    # Codon 0 is this ORF's start, and what matters there is whether the
    # trinucleotide still initiates — not what it translates to. The rule is
    # asymmetric (see start_codon_effect): destroying an ATG is start-loss,
    # while gaining one strengthens initiation and is not. "start_lost" is
    # already in varianteffect's LOF_CONSEQUENCES, so an override reaches the
    # loss-of-function branch that AlphaMissense and ESM-C — both missense-only,
    # and both blind here — cannot express.
    start_override, note = (
        start_codon_effect(codon_ref[:3], codon_alt[:3]) if first_codon == 0 else (None, "")
    )

    if start_override:
        consequence = start_override
    elif aa_ref == aa_alt:
        consequence = "synonymous_variant"
    elif "*" in aa_alt:
        # Membership, not equality: a substitution spanning two codons can turn
        # the second one into a stop while leaving the first a normal residue.
        consequence = "stop_gained"
    elif "*" in aa_ref:
        consequence = "stop_lost"
    else:
        consequence = "missense_variant"

    return {
        "consequence": consequence,
        "protein_pos": first_codon,
        "aa_ref": aa_ref,
        "aa_alt": aa_alt,
        "codon_ref": codon_ref,
        "codon_alt": codon_alt,
        "note": note,
        "validated": True,
    }


def _indel_term(ref_changed: str, alt_changed: str) -> str | None:
    """The consequence class of a length-changing variant, from the delta alone.

    Needs no sequence, so it is the one answer available even when the coding
    sequence is missing or the codon window runs off its end. ``None`` for an
    equal-length allele, whose class only translation can decide.
    """
    delta = len(alt_changed) - len(ref_changed)
    if delta == 0:
        return None
    if delta % 3 != 0:
        return "frameshift_variant"
    return "inframe_insertion" if delta > 0 else "inframe_deletion"


def _translate(seq: str, *, stop_at_first: bool = False) -> str:
    """Translate a coding sequence, dropping any trailing partial codon.

    A frameshifted sequence is rarely a multiple of three, and the residues past the
    last whole codon are not knowable from the CDS alone.

    Args:
        seq: Coding sequence in mRNA order.
        stop_at_first: Truncate at the first stop, which is what the mutant protein
            actually is — nothing downstream of a premature stop is translated.
    """
    whole = seq[: len(seq) - len(seq) % 3]
    protein = translate(whole)
    if stop_at_first and "*" in protein:
        return protein[: protein.index("*") + 1]
    return protein


@functools.lru_cache(maxsize=256)
def _translate_cached(seq: str) -> str:
    """:func:`_translate` of a reference CDS, which every indel in an ORF re-reads.

    A multi-allelic record fans out into one indel per ALT against the same ORF, and
    each used to retranslate the whole reference.
    """
    return _translate(seq)


def _span_edge_effect(
    *,
    pos_map: dict[int, int],
    coding_seq: str,
    strand: str,
    span_start: int,
    span_offsets: list[int | None],
) -> dict[str, Any] | None:
    """Classify a span that is partly coding by *where* it leaves the coding sequence.

    Two exits change the answer, and neither is visible in the REF/ALT length delta,
    which counts the non-coding bases too:

    * **Into an intron inside the ORF** — a splice site. A span that crosses an exon
      edge from the intron side always takes the intronic base next to the exon,
      which is the +1/+2 (donor) or -1/-2 (acceptor) dinucleotide, so the exon is no
      longer spliced as annotated. ``AAGC>A`` over an acceptor removes one coding
      base and two intronic ones; its delta of -3 read as ``inframe_deletion``,
      which is outside ``LOF_CONSEQUENCES``. Reported as VEP does,
      ``splice_acceptor_variant`` / ``splice_donor_variant``, whatever the delta.
    * **Out of the 5' end, through the start codon** — the bases removed or
      rewritten include the initiator, which a span reaching in from the 5'UTR
      destroys whatever the delta: ``c.-3_3del`` is ``p.Met1?``. The substitution
      and in-ORF indel paths ask :func:`start_codon_effect` about codon 0, and so
      does this one, with the lost bases unknown (``N``).

    "Inside the ORF" is the genomic envelope of the coding bases: a non-coding base
    between the first and last coding base is intronic to this ORF, one outside it is
    UTR or flank. Mapped offsets are in mRNA order, so the start-codon test needs no
    strand; the donor/acceptor side does.

    Returns:
        The result dict, or None when neither applies (a span leaving through the
        3' end) and the caller's length-delta answer stands.
    """
    positions = range(span_start, span_start + len(span_offsets))
    coding = [p for p, o in zip(positions, span_offsets, strict=True) if o is not None]
    noncoding = [p for p, o in zip(positions, span_offsets, strict=True) if o is None]
    mapped = [o for o in span_offsets if o is not None]
    gmin, gmax = min(pos_map), max(pos_map)

    intronic = [p for p in noncoding if gmin < p < gmax]
    if intronic:
        # Plus strand: an intron below the coding bases precedes the exon in mRNA
        # order, so it is the acceptor side. The minus strand reads the other way.
        # Intronic bases on both sides (or between them) mean the span takes a whole
        # exon edge pair; that is reported as the acceptor, the first one in mRNA
        # order.
        below = any(p < min(coding) for p in intronic)
        above = any(p > max(coding) for p in intronic)
        upstream = below if strand == "+" else above
        downstream = above if strand == "+" else below
        side = "donor" if downstream and not upstream else "acceptor"
        return {
            "consequence": f"splice_{side}_variant",
            "protein_pos": min(mapped) // 3,
            "aa_ref": None,
            "aa_alt": None,
            "codon_ref": None,
            "codon_alt": None,
            "note": (
                f"changes {len(mapped)} coding base(s) and {len(intronic)} intronic "
                f"base(s) across the exon's splice {side}"
            ),
            "validated": True,
        }

    if coding_seq and min(mapped) < 3:
        start_override, start_note = start_codon_effect(coding_seq[:3], "NNN")
        if start_override:
            return {
                "consequence": start_override,
                "protein_pos": 0,
                "aa_ref": _translate_cached(coding_seq)[:1],
                "aa_alt": "",
                "codon_ref": coding_seq[:3],
                "codon_alt": None,
                "note": f"{start_note} by a change running in from the 5'UTR",
                "validated": True,
            }
    return None


def _indel_result(
    *,
    coding_seq: str,
    coding_pos: int,
    first_codon: int,
    codon_ref: str,
    ref_bases: str,
    alt_bases: str,
) -> dict[str, Any]:
    """Classify a length-changing variant and name the residues it gains or loses.

    Splices the ALT into the coding sequence and translates both, which is what makes
    the amino acids available at all — the codon-window path cannot, because a
    frameshift changes every residue after the event and an in-frame indel changes
    the length of the protein.

    ``aa_ref`` and ``aa_alt`` read as "removed" and "gained": a deletion reports the
    residues it takes out and no replacement, an insertion the reverse. For an indel
    that is not codon-aligned this is an approximation — the residues at the seam are
    rewritten rather than cleanly removed — so the note says which residues the call
    is anchored on rather than implying a tidy excision.

    Frameshifts report the first affected residue and, when a premature stop falls
    inside the coding sequence, the distance to it. The new frame usually runs past
    the annotated stop into 3'UTR the CDS does not contain, so a stop distance is
    only quoted when one is genuinely reachable.
    """
    delta = len(alt_bases) - len(ref_bases)
    mutant = coding_seq[:coding_pos] + alt_bases + coding_seq[coding_pos + len(ref_bases) :]
    ref_prot = _translate_cached(coding_seq)
    alt_prot = _translate(mutant, stop_at_first=True)

    # The same codon-0 question the substitution path asks: does the trinucleotide at
    # the start still initiate? An indel can destroy an ATG as surely as an SNV can —
    # deleting it outright, or splitting it with an insertion — and ClinVar names
    # those ``p.Met1?`` / ``p.Met1del``. ``inframe_deletion`` is not in the LoF set, so
    # without this the classic deleted-start variant never reached it.
    if first_codon == 0:
        start_override, start_note = start_codon_effect(coding_seq[:3], mutant[:3])
        if start_override:
            return {
                "consequence": start_override,
                "protein_pos": 0,
                "aa_ref": ref_prot[:1],
                "aa_alt": "",
                "codon_ref": codon_ref,
                "codon_alt": None,
                "note": f"{start_note} by a {_indel_term(ref_bases, alt_bases)}",
                "validated": True,
            }

    if delta % 3 != 0:
        consequence = "frameshift_variant"
        aa_ref = ref_prot[first_codon] if first_codon < len(ref_prot) else ""
        note = f"frameshift from residue {first_codon + 1}"
        if alt_prot.endswith("*"):
            note += f"; premature stop {len(alt_prot) - first_codon} residues downstream"
        else:
            note += "; no premature stop inside the coding sequence"
        return {
            "consequence": consequence,
            "protein_pos": first_codon,
            "aa_ref": aa_ref,
            "aa_alt": "",
            "codon_ref": codon_ref,
            "codon_alt": None,
            "note": note,
            "validated": True,
        }

    n_codons = abs(delta) // 3
    if delta < 0:
        removed = ref_prot[first_codon : first_codon + n_codons]
        return {
            "consequence": "inframe_deletion",
            "protein_pos": first_codon,
            "aa_ref": removed,
            "aa_alt": "",
            "codon_ref": codon_ref,
            "codon_alt": None,
            "note": f"{n_codons} residue(s) removed from {first_codon + 1}",
            "validated": True,
        }

    gained = alt_prot[first_codon : first_codon + n_codons]
    return {
        "consequence": "inframe_insertion",
        "protein_pos": first_codon,
        "aa_ref": "",
        "aa_alt": gained,
        "codon_ref": codon_ref,
        "codon_alt": None,
        "note": f"{n_codons} residue(s) inserted at {first_codon + 1}",
        "validated": True,
    }


def null_result(validated: bool) -> dict[str, Any]:
    """The result for a variant the classifier cannot place: every field empty."""
    return {
        "consequence": None,
        "protein_pos": None,
        "aa_ref": None,
        "aa_alt": None,
        "codon_ref": None,
        "codon_alt": None,
        "note": "",
        "validated": validated,
    }
