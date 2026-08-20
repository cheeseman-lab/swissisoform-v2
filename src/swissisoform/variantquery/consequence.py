"""The consequence vocabulary the gene figure draws, and the sequence-free fallback.

Classification itself is **not** here. A variant is classified by the pipeline's own
:meth:`~swissisoform.clinical.validate.ConsequenceValidator.classify_against_orf`,
called with the coding sequence the ORF index carries — the same code path, the same
codon walk and the same terms an annotated variant goes through, so an uploaded VCF
and a ClinVar record cannot disagree about what a substitution does.

What remains here is what that classifier has no counterpart for: an index built
without a genome (``build_orf_index.py --no-cds``) carries no coding sequence, and a
variant resolved against it can still be classified by length alone.
"""

from __future__ import annotations

# Terms the gene figure knows how to draw (plots/protein.py). Anything outside this
# set lands in a generic row, so new terms must be added to both.
FRAMESHIFT = "frameshift_variant"
STOP_GAINED = "stop_gained"
STOP_LOST = "stop_lost"
START_LOST = "start_lost"
MISSENSE = "missense_variant"
SYNONYMOUS = "synonymous_variant"
INFRAME_DELETION = "inframe_deletion"
INFRAME_INSERTION = "inframe_insertion"
OTHER = "other"

_NO_SEQUENCE = "no coding sequence in the index"


def classify_without_sequence(ref: str, alt: str) -> tuple[str, str]:
    """Class from the length delta alone, for an index built without a genome.

    Frameshift vs in-frame is exact without any sequence, and matches what the
    classifier derives the same way (``validate.py:435-449``). Substitutions cannot
    be resolved — missense, synonymous and stop_gained are indistinguishable from
    REF/ALT alone — so they come back as ``other`` with a note rather than a guess.

    Returns:
        ``(term, note)``; the note explains an unresolved or absent answer and is
        empty when the term stands on its own.
    """
    ref, alt = ref.upper(), alt.upper()
    # Strip the shared leading base a VCF indel carries: it is padding, not part of
    # the change, and it would make CGT>C look like a 3-to-1 substitution.
    while len(ref) > 1 and len(alt) > 1 and ref[0] == alt[0]:
        ref, alt = ref[1:], alt[1:]
    if len(ref) != len(alt) and ref and alt and ref[0] == alt[0]:
        ref, alt = ref[1:], alt[1:]

    delta = len(alt) - len(ref)
    if delta % 3 != 0:
        return FRAMESHIFT, _NO_SEQUENCE
    if delta < 0:
        return INFRAME_DELETION, _NO_SEQUENCE
    if delta > 0:
        return INFRAME_INSERTION, _NO_SEQUENCE
    return OTHER, "substitution needs the codon; index was built without a genome"
