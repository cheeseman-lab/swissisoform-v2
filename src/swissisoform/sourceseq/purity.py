"""Window-purity decision: do a TIS's candidate transcripts agree near the start?

Given the ±W start-codon windows (:class:`swissisoform.sourceseq.mrna.TisWindow`)
of the candidate transcripts that survived expression/CAGE narrowing for one
TIS, decide whether they are **identical within ``W`` nt of the start codon**.

The point (see ``docs/plans/source_transcript_resolution.md``): if every
surviving candidate shares the same local sequence, the TIS maps to a unique
mRNA window regardless of which isoform a footprint came from, so the TIS is
kept with that consensus window. If they diverge inside the window the local
sequence is genuinely ambiguous and the TIS is discarded.
"""

from __future__ import annotations

from dataclasses import dataclass

from swissisoform.sourceseq.mrna import TisWindow


@dataclass
class PurityResult:
    """Outcome of the window-purity test for one TIS.

    Attributes:
        keep: Whether to keep the TIS (windows agree within ``W``).
        divergence_nt: Smallest radius (1..W, in nt from the A) at which the
            candidates first disagree, or ``None`` if they agree throughout.
        reason: ``"single_candidate"`` | ``"pure_window"`` |
            ``"ambiguous_window"`` | ``"no_candidates"``.
    """

    keep: bool
    divergence_nt: int | None
    reason: str


def _all_equal(values: list) -> bool:
    """True if every element equals the first (``None`` is a distinct value)."""
    first = values[0]
    return all(v == first for v in values[1:])


def divergence_radius(windows: list[TisWindow], w: int) -> int | None:
    """Smallest radius ``r`` in ``1..w`` at which the windows disagree.

    The start-codon A is the shared anchor at radius 0 and is not compared.
    At each radius ``r`` the ``r``-th base 3' of the A (``downstream[r]``) and
    the ``r``-th base 5' of the A (``upstream[-r]``) are compared across all
    candidates. A candidate that has run out of sequence at radius ``r`` (a
    shorter 5'UTR, or a 3' end) contributes ``None`` — so a *different
    transcript extent* within the window counts as a divergence, while
    candidates that all end at the same radius still agree.

    Args:
        windows: One :class:`TisWindow` per candidate transcript.
        w: Window radius in nt.

    Returns:
        The radius of the first disagreement, or ``None`` if all candidates
        are identical out to ``w`` on both sides (given available sequence).
    """
    for r in range(1, w + 1):
        downstream = [wn.downstream[r] if r < len(wn.downstream) else None for wn in windows]
        if not _all_equal(downstream):
            return r
        upstream = [wn.upstream[-r] if r <= len(wn.upstream) else None for wn in windows]
        if not _all_equal(upstream):
            return r
    return None


def purity_decision(windows: list[TisWindow], w: int) -> PurityResult:
    """Decide whether a TIS's candidate windows are pure within ``±w`` nt.

    Args:
        windows: One :class:`TisWindow` per surviving candidate transcript.
        w: Window radius in nt (the "±W" of the test).

    Returns:
        A :class:`PurityResult`. A single candidate is trivially pure; an empty
        set is reported as ``no_candidates`` (not kept).
    """
    if not windows:
        return PurityResult(keep=False, divergence_nt=None, reason="no_candidates")
    if len(windows) == 1:
        return PurityResult(keep=True, divergence_nt=None, reason="single_candidate")
    r = divergence_radius(windows, w)
    if r is None:
        return PurityResult(keep=True, divergence_nt=None, reason="pure_window")
    return PurityResult(keep=False, divergence_nt=r, reason="ambiguous_window")
