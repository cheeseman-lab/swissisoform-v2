"""Turning judgments into arm contrasts.

Six rules, each answering a way this comparison can lie:

1. **Absolute scores are centered within a cell, never averaged raw.** Isoform
   difficulty dominates raw means -- a conserved truncation outscores a uORF under
   every arm -- so a raw mean conflates arm quality with which isoforms are easy.
   Subtracting the cell mean leaves each arm's advantage over the field on that
   cell. A within-block contrast, no model required.
2. **Pairwise goes through Bradley-Terry**, per category, with the status quo
   pinned at strength 0 so every number reads as log-odds better than what we ship.
   A position-aware variant was built and removed. It modelled the slot-A
   advantage as a nuisance parameter -- ``P(slot-A wins) = sigmoid(s_A - s_B +
   delta)`` -- which kept all 21,035 parsed comparisons instead of the 4,156 that
   agreed, and measured delta at +0.797 pooled (0.29 in synthesis to 1.71 in L).
   It was correct and it validated: the replicate came out at +0.013 against
   +0.046 for the filter. But it changed only the *magnitudes*, shrinking every
   effect ~2.3x, and left the ordering identical. Given the judge discriminates so
   weakly that position explains 5-10x more of each verdict than arm quality, the
   ordering is the only durable output, so the extra machinery bought nothing the
   conclusion rests on. Disagreements also turned out to carry almost no
   information about which arm is better -- pairs that disagreed involved arms only
   1.07x closer than pairs that agreed, not the clear separation the model assumes.
3. **Order-inconsistent pairs are dropped, not split.** Prometheus 2 has
   documented position bias; a pair the judge decides differently by presentation
   order carries no information, and splitting it half-and-half would dilute real
   signal toward zero. The drop rate is reported as judge reliability.
4. **Uncertainty comes from a cluster bootstrap over isoforms**, resampling
   isoforms rather than judgments. The 7 units of one isoform share evidence and
   are not independent; a naive bootstrap would report intervals several times too
   narrow and manufacture significance.
5. **The replicate arm is the resolution floor.** ``criteria_hint_rep`` is the
   status quo run twice, and it competes in the fit like any other arm, so its
   interval is where zero sits when the same framing is judged against itself. An
   effect whose point estimate falls inside that half-width is not distinguishable
   from re-running one arm. It is per unit because the units do not resolve alike.

   This replaced a floor measured as the fraction of isoforms whose *verdict
   label* flipped between the two runs. That statistic is gone with the label, and
   the two are not interchangeable: the old one was on the output scale (11.7%
   overall, 4.0% in C to 20.0% in S), this one is in the judge's log-odds. Old
   ``Nx floor`` ratios cannot be reconstructed from new output.
6. **Nothing is pooled across categories in a headline.** ``tags`` carries 24 tags
   in S against 3 in D; pooled, vocabulary thinness reads as a framing effect.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from swissisoform.judge import ARMS, BASELINE, REPLICATE

# Bootstrap replicates. 2,000 is enough for a 95% percentile interval and cheap
# here, since each replicate refits on at most a few thousand comparisons.
N_BOOTSTRAP = 2_000
SEED = 20260915


@dataclass(frozen=True)
class Comparison:
    """One order-consistent pairwise verdict within one cell."""

    slug: str
    unit: str
    winner: str
    loser: str


@dataclass(frozen=True)
class Score:
    """One absolute score."""

    slug: str
    unit: str
    arm: str
    rubric: str
    score: int


@dataclass
class OrderCheck:
    """What both-orders agreement looked like, per unit."""

    consistent: int = 0
    inconsistent: int = 0
    unparseable: int = 0

    @property
    def total(self) -> int:
        """Every pair attempted."""
        return self.consistent + self.inconsistent + self.unparseable

    @property
    def inconsistency_rate(self) -> float:
        """Share of parseable pairs the judge decided both ways."""
        parseable = self.consistent + self.inconsistent
        return self.inconsistent / parseable if parseable else 0.0


def resolve_orders(
    forward: dict[tuple[str, str, str, str], str | None],
) -> tuple[list[Comparison], dict[str, OrderCheck]]:
    """Keep only pairs both presentation orders agree on.

    Args:
        forward: ``{(slug, unit, arm_a, arm_b): winner}`` where ``winner`` is the
            arm id the judge picked, or ``None`` when the completion did not
            parse. Both ``(a, b)`` and ``(b, a)`` are expected.

    Returns:
        ``(comparisons, checks)`` -- the surviving verdicts, and per-unit tallies
        of what was dropped and why.
    """
    checks: dict[str, OrderCheck] = defaultdict(OrderCheck)
    seen: set[tuple[str, str, frozenset[str]]] = set()
    out: list[Comparison] = []

    for (slug, unit, arm_a, arm_b), winner in forward.items():
        key = (slug, unit, frozenset({arm_a, arm_b}))
        if key in seen:
            continue
        reverse = forward.get((slug, unit, arm_b, arm_a), "__missing__")
        if reverse == "__missing__":
            continue
        seen.add(key)
        if winner is None or reverse is None:
            checks[unit].unparseable += 1
            continue
        if winner != reverse:
            checks[unit].inconsistent += 1
            continue
        checks[unit].consistent += 1
        loser = arm_b if winner == arm_a else arm_a
        out.append(Comparison(slug=slug, unit=unit, winner=winner, loser=loser))
    return out, dict(checks)


def center_scores(scores: Iterable[Score]) -> dict[tuple[str, str, str], float]:
    """Mean cell-centered score per ``(arm, unit, rubric)``.

    Centering happens inside ``(slug, unit, rubric)``: every arm judged on that
    cell contributes, and the cell's own mean is subtracted. A cell judged on
    fewer than two arms is dropped -- with one arm the centered value is 0 by
    construction and carries no comparison.
    """
    by_cell: dict[tuple[str, str, str], list[Score]] = defaultdict(list)
    for s in scores:
        by_cell[(s.slug, s.unit, s.rubric)].append(s)

    sums: dict[tuple[str, str, str], float] = defaultdict(float)
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for cell, group in by_cell.items():
        if len(group) < 2:
            continue
        mean = sum(s.score for s in group) / len(group)
        for s in group:
            sums[(s.arm, s.unit, s.rubric)] += s.score - mean
            counts[(s.arm, s.unit, s.rubric)] += 1
    return {k: sums[k] / counts[k] for k in sums}


def bradley_terry(
    comparisons: Sequence[Comparison],
    arms: Sequence[str] | None = None,
    *,
    baseline: str = BASELINE,
    prior: float = 0.5,
    iterations: int = 500,
    tol: float = 1e-9,
) -> dict[str, float]:
    """Log-strengths per arm, with *baseline* pinned at 0.

    Fitted by the standard MM update (Hunter 2004), which is monotone and needs no
    step size. ``prior`` adds a symmetric pseudo-win to every matched pair actually
    observed, so an arm that won or lost all of its comparisons gets a large finite
    strength rather than an infinite one -- with 50 isoforms a clean sweep is
    common and unregularised BT would diverge on it.

    Returns an empty dict when no comparison survives, which is a real outcome for
    a unit where the judge contradicted itself throughout.
    """
    # ``arms=None`` means every arm that actually competed, which is the right
    # default: an arm present in the comparisons but absent from *arms* used to be
    # counted in the numerator (it beat someone) and not the denominator (it was
    # not an opponent), inflating everyone else. The replicate hit exactly this --
    # it is not in ARMS, so 1,297 of 5,919 comparisons were half-counted and the
    # noise-floor control itself got no estimate. An explicit *arms* is still
    # honoured, and `pool` below drops the excluded arms' games so the numerator
    # and denominator always describe the same opponent set.
    observed = {c.winner for c in comparisons} | {c.loser for c in comparisons}
    candidates = sorted(observed) if arms is None else arms
    present = [a for a in candidates if a in observed]
    if not present or not comparisons:
        return {}

    wins: dict[tuple[str, str], float] = defaultdict(float)
    for c in comparisons:
        wins[(c.winner, c.loser)] += 1.0

    # Symmetric prior, applied only to pairs that actually met.
    pairs = {frozenset(k) for k in wins}
    for pair in pairs:
        a, b = sorted(pair)
        wins[(a, b)] += prior
        wins[(b, a)] += prior

    # Restricted to opponents in *present*, so the numerator can never count a game
    # the denominator below does not.
    pool = set(present)
    total_wins = {
        a: sum(v for (w, loser), v in wins.items() if w == a and loser in pool) for a in present
    }
    strength = {a: 1.0 for a in present}

    for _ in range(iterations):
        updated: dict[str, float] = {}
        for a in present:
            denominator = 0.0
            for b in present:
                if a == b:
                    continue
                n_ab = wins.get((a, b), 0.0) + wins.get((b, a), 0.0)
                if n_ab:
                    denominator += n_ab / (strength[a] + strength[b])
            updated[a] = total_wins[a] / denominator if denominator else strength[a]
        scale = sum(updated.values()) / len(updated)
        updated = {a: v / scale for a, v in updated.items()}
        shift = max(abs(updated[a] - strength[a]) for a in present)
        strength = updated
        if shift < tol:
            break

    anchor = strength.get(baseline)
    log = {a: math.log(v) for a, v in strength.items()}
    if anchor:
        offset = math.log(anchor)
        log = {a: v - offset for a, v in log.items()}
    return log


@dataclass
class Interval:
    """A point estimate with a bootstrap percentile interval."""

    point: float
    lo: float
    hi: float
    n: int = 0

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval is entirely on one side of zero."""
        return (self.lo > 0 and self.hi > 0) or (self.lo < 0 and self.hi < 0)

    @property
    def half_width(self) -> float:
        """Half the interval's span — the resolution this fit can offer.

        On the replicate arm this is the noise floor: the same framing judged
        twice, so an effect whose point estimate falls inside it is not
        distinguishable from running one arm again.
        """
        return (self.hi - self.lo) / 2


def cluster_bootstrap_bt(
    comparisons: Sequence[Comparison],
    arms: Sequence[str] | None = None,
    *,
    baseline: str = BASELINE,
    n: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict[str, Interval]:
    """Bradley-Terry strengths with isoform-clustered bootstrap intervals.

    Isoforms are the resampling unit. Resampling individual comparisons would
    treat the 36 pairs within one cell as independent when they are 9 arms read
    against one shared evidence payload.

    ``arms=None`` passes through to :func:`bradley_terry`, i.e. fit every arm that
    competed. Defaulting to ``ARMS`` here is what left the replicate without an
    interval even after that function was fixed.
    """
    point = bradley_terry(comparisons, arms, baseline=baseline)
    if not point:
        return {}

    by_slug: dict[str, list[Comparison]] = defaultdict(list)
    for c in comparisons:
        by_slug[c.slug].append(c)
    slugs = sorted(by_slug)

    rng = random.Random(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(n):
        picked = [slugs[rng.randrange(len(slugs))] for _ in range(len(slugs))]
        resampled = [c for s in picked for c in by_slug[s]]
        fit = bradley_terry(resampled, arms, baseline=baseline)
        for arm, value in fit.items():
            draws[arm].append(value)

    out: dict[str, Interval] = {}
    for arm, value in point.items():
        series = sorted(draws.get(arm, []))
        if not series:
            out[arm] = Interval(point=value, lo=value, hi=value, n=0)
            continue
        out[arm] = Interval(
            point=value,
            lo=series[int(0.025 * (len(series) - 1))],
            hi=series[int(0.975 * (len(series) - 1))],
            n=len(series),
        )
    return out


def factorial_effects(
    strengths: dict[str, Interval],
) -> dict[str, dict[str, float]]:
    """Grounding and hint main effects from the 8 design arms.

    The replicate is excluded: it is not a cell of the 4x2 design, and including
    it would invent a fifth grounding and double-weight ``criteria``.
    """
    usable = {a: v.point for a, v in strengths.items() if a in ARMS and a != REPLICATE}
    if not usable:
        return {}

    grounding: dict[str, list[float]] = defaultdict(list)
    hint: dict[str, list[float]] = defaultdict(list)
    for arm, value in usable.items():
        mode, _, level = arm.rpartition("_")
        grounding[mode].append(value)
        hint[level].append(value)

    return {
        "grounding": {k: sum(v) / len(v) for k, v in sorted(grounding.items())},
        "hint": {k: sum(v) / len(v) for k, v in sorted(hint.items())},
    }
