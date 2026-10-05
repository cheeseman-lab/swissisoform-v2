"""Turning judgments into arm contrasts.

Five rules, each answering a way this comparison can lie:

1. **Pairwise goes through Bradley-Terry with the presentation slot as a fitted
   covariate**, per category, with the status quo pinned at strength 0 so every
   number reads as log-odds better than what we ship:
   ``P(slot-A wins) = sigmoid(s_A - s_B + delta)``. Both orders of every pair are
   kept, and ``delta`` -- the judge's slot-A advantage -- is estimated and
   reported beside the arm strengths. An earlier round removed this model
   because it left the arm *ordering* unchanged while shrinking magnitudes
   ~2.3x; but slot A wins 68-76% of calls, and fitting only the order-agreeing
   pairs lets whatever else decides agreement carry the ranking unseen.
2. **Order-inconsistent pairs are counted, never silently dropped.** The
   consistent-only Bradley-Terry fit that earlier rounds published is still
   computed as a secondary readout, but its drops are reported per unit *and*
   per arm, together with how many of them went to slot A both times -- so a
   reader can see which arms the filter removed and that position, not a
   change of mind, removed them.
3. **Uncertainty comes from a cluster bootstrap over isoforms**, resampling
   isoforms rather than judgments. The 7 units of one isoform share evidence and
   are not independent; a naive bootstrap would report intervals several times too
   narrow and manufacture significance.
4. **The replicate arm is the resolution floor.** ``criteria_hint_rep`` is the
   status quo run twice, and it competes in the fit like any other arm, so its
   interval is where zero sits when the same framing is judged against itself. An
   effect whose point estimate falls inside that half-width is not distinguishable
   from re-running one arm. It is per unit because the units do not resolve alike.

   This replaced a floor measured as the fraction of isoforms whose *verdict
   label* flipped between the two runs. That statistic is gone with the label, and
   the two are not interchangeable: the old one was on the output scale (11.7%
   overall, 4.0% in C to 20.0% in S), this one is in the judge's log-odds. Old
   ``Nx floor`` ratios cannot be reconstructed from new output.
5. **Nothing is pooled across categories in a headline.** ``tags`` carries 24 tags
   in S against 3 in D; pooled, vocabulary thinness reads as a framing effect.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

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


@dataclass
class OrderCheck:
    """What both-orders agreement looked like, per unit."""

    consistent: int = 0
    inconsistent: int = 0
    unparseable: int = 0
    # Inconsistent pairs where slot A won both presentations -- position, not a
    # change of mind about the arms.
    inconsistent_slot_a: int = 0
    # Per arm: how many of its pairs were dropped as inconsistent, and how many kept.
    inconsistent_by_arm: dict[str, int] = field(default_factory=dict)
    consistent_by_arm: dict[str, int] = field(default_factory=dict)

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
    """Keep only pairs both presentation orders agree on, counting what was dropped.

    This is the secondary, consistent-only readout; :func:`fit_bradley_terry` is
    the primary one and keeps every parsed call.

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
        check = checks[unit]
        if winner != reverse:
            check.inconsistent += 1
            if winner == arm_a:
                check.inconsistent_slot_a += 1
            for arm in (arm_a, arm_b):
                check.inconsistent_by_arm[arm] = check.inconsistent_by_arm.get(arm, 0) + 1
            continue
        check.consistent += 1
        for arm in (arm_a, arm_b):
            check.consistent_by_arm[arm] = check.consistent_by_arm.get(arm, 0) + 1
        loser = arm_b if winner == arm_a else arm_a
        out.append(Comparison(slug=slug, unit=unit, winner=winner, loser=loser))
    return out, dict(checks)


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


@dataclass(frozen=True)
class Call:
    """One presentation of one pair: who sat in slot A, who in slot B, who won."""

    slug: str
    unit: str
    arm_a: str
    arm_b: str
    a_won: bool


def calls_from_forward(forward: dict[tuple[str, str, str, str], str | None]) -> list[Call]:
    """Every parsed call, both orders kept; unparseable ones are left out."""
    return [
        Call(slug=slug, unit=unit, arm_a=arm_a, arm_b=arm_b, a_won=winner == arm_a)
        for (slug, unit, arm_a, arm_b), winner in forward.items()
        if winner is not None
    ]


def slot_a_rate(calls: Sequence[Call]) -> float | None:
    """Raw share of calls won by slot A -- 0.5 for a judge without position bias."""
    return sum(c.a_won for c in calls) / len(calls) if calls else None


@dataclass
class CovariateFit:
    """Arm strengths plus the nuisance effects fitted beside them."""

    strengths: dict[str, float]
    position: float | None = None
    n_calls: int = 0


def fit_bradley_terry(
    calls: Sequence[Call],
    *,
    baseline: str = BASELINE,
    position: bool = True,
    ridge: float = 0.1,
    iterations: int = 100,
    tol: float = 1e-9,
) -> CovariateFit:
    """Logistic Bradley-Terry over individual calls, with a slot-A effect.

    ``logit P(slot A wins) = s_A - s_B + delta``, fitted by penalised Newton
    (the problem is a 9-parameter logistic regression, so it converges in a
    handful of steps). ``ridge`` is an L2 penalty on the arm strengths only --
    the counterpart of the MM fit's pseudo-win prior, so a clean sweep gets a
    large finite strength instead of an infinite one -- and never touches
    ``delta``. *baseline* is pinned at 0 when it competed; otherwise strengths are
    centred on their mean.

    Returns an empty fit when no call survives.
    """
    arms = sorted({c.arm_a for c in calls} | {c.arm_b for c in calls})
    if not calls or len(arms) < 2:
        return CovariateFit(strengths={}, n_calls=len(calls))

    anchor = baseline if baseline in arms else arms[0]
    free = [a for a in arms if a != anchor]
    col = {a: i for i, a in enumerate(free)}
    n_arm = len(free)
    n_par = n_arm + int(position)

    x = np.zeros((len(calls), n_par))
    y = np.empty(len(calls))
    for row, c in enumerate(calls):
        if c.arm_a in col:
            x[row, col[c.arm_a]] += 1.0
        if c.arm_b in col:
            x[row, col[c.arm_b]] -= 1.0
        if position:
            x[row, n_arm] = 1.0
        y[row] = 1.0 if c.a_won else 0.0

    penalty = np.zeros(n_par)
    penalty[:n_arm] = ridge
    theta = np.zeros(n_par)

    def objective(t: np.ndarray) -> float:
        eta = x @ t
        return float(y @ eta - np.logaddexp(0.0, eta).sum() - 0.5 * (penalty * t * t).sum())

    current = objective(theta)
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-(x @ theta)))
        grad = x.T @ (y - p) - penalty * theta
        hess = (x * (p * (1.0 - p))[:, None]).T @ x + np.diag(penalty)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        # Step-halving keeps Newton monotone on a near-separable bootstrap draw.
        scale = 1.0
        trial = theta + step
        value = objective(trial)
        while value < current - 1e-12 and scale >= 1e-6:
            scale /= 2
            trial = theta + scale * step
            value = objective(trial)
        if value < current - 1e-12:
            break  # no ascent direction left: converged as far as floats allow
        theta, previous, current = trial, current, value
        if abs(current - previous) < tol and float(np.abs(scale * step).max()) < 1e-6:
            break

    strengths = {anchor: 0.0, **{a: float(theta[col[a]]) for a in free}}
    if anchor != baseline:
        mean = sum(strengths.values()) / len(strengths)
        strengths = {a: v - mean for a, v in strengths.items()}
    return CovariateFit(
        strengths=strengths,
        position=float(theta[n_arm]) if position else None,
        n_calls=len(calls),
    )


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


@dataclass
class CovariateIntervals:
    """A covariate fit with isoform-clustered bootstrap intervals."""

    strengths: dict[str, Interval]
    position: Interval | None = None
    n_calls: int = 0


def cluster_bootstrap_fit(
    calls: Sequence[Call],
    *,
    baseline: str = BASELINE,
    position: bool = True,
    n: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> CovariateIntervals:
    """:func:`fit_bradley_terry` with isoform-clustered percentile intervals.

    Same resampling unit as :func:`cluster_bootstrap_bt`, and the nuisance
    effects get intervals too, so a reported position effect carries its own
    uncertainty.
    """
    kwargs = {"baseline": baseline, "position": position}
    point = fit_bradley_terry(calls, **kwargs)
    if not point.strengths:
        return CovariateIntervals(strengths={}, n_calls=point.n_calls)

    by_slug: dict[str, list[Call]] = defaultdict(list)
    for c in calls:
        by_slug[c.slug].append(c)
    slugs = sorted(by_slug)

    rng = random.Random(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    nuisance: dict[str, list[float]] = defaultdict(list)
    for _ in range(n):
        picked = [slugs[rng.randrange(len(slugs))] for _ in range(len(slugs))]
        fit = fit_bradley_terry([c for s in picked for c in by_slug[s]], **kwargs)
        for arm, value in fit.strengths.items():
            draws[arm].append(value)
        if fit.position is not None:
            nuisance["position"].append(fit.position)

    def interval(value: float, series: list[float]) -> Interval:
        series = sorted(series)
        if not series:
            return Interval(point=value, lo=value, hi=value, n=0)
        return Interval(
            point=value,
            lo=series[int(0.025 * (len(series) - 1))],
            hi=series[int(0.975 * (len(series) - 1))],
            n=len(series),
        )

    return CovariateIntervals(
        strengths={arm: interval(v, draws[arm]) for arm, v in point.strengths.items()},
        position=(
            None if point.position is None else interval(point.position, nuisance["position"])
        ),
        n_calls=point.n_calls,
    )


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
