"""Weigh the judged corpus into arm contrasts.

Reads ``results.jsonl``, applies the six rules in :mod:`swissisoform.judge.weigh`
and writes the tables. Nothing here fits a model to raw scores: the isoform is a
blocking factor throughout, and the replicate arm is carried in the fit so its own
interval shows the smallest effect this pipeline can resolve — a number inside it
is one framing judged twice, not a finding.

Usage:
    python scripts/judge/analyze.py
    python scripts/judge/analyze.py --bootstrap 500     # faster, wider intervals
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from swissisoform.judge import (  # noqa: E402
    ARMS,
    BASELINE,
    DEFAULT_CORPUS,
    REPLICATE,
    SYNTHESIS_UNIT,
    UNITS,
)
from swissisoform.judge import prompts as PR  # noqa: E402
from swissisoform.judge import weigh as W  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("judge.analyze")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI options."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--corpus", default=DEFAULT_CORPUS)
    p.add_argument("--dir", type=Path, default=None)
    p.add_argument("--bootstrap", type=int, default=W.N_BOOTSTRAP)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Parse results, weigh them, write tables."""
    args = parse_args(argv)
    work = args.dir or (ROOT / "data" / "output" / "judge" / args.corpus)
    results_path = work / "results.jsonl"
    if not results_path.exists():
        raise SystemExit(f"no results at {results_path}; run scripts/judge/run_judge.py")

    forward, parse_failures = _parse(results_path)
    logger.info("%d pairwise call(s), %d unparseable", len(forward), parse_failures)

    comparisons, checks = W.resolve_orders(forward)
    logger.info("%d order-consistent comparison(s)", len(comparisons))

    # Primary: every parsed call, both orders, with the slot-A effect fitted.
    calls = W.calls_from_forward(forward)
    per_unit_fit = {
        unit: W.cluster_bootstrap_fit([c for c in calls if c.unit == unit], n=args.bootstrap)
        for unit in UNITS
    }
    pooled_fit = W.cluster_bootstrap_fit(calls, n=args.bootstrap)
    position = _position_report(calls, per_unit_fit, pooled_fit)

    # Secondary: the consistent-only fit earlier rounds published.
    per_unit_bt = {
        unit: W.cluster_bootstrap_bt([c for c in comparisons if c.unit == unit], n=args.bootstrap)
        for unit in UNITS
    }
    pooled_bt = W.cluster_bootstrap_bt(comparisons, n=args.bootstrap)

    _write(work, checks, per_unit_fit, pooled_fit, position, per_unit_bt, pooled_bt, parse_failures)
    _print(checks, per_unit_fit, pooled_fit, position, per_unit_bt)
    return 0


def _interval(i) -> dict:
    """An :class:`~swissisoform.judge.weigh.Interval` as rounded JSON."""
    return {"point": round(i.point, 4), "lo": round(i.lo, 4), "hi": round(i.hi, 4)}


def _position_report(calls, per_unit_fit, pooled_fit) -> dict:
    """Raw slot-A win rate and the fitted slot-A log-odds, per unit and pooled."""
    out = {}
    for unit, fit in (*per_unit_fit.items(), ("pooled", pooled_fit)):
        subset = calls if unit == "pooled" else [c for c in calls if c.unit == unit]
        rate = W.slot_a_rate(subset)
        out[unit] = {
            "n_calls": len(subset),
            "slot_a_win_rate": None if rate is None else round(rate, 4),
            "delta": None if fit.position is None else _interval(fit.position),
        }
    return out


def _parse(path: Path):
    """``(forward, n_unparseable)`` from a results file."""
    forward: dict[tuple[str, str, str, str], str | None] = {}
    failures = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid, completion = row["id"], row.get("completion", "")
            parts = rid.split("|")
            if parts[0] == "pw":
                _, slug, unit, arm_a, arm_b, _order = parts
                winner = _winner(row, completion, arm_a, arm_b)
                if winner is None:
                    failures += 1
                forward[(slug, unit, arm_a, arm_b)] = winner
    return forward, failures


def _winner(row: dict, completion: str, arm_a: str, arm_b: str) -> str | None:
    """The arm this call preferred, read from logprobs and falling back to the regex.

    Logprobs first because they read far more of the corpus: 33 of 40 gate probes
    against the regex's 15, and 40 of 40 once the forced-verdict pass is counted.
    The regex fallback is what lets a pre-logprob results file still be analysed --
    the first 31,950-call run has no logprobs in it at all.

    An exact 0.5 is returned as ``None``. It means the two letters were equally
    likely at the verdict position, which is not a preference; with a saturated
    verdict token it should essentially never happen, and counting it as a win for
    whichever arm sorts first would be inventing data.
    """
    steps = row.get("token_logprobs") or []
    if steps or row.get("forced_logprobs"):
        prob = PR.preference(steps, completion, row.get("forced_logprobs") or [])
        if prob is None or prob == 0.5:
            return None
        return arm_a if prob > 0.5 else arm_b
    try:
        choice, _ = PR.parse_choice(completion)
    except PR.ParseError:
        return None
    return arm_a if choice == "A" else arm_b


def _resolution_floor(per_unit_bt, pooled_bt) -> dict:
    """The replicate arm's own interval, per unit — the smallest resolvable effect.

    No separate measurement: ``criteria_hint_rep`` competes in the fit like any
    other arm, so its interval already says where zero sits when one framing is
    judged against itself.
    """
    out = {}
    for unit, fit in per_unit_bt.items():
        i = fit.get(REPLICATE)
        if i is not None:
            out[unit] = {
                "point": round(i.point, 4),
                "lo": round(i.lo, 4),
                "hi": round(i.hi, 4),
                "half_width": round(i.half_width, 4),
            }
    i = pooled_bt.get(REPLICATE)
    if i is not None:
        out["pooled"] = {
            "point": round(i.point, 4),
            "lo": round(i.lo, 4),
            "hi": round(i.hi, 4),
            "half_width": round(i.half_width, 4),
        }
    return out


def _write(
    work, checks, per_unit_fit, pooled_fit, position, per_unit_bt, pooled_bt, parse_failures
) -> None:
    """Persist everything as JSON, plus a TSV of the headline table.

    ``bradley_terry_*`` is the order-aware fit over every parsed call;
    ``bradley_terry_consistent_only_*`` is the older fit over order-agreeing
    pairs, kept for comparison with earlier rounds and never the headline.
    """
    per_unit_strengths = {unit: fit.strengths for unit, fit in per_unit_fit.items()}
    payload = {
        "resolution_floor": _resolution_floor(per_unit_strengths, pooled_fit.strengths),
        "position_effect": position,
        "judge_reliability": {
            unit: {
                "consistent": c.consistent,
                "inconsistent": c.inconsistent,
                "inconsistent_slot_a_both_orders": c.inconsistent_slot_a,
                "unparseable": c.unparseable,
                "inconsistency_rate": round(c.inconsistency_rate, 4),
                "inconsistent_by_arm": dict(sorted(c.inconsistent_by_arm.items())),
                "consistent_by_arm": dict(sorted(c.consistent_by_arm.items())),
            }
            for unit, c in checks.items()
        },
        "n_unparseable_completions": parse_failures,
        "bradley_terry_by_unit": {
            unit: {arm: _interval(i) for arm, i in fit.strengths.items()}
            for unit, fit in per_unit_fit.items()
        },
        "bradley_terry_pooled": {arm: _interval(i) for arm, i in pooled_fit.strengths.items()},
        "factorial_pooled": W.factorial_effects(pooled_fit.strengths),
        "bradley_terry_consistent_only_by_unit": {
            unit: {arm: _interval(i) for arm, i in fit.items()} for unit, fit in per_unit_bt.items()
        },
        "bradley_terry_consistent_only_pooled": {arm: _interval(i) for arm, i in pooled_bt.items()},
    }
    (work / "analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    lines = ["unit\tarm\tbt_point\tbt_lo\tbt_hi\texcludes_zero"]
    for unit, fit in per_unit_fit.items():
        for arm, i in fit.strengths.items():
            lines.append(
                f"{unit}\t{arm}\t{i.point:.4f}\t{i.lo:.4f}\t{i.hi:.4f}\t{int(i.excludes_zero)}"
            )
    (work / "bradley_terry.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote %s", work)


def _print(checks, per_unit_fit, pooled_fit, position, per_unit_bt) -> None:
    """The tables a reader actually needs."""
    per_unit_strengths = {unit: fit.strengths for unit, fit in per_unit_fit.items()}
    print("\n=== resolution floor (status quo judged against its own replicate) ===")
    print("    (log-odds; an effect inside +/- half-width is one framing judged twice)")
    for unit in (*UNITS, "pooled"):
        source = pooled_fit.strengths if unit == "pooled" else per_unit_strengths.get(unit, {})
        i = source.get(REPLICATE)
        if i is not None:
            print(
                f"  {unit:10s} {i.point:+6.2f} [{i.lo:+5.2f}, {i.hi:+5.2f}]   +/-{i.half_width:.2f}"
            )
    print("  This should straddle zero — it is the same framing on both sides.")

    print("\n=== position effect (slot A's advantage, fitted beside the arms) ===")
    for unit in (*UNITS, "pooled"):
        entry = position.get(unit) or {}
        delta = entry.get("delta")
        if delta is None:
            continue
        print(
            f"  {unit:10s} slot A wins {entry['slot_a_win_rate']:6.1%}   "
            f"delta {delta['point']:+5.2f} [{delta['lo']:+5.2f}, {delta['hi']:+5.2f}]"
        )

    print("\n=== judge reliability (pairs decided both ways, by order) ===")
    for unit in UNITS:
        c = checks.get(unit)
        if c:
            print(
                f"  {unit:10s} {c.inconsistency_rate:6.1%} inconsistent "
                f"({c.consistent} consistent, {c.inconsistent} inconsistent of which "
                f"{c.inconsistent_slot_a} slot-A both times, {c.unparseable} unparsed)"
            )
    print("  Inconsistent pairs stay in the order-aware fit; per-arm counts are in analysis.json.")

    print("\n=== Bradley-Terry strength vs the status quo, per unit (order-aware) ===")
    print("    (log-odds; * = bootstrap CI excludes zero)")
    _print_table(per_unit_strengths)

    print("\n=== consistent-only Bradley-Terry (earlier rounds' readout; not the headline) ===")
    _print_table(per_unit_bt)

    print("\n=== factorial (pooled; grounding and hint main effects) ===")
    eff = W.factorial_effects(pooled_fit.strengths)
    for axis, values in eff.items():
        print(f"  {axis}:")
        for level, value in sorted(values.items(), key=lambda kv: -kv[1]):
            print(f"    {level:12s} {value:+.3f}")
    print(
        "\n  Pooled figures are for orientation only. Per-unit is the reading that "
        "counts:\n  `tags` carries 24 tags in S against 3 in D, so a pooled win can "
        "be one category."
    )
    print(f"\n  (synthesis unit = {SYNTHESIS_UNIT!r}, judged pairwise)")


def _print_table(per_unit: dict) -> None:
    """One arm-by-unit table of strengths."""
    header = f"  {'arm':20s}" + "".join(f"{u:>12s}" for u in UNITS)
    print(header)
    for arm in (*ARMS, REPLICATE):
        if arm == BASELINE:
            continue
        row = f"  {arm:20s}"
        for unit in UNITS:
            i = per_unit.get(unit, {}).get(arm)
            row += (
                f"{'':>12s}"
                if i is None
                else f"{i.point:+8.2f}{'*' if i.excludes_zero else ' '}   "
            )
        print(row)


if __name__ == "__main__":
    raise SystemExit(main())
