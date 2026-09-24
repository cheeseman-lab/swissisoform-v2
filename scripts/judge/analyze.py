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
    CATEGORY_LETTERS,
    DEFAULT_CORPUS,
    REPLICATE,
    SYNTHESIS_UNIT,
    UNITS,
)
from swissisoform.judge import prompts as PR  # noqa: E402
from swissisoform.judge import rubrics as RB  # noqa: E402
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

    scores, forward, parse_failures = _parse(results_path)
    logger.info(
        "%d absolute score(s), %d pairwise call(s), %d unparseable",
        len(scores),
        len(forward),
        parse_failures,
    )

    comparisons, checks = W.resolve_orders(forward)
    logger.info("%d order-consistent comparison(s)", len(comparisons))

    centered = W.center_scores(scores)
    per_unit_bt = {
        unit: W.cluster_bootstrap_bt([c for c in comparisons if c.unit == unit], n=args.bootstrap)
        for unit in UNITS
    }
    pooled_bt = W.cluster_bootstrap_bt(comparisons, n=args.bootstrap)

    _write(work, checks, centered, per_unit_bt, pooled_bt, parse_failures)
    _print(checks, centered, per_unit_bt, pooled_bt)
    return 0


def _parse(path: Path):
    """``(scores, forward, n_unparseable)`` from a results file."""
    scores: list[W.Score] = []
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
            if parts[0] == "abs":
                _, slug, unit, arm, rubric = parts
                try:
                    score, _ = PR.parse_score(completion)
                except PR.ParseError:
                    failures += 1
                    continue
                scores.append(W.Score(slug=slug, unit=unit, arm=arm, rubric=rubric, score=score))
            elif parts[0] == "pw":
                _, slug, unit, arm_a, arm_b, _order = parts
                winner = _winner(row, completion, arm_a, arm_b)
                if winner is None:
                    failures += 1
                forward[(slug, unit, arm_a, arm_b)] = winner
    return scores, forward, failures


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


def _write(work, checks, centered, per_unit_bt, pooled_bt, parse_failures) -> None:
    """Persist everything as JSON, plus a TSV of the headline table."""
    payload = {
        "resolution_floor": _resolution_floor(per_unit_bt, pooled_bt),
        "judge_reliability": {
            unit: {
                "consistent": c.consistent,
                "inconsistent": c.inconsistent,
                "unparseable": c.unparseable,
                "inconsistency_rate": round(c.inconsistency_rate, 4),
            }
            for unit, c in checks.items()
        },
        "n_unparseable_completions": parse_failures,
        "centered_absolute": {
            f"{arm}|{unit}|{rubric}": round(value, 4)
            for (arm, unit, rubric), value in centered.items()
        },
        "bradley_terry_by_unit": {
            unit: {
                arm: {"point": round(i.point, 4), "lo": round(i.lo, 4), "hi": round(i.hi, 4)}
                for arm, i in fit.items()
            }
            for unit, fit in per_unit_bt.items()
        },
        "bradley_terry_pooled": {
            arm: {"point": round(i.point, 4), "lo": round(i.lo, 4), "hi": round(i.hi, 4)}
            for arm, i in pooled_bt.items()
        },
        "factorial_pooled": W.factorial_effects(pooled_bt),
    }
    (work / "analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    lines = ["unit\tarm\tbt_point\tbt_lo\tbt_hi\texcludes_zero"]
    for unit, fit in per_unit_bt.items():
        for arm, i in fit.items():
            lines.append(
                f"{unit}\t{arm}\t{i.point:.4f}\t{i.lo:.4f}\t{i.hi:.4f}\t{int(i.excludes_zero)}"
            )
    (work / "bradley_terry.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote %s", work)


def _print(
    checks,
    centered,
    per_unit_bt,
    pooled_bt,
    pa_per_unit=None,
    pa_pooled=None,
    pa_delta=None,
    pa_delta_per_unit=None,
) -> None:
    """The tables a reader actually needs."""
    print("\n=== resolution floor (status quo judged against its own replicate) ===")
    print("    (log-odds; an effect inside +/- half-width is one framing judged twice)")
    for unit in (*UNITS, "pooled"):
        i = (pooled_bt if unit == "pooled" else per_unit_bt.get(unit, {})).get(REPLICATE)
        if i is not None:
            print(
                f"  {unit:10s} {i.point:+6.2f} [{i.lo:+5.2f}, {i.hi:+5.2f}]"
                f"   +/-{i.half_width:.2f}"
            )
    print("  This should straddle zero — it is the same framing on both sides.")

    print("\n=== judge reliability (pairs decided both ways, by order) ===")
    for unit in UNITS:
        c = checks.get(unit)
        if c:
            print(
                f"  {unit:10s} {c.inconsistency_rate:6.1%} inconsistent "
                f"({c.consistent} kept, {c.inconsistent} dropped, {c.unparseable} unparsed)"
            )

    print("\n=== Bradley-Terry strength vs the status quo, per unit ===")
    print("    (log-odds; * = bootstrap CI excludes zero)")
    header = f"  {'arm':20s}" + "".join(f"{u:>12s}" for u in UNITS)
    print(header)
    for arm in (*ARMS, REPLICATE):
        if arm == BASELINE:
            continue
        row = f"  {arm:20s}"
        for unit in UNITS:
            fit = per_unit_bt.get(unit, {})
            i = fit.get(arm)
            row += (
                f"{'':>12s}"
                if i is None
                else f"{i.point:+8.2f}{'*' if i.excludes_zero else ' '}   "
            )
        print(row)

    print("\n=== centered absolute scores (advantage over the field, per cell) ===")
    rubric_ids = list(RB.CATEGORY_IDS)
    print(f"  {'arm':20s}" + "".join(f"{r.split('_')[0]:>9s}" for r in rubric_ids))
    for arm in (*ARMS, REPLICATE):
        row = f"  {arm:20s}"
        for rubric in rubric_ids:
            vals = [
                centered[(arm, unit, rubric)]
                for unit in CATEGORY_LETTERS
                if (arm, unit, rubric) in centered
            ]
            row += f"{sum(vals) / len(vals):+9.3f}" if vals else f"{'':>9s}"
        print(row)

    # The replicate's own numbers now head the output as the resolution floor;
    # printing them twice invited reading them as two separate checks.

    print("\n=== factorial (pooled; grounding and hint main effects) ===")
    eff = W.factorial_effects(pooled_bt)
    for axis, values in eff.items():
        print(f"  {axis}:")
        for level, value in sorted(values.items(), key=lambda kv: -kv[1]):
            print(f"    {level:12s} {value:+.3f}")
    print(
        "\n  Pooled figures are for orientation only. Per-unit is the reading that "
        "counts:\n  `tags` carries 24 tags in S against 3 in D, so a pooled win can "
        "be one category."
    )
    print(f"\n  (synthesis unit = {SYNTHESIS_UNIT!r}, judged on its own 3 rubrics)")


if __name__ == "__main__":
    raise SystemExit(main())
