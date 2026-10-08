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
    TOOL_UNITS,
    UNITS,
)
from swissisoform.judge import prompts as PR  # noqa: E402
from swissisoform.judge import provenance as PV  # noqa: E402
from swissisoform.judge import weigh as W  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("judge.analyze")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI options."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--corpus", default=DEFAULT_CORPUS)
    p.add_argument("--dir", type=Path, default=None)
    p.add_argument("--bootstrap", type=int, default=W.N_BOOTSTRAP)
    p.add_argument(
        "--allow-provenance-mismatch",
        action="store_true",
        help=(
            "Fit even when results cannot all be tied to the current request build. "
            "The mismatch counts are still recorded in analysis.json."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Parse results, weigh them, write tables."""
    args = parse_args(argv)
    work = args.dir or (ROOT / "data" / "output" / "judge" / args.corpus)
    results_path = work / "results.jsonl"
    if not results_path.exists():
        raise SystemExit(f"no results at {results_path}; run scripts/judge/run_judge.py")

    rows = _load_rows(results_path)
    provenance = _provenance(work, rows, args)
    forward, parse_failures = _parse(rows)
    logger.info("%d pairwise call(s), %d unparseable", len(forward), parse_failures)

    comparisons, checks = W.resolve_orders(forward)
    logger.info("%d order-consistent comparison(s)", len(comparisons))

    # Primary: every parsed call, both orders, with the slot-A effect fitted.
    calls = W.calls_from_forward(forward, _lengths(work))
    per_unit_fit = {
        unit: W.cluster_bootstrap_fit([c for c in calls if c.unit == unit], n=args.bootstrap)
        for unit in UNITS
    }
    pooled_fit = W.cluster_bootstrap_fit(calls, n=args.bootstrap)
    position = _position_report(calls, per_unit_fit, pooled_fit)

    # Beside it, not instead: the same fit with a log-length covariate.
    per_unit_len = {
        unit: W.cluster_bootstrap_fit(
            [c for c in calls if c.unit == unit], length=True, n=args.bootstrap
        )
        for unit in UNITS
    }
    pooled_len = W.cluster_bootstrap_fit(calls, length=True, n=args.bootstrap)
    length = _length_report(calls, per_unit_len, pooled_len)

    # Secondary: the consistent-only fit earlier rounds published.
    per_unit_bt = {
        unit: W.cluster_bootstrap_bt([c for c in comparisons if c.unit == unit], n=args.bootstrap)
        for unit in UNITS
    }
    pooled_bt = W.cluster_bootstrap_bt(comparisons, n=args.bootstrap)

    _write(
        work,
        checks,
        per_unit_fit,
        pooled_fit,
        position,
        per_unit_bt,
        pooled_bt,
        parse_failures,
        provenance,
        per_unit_len,
        pooled_len,
        length,
    )
    _print(checks, per_unit_fit, pooled_fit, position, per_unit_bt, per_unit_len, length)
    blind = {u: v for u, v in _tool_blind(work).items() if v}
    if blind:
        print(
            "\n  !! tool-blind: "
            + ", ".join(f"{u} {v:.0%}" for u, v in blind.items())
            + " of calls were judged without the tool results the arms read"
            "\n     (rebuild with build_requests.py --tool-results-chars N to show them)."
        )
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


def _length_report(calls, per_unit_len, pooled_len) -> dict:
    """Longer-wins rates, median length per arm, and the fitted length log-odds."""
    out = {}
    for unit, fit in (*per_unit_len.items(), ("pooled", pooled_len)):
        subset = calls if unit == "pooled" else [c for c in calls if c.unit == unit]
        by_arm: dict[str, list[int]] = {}
        for c in subset:
            if c.has_lengths:
                by_arm.setdefault(c.arm_a, []).append(c.len_a)
                by_arm.setdefault(c.arm_b, []).append(c.len_b)
        out[unit] = {
            **W.length_preference(subset),
            "median_chars_by_arm": {
                arm: sorted(v)[len(v) // 2] for arm, v in sorted(by_arm.items())
            },
            "beta": None if fit.length is None else _interval(fit.length),
            "n_calls_fitted": fit.n_calls,
        }
    return out


def _tool_blind(work: Path) -> dict[str, float]:
    """Share of each tool unit's requests judged without the arms' tool results.

    A pre-provenance build has no flag, and every M/P call in it was tool-blind,
    so an empty index counts as fully blind rather than as unknown.
    """
    rows = [r for r in _load_index(work).values() if r.get("unit") in TOOL_UNITS]
    if not rows:
        return dict.fromkeys(TOOL_UNITS, 1.0)
    out = {}
    for unit in TOOL_UNITS:
        subset = [r for r in rows if r["unit"] == unit]
        if subset:
            out[unit] = round(sum(bool(r.get("tool_blind")) for r in subset) / len(subset), 4)
    return out


def _lengths(work: Path) -> dict[tuple[str, str, str, str], tuple[int, int]]:
    """``{(slug, unit, arm_a, arm_b): (len_a, len_b)}`` from the request index."""
    out = {}
    for row in _load_index(work).values():
        if row.get("len_a") and row.get("len_b"):
            key = (row["slug"], row["unit"], row["arm_a"], row["arm_b"])
            out[key] = (row["len_a"], row["len_b"])
    return out


def _load_rows(path: Path) -> list[dict]:
    """Every results row, in file order."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_index(work: Path) -> dict[str, dict]:
    """``{request_id: request-without-prompt}``, or empty when the build has none."""
    path = work / PV.INDEX_NAME
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        rows = (json.loads(line) for line in handle if line.strip())
        return {row["id"]: row for row in rows}


def _provenance(work: Path, rows: list[dict], args: argparse.Namespace) -> dict:
    """Tie every result to the current request build, or refuse to fit.

    A results file that resumes across a rebuild ends up holding judgments of two
    different request sets, and nothing downstream can tell them apart: the v3
    fit mixed 14,700 carried-over v2 rows with 10,500 fresh ones. So unless
    ``--allow-provenance-mismatch`` is passed, any row that is not in the current
    build, carries a different (or no) build id, or names different response text
    stops the analysis here. Also reports which judged responses have changed on
    disk since, which does not invalidate the fit but means the arm outputs are
    no longer the text that was scored.
    """
    meta_path = work / "requests_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    build_id = meta.get("build_id") or (meta.get("provenance") or {}).get("build_id") or ""
    index = _load_index(work)
    checks = PV.check_results(rows, index, build_id)
    bad = {k: v for k, v in checks.items() if k != "n_results" and v}
    if not build_id or not index:
        bad["no_build_record"] = 1
    # Arm-vs-reference problems the build was told to tolerate are not tolerated
    # here by default: a build override must be repeated at fit time.
    arm_side = meta.get("provenance") or {}
    if arm_side.get("problems"):
        bad["arm_provenance_problems"] = len(arm_side["problems"])
    elif not arm_side:
        bad["no_arm_provenance"] = 1
    report = {
        "build_id": build_id,
        "reference": arm_side,
        "results": checks,
        "mismatch": bool(bad),
        "allowed_by_override": bool(bad) and args.allow_provenance_mismatch,
        # The corpus this build was made from, as the build recorded it: --dir can
        # point at another corpus's work directory than --corpus names.
        "stale_on_disk": _stale_on_disk(index.values(), arm_side.get("corpus") or args.corpus),
    }
    if bad and not args.allow_provenance_mismatch:
        raise SystemExit(
            f"results in {work} cannot all be tied to request build {build_id or '(none)'}: "
            f"{bad}. Re-run scripts/judge/run_judge.py --force against the current "
            "requests, or pass --allow-provenance-mismatch to fit anyway."
        )
    if bad:
        logger.warning("provenance mismatch allowed by override: %s", bad)
    stale = report["stale_on_disk"]
    if stale.get("n_stale"):
        logger.warning(
            "%d of %d judged response(s) differ on disk now: %s",
            stale["n_stale"],
            stale["n_judged"],
            stale["stale_by_arm"],
        )
    return report


def _stale_on_disk(index, corpus_name: str) -> dict:
    """:func:`provenance.stale_on_disk` against the corpus as it is now, if loadable."""
    index = list(index)
    if not index:
        return {"checked": False, "reason": "no request index"}
    from swissisoform.judge.corpus import CorpusError, load_corpus

    try:
        corpus = load_corpus(corpus_name, require_complete=False)
    except CorpusError as exc:
        return {"checked": False, "reason": str(exc)}
    current = {key: PV.text_sha(out.text) for key, out in corpus.outputs.items()}
    return {"checked": True, **PV.stale_on_disk(index, current)}


def _parse(rows: list[dict]):
    """``(forward, n_unparseable)`` from results rows."""
    forward: dict[tuple[str, str, str, str], str | None] = {}
    failures = 0
    for row in rows:
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
    work,
    checks,
    per_unit_fit,
    pooled_fit,
    position,
    per_unit_bt,
    pooled_bt,
    parse_failures,
    provenance,
    per_unit_len,
    pooled_len,
    length,
) -> None:
    """Persist everything as JSON, plus a TSV of the headline table.

    ``bradley_terry_*`` is the order-aware fit over every parsed call;
    ``bradley_terry_consistent_only_*`` is the older fit over order-agreeing
    pairs, kept for comparison with earlier rounds and never the headline.
    """
    per_unit_strengths = {unit: fit.strengths for unit, fit in per_unit_fit.items()}
    tool_blind = _tool_blind(work)
    payload = {
        "provenance": provenance,
        # Share of each M/P unit's calls judged without the tool results the arms
        # read; any non-zero value means those scores rest on partial evidence.
        "tool_blind": tool_blind,
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
        "length_effect": length,
        "bradley_terry_length_adjusted_by_unit": {
            unit: {arm: _interval(i) for arm, i in fit.strengths.items()}
            for unit, fit in per_unit_len.items()
        },
        "bradley_terry_length_adjusted_pooled": {
            arm: _interval(i) for arm, i in pooled_len.strengths.items()
        },
        "factorial_length_adjusted_pooled": W.factorial_effects(pooled_len.strengths),
        "bradley_terry_consistent_only_by_unit": {
            unit: {arm: _interval(i) for arm, i in fit.items()} for unit, fit in per_unit_bt.items()
        },
        "bradley_terry_consistent_only_pooled": {arm: _interval(i) for arm, i in pooled_bt.items()},
    }
    (work / "analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    lines = ["unit\tarm\tbt_point\tbt_lo\tbt_hi\texcludes_zero\ttool_blind"]
    for unit, fit in per_unit_fit.items():
        blind = int(tool_blind.get(unit, 0.0) > 0)
        for arm, i in fit.strengths.items():
            lines.append(
                f"{unit}\t{arm}\t{i.point:.4f}\t{i.lo:.4f}\t{i.hi:.4f}"
                f"\t{int(i.excludes_zero)}\t{blind}"
            )
    (work / "bradley_terry.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote %s", work)


def _print(checks, per_unit_fit, pooled_fit, position, per_unit_bt, per_unit_len, length) -> None:
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

    print("\n=== length preference (the rubric says length should not matter) ===")
    for unit in (*UNITS, "pooled"):
        entry = length.get(unit) or {}
        beta = entry.get("beta")
        if not entry.get("n") or beta is None:
            continue
        top = entry["by_quartile"][-1]["longer_wins_rate"] if entry["by_quartile"] else None
        print(
            f"  {unit:10s} longer wins {entry['longer_wins_rate']:6.1%} "
            f"(top quartile {top:.1%})   "
            f"beta {beta['point']:+5.2f} [{beta['lo']:+5.2f}, {beta['hi']:+5.2f}] per log-ratio"
        )

    print("\n=== Bradley-Terry, length-adjusted (same fit + beta * log(len_A/len_B)) ===")
    print("    (an arm effect that vanishes here is a length effect until shown otherwise)")
    _print_table({unit: fit.strengths for unit, fit in per_unit_len.items()})

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
