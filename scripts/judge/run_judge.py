"""Score the request file with Prometheus, resumably.

Reads ``requests.jsonl``, appends to ``results.jsonl``, and skips ids already
there, so a preempted run restarts where it stopped.

Verification gates live here rather than in a separate script, because the
expensive thing must not start until they pass:

* ``--check-context`` counts every prompt with the real tokenizer and refuses to
  run if any leaves no room for the feedback it has to generate. The builder's
  4-chars/token estimate is a screen, not an assertion.
* ``--self-consistency`` feeds a response against itself. The judge must split
  those ~50/50; a systematic winner means the harness leaks position or identity
  and every number is worthless.
* ``--sanity-anchor`` judges the anchor pairs, which differ only in whether they
  relate the measurements or list them. The rubric must prefer the terse read.

Usage:
    # gates only, no scoring
    python scripts/judge/run_judge.py --check-context
    python scripts/judge/run_judge.py --self-consistency --limit 40
    python scripts/judge/run_judge.py --sanity-anchor

    # the run
    python scripts/judge/run_judge.py --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from swissisoform.judge import DEFAULT_CORPUS  # noqa: E402
from swissisoform.judge import prompts as PR  # noqa: E402
from swissisoform.judge import rubrics as RB  # noqa: E402
from swissisoform.judge.serve import (  # noqa: E402
    MAX_MODEL_LEN,
    Judge,
    Request,
    completed_ids,
    fits_context,
    order_by_cell,
    read_requests,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("judge.run")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI options."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--corpus", default=DEFAULT_CORPUS)
    p.add_argument("--dir", type=Path, default=None, help="Where requests.jsonl lives")
    p.add_argument("--model", type=Path, default=None, help="Override the weights path")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    p.add_argument(
        "--quantization",
        default=None,
        help=(
            "vLLM quantization method, e.g. compressed-tensors. Omit for bf16, "
            "which is the reference configuration and needs no licence; a 4-bit "
            "copy does (see --compare-backends)."
        ),
    )
    p.add_argument(
        "--compare-backends",
        type=Path,
        default=None,
        help=(
            "Path to a backend_<cell>.json written by the other precision; "
            "scores one cell and diffs the pairwise ordering"
        ),
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help=(
            "Override the context window. The full run needs the default 32,768 "
            "(worst prompt 30,517), but the gates use prompts of ~5k, and bf16 on "
            "2x A6000 leaves no room for KV cache at 32k -- vLLM fails outright "
            "with 'No available memory for the cache blocks'. Lowering this is "
            "what lets a gate run on A6000s while the full run waits for A100s."
        ),
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=None, help="First N requests")
    p.add_argument(
        "--check-context",
        action="store_true",
        help="Tokenize every prompt and exit; no scoring",
    )
    p.add_argument(
        "--self-consistency",
        action="store_true",
        help="Judge responses against themselves and exit; must be ~50/50",
    )
    p.add_argument(
        "--sanity-anchor",
        action="store_true",
        help=(
            "Judge the twelve anchor pairs and exit. The rubric must prefer the "
            "terse relational read over the fluent inventory in at least 8 of 12, "
            "in both presentation orders"
        ),
    )
    p.add_argument("--cell", default=None, help="Restrict a gate to one slug|unit")
    p.add_argument("--force", action="store_true", help="Ignore existing results")
    return p.parse_args(argv)


def _build_judge(args: argparse.Namespace) -> Judge:
    """The served model, configured from the CLI."""
    kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if args.max_model_len:
        kwargs["max_model_len"] = args.max_model_len
    if args.quantization:
        kwargs["quantization"] = args.quantization
    return Judge(**kwargs)


def main(argv: list[str] | None = None) -> int:
    """Run the gates, or the scoring."""
    args = parse_args(argv)
    work = args.dir or (ROOT / "data" / "output" / "judge" / args.corpus)

    if args.sanity_anchor:
        # Self-contained: the anchor pairs carry their own instructions, so this
        # skips parsing a 1.3 GB request file it would not read.
        return _sanity_anchor(_build_judge(args), work)

    requests_path = work / "requests.jsonl"
    if not requests_path.exists():
        raise SystemExit(
            f"no requests at {requests_path}. Build them first:\n"
            f"  python scripts/judge/build_requests.py --corpus {args.corpus}"
        )

    requests = list(read_requests(requests_path))
    logger.info("%d request(s) in %s", len(requests), requests_path)

    judge = _build_judge(args)

    if args.check_context:
        return _check_context(judge, requests, work)
    if args.self_consistency:
        return _self_consistency(judge, requests, work, limit=args.limit or 40)
    if args.compare_backends:
        return _compare_backends(judge, requests, work, args.compare_backends, args.cell)

    results_path = work / "results.jsonl"
    done = set() if args.force else completed_ids(results_path)
    todo = [r for r in order_by_cell(requests) if r.id not in done]
    if args.limit:
        todo = todo[: args.limit]
    logger.info("%d done, %d to run", len(done), len(todo))

    mode = "w" if args.force else "a"
    with results_path.open(mode, encoding="utf-8") as sink:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start : start + args.batch_size]
            for result in judge.run(batch):
                sink.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
            sink.flush()
            logger.info(
                "%d/%d (%.1f%%)",
                min(start + args.batch_size, len(todo)),
                len(todo),
                100 * min(start + args.batch_size, len(todo)) / max(len(todo), 1),
            )
    logger.info("wrote %s", results_path)
    return 0


def _check_context(judge: Judge, requests: list[Request], work: Path) -> int:
    """Tokenize every prompt; non-zero exit if any will not fit."""
    over: list[dict] = []
    longest = 0
    for request in requests:
        tokens = judge.token_count(request.prompt)
        longest = max(longest, tokens)
        if not fits_context(tokens):
            over.append(
                {
                    "id": request.id,
                    "tokens": tokens,
                    "slug": request.slug,
                    "unit": request.unit,
                }
            )
    (work / "context_check.json").write_text(
        json.dumps(
            {
                "n_requests": len(requests),
                "longest_prompt_tokens": longest,
                "max_model_len": MAX_MODEL_LEN,
                "n_over": len(over),
                "over": over[:50],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"\n{len(requests):,} prompts, longest {longest:,} tokens (ctx {MAX_MODEL_LEN:,})")
    if over:
        print(f"!! {len(over)} prompt(s) do not fit — worst {over[0]['tokens']:,}")
        for entry in over[:5]:
            print(f"   {entry['unit']:9s} {entry['slug'][:40]} {entry['tokens']:,}")
        return 1
    print("all prompts fit with room for the feedback")
    return 0


# Below this many readable calls the gate reports INCONCLUSIVE instead of a
# verdict: the run it authorises is 31,950 calls, so a mean over a handful is not
# evidence either way.
_MIN_GATE_CALLS = 20


def _self_consistency(judge: Judge, requests: list[Request], work: Path, *, limit: int) -> int:
    """Judge a response against itself; a systematic winner is a harness leak.

    With identical text in both slots there is nothing to choose between, so the
    only thing a preference can reflect is position. Prometheus 2 is known to have
    some; this measures how much, on this corpus, before any of it is attributed
    to an arm.
    """
    pairwise = [r for r in requests if r.kind == "pairwise" and r.order == 0][:limit]
    if not pairwise:
        print("no pairwise requests to check")
        return 0

    probes: list[Request] = []
    for request in pairwise:
        # Rebuild with response A duplicated into both slots.
        body = request.prompt
        marker_a = "###Response A:\n"
        try:
            a_text = body.split(marker_a, 1)[1].split("\n\n###Response B:", 1)[0]
        except IndexError:  # pragma: no cover - template is fixed
            continue
        rebuilt = PR.chat(
            PR.relative_prompt(
                instruction=body.split("###Instruction:\n", 1)[1].split("\n\n###Response A:", 1)[0],
                response_a=a_text,
                response_b=a_text,
                rubric=RB.pairwise().criterion,
            )
        )
        probes.append(
            Request(
                id=f"self|{request.slug}|{request.unit}|{request.arm_a}",
                kind="pairwise",
                slug=request.slug,
                unit=request.unit,
                rubric=RB.PAIRWISE_ID,
                prompt=rebuilt,
                arm_a=request.arm_a,
                arm_b=request.arm_a,
            )
        )

    picks = {"A": 0, "B": 0, "unparsed": 0}
    probabilities: list[float] = []
    no_logprob = 0
    # Every probe is written out whole. The aggregates alone cannot say *why* a
    # call yielded no probability, so a bad reader costs one 8-minute GPU job per
    # hypothesis; with this, the second hypothesis onward is free and offline.
    raw_path = work / "self_consistency_raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as raw:
        for result in judge.run(probes):
            # The probability is the number that matters; the parsed letter is kept
            # alongside only to show how the two channels compare on the same calls.
            prob = PR.preference(result.token_logprobs, result.completion, result.forced_logprobs)
            if prob is None:
                no_logprob += 1
            else:
                probabilities.append(prob)
            try:
                winner, _ = PR.parse_choice(result.completion)
                picks[winner] += 1
                parsed: str | None = winner
            except PR.ParseError:
                picks["unparsed"] += 1
                parsed = None
            raw.write(
                json.dumps(
                    {
                        "id": result.id,
                        "completion": result.completion,
                        "parsed": parsed,
                        "probability": prob,
                        "emitted_text": PR.emitted_text(result.token_logprobs),
                        "forced": bool(result.forced_logprobs),
                        "token_logprobs": result.token_logprobs,
                        "forced_logprobs": result.forced_logprobs,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    decided = picks["A"] + picks["B"]
    share_a = picks["A"] / decided if decided else 0.0
    mean_p = sum(probabilities) / len(probabilities) if probabilities else None
    (work / "self_consistency.json").write_text(
        json.dumps(
            {
                **picks,
                "n": len(probes),
                "share_a": round(share_a, 4),
                "mean_p_a": None if mean_p is None else round(mean_p, 4),
                "n_with_logprobs": len(probabilities),
                "n_without_logprobs": no_logprob,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nself-comparison on {len(probes)} identical pairs:")
    print("  Both slots hold the SAME text, so there is nothing to choose between.")
    print("  Any preference here is purely positional.\n")
    print(
        "  hard votes:  A {} / B {} / unparsed {}  -> share A {:.1%}".format(
            picks["A"], picks["B"], picks["unparsed"], share_a
        )
    )
    if mean_p is None:
        print(f"  logprobs:    unavailable on all {no_logprob} call(s)")
        print("!! no verdict position was readable; logprob scoring cannot run")
        print(f"   inspect {work / 'self_consistency_raw.jsonl'}")
        return 1

    print(
        f"  logprobs:    mean P(A) = {mean_p:.3f} over {len(probabilities)} call(s)"
        + (f", {no_logprob} without" if no_logprob else "")
    )
    print("\n  THE GATE: mean P(A) must be near 0.500.")
    if len(probabilities) < _MIN_GATE_CALLS:
        # A mean over a handful of calls is not a measurement, and reporting one
        # as PASS/FAIL is how a reader bet 31,950 calls on n=1.
        print(
            f"  INCONCLUSIVE -- only {len(probabilities)} of {len(probes)} call(s) "
            f"yielded a probability ({_MIN_GATE_CALLS} needed)."
        )
        print("  The reader could not locate the verdict, so this says nothing about")
        print(f"  position bias. Inspect {work / 'self_consistency_raw.jsonl'}")
        return 1
    if abs(mean_p - 0.5) <= 0.15:
        print(f"  PASS ({mean_p:.3f}) -- averaging the two orders cancels the offset,")
        print("  so the full re-run can proceed.")
        return 0
    print(f"  FAIL ({mean_p:.3f}) -- the preference sits in the first sampled token")
    print("  regardless of content, so logprobs measure the bias more precisely")
    print("  rather than removing it. Use the fixed-anchor design instead of")
    print("  re-running 31,950 calls into the same wall.")
    return 1


# Each anchor pair states the same conclusion from the same numbers, one relating
# the measurements and one listing them, so support is equal by construction and
# only economy can decide. Self-contained, carrying the real payload schema in
# miniature, so the probe looks like the task.
PAIRWISE_ANCHOR_DIR = RB.PROMPTS_DIR / "pairwise_anchors"

# A judge with no preference that always picks slot A wins nothing, and one
# picking at random wins both orders a quarter of the time -- so the null is well
# under half and the mark sits high.
PAIRWISE_ANCHOR_PASS = 8


def load_pairwise_anchors(directory: Path | None = None) -> list[dict[str, Any]]:
    """The hand-written terse/verbose pairs, ordered by filename."""
    root = directory or PAIRWISE_ANCHOR_DIR
    pairs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("*.json"))]
    if not pairs:
        raise SystemExit(f"no pairwise anchor pairs in {root}")
    return pairs


def _sanity_anchor(judge: Judge, work: Path) -> int:
    """Judge the anchor pairs and report whether the rubric prefers the terse read.

    Writes ``sanity_anchor.json`` and returns non-zero when the rubric misses the
    pass mark, so the gate can stop a round before it starts.
    """
    pairwise = _pairwise_anchor(judge)

    (work / "sanity_anchor.json").write_text(
        json.dumps(pairwise, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"\npairwise anchors ({pairwise['n_pairs']} terse/verbose pairs, each judged in both "
        f"orders; a pair counts only when the two orders agree):"
    )
    for pair_id, outcome in pairwise["per_pair"].items():
        a, b = pairwise["orders"][pair_id]
        print(f"  {pair_id:22s} {outcome:6s}  (terse as A -> {a}, terse as B -> {b})")
    print(
        f"  {pairwise['wins']} win / {pairwise['losses']} loss / {pairwise['splits']} split "
        f"on position / {pairwise['no_verdict']} no verdict — pass at {pairwise['pass_mark']}"
    )
    if not pairwise["ok"]:
        print("  !! the rubric does not prefer the terse reads; the fluency hole is open")
        return 1
    return 0


def _pairwise_anchor(judge: Judge) -> dict[str, Any]:
    """Score every terse/verbose pair in both orders; report the win rate.

    A pair counts only when it agrees with itself across the two presentation
    orders -- the same rule the analysis applies to the corpus, and the only way
    to keep position bias out of the count. Disagreement is reported as ``split``
    rather than folded into either side, so a judge deciding on slot rather than
    content stays visible.
    """
    pairs = load_pairwise_anchors()
    rubric = RB.pairwise().criterion
    probes: list[Request] = []
    for pair in pairs:
        for order, (a, b) in enumerate(
            ((pair["terse"], pair["verbose"]), (pair["verbose"], pair["terse"]))
        ):
            probes.append(
                Request(
                    id=f"anchor|pairwise|{pair['id']}|{order}",
                    slug=pair["id"],
                    unit=pair["instruction"]["category"],
                    rubric=RB.PAIRWISE_ID,
                    prompt=PR.chat(
                        PR.relative_prompt(
                            instruction=json.dumps(pair["instruction"], indent=2),
                            response_a=a,
                            response_b=b,
                            rubric=rubric,
                        )
                    ),
                    arm_a="terse" if order == 0 else "verbose",
                    arm_b="verbose" if order == 0 else "terse",
                    order=order,
                )
            )

    picked: dict[str, list[str | None]] = {p["id"]: [None, None] for p in pairs}
    for probe, result in zip(probes, judge.run(probes)):
        # The same entry point the analysis uses: a confident completion that never
        # wrote the marker still carries its verdict in the logprobs.
        prob = PR.preference(
            result.token_logprobs, result.completion, result.forced_logprobs
        )
        if prob is None:
            try:
                letter, _ = PR.parse_choice(result.completion)
            except PR.ParseError:
                continue
            prob = 1.0 if letter == "A" else 0.0
        picked[probe.slug][probe.order] = probe.arm_a if prob > 0.5 else probe.arm_b

    per_pair: dict[str, str] = {}
    for pair in pairs:
        got = picked[pair["id"]]
        if None in got:
            # Kept apart from `split`: an abstention is not the two orders
            # disagreeing.
            per_pair[pair["id"]] = "no_verdict"
        elif got == ["terse", "terse"]:
            per_pair[pair["id"]] = "win"
        elif got == ["verbose", "verbose"]:
            per_pair[pair["id"]] = "loss"
        else:
            per_pair[pair["id"]] = "split"

    tally = {k: sum(1 for v in per_pair.values() if v == k) for k in
             ("win", "loss", "split", "no_verdict")}
    return {
        "n_pairs": len(pairs),
        "wins": tally["win"],
        "losses": tally["loss"],
        "splits": tally["split"],
        "no_verdict": tally["no_verdict"],
        "pass_mark": PAIRWISE_ANCHOR_PASS,
        "per_pair": per_pair,
        "orders": {k: [v[0] or "no_verdict", v[1] or "no_verdict"] for k, v in picked.items()},
        "ok": tally["win"] >= PAIRWISE_ANCHOR_PASS,
    }


def _compare_backends(
    judge: Judge,
    requests: list[Request],
    work: Path,
    other: Path,
    cell: str | None,
) -> int:
    """Score one cell and diff the pairwise ordering against the other precision.

    This is what licenses serving a 4-bit judge. 4-bit is a lossy approximation of
    the model doing the scoring, so the question is not whether individual scores
    shift but whether the *ranking* survives -- the ranking is all the analysis
    reads. Run once under each precision pointing at the other's output file.
    """
    if cell:
        slug, _, unit = cell.partition("|")
    else:
        first = next((r for r in requests if r.kind == "pairwise"), None)
        if first is None:
            print("no pairwise requests to compare")
            return 0
        slug, unit = first.slug, first.unit

    subset = [r for r in requests if r.slug == slug and r.unit == unit and r.kind == "pairwise"]
    if not subset:
        print(f"no pairwise requests for cell ({slug}, {unit})")
        return 1
    print(f"comparing cell ({slug}, {unit}): {len(subset)} pairwise call(s)")

    mine: dict[str, str | None] = {}
    for request, result in zip(subset, judge.run(subset)):
        try:
            mine[request.id], _ = PR.parse_choice(result.completion)
        except PR.ParseError:
            mine[request.id] = None

    out_path = work / f"backend_{slug}_{unit}.json"
    out_path.write_text(json.dumps(mine, indent=2, sort_keys=True), encoding="utf-8")

    if not other.exists() or other.resolve() == out_path.resolve():
        print(f"\nwrote {out_path}")
        print("Now run the other precision with --compare-backends pointing here.")
        return 0

    theirs = json.loads(other.read_text(encoding="utf-8"))
    shared = sorted(set(mine) & set(theirs))
    if not shared:
        print(f"!! no overlapping request ids with {other}")
        return 1
    agree = sum(mine[i] == theirs[i] for i in shared)
    rate = agree / len(shared)
    print(f"\n{agree}/{len(shared)} identical choices ({rate:.1%})")
    if rate < 0.9:
        print("!! the two precisions disagree on the ordering. The quantized judge")
        print("   is not a substitute for this corpus; use the bf16 run.")
        return 1
    print("ordering agrees; the quantized judge is usable for this corpus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
