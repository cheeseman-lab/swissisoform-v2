"""Score the request file with Prometheus, resumably.

Reads ``requests.jsonl``, appends to ``results.jsonl``, and skips ids already
there -- 37,800 calls is far too long to lose to one preemption, and the run is
the only part of this pipeline that cannot be redone in seconds.

Verification gates live here rather than in a separate script, because the
expensive thing must not start until they pass:

* ``--check-context`` counts every prompt with the real tokenizer and refuses to
  run if any leaves no room for the feedback it has to generate. The builder's
  4-chars/token estimate is a screen, not an assertion.
* ``--self-consistency`` feeds a response against itself. The judge must split
  those ~50/50; a systematic winner means the harness leaks position or identity
  and every pairwise number is worthless.

Usage:
    # gates only, no scoring
    python scripts/judge/run_judge.py --check-context
    python scripts/judge/run_judge.py --self-consistency --limit 40

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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from swissisoform.judge import DEFAULT_CORPUS, SYNTHESIS_UNIT  # noqa: E402
from swissisoform.judge import prompts as PR  # noqa: E402
from swissisoform.judge import rubrics as RB  # noqa: E402
from swissisoform.judge.serve import (  # noqa: E402
    MAX_MODEL_LEN,
    MAX_NEW_TOKENS,
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
    p.add_argument(
        "--kind",
        choices=("pairwise", "absolute", "both"),
        default="both",
        help=(
            "Which call kind to run. 'pairwise' (25,200 of 31,950) carries the whole "
            "Bradley-Terry ranking; 'absolute' (6,750) carries only the centered "
            "rubric table, and still depends on the [RESULT] 1-5 regex."
        ),
    )
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
        help="Score a deliberately bad verdict; R1/R2 must give it 1-2",
    )
    p.add_argument("--cell", default=None, help="Restrict a gate to one slug|unit")
    p.add_argument("--force", action="store_true", help="Ignore existing results")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the gates, or the scoring."""
    args = parse_args(argv)
    work = args.dir or (ROOT / "data" / "output" / "judge" / args.corpus)
    requests_path = work / "requests.jsonl"
    if not requests_path.exists():
        raise SystemExit(
            f"no requests at {requests_path}. Build them first:\n"
            f"  python scripts/judge/build_requests.py --corpus {args.corpus}"
        )

    requests = list(read_requests(requests_path))
    logger.info("%d request(s) in %s", len(requests), requests_path)

    judge_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if args.max_model_len:
        judge_kwargs["max_model_len"] = args.max_model_len
    if args.quantization:
        judge_kwargs["quantization"] = args.quantization
    judge = Judge(**judge_kwargs)

    if args.check_context:
        return _check_context(judge, requests, work)
    if args.self_consistency:
        return _self_consistency(judge, requests, work, limit=args.limit or 40)
    if args.sanity_anchor:
        return _sanity_anchor(judge, requests, work)
    if args.compare_backends:
        return _compare_backends(judge, requests, work, args.compare_backends, args.cell)

    results_path = work / "results.jsonl"
    done = set() if args.force else completed_ids(results_path)
    todo = [r for r in order_by_cell(requests) if r.id not in done]
    if args.kind != "both":
        # Filtered here rather than in the request file so one requests.jsonl keeps
        # serving both, and a later absolute run appends to the same results file.
        before = len(todo)
        todo = [r for r in todo if r.kind == args.kind]
        logger.info("--kind %s: %d of %d request(s)", args.kind, len(todo), before)
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


# One anchor per axis, each violating only its own rubric and otherwise clean.
#
# The first version was a single verdict that broke all four axes at once, and it
# could not tell us anything: R1 correctly scored it 1, but R3 -- which judges
# proportionality and leaves number-checking to R1 -- scored the same response 4,
# which is R3 complying with its rubric rather than failing. Two other claims were
# aimed at secondary structure and clinical variants, neither of which appears in
# the Conservation cell it was scored against, so R2 never received the violation
# it was meant to catch.
#
# Each anchor below is wrong in exactly one way, so a score above 2 is
# unambiguously that rubric's problem. `needs` says what the cell must look like
# for the violation to be possible at all.
#
# R3 has no anchor because it is no longer scored: it gave its own anchor a 4,
# and calibration has no checkable referent in the evidence for the judge to be
# wrong about. See RETIRED_RUBRICS in rubrics.py.
ANCHORS: dict[str, dict[str, str]] = {
    "R1_evidential_support": {
        "needs": "any",
        "response": (
            "verdict: interesting\n\n"
            "reasoning: The unique region is under strong constraint: mean phyloP "
            "is 88.412 and primate amino-acid identity reaches 99.97%, both far "
            "above the thresholds used here. Those two figures alone put this "
            "isoform among the most constrained in the set, which is why it is "
            "worth following up rather than setting aside."
        ),
        "why": "cites two figures that appear nowhere in the evidence",
    },
    "R2_not_evaluable_discipline": {
        "needs": "unmeasured",
        "response": (
            "verdict: not_interesting\n\n"
            "reasoning: Every line of evidence here comes back negative. The "
            "measurements that were run show no signal, and the remaining fields "
            "are likewise absent -- nothing was detected on any axis. Taken "
            "together the evidence is uniformly negative, so there is no case for "
            "this isoform being functionally distinct from the canonical."
        ),
        "why": "calls fields the evidence marks unmeasured 'absent' and 'negative'",
    },
}


# Synthesis anchors. The synthesis reference is that arm's six category verdicts
# plus the criteria evidence, so `needs: "synthesis"` picks a synthesis cell and
# the violations are aimed at that payload.
SYNTHESIS_ANCHORS: dict[str, dict[str, str]] = {
    "SY1_coherence": {
        "response": (
            "headline: Every category read supports a clear functional change.\n"
            "divergence_hypothesis: All six category verdicts agree that this "
            "isoform diverges functionally from the canonical, with none of them "
            "returning a negative or neutral read, so the picture is consistent "
            "across conservation, detection, localization, mutation landscape, "
            "structure and biophysics alike.\n"
            "function_relevance: Because nothing in the inputs disagrees, the "
            "isoform can be treated as a confirmed functional variant.\n"
            "tags: Domain loss\nconfidence: high"
        ),
        "why": "claims all six verdicts agree when the inputs include neutral/negative reads",
    },
    "SY2_no_new_claims": {
        "response": (
            "headline: A ribosome-profiling-validated interaction switch at the "
            "ADAR N-terminus.\n"
            "divergence_hypothesis: This isoform loses the Z-alpha domain that "
            "mediates binding to STAT3 and to the NF-kB p65 subunit, and published "
            "co-immunoprecipitation work places that contact within residues "
            "140-180. The interferon-stimulated promoter upstream drives the "
            "shorter form specifically in myeloid lineages.\n"
            "function_relevance: Loss of that interaction surface would release "
            "STAT3 inhibition and raise interferon output roughly threefold.\n"
            "tags: Domain loss\nconfidence: high"
        ),
        "why": (
            "imports named interactors, a residue range and a fold-change "
            "found nowhere in the inputs"
        ),
    },
    "SY3_hypothesis_quality": {
        "response": (
            "headline: This isoform may differ functionally from the canonical.\n"
            "divergence_hypothesis: The isoform is different from the canonical "
            "protein and that difference could matter. It may affect the "
            "protein's behaviour in the cell, or it may not. Further work would "
            "be needed to say which.\n"
            "function_relevance: If the difference is real it could be relevant "
            "to the gene's function.\n"
            "tags: \nconfidence: low"
        ),
        "why": "a hypothesis so generic it would fit any isoform in the corpus",
    },
}


def _cell_kind(instruction: str) -> set[str]:
    """What violations this cell's evidence can support.

    Returns a set, always containing ``"any"``. The `str` annotation this had
    first was wrong in a way that would have run: the error path returned the
    bare string ``"any"``, which iterates as three characters and would have
    filed the cell under "a", "n" and "y".
    """
    try:
        payload = json.loads(instruction)
    except json.JSONDecodeError:
        return {"any"}
    orf = str((payload.get("isoform") or {}).get("orf_type") or "").lower()
    kinds = {"any"}
    if orf in ("uorf", "uoorf", "internal_oof", "internal-oof", "3utr_orf", "3utr-orf"):
        kinds.add("separate")
    blob = json.dumps(payload).lower()
    if any(m in blob for m in ("not_evaluable", "no_cache", "not_run", "null")):
        kinds.add("unmeasured")
    return kinds


def _sanity_anchor(judge: Judge, requests: list[Request], work: Path) -> int:
    """Score each axis against a verdict wrong in exactly that axis.

    A rubric that cannot score its own anchor 1-2 is not measuring anything, and
    that is far cheaper to learn here than after the full run.
    """
    # Group candidate cells by what they can test, then give each rubric a cell
    # whose evidence actually supports its violation.
    candidates: dict[str, list[Request]] = {}
    for request in requests:
        if request.kind != "absolute":
            continue
        instruction = request.prompt.split("###The instruction to evaluate:\n", 1)[-1]
        instruction = instruction.split("\n\n###Response to evaluate:", 1)[0]
        for kind in _cell_kind(instruction):
            candidates.setdefault(kind, []).append(request)

    probes: list[Request] = []
    chosen: dict[str, str] = {}
    specs: dict[str, dict[str, str]] = {
        **{k: {**v, "unit": "category"} for k, v in ANCHORS.items()},
        **{k: {**v, "needs": "any", "unit": SYNTHESIS_UNIT} for k, v in SYNTHESIS_ANCHORS.items()},
    }
    for rubric_id, spec in specs.items():
        rubric = RB.by_id(rubric_id)
        want_unit = spec.get("unit")
        pool = candidates.get(spec["needs"]) or candidates.get("any") or []
        if want_unit == SYNTHESIS_UNIT:
            pool = [r for r in pool if r.unit == SYNTHESIS_UNIT]
        else:
            pool = [r for r in pool if r.unit != SYNTHESIS_UNIT]
        if rubric is None or not pool:
            logger.warning("no cell available to test %s (needs %s)", rubric_id, spec["needs"])
            continue
        template = pool[0]
        instruction = template.prompt.split("###The instruction to evaluate:\n", 1)[1]
        instruction = instruction.split("\n\n###Response to evaluate:", 1)[0]
        chosen[rubric_id] = f"{template.slug}|{template.unit}|needs={spec['needs']}"
        probes.append(
            Request(
                id=f"anchor|{rubric_id}",
                kind="absolute",
                slug=template.slug,
                unit=template.unit,
                rubric=rubric_id,
                prompt=PR.chat(
                    PR.absolute_prompt(
                        instruction=instruction,
                        response=spec["response"],
                        rubric=rubric.render(),
                    )
                ),
                arm="__anchor__",
            )
        )

    # Drop probes the served context cannot hold. The synthesis anchors are ~25k
    # tokens, so on 2x A6000 bf16 (which vLLM caps at 15,040) they cannot run at
    # all -- and one oversized probe used to abort the whole gate, taking the
    # category anchors down with it.
    limit = judge.max_model_len - MAX_NEW_TOKENS
    fits, skipped = [], []
    for probe in probes:
        if judge.token_count(probe.prompt) <= limit:
            fits.append(probe)
        else:
            skipped.append(probe.rubric)
    if skipped:
        logger.warning(
            "skipping %d anchor(s) too long for a %d-token context: %s",
            len(skipped),
            judge.max_model_len,
            skipped,
        )

    scores: dict[str, int | None] = {}
    for probe, result in zip(fits, judge.run(fits)):
        try:
            scores[probe.rubric], _ = PR.parse_score(result.completion)
        except PR.ParseError:
            scores[probe.rubric] = None

    (work / "sanity_anchor.json").write_text(
        json.dumps(
            {
                "scores": scores,
                "cells": chosen,
                "violations": {k: v["why"] for k, v in specs.items()},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print("\nper-axis anchors (each wrong in ONE way; want 1-2):")
    failed = []
    for rubric_id, spec in specs.items():
        if rubric_id in skipped:
            print(f"  {rubric_id:28s} -  skipped   (needs more context than served)")
            continue
        score = scores.get(rubric_id)
        verdict = "ok" if score is not None and score <= 2 else "TOO HIGH"
        print(f"  {rubric_id:28s} {score}  {verdict:8s}  ({spec['why']})")
        if score is None or score > 2:
            failed.append(rubric_id)
    if failed:
        print(f"!! {len(failed)} rubric(s) cannot fail their own anchor: {failed}")
        return 1
    return 0


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
