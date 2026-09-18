"""Score every arm output with a reward model instead of a generative judge.

Runs the same 50-isoform corpus the Prometheus arms were judged on, but scores each
response *alone* against its instruction, so position bias is undefined rather than
merely reduced. See ``swissisoform.judge.reward`` for why that matters here.

The instruction and response are lifted out of the existing pairwise prompts rather
than rebuilt: ``requests.jsonl`` already carries the exact ``###Instruction:`` block
the arm was shown and the ``###Response A:`` text it produced, so the reward model
reads byte-identical input to what the judge read. Rebuilding the payload would
risk scoring against a different reference than Prometheus saw, which would make
the two judges incomparable -- the one thing this run exists to test.

Usage:

    python scripts/judge/run_reward.py                      # all 3,150
    python scripts/judge/run_reward.py --limit 20           # smoke test
    python scripts/judge/run_reward.py --check-context      # no GPU, fit check
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from swissisoform.judge import DEFAULT_CORPUS  # noqa: E402
from swissisoform.judge.reward import MODEL_ID, RewardError, RewardScore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("judge.run_reward")

INSTRUCTION_MARKER = "###Instruction:\n"
RESPONSE_MARKER = "###Response A:\n"
RESPONSE_END = "\n\n###Response B:"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI options."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--corpus", default=DEFAULT_CORPUS)
    p.add_argument("--dir", type=Path, default=None)
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--limit", type=int, default=None, help="First N pairs")
    p.add_argument(
        "--check-context",
        action="store_true",
        help="Tokenize every pair and report fit; loads the tokenizer, not the model",
    )
    p.add_argument("--force", action="store_true", help="Ignore existing scores")
    return p.parse_args(argv)


def extract_pairs(requests_path: Path) -> list[tuple[str, str, str, str, str]]:
    """``(slug, unit, arm, instruction, response)`` once per arm output.

    Every arm appears in many pairwise requests (36 pairings per cell), so the first
    sighting wins and the rest are skipped -- the response text is identical across
    them by construction.
    """
    seen: set[tuple[str, str, str]] = set()
    out: list[tuple[str, str, str, str, str]] = []
    with requests_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("kind") != "pairwise":
                continue
            key = (row["slug"], row["unit"], row["arm_a"])
            if key in seen:
                continue
            # partition, not split+index: a template change should yield an empty
            # string and a warning, not an IndexError mid-run.
            rest = row["prompt"].partition(INSTRUCTION_MARKER)[2]
            instruction = rest.partition("\n\n" + RESPONSE_MARKER.strip())[0]
            response = rest.partition(RESPONSE_MARKER)[2].partition(RESPONSE_END)[0]
            if not instruction or not response:
                logger.warning("could not split prompt for %s", row["id"])
                continue
            seen.add(key)
            out.append((*key, instruction, response))
    return out


def main(argv: list[str] | None = None) -> int:
    """Score every arm output and write one JSONL row per score."""
    args = parse_args(argv)
    work = args.dir or (ROOT / "data" / "output" / "judge" / args.corpus)
    requests_path = work / "requests.jsonl"
    if not requests_path.exists():
        raise SystemExit(f"no requests at {requests_path}; run scripts/judge/build_requests.py")

    pairs = extract_pairs(requests_path)
    by_arm = Counter(arm for _, _, arm, _, _ in pairs)
    logger.info("%d arm output(s) over %d arm(s)", len(pairs), len(by_arm))
    for arm, n in sorted(by_arm.items()):
        logger.info("  %-20s %d", arm, n)

    scores_path = work / "reward_scores.jsonl"
    done: set[str] = set()
    if scores_path.exists() and not args.force:
        with scores_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                done.add(f"{row['slug']}|{row['unit']}|{row['arm']}")
        logger.info("%d already scored", len(done))

    todo = [p for p in pairs if f"{p[0]}|{p[1]}|{p[2]}" not in done]
    if args.limit:
        todo = todo[: args.limit]

    from swissisoform.judge.reward import RewardScorer

    if args.check_context:
        return _check_context(args, work, todo)

    scorer = RewardScorer(args.model)
    mode = "w" if args.force else "a"
    oversized = 0
    with scores_path.open(mode, encoding="utf-8") as sink:
        for index, (slug, unit, arm, instruction, response) in enumerate(todo, 1):
            try:
                value = scorer.score(instruction, response)
            except RewardError as exc:
                oversized += 1
                logger.warning("skipping %s|%s|%s: %s", slug, unit, arm, exc)
                continue
            sink.write(
                json.dumps(asdict(RewardScore(slug=slug, unit=unit, arm=arm, score=value))) + "\n"
            )
            if index % 100 == 0:
                sink.flush()
                logger.info("%d/%d (%.1f%%)", index, len(todo), 100 * index / len(todo))
    logger.info("wrote %s (%d oversized skipped)", scores_path, oversized)
    return 0


def _check_context(args: argparse.Namespace, work: Path, todo: list) -> int:
    """Tokenize every pair without loading the model; non-zero if any will not fit."""
    from swissisoform.judge.reward import MAX_LENGTH, RewardScorer

    # Same class, same chat template, no model on the GPU.
    scorer = RewardScorer(args.model, load_model=False)
    lengths: list[tuple[int, str]] = []
    for slug, unit, arm, instruction, response in todo:
        lengths.append((scorer.token_count(instruction, response), f"{slug}|{unit}|{arm}"))
    lengths.sort(reverse=True)
    over = [entry for entry in lengths if entry[0] > MAX_LENGTH]
    (work / "reward_context_check.json").write_text(
        json.dumps(
            {
                "n": len(lengths),
                "max_length": MAX_LENGTH,
                "longest": lengths[0][0] if lengths else 0,
                "n_over": len(over),
                "over": [{"tokens": t, "cell": c} for t, c in over[:50]],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n{len(lengths):,} pair(s), longest {lengths[0][0]:,} tokens (cap {MAX_LENGTH:,})")
    for tokens, cell in lengths[:5]:
        print(f"   {tokens:7,}  {cell}")
    if over:
        print(f"!! {len(over)} pair(s) will not fit")
        return 1
    print("all pairs fit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
