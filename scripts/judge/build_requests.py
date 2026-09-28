"""Turn the arm corpus into a JSONL of judge requests.

Per cell (one isoform, one output unit), every arm present:

  C(9,2) = 36 pairs x 2 presentation orders = 72 calls

Both orders is not optional. Prometheus 2 relative grading has documented position
bias, so a pair is only a verdict when the judge picks the same arm whichever slot
it sits in; the rest are dropped and the drop rate reported as judge reliability.

Requests come out ordered by cell, so vLLM prefills each ~5-24k reference prefix
once for all 72 calls that share it rather than 72 times.

Usage:
    python scripts/judge/build_requests.py --limit 2      # smoke test
    python scripts/judge/build_requests.py                # all 350 cells
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from swissisoform.judge import (  # noqa: E402
    CATEGORY_LETTERS,
    DEFAULT_CORPUS,
    SYNTHESIS_UNIT,
)
from swissisoform.judge import prompts as PR  # noqa: E402
from swissisoform.judge import rubrics as RB  # noqa: E402
from swissisoform.judge.corpus import Corpus, load_corpus  # noqa: E402
from swissisoform.judge.reference import (  # noqa: E402
    REFERENCE_BUDGET_TOKENS,
    ReferenceBuilder,
    fit_reference,
    isoform_records,
    render,
)
from swissisoform.judge.serve import (  # noqa: E402
    MAX_MODEL_LEN,
    MAX_NEW_TOKENS,
    Request,
    estimate_tokens,
    write_requests,
)
from swissisoform.site.llm import load_records  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("judge.build")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI options."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--corpus", default=DEFAULT_CORPUS)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None, help="First N isoforms")
    p.add_argument(
        "--units",
        default=",".join((*CATEGORY_LETTERS, SYNTHESIS_UNIT)),
        help="Comma-separated units to build",
    )
    p.add_argument(
        "--budget",
        type=int,
        default=REFERENCE_BUDGET_TOKENS,
        help="Token budget for the reference payload",
    )
    return p.parse_args(argv)


def _tokenizer():
    """Prometheus's own tokenizer, loaded offline from the staged weights.

    Counting with the real tokenizer rather than chars/4 is the whole point: the
    estimate ran 1.8x optimistic on these JSON payloads, so 414 prompts sat over
    the 32,768 context and vLLM would have silently truncated them.
    """
    from transformers import AutoTokenizer

    from swissisoform.judge.serve import snapshot_dir

    return AutoTokenizer.from_pretrained(str(snapshot_dir()))


def _reference_for(
    builder: ReferenceBuilder, record: dict, unit: str, arm: str, corpus: Corpus, slug: str
) -> dict:
    """The reference for one cell.

    Category cells share one reference across arms -- that is the whole design.
    Synthesis cannot: it is judged on coherence with *its own* inherited category
    verdicts, which are the only thing that differs between arms, so each arm gets
    its own synthesis reference.
    """
    if unit != SYNTHESIS_UNIT:
        return builder.category(record, unit)
    reads = {
        letter: corpus.get(arm, slug, letter).payload
        for letter in CATEGORY_LETTERS
        if corpus.get(arm, slug, letter) is not None
    }
    return builder.synthesis(record, reads)


def main(argv: list[str] | None = None) -> int:
    """Build and write the request file."""
    args = parse_args(argv)
    units = [u.strip() for u in args.units.split(",") if u.strip()]
    out_dir = args.out or (ROOT / "data" / "output" / "judge" / args.corpus)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus(args.corpus)
    records = load_records(ROOT / "data" / "output" / args.corpus / "llm_evidence")
    isos = isoform_records(records)
    builder = ReferenceBuilder.build()

    tokenizer = _tokenizer()

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))

    slugs = list(corpus.slugs)[: args.limit] if args.limit else list(corpus.slugs)
    requests: list[Request] = []
    at_risk: list[dict] = []
    trimmed: list[dict] = []

    for slug in slugs:
        record = isos.get(slug)
        if record is None:
            logger.warning("no evidence record for %s — skipping", slug)
            continue
        for unit in units:
            arms = corpus.arms_for(slug, unit)
            if len(arms) < 2:
                logger.warning("cell (%s, %s) has %d arm(s) — skipping", slug, unit, len(arms))
                continue

            # One fitted reference per cell, reused by all 9 arms -- trimming has
            # to be identical across arms or they stop being comparable.
            shared = None
            if unit != SYNTHESIS_UNIT:
                fitted = fit_reference(
                    builder.category(record, unit), count_tokens, budget=args.budget
                )
                shared = render(fitted.reference)
                if fitted.trimmed:
                    trimmed.append(
                        {
                            "slug": slug,
                            "unit": unit,
                            "tokens": fitted.tokens,
                            "steps": fitted.steps_applied,
                        }
                    )

            # Every unordered pair, both presentation orders.
            for arm_a, arm_b in itertools.combinations(arms, 2):
                for order, (first, second) in enumerate(((arm_a, arm_b), (arm_b, arm_a))):
                    if shared is not None:
                        instruction = shared
                    else:
                        instruction = render(
                            fit_reference(
                                _reference_for(builder, record, unit, first, corpus, slug),
                                count_tokens,
                                budget=args.budget,
                            ).reference
                        )
                    prompt = PR.chat(
                        PR.relative_prompt(
                            instruction=instruction,
                            response_a=corpus.get(first, slug, unit).text,
                            response_b=corpus.get(second, slug, unit).text,
                            rubric=RB.pairwise().criterion,
                        )
                    )
                    requests.append(
                        Request(
                            id=f"pw|{slug}|{unit}|{first}|{second}|{order}",
                            slug=slug,
                            unit=unit,
                            rubric=RB.PAIRWISE_ID,
                            prompt=prompt,
                            arm_a=first,
                            arm_b=second,
                            order=order,
                        )
                    )
                    _note_if_at_risk(prompt, at_risk, slug, unit)

    path = out_dir / "requests.jsonl"
    n = write_requests(requests, path)
    _summarise(requests, at_risk, path, n, out_dir, trimmed, count_tokens)
    return 1 if at_risk else 0


def _note_if_at_risk(prompt: str, sink: list[dict], slug: str, unit: str) -> None:
    """Record a prompt whose estimated length leaves no room for feedback.

    Estimated at 4 chars/token, so this is a screen rather than the assertion --
    ``run_judge.py`` re-checks with the real tokenizer, which is the only count
    that decides anything.
    """
    tokens = estimate_tokens(prompt)
    if tokens + MAX_NEW_TOKENS > MAX_MODEL_LEN:
        sink.append({"slug": slug, "unit": unit, "est_tokens": tokens})


def _summarise(
    requests: list[Request],
    at_risk: list[dict],
    path: Path,
    n: int,
    out_dir: Path,
    trimmed: list[dict],
    count_tokens,
) -> None:
    """Print and persist the shape of what was built."""
    by_unit: dict[str, int] = {}
    longest = 0
    longest_real = 0
    for r in requests:
        by_unit[r.unit] = by_unit.get(r.unit, 0) + 1
        longest = max(longest, estimate_tokens(r.prompt))
    # Real count on the longest few only -- tokenizing all 37k here would double
    # the build time, and run_judge.py --check-context is the actual assertion.
    for r in sorted(requests, key=lambda r: -len(r.prompt))[:40]:
        longest_real = max(longest_real, count_tokens(r.prompt))

    meta = {
        "n_requests": n,
        "by_unit": by_unit,
        "longest_prompt_est_tokens": longest,
        "longest_prompt_real_tokens": longest_real,
        "n_cells_trimmed": len(trimmed),
        "trimmed": trimmed[:100],
        "max_model_len": MAX_MODEL_LEN,
        "max_new_tokens": MAX_NEW_TOKENS,
        "n_at_risk": len(at_risk),
        "at_risk": at_risk[:50],
    }
    (out_dir / "requests_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"\n{n:,} pairwise request(s) -> {path}")
    print("\nper unit:")
    for unit, count in sorted(by_unit.items()):
        print(f"  {unit:10s} {count:>8,}")
    print(
        f"\nlongest prompt ~{longest:,} est / {longest_real:,} real tokens (ctx {MAX_MODEL_LEN:,})"
    )
    if trimmed:
        steps: dict[str, int] = {}
        for entry in trimmed:
            for step in entry["steps"]:
                steps[step] = steps.get(step, 0) + 1
        print(f"{len(trimmed)} reference(s) trimmed to fit: {steps}")
        print("  (a trimmed cell is weaker evidence; requests_meta.json lists them)")
    else:
        print("no reference needed trimming")
    if at_risk:
        print(f"!! {len(at_risk)} prompt(s) leave no room for {MAX_NEW_TOKENS} new tokens")
    else:
        print(f"all prompts leave room for {MAX_NEW_TOKENS} new tokens")


if __name__ == "__main__":
    raise SystemExit(main())
