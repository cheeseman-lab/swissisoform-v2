r"""The Prometheus 2 prompt templates, verbatim, and the ``[RESULT]`` parser.

These strings are the ``prometheus-eval`` package's ``ABSOLUTE_PROMPT_WO_REF`` and
``RELATIVE_PROMPT_WO_REF``, vendored rather than imported because the request
builder runs in an env without that package. ``test_judge.py`` asserts equality
wherever ``prometheus_eval`` *is* importable, so the copy cannot drift silently --
which it already had: the first version of this file carried a ``"Feedback: "``
prefix in the output-format line that the published template does not have.
They must not be paraphrased. The model was trained to complete exactly this shape, and the usual
failure mode of an open judge is a lightly edited template: it still answers, so
nothing errors, but the scores stop meaning what the benchmarks measured. The
``###`` headers, the numbered instructions and the trailing ``###Feedback: `` are
all load-bearing.

**No reference answer.** The absolute template accepts a
``###Reference Answer (Score 5):`` block and scores are less noisy with one, but
we have no gold verdicts for these isoforms. A model-written reference would be a
ninth framing smuggled in as ground truth -- the one thing the experiment cannot
afford. So the reference-free variant is used and the cost is paid in precision,
which is exactly what the noise floor is there to measure.
"""

from __future__ import annotations

import math
import re
from typing import Literal

# Mistral/Mixtral has no system role, so the judge persona is prepended to the
# user turn -- which is what the prometheus-eval library does.
ABS_SYSTEM = (
    "You are a fair judge assistant tasked with providing clear, objective feedback "
    "based on specific criteria, ensuring each assessment reflects the absolute "
    "standards set for performance."
)
REL_SYSTEM = (
    "You are a fair judge assistant assigned to deliver insightful feedback that "
    "compares individual performances, highlighting how each stands relative to "
    "others within the same cohort."
)

# Assembled from short source lines rather than written out as one long literal:
# the template must stay byte-identical to the release, and the repo's line limit
# is 100. Implicit concatenation keeps both -- `test_prompts.py` pins the
# reconstructed text against the expected header and instruction lines.
_TASK_INTRO = (
    "An instruction (might include an Input inside it), a response to evaluate, "
    "and a score rubric representing a evaluation criteria are given.\n"
)
_ABS_STEPS = (
    "1. Write a detailed feedback that assess the quality of the response strictly "
    "based on the given score rubric, not evaluating in general.\n"
    "2. After writing a feedback, write a score that is an integer between 1 and 5. "
    "You should refer to the score rubric.\n"
    '3. The output format should look as follows: "(write a feedback for criteria) '
    '[RESULT] (an integer number between 1 and 5)"\n'
    "4. Please do not generate any other opening, closing, and explanations.\n"
)
_REL_STEPS = (
    "1. Write a detailed feedback that assess the quality of two responses strictly "
    "based on the given score rubric, not evaluating in general.\n"
    "2. After writing a feedback, choose a better response between Response A and "
    "Response B. You should refer to the score rubric.\n"
    '3. The output format should look as follows: "(write a feedback for criteria) '
    '[RESULT] (A or B)"\n'
    "4. Please do not generate any other opening, closing, and explanations.\n"
)

ABSOLUTE_TEMPLATE = (
    "###Task Description:\n" + _TASK_INTRO + _ABS_STEPS + "\n"
    "###The instruction to evaluate:\n{instruction}\n\n"
    "###Response to evaluate:\n{response}\n\n"
    "###Score Rubrics:\n{rubric}\n\n"
    "###Feedback: "
)

RELATIVE_TEMPLATE = (
    "###Task Description:\n" + _TASK_INTRO + _REL_STEPS + "\n"
    "###Instruction:\n{instruction}\n\n"
    "###Response A:\n{response_A}\n\n"
    "###Response B:\n{response_B}\n\n"
    "###Score Rubric:\n{rubric}\n\n"
    "###Feedback: "
)


def absolute_prompt(*, instruction: str, response: str, rubric: str) -> str:
    """The user turn for one direct assessment."""
    body = ABSOLUTE_TEMPLATE.format(instruction=instruction, response=response, rubric=rubric)
    return f"{ABS_SYSTEM}\n\n{body}"


def relative_prompt(*, instruction: str, response_a: str, response_b: str, rubric: str) -> str:
    """The user turn for one pairwise comparison."""
    body = RELATIVE_TEMPLATE.format(
        instruction=instruction,
        response_A=response_a,
        response_B=response_b,
        rubric=rubric,
    )
    return f"{REL_SYSTEM}\n\n{body}"


def chat(user_turn: str) -> str:
    """Wrap a user turn in Mixtral's instruct format.

    Written out rather than taken from the tokenizer's chat template so the
    prompt is inspectable in a request file and identical whatever transformers
    does to its templates between versions.
    """
    return f"<s>[INST] {user_turn} [/INST]"


# ── Parsing ────────────────────────────────────────────────────────────────

_SCORE = re.compile(r"\[RESULT\]\s*\(?\s*([1-5])\s*\)?", re.I)
_CHOICE = re.compile(r"\[RESULT\]\s*\(?\s*([AB])\s*\)?\b", re.I)

# The verdict sits within a couple of tokens of the marker; a wider scan drifts
# into the "Response A/B" prose that follows it in 283 of 7,261 completions.
_RESULT_MARKER = "[RESULT]"
_VERDICT_WINDOW = 4


# ── Preference from logprobs ───────────────────────────────────────────────


def emitted_text(token_logprobs: list[dict[str, float]]) -> str:
    """The greedily-decoded text, rebuilt from the per-step top-k alternatives.

    Decoding runs at ``temperature=0.0``, so the token actually emitted at each
    step is that step's argmax, and vLLM always includes the emitted token in the
    logprob dict. Rebuilding the text this way is what lets a character offset in
    the completion be mapped back to a step index -- there is no other alignment
    between :attr:`Result.completion` and :attr:`Result.token_logprobs`.
    """
    return "".join(max(step, key=step.__getitem__) for step in token_logprobs or [] if step)


def choice_probability(
    token_logprobs: list[dict[str, float]],
    completion: str = "",
) -> float | None:
    """P(the judge prefers slot A), from the logprobs at the verdict token.

    Returns ``None`` when the verdict position cannot be located or offers
    neither letter -- caller decides whether to fall back to :func:`parse_choice`
    or drop the call.

    **Why a probability rather than a letter.** A parsed letter has two failure
    modes that cost this corpus most of its data: two presentation orders can
    disagree outright (43-72% per category, so the pair is discarded), and the
    completion can omit ``[RESULT]`` entirely (4,303 of 31,950, rising 7% -> 24%
    with prompt length). A probability has neither: two orders average, and a
    confident-but-unparseable completion still yields its preference. It also
    keeps magnitude -- 0.52 and 0.98 are different evidence, where a vote flattens
    both to "A".

    **The position must be anchored to ``[RESULT]``.** An earlier version scanned
    for the first step offering both letters, which is wrong: Prometheus opens its
    feedback with "Response A ..." as narrative order, and in 3,014 of 4,000
    sampled completions that prose "A" is the first both-letters position. Reading
    it gave mean P(A) = 0.999 on byte-identical responses while the parsed verdicts
    on those same calls ran 8/7 -- it was measuring which response gets discussed
    first, not which one wins. So the scan starts after the last ``[RESULT]``
    (last, matching :func:`parse_choice`) and looks only inside a short window,
    since 283 of those completions carry more "Response A/B" prose afterwards.
    """
    steps = [step for step in token_logprobs or [] if step]
    if not steps:
        return None

    start = _verdict_step(steps, completion)
    if start is None:
        return None

    for step in steps[start : start + _VERDICT_WINDOW]:
        pair = _letter_logprobs(step)
        if pair is None:
            continue
        a, b = pair
        top = max(a, b)
        ea, eb = math.exp(a - top), math.exp(b - top)
        return ea / (ea + eb)
    return None


def _verdict_step(steps: list[dict[str, float]], completion: str) -> int | None:
    """Index of the first step at or after the last ``[RESULT]`` marker.

    Refuses (``None``) when the rebuilt text disagrees with *completion* about
    whether a marker exists at all: a mismatch means the argmax reconstruction is
    not the text that was returned, and silently scoring the wrong position is the
    exact failure this function was written to remove.
    """
    text = emitted_text(steps)
    marker = text.rfind(_RESULT_MARKER)
    if marker < 0:
        return None
    if completion and _RESULT_MARKER not in completion:
        return None

    cutoff = marker + len(_RESULT_MARKER)
    consumed = 0
    for index, step in enumerate(steps):
        consumed += len(max(step, key=step.__getitem__))
        if consumed >= cutoff:
            return index + 1
    return None


def _letter_logprobs(step: dict[str, float]) -> tuple[float, float] | None:
    """``(logprob_A, logprob_B)`` at one position, or ``None`` if neither appears.

    A letter missing from the top-k is scored at that step's floor -- the smallest
    logprob present -- rather than dropping the call. The losing letter falls out
    of the top-20 precisely when the judge is most certain, so dropping those would
    discard the most decisive verdicts and bias what survives toward the
    indecisive ones. The floor understates the winner's margin, which is the safe
    direction.
    """
    a = _letter_logprob(step, "A")
    b = _letter_logprob(step, "B")
    if a is None and b is None:
        return None
    floor = min(step.values())
    return (a if a is not None else floor, b if b is not None else floor)


def _letter_logprob(step: dict[str, float], letter: str) -> float | None:
    """The logprob of *letter* at one position, tolerating tokenizer whitespace.

    Mixtral's tokenizer may emit "A", " A", or the SentencePiece form. Taking the
    max over those spellings avoids missing the candidate on a detail of encoding.
    """
    best: float | None = None
    for token, logprob in step.items():
        if token.strip().strip("()[]") == letter:
            best = logprob if best is None else max(best, logprob)
    return best


def verdict_probability(forced_logprobs: list[dict[str, float]] | None) -> float | None:
    """P(slot A) from a forced-verdict step, where the position needs no anchoring.

    The forced pass appends ``[RESULT]`` to the prompt itself, so the very first
    generated position *is* the verdict -- there is no feedback prose to scan past
    and no marker to locate. Kept separate from :func:`choice_probability` for that
    reason: mixing them would mean the anchored path silently accepting a position
    it had not actually verified.
    """
    for step in forced_logprobs or []:
        pair = _letter_logprobs(step)
        if pair is None:
            continue
        a, b = pair
        top = max(a, b)
        ea, eb = math.exp(a - top), math.exp(b - top)
        return ea / (ea + eb)
    return None


def preference(
    token_logprobs: list[dict[str, float]],
    completion: str = "",
    forced_logprobs: list[dict[str, float]] | None = None,
) -> float | None:
    """P(slot A) for one call, from whichever channel carried a verdict.

    The single entry point every consumer should use. Tries the normal completion
    first, then the forced-verdict pass. Together they read 40 of 40 gate probes
    where the ``[RESULT]`` regex read 15.
    """
    direct = choice_probability(token_logprobs, completion)
    if direct is not None:
        return direct
    return verdict_probability(forced_logprobs)


def average_orders(forward: float | None, reverse: float | None) -> float | None:
    """One preference for a pair, from its two presentations.

    *forward* is P(slot A) with arm X in slot A; *reverse* is P(slot A) with arm Y
    there. So P(X preferred) = (forward + (1 - reverse)) / 2, which cancels a
    slot-position offset to first order -- the reason this replaces "keep the pair
    only if both orders agree".

    Returns whichever side is available when only one presentation parsed, and
    ``None`` when neither did.
    """
    if forward is None and reverse is None:
        return None
    if reverse is None:
        return forward
    if forward is None:
        return 1.0 - reverse
    return (forward + (1.0 - reverse)) / 2.0


class ParseError(ValueError):
    """The judge's output carried no usable verdict."""


def parse_score(text: str) -> tuple[int, str]:
    """``(score, feedback)`` from an absolute completion.

    Raises:
        ParseError: No ``[RESULT] <1-5>`` present. Raised rather than defaulted:
            a silent 3 would be indistinguishable from a real middling score and
            would drag every mean toward the centre.
    """
    matches = _SCORE.findall(text or "")
    if not matches:
        raise ParseError(f"no '[RESULT] <1-5>' in completion: {(text or '')[-200:]!r}")
    # Last match: the template puts the score at the end, and feedback prose can
    # quote the format string from the instructions.
    score = int(matches[-1])
    feedback = (text or "").split("[RESULT]")[0].removeprefix("Feedback:").strip()
    return score, feedback


def parse_choice(text: str) -> tuple[Literal["A", "B"], str]:
    """``(winner, feedback)`` from a relative completion.

    Raises:
        ParseError: No ``[RESULT] <A|B>`` present. A tie is not representable --
            the template offers only A or B -- so an unparseable comparison is
            dropped, and the drop rate is reported as judge reliability.
    """
    matches = _CHOICE.findall(text or "")
    if not matches:
        raise ParseError(f"no '[RESULT] <A|B>' in completion: {(text or '')[-200:]!r}")
    winner = matches[-1].upper()
    feedback = (text or "").split("[RESULT]")[0].removeprefix("Feedback:").strip()
    return winner, feedback  # type: ignore[return-value]
