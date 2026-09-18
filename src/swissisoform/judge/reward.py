"""Reward-model scoring, as a judge whose failure modes are structurally absent.

Prometheus is a *generative* judge: it reads two responses in two slots and emits a
letter. Measured on this corpus that design failed three ways at once -- 75.8%
slot-A preference on byte-identical text, a verdict token saturated at P in
{0, 1} so no graded preference exists, and 43-72% of pairs discarded because the
two presentation orders disagreed.

A reward model removes all three by construction rather than by compensation:

* **No slots.** Each response is scored alone against the instruction, so there is
  no position for a preference to attach to. Position bias is not reduced, it is
  undefined.
* **No verdict token.** The output is a scalar from a regression head, so
  magnitude is what the model natively produces -- the thing logprobs were
  supposed to recover and could not.
* **Nothing to discard.** Without two orders there is no consistency test, so
  every cell contributes.

The scores are cell-centered and compared exactly like the absolute rubric scores:
:func:`weigh.center_scores` already takes ``Score`` objects keyed by
``(slug, unit, arm, rubric)`` and subtracts each cell's own mean, so a reward run
needs no new weighing math.

**Model choice: Skywork-Reward-V2-Llama-3.1-8B.** Picked on context length, which
is the binding constraint here -- its 131,072 positions hold this corpus's 12-30k
token instructions whole. The alternative worth wanting was ArmoRM-Llama3-8B, whose
multi-objective head reports a *verbosity* dimension that would let length be
discounted explicitly instead of asked away in a rubric; it maxes out at 8,192
positions, so using it would mean truncating the reference payload -- reintroducing
the exact confound the pairwise design was built to avoid. Context won.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("judge.reward")

MODEL_ID = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
# 131,072 in the config; held well below it because the reward head reads the final
# position and a silently right-truncated instruction would score a response
# against evidence it was never shown.
MAX_LENGTH = 65_536


class RewardError(RuntimeError):
    """Raised when the scorer cannot be built or a sequence will not fit."""


@dataclass
class RewardScore:
    """One response's scalar reward, keyed like an absolute rubric score."""

    slug: str
    unit: str
    arm: str
    score: float
    tokens: int = 0


class RewardScorer:
    """Scores ``(instruction, response)`` pairs with a sequence-classification head.

    One model, held on GPU for the life of the process. Built lazily so importing
    this module costs nothing in the CPU env.
    """

    def __init__(
        self,
        model: str | Path = MODEL_ID,
        *,
        device: str = "cuda",
        max_length: int = MAX_LENGTH,
        dtype: str = "bfloat16",
        load_model: bool = True,
    ) -> None:
        """Load the tokenizer, and the model onto *device* unless ``load_model``.

        ``load_model=False`` gives a tokenizer-only scorer for the context check, so
        that check goes through this class's own :meth:`token_count` rather than
        re-implementing the chat template. The duplicate implementation is how the
        BatchEncoding bug below survived in two places at once.
        """
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RewardError(
                f"transformers/torch unavailable in this env ({exc}). The reward "
                "scorer runs in swissisoform-v2-judge."
            ) from exc

        self._torch = torch
        self.max_length = max_length
        self._tokenizer = AutoTokenizer.from_pretrained(str(model))
        self._model = None
        if not load_model:
            return
        logger.info("loading %s (%s, max_len=%d)", model, dtype, max_length)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(model),
            torch_dtype=getattr(torch, dtype),
            device_map=device,
            num_labels=1,
            # Standard LlamaForSequenceClassification -- no custom modelling code,
            # unlike ArmoRM's LlamaForRewardModelWithGating.
            trust_remote_code=False,
        )
        self._model.eval()

    def token_count(self, instruction: str, response: str) -> int:
        """Tokens in the chat-formatted pair, for a fit check before scoring."""
        return int(self._encode(instruction, response).shape[-1])

    def _encode(self, instruction: str, response: str):
        """Chat-template the pair the way the reward model was trained to see it.

        Rendered to text and tokenized in a second step, rather than with
        ``apply_chat_template(tokenize=True)``: on transformers 5.14.1 that returns
        a ``BatchEncoding``, so ``len()`` yields 2 -- its key count -- and a 5,000
        token pair silently measures as 2 tokens. ``add_special_tokens=False``
        because the template already emits ``<|begin_of_text|>``; without it every
        sequence carries two BOS tokens.
        """
        text = self._tokenizer.apply_chat_template(
            [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ],
            tokenize=False,
        )
        encoded = self._tokenizer(text, add_special_tokens=False, return_tensors="pt")
        return encoded["input_ids"]

    def score(self, instruction: str, response: str) -> float:
        """The scalar reward for one response.

        Raises:
            RewardError: The pair exceeds ``max_length``, or the model was not
                loaded. Oversized pairs raise rather than truncate -- a
                right-truncated instruction silently scores a response against
                evidence the model never read, which is a wrong number rather than
                a missing one.
        """
        if self._model is None:
            raise RewardError("scorer built with load_model=False")
        ids = self._encode(instruction, response)
        if ids.shape[-1] > self.max_length:
            raise RewardError(f"{ids.shape[-1]:,} tokens exceeds max_length {self.max_length:,}")
        with self._torch.no_grad():
            out = self._model(ids.to(self._model.device))
        return float(out.logits[0][0].item())
