"""Running Prometheus locally through vLLM.

Two things here are not conveniences:

**Requests are ordered by cell.** All 108 calls for one ``(isoform, unit)`` share
the same reference payload -- up to 24k tokens of it -- so grouping them lets
vLLM's prefix cache prefill it once instead of 108 times. Shuffled, prefill
dominates the run and the 37,800 calls stop being feasible on one GPU.

**Generation is greedy and seeded.** A judge that scores differently on re-run
cannot establish a noise floor for anything else. ``temperature=0`` is the
setting; the seed is belt-and-braces for any sampling path vLLM takes internally.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]

# The local weights. bf16, 88 GB, 19 shards, sha256-verified against their HF blob
# names. Lives beside the ESM-C / ESMFold2 caches so HF_HOME resolves all of them.
MODEL_DIR = ROOT / ".torch_cache" / "hf" / "hub" / "models--prometheus-eval--prometheus-8x7b-v2.0"

# Prometheus 2 is Mixtral 8x7B: 32k trained context. The worst cell measured on
# this corpus is a ~26k-token pairwise call (synthesis, 24.4k reference), so the
# full window is needed and there is ~6.8k of headroom.
MAX_MODEL_LEN = 32_768

# Feedback runs a few hundred tokens; 512 leaves room without inviting an essay.
MAX_NEW_TOKENS = 512

# How many alternatives to return per generated position. 20 is ample for finding
# "A" and "B" among the candidates at the decision token: both are single tokens in
# a Llama/Mixtral vocabulary, and at temperature 0 the decision position is
# strongly peaked, so the pair is near the top when it is present at all.
LOGPROBS_TOP_K = 20

# Appended to prompt+completion to force the verdict to generated position 0.
FORCED_VERDICT_SUFFIX = "[RESULT]"

SEED = 20260915


class ServeError(RuntimeError):
    """The judge could not be served."""


@dataclass(frozen=True)
class Request:
    """One judge call, resumable by ``id``."""

    id: str
    kind: str  # "absolute" | "pairwise"
    slug: str
    unit: str
    rubric: str
    prompt: str
    arm: str = ""  # absolute
    arm_a: str = ""  # pairwise
    arm_b: str = ""
    order: int = 0  # 0 = (a, b) as listed; 1 = swapped

    @property
    def cell(self) -> tuple[str, str]:
        """The cell this request belongs to, for prefix-cache ordering."""
        return (self.slug, self.unit)


@dataclass
class Result:
    """One completion, parsed downstream rather than here.

    ``token_logprobs`` carries the top-k alternatives at each generated position:
    ``[{token_text: logprob, ...}, ...]``. It is what lets a pairwise verdict be
    read as a probability rather than a parsed letter.
    """

    id: str
    completion: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    token_logprobs: list[dict[str, float]] = field(default_factory=list)
    # Populated only when the completion never wrote a verdict, by the forced pass
    # below. Empty is the normal case, so it costs a key per row and nothing more.
    forced_logprobs: list[dict[str, float]] = field(default_factory=list)


def _flatten_logprobs(completion: object) -> list[dict[str, float]]:
    """vLLM's per-position logprob objects as plain ``{token: logprob}`` dicts.

    Returned as text keys rather than token ids so the decision position can be
    found without a tokenizer, and so a results file stays readable. Missing
    logprobs give an empty list rather than raising: a run that forgot to request
    them should fall back to parsing the text, not die.

    An empty ``decoded_token`` is kept as ``""``, not replaced. Mixtral's
    SentencePiece decodes a lone leading-space token to the empty string, and
    substituting the numeric token id for it (``x or str(token_id)``) injected
    digits into the text that :func:`prompts.emitted_text` rebuilds -- turning
    ``[RESULT]`` into ``[28705RESULT]`` and costing 14 of 40 gate probes their
    verdict position. Only a genuinely absent attribute falls back to the id, and
    it is marked as one so it cannot be mistaken for model output.
    """
    per_position = getattr(completion, "logprobs", None)
    if not per_position:
        return []
    out: list[dict[str, float]] = []
    for step in per_position:
        if not step:
            out.append({})
            continue
        entry: dict[str, float] = {}
        for token_id, info in step.items():
            text = getattr(info, "decoded_token", None)
            if text is None:
                text = f"<id:{token_id}>"
            logprob = float(getattr(info, "logprob", 0.0))
            # Two ids can decode to the same text; keep the likelier one rather
            # than whichever iterated last.
            if text not in entry or logprob > entry[text]:
                entry[text] = logprob
        out.append(entry)
    return out


def snapshot_dir(model_dir: Path = MODEL_DIR) -> Path:
    """The single revision inside an HF cache directory.

    Raises:
        ServeError: No snapshot, or more than one (ambiguous -- refuse rather than
            silently pick, since two revisions score differently).
    """
    snapshots = sorted(p for p in (model_dir / "snapshots").glob("*") if p.is_dir())
    if not snapshots:
        raise ServeError(f"no snapshot under {model_dir}; the weights are not staged")
    if len(snapshots) > 1:
        raise ServeError(
            f"{len(snapshots)} revisions under {model_dir}: {[p.name for p in snapshots]}. "
            "Pick one explicitly -- two revisions do not score identically."
        )
    return snapshots[0]


def order_by_cell(requests: Iterable[Request]) -> list[Request]:
    """Group requests so one cell's reference prefix is prefilled once.

    Within a cell, absolute calls come first: they are shorter, so any early
    failure surfaces before the expensive pairwise block runs.
    """
    return sorted(
        requests,
        key=lambda r: (r.slug, r.unit, r.kind != "absolute", r.rubric, r.id),
    )


def write_requests(requests: Iterable[Request], path: Path) -> int:
    """Write requests as JSONL. Returns the count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as handle:
        for request in order_by_cell(requests):
            handle.write(json.dumps(asdict(request), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_requests(path: Path) -> Iterator[Request]:
    """Stream requests back from JSONL."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield Request(**json.loads(line))


def completed_ids(path: Path) -> set[str]:
    """Ids already in a results file, so a killed run resumes.

    Tolerates a truncated final line: a job killed mid-write leaves one, and
    refusing to resume over it would mean redoing the whole run.
    """
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                logger.warning("ignoring unparseable results line (truncated write?)")
    return done


class Judge:
    """A vLLM engine wrapped for batch scoring.

    ``vllm`` is imported lazily so the request builder, the checks and the
    analysis all run in an environment without it -- which is most of this
    package's use.
    """

    def __init__(
        self,
        *,
        model: Path | str | None = None,
        max_model_len: int = MAX_MODEL_LEN,
        tensor_parallel_size: int = 1,
        # None = bf16. A plain passthrough to vLLM, so a quantized copy is an
        # *input you may supply* rather than a separate code path -- bf16 on
        # 2x A100-80 stays the reference configuration.
        quantization: str | None = None,
        gpu_memory_utilization: float = 0.92,
        seed: int = SEED,
    ) -> None:
        """Start the engine. Raises ServeError when vLLM is absent."""
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:  # pragma: no cover - env-dependent
            # Distinguish absent-package from broken-import. The first version of
            # this reported "vllm is not installed" for both, which hid a real
            # CXXABI_1.3.15 / libstdc++ mismatch behind a wrong instruction --
            # vllm was installed and the message sent the reader off to rebuild
            # the env that was already there.
            import importlib.util

            if importlib.util.find_spec("vllm") is None:
                raise ServeError(
                    "vllm is not installed. Create the judge env:\n"
                    "  bash scripts/setup/create_envs.sh --judge"
                ) from exc
            raise ServeError(
                f"vllm is installed but failed to import: {exc}\n"
                f"If this mentions CXXABI or libstdc++, the node's system library "
                f"is older than the env's; put the env first:\n"
                f'  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"'
            ) from exc

        path = Path(model) if model else snapshot_dir()
        # Exposed so callers can tell whether a prompt will fit before sending it;
        # the anchor gate uses it to skip oversized probes instead of aborting.
        self.max_model_len = max_model_len
        logger.info(
            "loading %s (%s, tp=%d, max_len=%d)",
            path,
            quantization or "bf16",
            tensor_parallel_size,
            max_model_len,
        )
        self._llm = LLM(
            model=str(path),
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            quantization=quantization,
            enable_prefix_caching=True,
            seed=seed,
            trust_remote_code=False,
        )
        # Greedy: a judge that drifts between runs cannot be a yardstick.
        self._params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=MAX_NEW_TOKENS,
            seed=seed,
            # Top-k logprobs at every generated position, so a pairwise verdict can
            # be read as P("A") vs P("B") instead of regex-parsed out of prose.
            #
            # The hard-vote path cost this run most of its data: 43-72% of pairs
            # per category disagreed between presentation orders and were dropped,
            # and 4,303 of 31,950 completions never emitted a parseable [RESULT] at
            # all. A probability has neither failure mode -- two orders average
            # instead of agreeing-or-not, and there is no token to find.
            logprobs=LOGPROBS_TOP_K,
        )
        # One token, because the prompt already ends in "[RESULT]" -- see
        # _force_missing_verdicts.
        self._forced_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
            seed=seed,
            logprobs=LOGPROBS_TOP_K,
        )

    def run(self, requests: list[Request]) -> list[Result]:
        """Score a batch, preserving input order in the output."""
        prompts = [r.prompt for r in requests]
        outputs = self._llm.generate(prompts, self._params)
        results: list[Result] = []
        for request, output in zip(requests, outputs):
            first = output.outputs[0] if output.outputs else None
            completion = first.text if first else ""
            results.append(
                Result(
                    id=request.id,
                    completion=completion,
                    prompt_tokens=len(output.prompt_token_ids or []),
                    completion_tokens=len(first.token_ids) if first else 0,
                    token_logprobs=_flatten_logprobs(first),
                )
            )
        self._force_missing_verdicts(requests, results)
        return results

    def _force_missing_verdicts(self, requests: list[Request], results: list[Result]) -> None:
        """Re-ask, with ``[RESULT]`` already written, where no verdict was emitted.

        Measured on the 40 gate probes: 7 completions stopped at 341 tokens of a
        512 cap, ended a complete sentence, and simply never wrote the marker. They
        are not truncated -- the judge finished and declined to commit. Appending
        the marker to the prompt puts the verdict at generated position 0, so it
        cannot be skipped, and the whole prefix is already in the KV cache
        (``enable_prefix_caching``), making this ~1 token of decode per call.

        Pairwise only: absolute rubrics emit ``[RESULT] 1-5``, which
        :func:`prompts.choice_probability` never reads, so every absolute call
        would otherwise queue a pointless second pass.
        """
        from . import prompts as PR

        todo = [
            index
            for index, (request, result) in enumerate(zip(requests, results))
            if request.kind == "pairwise"
            and PR.choice_probability(result.token_logprobs, result.completion) is None
        ]
        if not todo:
            return
        logger.info("forcing a verdict on %d of %d call(s)", len(todo), len(results))
        prompts = [requests[i].prompt + results[i].completion + FORCED_VERDICT_SUFFIX for i in todo]
        outputs = self._llm.generate(prompts, self._forced_params)
        for index, output in zip(todo, outputs):
            first = output.outputs[0] if output.outputs else None
            results[index].forced_logprobs = _flatten_logprobs(first) if first else []

    def token_count(self, text: str) -> int:
        """Tokens in *text*, for the context-fit assertion."""
        return len(self._llm.get_tokenizer().encode(text))


def fits_context(prompt_tokens: int, *, max_model_len: int = MAX_MODEL_LEN) -> bool:
    """Whether a prompt leaves room for the feedback it must generate."""
    return prompt_tokens + MAX_NEW_TOKENS <= max_model_len


def estimate_tokens(text: str) -> int:
    """Rough token count without loading a tokenizer.

    4 chars/token, deliberately crude: used only to flag a cell as at-risk before
    the GPU env exists. The real assertion uses :meth:`Judge.token_count`.
    """
    return len(text) // 4
