"""Produce a 4-bit copy of Prometheus-8x7B with llm-compressor.

Run as ``python -m swissisoform.judge.quantize`` in the ``swissisoform-v2-quant``
env (see ``scripts/slurm/quantize_prometheus.sbatch``).

**Why a 4-bit copy exists.** bf16 Prometheus is 87 GiB. It does not fit on one
A100-80 (11 GiB short before any KV cache), and on 2x A6000 vLLM reports 0.92 GiB
of KV cache and caps context at 15,040 tokens -- against a corpus whose worst
prompt is 30,517. So on A6000s, 4-bit is not an optimisation; it is the only way
the judge runs at all. At W4A16 the weights are ~22 GiB, leaving 23.6 GiB of KV on
a single 48 GiB card: 6 concurrent requests at worst-case length, ~21 at the median.

bf16 on 2x A100-80 remains the reference configuration and needs no licence; this
copy needs one (see the module's verification steps in the plan), because 4-bit is
a lossy approximation of the model doing the scoring.

**Why llm-compressor and not AutoAWQ.** AutoAWQ hardcodes Mixtral's
``block_sparse_moe``, which transformers 5.x renamed to ``mlp``; its last tested
transformers was 4.51.3 while vLLM requires >=4.56.0. Non-overlapping, and the
quantization died at layer 0 of 32. Its own deprecation notice redirects here.

Three decisions, two of them carried over from the AutoAWQ attempt because they
were validated there:

1. **Calibration is our own judge prompts**, not ``pile-val``. AWQ/GPTQ calibration
   measures activation scale on real inputs to decide which weight channels to
   protect from rounding, so in-distribution data is the point, not a convenience
   -- and generic web text is far from a 16k-token JSON evidence payload. It also
   keeps the job offline.
2. **Prompts are chunked to 512-token blocks, interleaved across prompts.** Our
   prompts are 3k-25k tokens. Round-robin matters: requests are written ordered by
   cell, so draining one prompt at a time would fill the calibration set from a
   single isoform's C and D payloads, and an M payload with its variant tables
   tokenizes nothing like a C one.
3. **The MoE router is never quantized, and that is asserted.** Mixtral routes each
   token to 2 of 8 experts through a gate; 4-bit rounding there changes *which*
   experts fire rather than perturbing a value. The model would still load and
   still answer, just wrongly. AutoAWQ excluded it by default; llm-compressor does
   not, so it is named explicitly and then checked.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from swissisoform.judge.serve import MODEL_DIR, ROOT, snapshot_dir

logger = logging.getLogger(__name__)

DEFAULT_OUT = ROOT / ".torch_cache" / "hf" / "hub" / "prometheus-8x7b-v2.0-w4a16"

# W4A16: 4-bit weights, 16-bit activations. The established scheme for Mixtral in
# llm-compressor, and what vLLM reads through `--quantization compressed-tensors`.
SCHEME = "W4A16"

# Never quantize these. `re:.*gate$` is Mixtral's expert router -- see the module
# docstring. `lm_head` is the standard exclusion: it is large, numerically
# sensitive, and quantizing it degrades output quality for no memory win worth
# having.
IGNORE: tuple[str, ...] = ("lm_head", "re:.*gate$")

# The router substring that MUST appear in `IGNORE`. Checked rather than trusted,
# because a quantized router fails silently.
ROUTER_PATTERN = "gate"

N_CALIB_SAMPLES = 512
CALIB_SEQ_LEN = 512
SEED = 20260917


class QuantizeError(RuntimeError):
    """Quantization could not run, or would produce something unusable."""


def load_calibration(
    requests_path: Path,
    *,
    n_samples: int = N_CALIB_SAMPLES,
    seed: int = SEED,
) -> list[str]:
    """Calibration prompts drawn evenly across the 7 output units.

    Raises:
        QuantizeError: The request file is missing or carries no prompts. Refused
            rather than falling back to a public dataset, which is the failure this
            function exists to prevent -- and the job runs offline anyway.
    """
    if not requests_path.exists():
        raise QuantizeError(
            f"no requests at {requests_path}. Build them first:\n"
            f"  python scripts/judge/build_requests.py\n"
            f"They are the calibration set, so this is not optional."
        )

    by_unit: dict[str, list[str]] = defaultdict(list)
    with requests_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("prompt"):
                by_unit[row.get("unit", "?")].append(row["prompt"])
    if not by_unit:
        raise QuantizeError(f"{requests_path} contained no prompts")

    rng = random.Random(seed)
    for prompts in by_unit.values():
        rng.shuffle(prompts)

    units = sorted(by_unit)
    cursor = {u: 0 for u in units}
    out: list[str] = []
    while len(out) < n_samples:
        progressed = False
        for unit in units:
            if len(out) >= n_samples:
                break
            if cursor[unit] < len(by_unit[unit]):
                out.append(by_unit[unit][cursor[unit]])
                cursor[unit] += 1
                progressed = True
        if not progressed:
            break
    logger.info(
        "calibration: %d prompt(s) across %d unit(s) %s", len(out), len(units), dict(cursor)
    )
    return out


def chunk_prompts(
    prompts: list[str],
    tokenizer: Any,
    *,
    seq_len: int = CALIB_SEQ_LEN,
    n_chunks: int = N_CALIB_SAMPLES,
) -> list[list[int]]:
    """Cut prompts into ``seq_len`` token blocks, round-robin across prompts.

    Interleaved rather than draining each prompt: one 25k-token M prompt would
    otherwise contribute 50 consecutive blocks and swamp the unit spread that
    :func:`load_calibration` exists to create.

    Raises:
        QuantizeError: No prompt yielded a full block.
    """
    per_prompt: list[list[list[int]]] = []
    for text in prompts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        blocks = [ids[i : i + seq_len] for i in range(0, len(ids) - seq_len + 1, seq_len)]
        if blocks:
            per_prompt.append(blocks)
    if not per_prompt:
        raise QuantizeError(
            f"no calibration prompt yielded a full {seq_len}-token block; "
            f"the sampled text is shorter than one block"
        )

    out: list[list[int]] = []
    depth = 0
    while len(out) < n_chunks:
        progressed = False
        for blocks in per_prompt:
            if len(out) >= n_chunks:
                break
            if depth < len(blocks):
                out.append(blocks[depth])
                progressed = True
        if not progressed:
            break
        depth += 1
    logger.info(
        "calibration: %d block(s) of %d tokens from %d prompt(s)",
        len(out),
        seq_len,
        len(per_prompt),
    )
    return out


def assert_router_excluded(ignore: tuple[str, ...] = IGNORE) -> None:
    """Fail before the expensive pass if the router would be quantized.

    Raises:
        QuantizeError: No entry in *ignore* covers Mixtral's gate. The failure this
            prevents is silent: a quantized router still loads and still answers,
            it just routes tokens to different experts than bf16 would.
    """
    if not any(ROUTER_PATTERN in entry for entry in ignore):
        raise QuantizeError(
            f"ignore list {list(ignore)} does not exclude Mixtral's router "
            f"({ROUTER_PATTERN!r}). Each token is routed to 2 of 8 experts through "
            f"that gate, so quantizing it changes which experts fire."
        )
    logger.info("router excluded: %s", list(ignore))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI options."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", type=Path, default=None, help="bf16 source (default: staged)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--requests",
        type=Path,
        default=ROOT / "data" / "output" / "judge" / "cheeseman50" / "requests.jsonl",
        help="Calibration source -- the judge's own prompts",
    )
    p.add_argument("--n-samples", type=int, default=N_CALIB_SAMPLES)
    p.add_argument("--seq-len", type=int, default=CALIB_SEQ_LEN)
    p.add_argument("--force", action="store_true", help="Overwrite an existing output")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the calibration set and recipe without loading the model",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Quantize, or report what would be done."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args(argv)

    assert_router_excluded()
    source = Path(args.model) if args.model else snapshot_dir()
    prompts = load_calibration(args.requests, n_samples=args.n_samples, seed=SEED)

    if args.dry_run:
        lengths = sorted(len(t) for t in prompts)
        print(f"\n{len(prompts)} calibration prompt(s) from {args.requests}")
        print(
            f"  chars: min {lengths[0]:,}  median {lengths[len(lengths) // 2]:,}  "
            f"max {lengths[-1]:,}"
        )
        print(f"  source:  {source}")
        print(f"  out:     {args.out}")
        print(f"  scheme:  {SCHEME}   ignore: {list(IGNORE)}")
        print(f"  blocks:  {args.n_samples} x {args.seq_len} tokens")
        return 0

    if args.out.exists() and not args.force:
        raise QuantizeError(
            f"{args.out} already exists. Pass --force to overwrite -- re-quantizing "
            f"silently would make it unclear which weights were served."
        )

    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import GPTQModifier
    except ImportError as exc:
        raise QuantizeError(
            "llmcompressor is not installed. Create its env:\n"
            "  bash scripts/setup/create_envs.sh --quant\n"
            "It cannot share the judge env: it needs compressed-tensors==0.18.0 and "
            "transformers<=5.14.1, while vLLM pins 0.15.0.1 and 5.17.0."
        ) from exc
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("loading bf16 from %s", source)
    tokenizer = AutoTokenizer.from_pretrained(str(source))
    # device_map="cpu", NOT "auto". With "auto" accelerate spreads the 87 GiB of
    # bf16 weights onto the single 48 GiB GPU and llm-compressor OOMs ~20 minutes
    # in ("GPU 0 has a total capacity of 47.43 GiB of which 110.31 MiB is free").
    # The weights belong in host RAM -- the sbatch asks for 300 GB -- while
    # llm-compressor onloads one layer at a time to quantize it.
    model = AutoModelForCausalLM.from_pretrained(
        str(source), torch_dtype="auto", device_map="cpu", low_cpu_mem_usage=True
    )

    blocks = chunk_prompts(prompts, tokenizer, seq_len=args.seq_len, n_chunks=args.n_samples)
    dataset = Dataset.from_dict({"input_ids": blocks})

    logger.info("quantizing: scheme=%s ignore=%s", SCHEME, list(IGNORE))
    oneshot(
        model=model,
        dataset=dataset,
        recipe=GPTQModifier(targets="Linear", scheme=SCHEME, ignore=list(IGNORE)),
        max_seq_length=args.seq_len,
        num_calibration_samples=len(blocks),
        output_dir=str(args.out),
    )
    tokenizer.save_pretrained(str(args.out))
    _write_provenance(args.out, source, args.requests, blocks, args.seq_len)

    total = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file())
    logger.info("wrote %s (%.1f GiB)", args.out, total / 2**30)
    print(f"\nserve it with:\n  --model {args.out} --quantization compressed-tensors")
    return 0


def _write_provenance(
    out: Path, source: Path, requests: Path, blocks: list[list[int]], seq_len: int
) -> None:
    """Record what produced these weights.

    The calibration set is the part of a quantized judge that is invisible from the
    weights themselves, and the reason the community 4-bit Prometheus repos were
    rejected (all <20 downloads, none documenting theirs). Not recording ours would
    reproduce that problem one directory over.
    """
    (out / "_quantize.json").write_text(
        json.dumps(
            {
                "artifact": f"Prometheus-8x7b-v2.0, {SCHEME} via llm-compressor",
                "scheme": SCHEME,
                "ignore": list(IGNORE),
                "ignore_note": (
                    "re:.*gate$ is Mixtral's expert router. Quantizing it changes "
                    "which of the 8 experts each token reaches, which the model "
                    "would not report as an error."
                ),
                "source_weights": str(source),
                "source_model_dir": str(MODEL_DIR),
                "calibration_source": str(requests),
                "calibration_blocks": len(blocks),
                "calibration_seq_len": seq_len,
                "calibration_note": (
                    "The judge's own prompts, sampled round-robin across the 7 "
                    "output units, then cut into fixed-length blocks interleaved "
                    "across prompts. Not pile-val: generic web text is out of "
                    "distribution for 16k-token JSON evidence payloads."
                ),
                "seed": SEED,
                "serve_with": "--quantization compressed-tensors",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
