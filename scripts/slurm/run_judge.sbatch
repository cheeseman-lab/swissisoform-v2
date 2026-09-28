#!/bin/bash
#SBATCH --job-name=swissiso_judge
#SBATCH --partition=nvidia-A100-20
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16   # A100 partition cap is 32/user
#SBATCH --mem=120G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --requeue

# Score the judge request file with Prometheus-8x7B, unquantized.
#
# bf16 on 2x A100 via tensor parallel. 88 GB of weights across two 80 GB cards
# leaves ~68 GB for KV cache, and Mixtral's KV is 128 KB/token, so ~34 concurrent
# 16k-token requests -- more than 4-bit on a single A6000 would give (~10), with
# no precision question to license. 2 GPUs is the per-user cap on this partition.
#
# Unquantized on purpose: 4-bit on a single A6000 would give ~10 concurrent
# requests against this configuration's ~34, and it would need a quantization
# pass this toolchain cannot run (autoawq hardcodes Mixtral's `block_sparse_moe`,
# which transformers 5.x renamed to `mlp`, and its last tested transformers was
# 4.51.3 while vllm requires >=4.56.0).
#
# Requests arrive ordered by cell, so each reference payload (5-24k tokens, shared
# by the ~108 calls in that cell) is prefilled once instead of 108 times.
#
# --requeue with the resumable driver: results.jsonl is append-only and the driver
# skips ids already in it, so a preempted job restarts where it stopped. 37,350
# calls is too long to lose to one preemption.
#
# Defaults here are bf16 on 2x A100 and stay that way. The 4-bit option is an
# override, not a different script -- it supplies weights and a flag:
#
#   sbatch --partition=nvidia-A6000-20 --gres=gpu:1 --mem=96G \
#     scripts/slurm/run_judge.sbatch --batch-size 64 \
#     --quantization compressed-tensors \
#     --model .torch_cache/hf/hub/prometheus-8x7b-v2.0-w4a16
#
# 4-bit needs a licence bf16 does not: score one cell both ways with
# --compare-backends and confirm the pairwise ordering agrees.
#
# Usage (gates first -- they are cheap and the run is not):
#   sbatch ... scripts/slurm/run_judge.sbatch --check-context
#   sbatch ... scripts/slurm/run_judge.sbatch --sanity-anchor
#   sbatch ... scripts/slurm/run_judge.sbatch --self-consistency
#   sbatch ... scripts/slurm/run_judge.sbatch --batch-size 64

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

eval "$(conda shell.bash hook)"
conda activate swissisoform-v2-judge

export HF_HOME="$PWD/.torch_cache/hf"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# The env's own libstdc++ has to come first. conda's icu (pulled in by sqlite3,
# which vLLM reaches via diskcache when it builds the engine) needs CXXABI_1.3.15,
# and no compute node's system /lib/x86_64-linux-gnu/libstdc++.so.6 has it -- both
# vger and rodeo report 0 matches, while the env ships libstdc++.so.6.0.36 which
# does. Without this the engine dies at construction with
# `ImportError: libstdc++.so.6: version CXXABI_1.3.15 not found`.
#
# Note a bare `import vllm` does NOT reach that path, so an import smoke test
# passes and the failure only shows up when the engine is actually built.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "[$(date -u +%FT%TZ)] args='$*'"
t0=$SECONDS
python scripts/judge/run_judge.py --tensor-parallel-size 2 "$@"
took=$((SECONDS - t0))
printf '[%s] done in %ds (%dh%02dm)\n' \
    "$(date -u +%FT%TZ)" "$took" $((took / 3600)) $(((took % 3600) / 60))
