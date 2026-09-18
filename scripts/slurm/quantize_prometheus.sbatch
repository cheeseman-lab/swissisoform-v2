#!/bin/bash
#SBATCH --job-name=swissiso_prom_w4a16
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --requeue

# Produce the 4-bit (W4A16) copy of Prometheus-8x7B with llm-compressor.
#
# Why this exists: bf16 is 87 GiB, which does not fit on one A100-80 and cannot
# serve this corpus on A6000s at all -- vLLM reports 0.92 GiB of KV cache on
# 2x A6000 and caps context at 15,040 tokens against a worst prompt of 30,517.
# At W4A16 the weights are ~22 GiB, leaving 23.6 GiB of KV on a single 48 GiB
# card: 6 concurrent requests at worst-case length, ~21 at the median.
#
# bf16 on 2x A100-80 remains the reference configuration. This artifact is an
# option, and it needs a licence the reference does not:
#   run_judge.sbatch ... --compare-backends   (scores one cell both ways)
#
# 300G of host RAM because llm-compressor holds the model on CPU and moves one
# layer at a time to the GPU; the 87 GiB of bf16 weights live in host memory
# throughout. Both A6000 nodes have plenty (rodeo 756G, smackdown 491G).
#
# Runs in the *judge* env, not a separate one. A dedicated env resolves torch
# 2.14.0+cu130, which cannot initialise on this driver (CUDA 12.6); the judge env's
# torch 2.10.0+cu128 works. Sharing the env also means one compressed-tensors
# version writes and reads the weights.
#
# Usage:
#   sbatch --job-name=${USER}_prom_w4a16 scripts/slurm/quantize_prometheus.sbatch
#   # inspect the calibration set without loading the model:
#   sbatch ... scripts/slurm/quantize_prometheus.sbatch --dry-run

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

eval "$(conda shell.bash hook)"
conda activate swissisoform-v2-judge

export HF_HOME="$PWD/.torch_cache/hf"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1     # weights are staged; calibration is our own prompts
# The env's libstdc++ must precede the node's: conda's icu needs CXXABI_1.3.15 and
# no compute node's system copy has it. Same reason as run_judge.sbatch.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "[$(date -u +%FT%TZ)] quantizing  args='$*'"
t0=$SECONDS
python -m swissisoform.judge.quantize "$@"
took=$((SECONDS - t0))
printf '[%s] done in %ds (%dh%02dm)\n' \
    "$(date -u +%FT%TZ)" "$took" $((took / 3600)) $(((took % 3600) / 60))
