#!/bin/bash
#SBATCH --job-name=swissiso_reward
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --requeue

# Score every arm output with Skywork-Reward-V2-Llama-3.1-8B.
#
# A6000, not A100, for two reasons: the model is 15 GB in bf16 so one 48 GB card
# is ample, and the 2-GPU cap is per-partition -- so this runs alongside the
# Prometheus pairwise job holding both A100s rather than queueing behind it.
#
# Why a reward model at all: Prometheus's verdict token is saturated (P(A) in
# {0, 1} on 33 of 33 readable gate probes) and 75.8% slot-A biased on
# byte-identical text. A sequence-classification head scores each response alone,
# so there is no slot for a preference to attach to and the output is natively a
# scalar. See src/swissisoform/judge/reward.py.
#
# 3,150 forward passes (9 arms x 350 cells), one per arm output, at 12-30k tokens
# each. Resumable: reward_scores.jsonl is append-only and the driver skips
# (slug, unit, arm) keys already in it.
#
# Usage:
#   sbatch scripts/slurm/run_reward.sbatch                     # all 3,150
#   sbatch scripts/slurm/run_reward.sbatch --limit 20          # smoke test

set -euo pipefail

ROOT=/lab/barcheese01/ating/swissisoform-v2
cd "$ROOT"

# Weights and datasets off /home -- it has a 48 GB quota that a 15 GB model plus
# caches has already overrun once.
export HF_HOME="$ROOT/.torch_cache/hf"
export TRANSFORMERS_CACHE="$HF_HOME"
export TMPDIR="$ROOT/.cache/tmp"
mkdir -p "$TMPDIR"

eval "$(conda shell.bash hook)"
conda activate swissisoform-v2-judge

# conda's libstdc++ carries CXXABI_1.3.15, which conda's icu needs and no node's
# system libstdc++ provides. Without this the engine dies at construction while a
# bare `import torch` still succeeds.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/judge/run_reward.py "$@"
