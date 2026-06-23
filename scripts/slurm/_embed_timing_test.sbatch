#!/bin/bash
#SBATCH --job-name=ating_embed_timing
#SBATCH --partition=nvidia-A100-20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# TIMING TEST ONLY — re-embed the 37 cheeseman proteins with ESM-C 6B to measure
# wall time, WITHOUT touching the production cache (data/cache/plm_esmc/6B/).
# Writes throwaway .npz into esm_swissisoform_doc/embed_timing_test/6B/.
#
# Usage: sbatch scripts/slurm/_embed_timing_test.sbatch [300m|600m|6b] [full|embedonly]
#   full      = embeddings + SAE + masked-marginal LLR (default; matches prod)
#   embedonly = single forward (embeddings + SAE only, --no-require-aa-logprobs)

set -euo pipefail
cd /lab/barcheese01/ating/swissisoform-v2

if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
else
    eval "$(conda shell.bash hook)"
fi
conda activate swissisoform-v2

MODEL_SIZE="${1:-6b}"
MODE="${2:-full}"
FASTA="data/cache/proteins.fa"
TEST_CACHE="/lab/barcheese01/ating/esm_swissisoform_doc/embed_timing_test/${MODEL_SIZE}_${MODE}"
mkdir -p "$TEST_CACHE"

MODE_ARGS=()
if [[ "$MODE" == "embedonly" ]]; then
    MODE_ARGS+=(--no-llr)   # genuinely skips Pass 2 (single forward/protein)
fi

export TORCH_HOME="$PWD/.torch_cache"
export HF_HOME="$PWD/.torch_cache/hf"
export XDG_CACHE_HOME="$PWD/.torch_cache/xdg"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$XDG_CACHE_HOME"

EXTRA_ARGS=()
LOCAL_6B="/lab/barcheese01/ating/esm_swissisoform_doc/upstream/hf-ESMC-6B"
if [[ "$MODEL_SIZE" == "6b" && -d "$LOCAL_6B" ]]; then
    EXTRA_ARGS+=(--model-id "$LOCAL_6B")
fi

echo "=== embed timing test: size=$MODEL_SIZE  mode=$MODE  fasta=$FASTA  cache=$TEST_CACHE ==="
echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "T_START=$(date +%s)  $(date)"

/usr/bin/time -v python -m swissisoform.plm.cli "$FASTA" \
    --model-size "$MODEL_SIZE" --cache-dir "$TEST_CACHE" \
    "${MODE_ARGS[@]+"${MODE_ARGS[@]}"}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

echo "T_END=$(date +%s)  $(date)"
echo "=== done; test embeddings in $TEST_CACHE (safe to delete) ==="
