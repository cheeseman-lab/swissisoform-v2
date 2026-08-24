#!/bin/bash
#SBATCH --job-name=swissiso_full_fold
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Phase B (fold) of the full genome-wide run: one array task per protein chunk.
# ESMFold2 structures cached under data/cache/structure/esmfold2/<hash>/. Cache
# skips already-folded proteins, so re-submitting this array resumes. Mirrors
# scripts/slurm/run_fold.sbatch but consumes chunks/chunk_${SLURM_ARRAY_TASK_ID}.fa.
#
# Submitted (with the right --array size) by 00_prepare.sbatch. To resume, just
# resubmit the same array — the hash cache skips finished proteins:
#   sbatch --partition=nvidia-A6000-20 --export=ALL,SWISSISO_CAMPAIGN=<campaign> \
#          --array=0-<C-1>%3 <this script>
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

eval "$(conda shell.bash hook)"
# Pick the env by what it can import, not by name — an env that exists but lacks
# ESMFold2 would otherwise be activated and fail mid-array.
source "$(dirname "$(readlink -f "$0")")/activate_gpu_env.sh"
activate_gpu_env "fold" "swissisoform-v2-fold,swissisoform-v2" esm.models.esmfold2

export TORCH_HOME="$PWD/.torch_cache"
export HF_HOME="$PWD/.torch_cache/hf"
export XDG_CACHE_HOME="$PWD/.torch_cache/xdg"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$XDG_CACHE_HOME"

CAMPAIGN="${SWISSISO_CAMPAIGN:?SWISSISO_CAMPAIGN is not set — resubmit with \
    sbatch --export=ALL,SWISSISO_CAMPAIGN=<campaign> ... (see 00_prepare.sbatch)}"
CHUNK="data/output/${CAMPAIGN}/chunks/chunk_${SLURM_ARRAY_TASK_ID}.fa"
[[ -f "$CHUNK" ]] || { echo "chunk not found: $CHUNK" >&2; exit 1; }

echo "[$(date -u +%FT%TZ)] fold chunk ${SLURM_ARRAY_TASK_ID}: $(grep -c '^>' "$CHUNK") proteins"
python -m swissisoform.structure.cli "$CHUNK" --backend esmfold2
echo "[$(date -u +%FT%TZ)] fold chunk ${SLURM_ARRAY_TASK_ID} done"
