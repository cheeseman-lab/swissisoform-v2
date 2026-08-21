#!/bin/bash
#SBATCH --job-name=swissiso_full_embed
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Phase B (embed) of the full genome-wide run: one array task per protein chunk.
# ESM-C embeddings + masked-marginal LLR (plm_esmc/6B/) + the SAE encode
# (sae_esmc/6B/), keyed by protein hash. Caches skip already-done proteins, so
# re-submitting this array resumes — a task re-runs its chunk but cache-hits
# whatever's already computed. Mirrors scripts/slurm/run_plm_embed.sbatch, but
# consumes chunks/chunk_${SLURM_ARRAY_TASK_ID}.fa instead of the whole FASTA.
#
# Submitted (with the right --array size) by 00_prepare.sbatch. To resume, just
# resubmit the same array — the hash cache skips finished proteins:
#   sbatch --partition=nvidia-A6000-20 --export=ALL,SWISSISO_CAMPAIGN=<campaign> \
#          --array=0-<C-1>%3 <this script>
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

eval "$(conda shell.bash hook)"
# Pick the env by what it can import, not by name: -plm still holds the legacy
# fair-esm stack from before the ESM-C migration and cannot run this workload.
# The SAE encode below rides the same env, so require its module too unless the
# encode is skipped. Fails with a pointer to the setup job if nothing qualifies.
source "$(dirname "$(readlink -f "$0")")/activate_gpu_env.sh"
EMBED_MODULES=(transformers.models.esmc.modeling_esmc)
[[ "${SWISSISO_SKIP_SAE:-0}" == "1" ]] || EMBED_MODULES+=(transformers.models.esmc.modeling_esmc_sae)
activate_gpu_env "embed" "swissisoform-v2-plm,swissisoform-v2-fold,swissisoform-v2" \
    "${EMBED_MODULES[@]}"

export TORCH_HOME="$PWD/.torch_cache"
export HF_HOME="$PWD/.torch_cache/hf"
export XDG_CACHE_HOME="$PWD/.torch_cache/xdg"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$XDG_CACHE_HOME"

CAMPAIGN="${SWISSISO_CAMPAIGN:?SWISSISO_CAMPAIGN is not set — resubmit with \
    sbatch --export=ALL,SWISSISO_CAMPAIGN=<campaign> ... (see 00_prepare.sbatch)}"
CHUNK="data/output/${CAMPAIGN}/chunks/chunk_${SLURM_ARRAY_TASK_ID}.fa"
[[ -f "$CHUNK" ]] || { echo "chunk not found: $CHUNK" >&2; exit 1; }

MODEL_SIZE=6b
EXTRA_ARGS=()
LOCAL_6B="/lab/barcheese01/ating/esm_swissisoform_doc/upstream/hf-ESMC-6B"
if [[ -d "$LOCAL_6B" ]]; then
    EXTRA_ARGS+=(--model-id "$LOCAL_6B")
fi

echo "[$(date -u +%FT%TZ)] embed chunk ${SLURM_ARRAY_TASK_ID}: $(grep -c '^>' "$CHUNK") proteins"
python -m swissisoform.plm.cli "$CHUNK" --model-size "$MODEL_SIZE" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

if [[ "${SWISSISO_SKIP_SAE:-0}" == "1" ]]; then
    echo "=== SAE encode SKIPPED (SWISSISO_SKIP_SAE=1) ==="
else
    echo "=== SAE encode ($MODEL_SIZE) -> sae_esmc/${MODEL_SIZE^^}/ ==="
    python -m swissisoform.plm.sae "$CHUNK" --model-size "$MODEL_SIZE" --device cuda
fi
echo "[$(date -u +%FT%TZ)] embed chunk ${SLURM_ARRAY_TASK_ID} done"
