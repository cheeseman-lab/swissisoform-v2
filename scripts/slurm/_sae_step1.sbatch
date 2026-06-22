#!/bin/bash
#SBATCH --job-name=ating_sae_6b
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ESM-C SAE pipeline for a FASTA, one model size end-to-end:
#   1. embed   — ESM-C forward → plm_esmc/<SIZE>/ (llr, aa_logprobs, embedding,
#                embedding_sae = the SAE-target layer; 6B→60, 600M→27)
#   2. verify  — SAE recon loss: the SAE-target layer must beat the final layer
#   3. encode  — SAE Top-K features → sae_esmc/<SIZE>/
#   4. terms   — whole-protein top-5 functional terms CSV (Atlas labels)
#
# Usage:
#   sbatch scripts/slurm/_sae_step1.sbatch <fasta_path> [600m|6b]
# Defaults: data/cache/proteins.fa, model 6b.

set -euo pipefail
cd /lab/barcheese01/ating/swissisoform-v2

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate swissisoform-v2

export TORCH_HOME="$PWD/.torch_cache"
export HF_HOME="$PWD/.torch_cache/hf"
export XDG_CACHE_HOME="$PWD/.torch_cache/xdg"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$XDG_CACHE_HOME"

FASTA="${1:-data/cache/proteins.fa}"
MODEL_SIZE="${2:-6b}"

# For 6B, prefer the local clone of the model weights (~28 GB) to skip a
# re-download on the node. The SAE itself is small (~321 MB) and pulled from HF.
EXTRA_ARGS=()
LOCAL_6B="/lab/barcheese01/ating/esm_swissisoform_doc/upstream/hf-ESMC-6B"
if [[ "$MODEL_SIZE" == "6b" && -d "$LOCAL_6B" ]]; then
    EXTRA_ARGS+=(--model-id "$LOCAL_6B")
fi

echo "=== STEP 1: embed ($MODEL_SIZE) -> plm_esmc/${MODEL_SIZE^^}/ ==="
python -m swissisoform.plm.cli "$FASTA" --model-size "$MODEL_SIZE" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

echo "=== STEP 2: verify SAE-target layer (recon loss vs final layer) ==="
python scripts/_verify_sae_layer.py --model-size "$MODEL_SIZE"

echo "=== STEP 3: SAE encode ($MODEL_SIZE) -> sae_esmc/${MODEL_SIZE^^}/ ==="
python -m swissisoform.plm.sae "$FASTA" --model-size "$MODEL_SIZE" --device cuda

echo "=== STEP 4: top-5 functional terms CSV ==="
python scripts/_sae_top_terms.py
