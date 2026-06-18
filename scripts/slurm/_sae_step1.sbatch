#!/bin/bash
#SBATCH --job-name=ating_sae_step1
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Part-3 Step 1: backfill the layer-27 SAE-target embedding (embedding_sae)
# into data/cache/plm_esmc/*.npz, then verify the right layer was captured via
# the SAE's reconstruction loss (layer 27 must beat the final layer).

set -euo pipefail
cd /lab/barcheese01/ating/swissisoform-v2

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate swissisoform-v2

export TORCH_HOME="$PWD/.torch_cache"
export HF_HOME="$PWD/.torch_cache/hf"
export XDG_CACHE_HOME="$PWD/.torch_cache/xdg"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$XDG_CACHE_HOME"

echo "=== STEP 1a: backfill embedding_sae (layer 27) for data/cache/proteins.fa ==="
python -m swissisoform.plm.cli data/cache/proteins.fa --model-size 600m

echo "=== STEP 1b: reconstruction-loss verification (layer 27 vs final layer) ==="
python scripts/_verify_sae_layer27.py
