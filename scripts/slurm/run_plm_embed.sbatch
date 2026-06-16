#!/bin/bash
#SBATCH --job-name=ating_plm_embed
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Precompute ESM-C per-residue embeddings + masked-marginal LLR for a
# FASTA of proteins. Caches per-protein .npz under data/cache/plm_esmc/
# keyed by sha1(stop-stripped, uppercased seq).
#
# Usage:
#   sbatch scripts/slurm/run_plm_embed.sbatch <fasta_path> [300m|600m|6b]
# Or, with defaults = E2E proteins from data/cache/proteins.fa, model 600m:
#   sbatch scripts/slurm/run_plm_embed.sbatch

set -euo pipefail

cd /lab/barcheese01/ating/swissisoform-v2

# Initialize conda — batch shells don't source ~/.bashrc, so conda isn't on PATH.
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    eval "$(conda shell.bash hook)"
fi
# Prefer a dedicated fold/plm env if present; else the main project env.
if conda env list | grep -q "^swissisoform-v2-fold "; then
    conda activate swissisoform-v2-fold
else
    conda activate swissisoform-v2
fi

FASTA="${1:-data/cache/proteins.fa}"
MODEL_SIZE="${2:-600m}"
if [[ ! -f "$FASTA" ]]; then
    echo "FASTA not found: $FASTA  (run: python scripts/run.py <mode> --emit-fasta)" >&2
    exit 1
fi

# Cache torch/HF weights inside the repo (HOME quota is small on shared HPC).
export TORCH_HOME="$PWD/.torch_cache"
export HF_HOME="$PWD/.torch_cache/hf"
export XDG_CACHE_HOME="$PWD/.torch_cache/xdg"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$XDG_CACHE_HOME"

python -m swissisoform.plm.cli "$FASTA" --model-size "$MODEL_SIZE"
