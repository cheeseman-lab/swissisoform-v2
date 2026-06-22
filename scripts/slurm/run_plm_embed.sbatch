#!/bin/bash
#SBATCH --job-name=swissiso_plm_embed
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Precompute ESM-2 650M per-residue embeddings + masked-marginal LLR
# for a FASTA of proteins. Caches per-protein .npz under
# data/cache/plm_esm2/ keyed by sha1(stop-stripped, uppercased seq).
#
# Usage:
#   sbatch scripts/slurm/run_plm_embed.sbatch <fasta_path>
# Or, with default = E2E proteins from data/cache/proteins.fa:
#   sbatch scripts/slurm/run_plm_embed.sbatch

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"

eval "$(conda shell.bash hook)"
if conda env list | grep -q "^swissisoform-v2-plm "; then
    conda activate swissisoform-v2-plm
else
    echo "swissisoform-v2-plm env missing — create with:" >&2
    echo "  conda create -n swissisoform-v2-plm -c conda-forge python=3.12 uv pip -y" >&2
    echo "  conda activate swissisoform-v2-plm" >&2
    echo "  uv pip install -e '.[plm]'" >&2
    exit 2
fi

FASTA="${1:-data/cache/proteins.fa}"
if [[ ! -f "$FASTA" ]]; then
    echo "FASTA not found: $FASTA  (run: python scripts/run.py <mode> --emit-fasta)" >&2
    exit 1
fi

# Cache torch/HF weights inside the repo (HOME quota is small on shared HPC).
export TORCH_HOME="$PWD/.torch_cache"
export HF_HOME="$PWD/.torch_cache/hf"
export XDG_CACHE_HOME="$PWD/.torch_cache/xdg"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$XDG_CACHE_HOME"

python -m swissisoform.plm.cli "$FASTA"
