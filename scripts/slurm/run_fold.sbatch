#!/bin/bash
#SBATCH --job-name=mdiberna_fold
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Precompute Boltz-2 (default) or Chai-1 structures for a FASTA of
# proteins. Caches per-protein under
# data/cache/structure/<backend>/<hash>/{model.cif, confidence.json, metrics.json}.
#
# Usage:
#   sbatch scripts/slurm/run_fold.sbatch <fasta_path> [boltz|chai]
# Defaults: data/cache/proteins.fa, backend=boltz.

set -euo pipefail

cd /lab/barcheese01/mdiberna/swissisoform-v2

eval "$(conda shell.bash hook)"
if conda env list | grep -q "^swissisoform-v2-fold "; then
    conda activate swissisoform-v2-fold
else
    echo "swissisoform-v2-fold env missing — create with:" >&2
    echo "  conda create -n swissisoform-v2-fold -c conda-forge python=3.12 uv pip -y" >&2
    echo "  conda activate swissisoform-v2-fold" >&2
    echo "  uv pip install -e '.[fold]'" >&2
    exit 2
fi

FASTA="${1:-data/cache/proteins.fa}"
BACKEND="${2:-boltz}"
if [[ ! -f "$FASTA" ]]; then
    echo "FASTA not found: $FASTA  (run: python scripts/run.py <mode> --emit-fasta)" >&2
    exit 1
fi

# Cache torch/HF weights inside the repo (HOME quota is small on shared HPC).
export TORCH_HOME="$PWD/.torch_cache"
export HF_HOME="$PWD/.torch_cache/hf"
export XDG_CACHE_HOME="$PWD/.torch_cache/xdg"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$XDG_CACHE_HOME"

python -m swissisoform.structure.cli "$FASTA" --backend "$BACKEND"
