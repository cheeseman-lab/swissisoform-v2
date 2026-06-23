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

# Precompute ESM-C per-residue embeddings + masked-marginal LLR for a
# FASTA of proteins. Caches per-protein .npz under data/cache/plm_esmc/<SIZE>/
# (e.g. 6B/, 600M/) keyed by sha1(stop-stripped, uppercased seq).
#
# Then runs the SAE encode in the SAME job: the cheap Top-K autoencoder forward
# that turns the just-written `embedding_sae` layer into sparse features under
# data/cache/sae_esmc/<SIZE>/, so a fresh run lands SAE features in the cache for
# SAEFeatureModule (annotate stage) without a separate GPU job. Skip it by
# exporting SWISSISO_SKIP_SAE=1 (run.sbatch does this on `--skip-modules sae`).
#
# Usage:
#   sbatch scripts/slurm/run_plm_embed.sbatch <fasta_path> [300m|600m|6b]
# Or, with defaults = E2E proteins from data/cache/proteins.fa, model 6b:
#   sbatch scripts/slurm/run_plm_embed.sbatch

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"

eval "$(conda shell.bash hook)"
if conda env list | grep -q "^swissisoform-v2-plm "; then
    conda activate swissisoform-v2-plm
else
    echo "No swissisoform-v2[-plm|-fold] env found. Create one with:" >&2
    echo "  conda create -n swissisoform-v2-plm -c conda-forge python=3.12 uv pip -y" >&2
    echo "  conda activate swissisoform-v2-plm && uv pip install -e '.[plm]'" >&2
    exit 2
fi

FASTA="${1:-data/cache/proteins.fa}"
MODEL_SIZE="${2:-6b}"
if [[ ! -f "$FASTA" ]]; then
    echo "FASTA not found: $FASTA  (run: python scripts/run.py <mode> --emit-fasta)" >&2
    exit 1
fi

# Cache torch/HF weights inside the repo (HOME quota is small on shared HPC).
export TORCH_HOME="$PWD/.torch_cache"
export HF_HOME="$PWD/.torch_cache/hf"
export XDG_CACHE_HOME="$PWD/.torch_cache/xdg"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$XDG_CACHE_HOME"

# For 6B, prefer the local clone of the weights (~28 GB already on disk) so the
# compute node doesn't re-download from HF (and runs offline). Falls back to the
# HF repo id baked into ESMC_MODEL_IDS if the clone isn't present.
EXTRA_ARGS=()
LOCAL_6B="/lab/barcheese01/ating/esm_swissisoform_doc/upstream/hf-ESMC-6B"
if [[ "$MODEL_SIZE" == "6b" && -d "$LOCAL_6B" ]]; then
    EXTRA_ARGS+=(--model-id "$LOCAL_6B")
fi

python -m swissisoform.plm.cli "$FASTA" --model-size "$MODEL_SIZE" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

# ── SAE encode (rides this job): reads the embedding_sae layer just written to
#    plm_esmc/<SIZE>/ and writes sparse Top-K features to sae_esmc/<SIZE>/. Cheap
#    (one linear layer + Top-K) — no ESM-C inference, so NO --model-id here. Only
#    pulls the small (~321 MB) SAE codebook from HF on first use. A failure aborts
#    this job loudly; embeddings are already cached, so a re-run cache-hits the
#    expensive embed and just redoes the encode.
if [[ "${SWISSISO_SKIP_SAE:-0}" == "1" ]]; then
    echo "=== SAE encode SKIPPED (SWISSISO_SKIP_SAE=1) ==="
else
    echo "=== SAE encode ($MODEL_SIZE) -> sae_esmc/${MODEL_SIZE^^}/ ==="
    python -m swissisoform.plm.sae "$FASTA" --model-size "$MODEL_SIZE" --device cuda
fi
