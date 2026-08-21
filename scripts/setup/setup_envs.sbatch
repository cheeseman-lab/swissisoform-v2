#!/bin/bash
#SBATCH --job-name=swissiso_envsetup
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Build the dedicated GPU conda environments from the pyproject extras via the
# tracked builder scripts/setup/create_envs.sh:
#   swissisoform-v2-plm   (.[plm])  — torch + esm (Biohub transformers fork) + xformers
#                                     (ESM-C embed / SAE encode)
#   swissisoform-v2-fold  (.[fold]) — Boltz-2 / Chai-1 (ESMFold2 structures)
#
# flash-attn is not built here (from-source compile, needs a GPU node); add it with:
#   bash scripts/setup/create_envs.sh --plm --flash   # on a GPU node
#
# Usage:  mkdir -p logs; sbatch scripts/setup/setup_envs.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

echo "[$(date -u +%FT%TZ)] building swissisoform-v2-plm + -fold from pyproject extras"
bash scripts/setup/create_envs.sh --plm --fold

echo "[$(date -u +%FT%TZ)] env setup done: swissisoform-v2-plm (embed) + -fold (structures) ready."
echo "  Add flash-attn: bash scripts/setup/create_envs.sh --plm --flash   # on a GPU node"
