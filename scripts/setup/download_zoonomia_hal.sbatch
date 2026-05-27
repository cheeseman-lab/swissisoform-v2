#!/bin/bash
#SBATCH --job-name=mdiberna_zoonomia_hal
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/zoonomia_hal_%j.out
#SBATCH --error=logs/zoonomia_hal_%j.err

# Slurm wrapper around scripts/setup/download_zoonomia_hal.sh. The HAL is
# 200-600 GB; a single curl can take many hours, so we run it inside a
# long-lived Slurm job rather than a head-node shell that would die on
# logout.
#
# Usage (from repo root):
#   mkdir -p logs
#   sbatch scripts/setup/download_zoonomia_hal.sbatch
#
# Optional first arg overrides the destination directory. Defaults to the
# shell script's own default (data/reference/zoonomia/).

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

DEST_DIR="${1:-data/reference/zoonomia}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting Zoonomia HAL download on $(hostname)"
echo "                            → $DEST_DIR"

bash scripts/setup/download_zoonomia_hal.sh "$DEST_DIR"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] finished. Size:"
du -h "$DEST_DIR"/*.hal 2>/dev/null || true
