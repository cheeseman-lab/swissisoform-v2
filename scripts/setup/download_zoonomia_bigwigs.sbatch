#!/bin/bash
#SBATCH --job-name=mdiberna_zoonomia_bigwigs
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --output=logs/zoonomia_bigwigs_%j.out
#SBATCH --error=logs/zoonomia_bigwigs_%j.err

# Slurm wrapper around scripts/setup/download_zoonomia_bigwigs.sh. Combined
# ~13 GB, typically finishes in a handful of minutes, but running
# through Slurm keeps the head node free and matches the HAL download
# pattern.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting BigWig download on $(hostname)"
bash scripts/setup/download_zoonomia_bigwigs.sh
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done."
du -h data/reference/zoonomia/*.bw 2>/dev/null || true
