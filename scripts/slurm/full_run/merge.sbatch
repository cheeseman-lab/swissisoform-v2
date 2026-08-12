#!/bin/bash
#SBATCH --job-name=swissiso_full_merge
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=256G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Phase D of the full genome-wide run: concatenate the per-shard paired parquets
# into data/output/full_catalog/all_paired.parquet. Depends (afterany) on the
# annotate array, so it runs once every shard has reached a terminal state; a
# partial merge is fine and re-runnable once missing shards are resubmitted.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
eval "$(conda shell.bash hook)"
conda activate swissisoform-v2

echo "[$(date -u +%FT%TZ)] merging shard outputs"
python scripts/slurm/full_run/merge.py
echo "[$(date -u +%FT%TZ)] merge done. Build the site with: RUN=full_catalog bash scripts/build_website.sh"
