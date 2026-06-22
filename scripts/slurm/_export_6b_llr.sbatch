#!/bin/bash
#SBATCH --job-name=ating_export_6b_llr
#SBATCH --partition=24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Regenerate the ESM-C ΔLLR exports off the 6B cache (CPU; no model inference):
#   1. export_plm_delta_llr     — constraint ΔLLR from `llr`        → plm_vep_delta_llr.csv
#   2. export_variant_delta_llr — per-variant ΔLLR from aa_logprobs → vep_* cols + clinical_variants_esmc.parquet
# Both read data/cache/plm_esmc/6B (the default now). The variant export's
# comparison twins (vep_*_esm2) are read from the true ESM-2 baseline parquet.

set -euo pipefail
cd /lab/barcheese01/ating/swissisoform-v2

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate swissisoform-v2

PRESET="${1:-cheeseman13}"
ESM2_BASELINE="data/output/cheeseman_13gene/all_paired_esm2boltz.parquet"

echo "=== STEP 1: constraint ΔLLR (6B) → plm_vep_delta_llr.csv ==="
python scripts/export/export_plm_delta_llr.py --preset "$PRESET"

echo "=== STEP 2: per-variant ΔLLR / probabilities (6B) → vep_* + clinical_variants_esmc.parquet ==="
python scripts/export/export_variant_delta_llr.py --preset "$PRESET" \
    --baseline-parquet "$ESM2_BASELINE"
