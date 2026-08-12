#!/bin/bash
#SBATCH --job-name=swissiso_pepquery_spectra
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# One-time, contract-legal reference provisioning: mirror the two PepQueryDB
# tissue datasets the runner searches (-b) from public S3 to a local store
# (data/reference/pepquery/spectra/, ~196 GiB) so massspec can search them via
# -ms with ZERO per-search download. Removes the 117x re-download the sharded
# annotate array would otherwise incur with -b. Anonymous S3 (--no-sign-request,
# handled inside setup_pepquery_spectra); idempotent + resumable (aws s3 sync
# transfers only missing/changed objects, so a requeue continues where it left
# off).
#
# Usage:  mkdir -p logs; sbatch scripts/setup/setup_pepquery_spectra.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate swissisoform-v2

echo "[$(date -u +%FT%TZ)] staging PepQueryDB spectra -> data/reference/pepquery/spectra/"
python scripts/setup/setup_databases.py pepquery-spectra
echo "[$(date -u +%FT%TZ)] done"
du -sh data/reference/pepquery/spectra/ 2>/dev/null || true
