#!/bin/bash
#SBATCH --job-name=mdiberna_small_e2e
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /lab/barcheese01/mdiberna/swissisoform-v2

eval "$(conda shell.bash hook)"
conda activate swissisoform-v2

export NXF_HOME="$PWD/.nextflow"
export NXF_SINGULARITY_CACHEDIR="$PWD/.singularity_cache"

python scripts/inspect_e2e.py
