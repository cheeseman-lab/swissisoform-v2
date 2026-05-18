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

# Force Nextflow to use the conda env's Java 17 (system default is Java 11)
export JAVA_HOME="$CONDA_PREFIX"
export JAVA_CMD="$CONDA_PREFIX/bin/java"
echo "Java: $($JAVA_CMD -version 2>&1 | head -1)"

export NXF_HOME="$PWD/.nextflow"
export NXF_SINGULARITY_CACHEDIR="$PWD/.singularity_cache"

python scripts/inspect_e2e.py

