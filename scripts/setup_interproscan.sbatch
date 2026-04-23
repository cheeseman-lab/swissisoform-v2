#!/bin/bash
#SBATCH --job-name=mdiberna_interproscan_setup
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /lab/barcheese01/mdiberna/swissisoform-v2

eval "$(conda shell.bash hook)"
conda activate swissisoform-v2

# Redirect caches into the repo to keep $HOME clean
export NXF_HOME="$PWD/.nextflow"
export NXF_SINGULARITY_CACHEDIR="$PWD/.singularity_cache"
mkdir -p "$NXF_HOME" "$NXF_SINGULARITY_CACHEDIR"

python scripts/setup_databases.py interproscan
