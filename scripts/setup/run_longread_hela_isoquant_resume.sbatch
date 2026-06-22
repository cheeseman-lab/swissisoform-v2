#!/bin/bash
# Resume the HeLa long-read run that hit the 12h TIMEOUT mid-quantification.
# The expensive upstream work is already on disk and reused:
#   - data/reference/longread/HeLa/HeLa.bam        (minimap2 splice-aligned + indexed, 95.9% mapped)
#   - data/reference/longread/isoquant_HeLa/*.db   (prebuilt GENCODE v49 gene database)
# IsoQuant --resume continues from the output folder, reusing the BAM + DB and
# skipping align/DB-build entirely -> only the per-chromosome quantification runs.
# Bigger walltime (quant alone, generous headroom) and more RAM (the prior run
# peaked at 67G against a 64G ceiling).
#SBATCH --job-name=eblack_longread_hela_iq
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

source /lab/barcheese01/${USER}/miniforge3/etc/profile.d/conda.sh
conda activate isoquant

ROOT=/lab/barcheese01/${USER}/swissisoform-v2
REF=${ROOT}/data/reference
IQ_OUT=${REF}/longread/isoquant_HeLa

# Safety: the resume requires the completed alignment + the started IsoQuant dir.
test -s "${REF}/longread/HeLa/HeLa.bam"     || { echo "FATAL: HeLa.bam missing"; exit 1; }
test -s "${REF}/longread/HeLa/HeLa.bam.bai" || { echo "FATAL: HeLa.bam.bai missing"; exit 1; }
test -d "${IQ_OUT}"                         || { echo "FATAL: isoquant_HeLa output dir missing"; exit 1; }

isoquant --resume -o "${IQ_OUT}" --threads "${SLURM_CPUS_PER_TASK}"

echo "Done. IsoQuant transcript table under ${IQ_OUT}/OUT/"
