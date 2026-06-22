#!/bin/bash
# Sweep start-codon window radius (5..500 nt) and measure how it changes
# candidate-transcript purity. Heavy step = load_exon_skeletons over the 3 GB
# GTF, so run on Slurm. Produces a per-TIS radius parquet, a curve CSV, and a
# PNG figure under data/output/source_transcripts/.
#SBATCH --job-name=eblack_purity_vs_window
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

source /lab/barcheese01/${USER}/miniforge3/etc/profile.d/conda.sh
conda activate swissisoform-v2

REF=/lab/barcheese01/${USER}/swissisoform-v2/data/reference
[ -s "${REF}/Gencode_v49_GRCh38.primary_assembly.genome.fa.fai" ] || \
  samtools faidx "${REF}/Gencode_v49_GRCh38.primary_assembly.genome.fa"

python scripts/analysis/window_purity_vs_length.py \
  --cell-line HeLa \
  --min-tpm 0.1 \
  --min-w 5 --max-w 500 \
  --out-dir data/output/source_transcripts

echo "Done. See data/output/source_transcripts/HeLa_purity_vs_window.png"
