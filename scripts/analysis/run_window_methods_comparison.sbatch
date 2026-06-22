#!/bin/bash
# Unambiguous-sequence counts across all methods (sequence-only, salmon TPM>=1,
# salmon TPM>=0.1, and the union at each) and window radii W=5..500. Heavy step:
# load_exon_skeletons over the 3 GB GTF -> Slurm.
#SBATCH --job-name=eblack_methods_vs_window
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

python scripts/analysis/window_methods_comparison.py \
  --cell-line HeLa \
  --thresholds 1.0 0.1 \
  --isoquant-table data/reference/longread/isoquant_HeLa/OUT/OUT.transcript_counts.tsv \
  --isoquant-min-counts 3 \
  --max-w 500 \
  --out-dir data/output/source_transcripts

echo "Done. See data/output/source_transcripts/HeLa_methods_vs_window.{csv,png}"
