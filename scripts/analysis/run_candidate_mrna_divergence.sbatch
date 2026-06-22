#!/bin/bash
# Pull candidate-transcript mRNAs per HeLa TIS and test start-codon-window
# divergence. The heavy step is load_exon_skeletons over the 3 GB GENCODE v49
# GTF (~a few min, ~GB RAM) -> run on Slurm, not the head node.
#SBATCH --job-name=eblack_mrna_divergence
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

ROOT=/lab/barcheese01/${USER}/swissisoform-v2
REF=${ROOT}/data/reference

# pysam needs .fai for random access on both FASTAs (idempotent)
[ -s "${REF}/gencode.v49.transcripts.fa.fai" ] || samtools faidx "${REF}/gencode.v49.transcripts.fa"
[ -s "${REF}/Gencode_v49_GRCh38.primary_assembly.genome.fa.fai" ] || \
  samtools faidx "${REF}/Gencode_v49_GRCh38.primary_assembly.genome.fa"

python scripts/analysis/candidate_mrna_divergence.py \
  --cell-line HeLa \
  --min-candidates 2 \
  -W 100 \
  --out-dir data/output/source_transcripts

echo "Done. See data/output/source_transcripts/HeLa_tis_window_summary.parquet"
