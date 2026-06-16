#!/bin/bash
# Long-read transcript existence for HeLa: pull public ONT reads (GSE277764),
# align splice-aware with minimap2, and quantify isoforms with IsoQuant vs
# GENCODE v49. Output transcript table -> consumed by
# swissisoform.sourceseq.expression.load_isoquant_abundance.
#
# Accessions come from data/reference/longread_manifest.csv (cell_line == HeLa).
# Library: ONT polyA+ cDNA (confirmed from GSE277764 / Cell 2025, PMID 40436014) →
# minimap2 `-ax splice` WITHOUT `-uf`; annotated junctions guide mapping via --junc-bed.
#SBATCH --job-name=eblack_longread_hela
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

source /lab/barcheese01/${USER}/miniforge3/etc/profile.d/conda.sh
conda activate swissisoform-v2          # sra-tools (prefetch/fasterq-dump) for the pull
# minimap2 + samtools + isoquant run in the dedicated `isoquant` env (activated
# below). Build it once: bash scripts/setup/create_conda_env.sh --with-isoquant

ROOT=/lab/barcheese01/${USER}/swissisoform-v2
REF=${ROOT}/data/reference
GENOME=${REF}/Gencode_v49_GRCh38.primary_assembly.genome.fa
GTF=${REF}/gencode.v49.primary_assembly.annotation.gtf
OUT=${REF}/longread/HeLa
mkdir -p "${OUT}/sra"

# --- 1) pull raw ONT reads for every HeLa accession in the manifest ---
ACCS=$(awk -F, '$1=="HeLa"{print $3}' "${REF}/longread_manifest.csv")   # SRX accessions
for SRX in ${ACCS}; do
  prefetch "${SRX}" -O "${OUT}/sra"                                     # download SRA archive
  fasterq-dump --threads "${SLURM_CPUS_PER_TASK}" "${OUT}/sra/${SRX}" -O "${OUT}"  # -> FASTQ
done
gzip -f "${OUT}"/*.fastq
cat "${OUT}"/*.fastq.gz > "${OUT}/HeLa_all.fastq.gz"                    # pool replicates for quant

# --- 2) splice-aware alignment (ONT cDNA) ---
conda activate isoquant                 # minimap2 + samtools + isoquant + paftools.js/k8

# Annotated GENCODE v49 junctions guide noisy ONT splice mapping (built once).
JUNCBED=${REF}/longread/gencode.v49.junctions.bed
[ -s "${JUNCBED}" ] || paftools.js gff2bed "${GTF}" > "${JUNCBED}"

minimap2 -ax splice -k14 --junc-bed "${JUNCBED}" -t "${SLURM_CPUS_PER_TASK}" \
  "${GENOME}" "${OUT}/HeLa_all.fastq.gz" \
  | samtools sort -@4 -o "${OUT}/HeLa.bam"
samtools index "${OUT}/HeLa.bam"

# --- 3) reference-ONLY isoform quantification vs GENCODE v49 (no novel discovery) ---
isoquant \
  --reference "${GENOME}" \
  --genedb "${GTF}" \
  --complete_genedb \
  --bam "${OUT}/HeLa.bam" \
  --data_type nanopore \
  --stranded none \
  --no_model_construction \
  --threads "${SLURM_CPUS_PER_TASK}" \
  -o "${REF}/longread/isoquant_HeLa"

echo "Done. IsoQuant transcript table under ${REF}/longread/isoquant_HeLa/"
