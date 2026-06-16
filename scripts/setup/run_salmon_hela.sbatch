#!/bin/bash
# Short-read transcript existence for HeLa: build a decoy-aware GENCODE v49
# salmon index (once) and quantify the two HeLa Illumina replicates.
# Output quant.sf -> consumed by swissisoform.sourceseq.expression.load_salmon_*.
#SBATCH --job-name=eblack_salmon_hela
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

source /lab/barcheese01/${USER}/miniforge3/etc/profile.d/conda.sh
conda activate swissisoform-v2

ROOT=/lab/barcheese01/${USER}/swissisoform-v2
REF=${ROOT}/data/reference
GENOME=${REF}/Gencode_v49_GRCh38.primary_assembly.genome.fa
TX=${REF}/gencode.v49.transcripts.fa            # transcriptome FASTA (download via download_references.sh)
SALMON=${REF}/salmon
IDX=${SALMON}/idx_v49

# --- 1) build decoy-aware index once ---
if [ ! -d "${IDX}" ]; then
  mkdir -p "${SALMON}"
  cat "${TX}" "${GENOME}" > "${SALMON}/gentrome.fa"                       # transcripts + genome decoy
  grep '^>' "${GENOME}" | sed 's/^>//; s/ .*//' > "${SALMON}/decoys.txt" # decoy = genome contig names
  salmon index -t "${SALMON}/gentrome.fa" -d "${SALMON}/decoys.txt" \
    -i "${IDX}" --keepDuplicates -k 31 -p "${SLURM_CPUS_PER_TASK}"
fi

# --- 2) per replicate: QC/trim with fastp, then quantify the cleaned reads ---
declare -A REPS=(
  [HeLa_rep1]=/lab/barcheese01/aTIS_data/rnaseq/BVTGPN_3_HeLa_1.fastq.gz
  [HeLa_rep2]=/lab/barcheese01/aTIS_data/rnaseq/BVTGPN_21_HeLa_2.fastq.gz
)
QC=${REF}/fastp/HeLa
mkdir -p "${QC}"
for name in "${!REPS[@]}"; do
  RAW="${REPS[$name]}"
  TRIM="${QC}/${name}.trimmed.fastq.gz"

  # Standard short-read pre-processing (single-end): fastp does, by default,
  # adapter auto-detection + trimming, quality filtering (Q15), polyG/polyX tail
  # trim, and drops reads failing quality; we also require >=20 nt post-trim.
  # JSON/HTML are the per-sample QC reports (before/after metrics).
  fastp \
    -i "${RAW}" -o "${TRIM}" \
    --length_required 20 \
    --json "${QC}/${name}.fastp.json" \
    --html "${QC}/${name}.fastp.html" \
    --thread "${SLURM_CPUS_PER_TASK}"

  # Quantify the cleaned reads.
  salmon quant -i "${IDX}" -l A \
    -r "${TRIM}" \
    --fldMean 250 --fldSD 25 \
    --seqBias --rangeFactorizationBins 4 --numBootstraps 30 \
    -p "${SLURM_CPUS_PER_TASK}" -o "${SALMON}/${name}"
done

echo "Done. fastp QC under ${QC}/ ; quant.sf at ${SALMON}/HeLa_rep1 and ${SALMON}/HeLa_rep2"
