#!/bin/bash
#SBATCH --job-name=swissiso_full_annot
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Phase C of the full genome-wide run: one array task per gene shard. Runs the
# FULL annotation (all modules on; GPU modules read the warm caches from Phase B)
# over the shard's ~100 genes, writing data/output/${CAMPAIGN}_shard_<k>/.
# HeLa long-read source resolution is already baked into the cached combined
# catalog, so it carries through untouched (no --skip-source-resolution).
#
# Campaign-scoped: SWISSISO_CAMPAIGN must be set (00_prepare.sbatch passes it
# down; a manual resubmit needs --export=ALL,SWISSISO_CAMPAIGN=<campaign>). Every
# path below is namespaced by it, so two campaigns cannot overwrite or blend.
#
# Resumable: a task whose shard output already exists exits immediately, so
# re-submitting the array (or just the missing indices) only redoes unfinished
# shards. Per-shard tmp/Nextflow dirs keep concurrent InterProScan/PepQuery runs
# from colliding.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

eval "$(conda shell.bash hook)"
conda activate swissisoform-v2

CAMPAIGN="${SWISSISO_CAMPAIGN:?SWISSISO_CAMPAIGN is not set — resubmit with \
    sbatch --export=ALL,SWISSISO_CAMPAIGN=<campaign> ... (see 00_prepare.sbatch)}"
K="${SLURM_ARRAY_TASK_ID}"
SHARD="data/output/${CAMPAIGN}/shards/shard_${K}.txt"
RUN_NAME="${CAMPAIGN}_shard_${K}"
OUTDIR="data/output/${RUN_NAME}"
[[ -f "$SHARD" ]] || { echo "shard list not found: $SHARD" >&2; exit 1; }

# Resume: skip a shard that already produced a COMPLETE paired parquet. run.py
# renames this file into place last, after the per-gene slices, so its presence
# means the whole shard finished. The trailing PAR1 magic additionally catches
# files truncated by a pre-atomic-write run — without it a shard killed mid-write
# is skipped forever and the damage only surfaces at merge time.
if [[ -f "$OUTDIR/all_paired.parquet" ]]; then
    if [[ "$(tail -c 4 "$OUTDIR/all_paired.parquet" 2>/dev/null)" == "PAR1" ]]; then
        echo "[$(date -u +%FT%TZ)] shard $K already complete ($OUTDIR/all_paired.parquet) — skipping"
        exit 0
    fi
    echo "[$(date -u +%FT%TZ)] shard $K has a truncated all_paired.parquet — recomputing" >&2
    rm -f "$OUTDIR/all_paired.parquet" "$OUTDIR/shard_meta.json"
fi

# Per-shard isolation for the subprocess-heavy modules (InterProScan/Nextflow,
# PepQuery, HAL tmp). Singularity image cache is shared read-only.
export JAVA_HOME="$CONDA_PREFIX"
export JAVA_CMD="$CONDA_PREFIX/bin/java"
export NXF_HOME="$PWD/.nextflow_shards/${CAMPAIGN}_shard_${K}"
export NXF_SINGULARITY_CACHEDIR="$PWD/.singularity_cache"
export TMPDIR="$PWD/data/cache/tmp/${CAMPAIGN}_shard_${K}"
mkdir -p "$NXF_HOME" "$TMPDIR"

# NOTE: PepQuery already multithreads (its -cpu default = all available CPUs,
# which the JVM sizes from this job's grant), and the runtime search is `-b`
# (S3 PepQueryDB) download/IO-bound, not CPU-bound — so no -cpu pin is set. The
# real speedup is local spectra staging (setup_databases.py pepquery-spectra +
# -ms); CPU/sharding only helps once the search is local and CPU-bound.

echo "[$(date -u +%FT%TZ)] annotate shard ${K}: $(wc -l < "$SHARD") genes"
python scripts/run.py \
    --gene-list "$SHARD" \
    --run-name "$RUN_NAME" \
    --drop-unsupported-tis \
    --fasta-out "data/output/${CAMPAIGN}/shard_fastas/shard_${K}.fa"
# Stamp the shard's gene membership next to its results. merge.py checks this
# against shards/shard_${K}.txt and refuses to merge a shard whose gene list has
# since changed — the one way a re-split could otherwise blend old and new rows.
cat > "$OUTDIR/shard_meta.json" <<EOF
{
  "campaign": "${CAMPAIGN}",
  "shard": ${K},
  "shard_list_sha1": "$(sha1sum "$SHARD" | cut -d' ' -f1)",
  "n_genes": $(grep -c . "$SHARD"),
  "finished_utc": "$(date -u +%FT%TZ)"
}
EOF
echo "[$(date -u +%FT%TZ)] annotate shard ${K} done"
