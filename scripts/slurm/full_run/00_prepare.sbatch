#!/bin/bash
#SBATCH --job-name=swissiso_full_prepare
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Phase A of the full genome-wide run, and the chaining hub for B/C/D. Runs on a
# compute node, so the entire pipeline is detached from Claude Code / the login
# shell — closing Claude Code cannot stall or cancel it.
#
#   A  emit the deduped full-catalog protein FASTA (run.py --all --emit-fasta)
#      + split into GPU chunks and gene shards
#   B  submit gpu_embed_array + gpu_fold_array  (--array sized to the chunk count)
#   C  submit annotate_array  (afterany on B; gene-sharded full annotation)
#   D  submit merge           (afterany on C)
#
# CAMPAIGN scoping: every path this chain touches is named for one campaign
# ($SWISSISO_CAMPAIGN, default full_catalog_<UTC date>) — inputs under
# data/output/$CAMPAIGN/, shard outputs at data/output/${CAMPAIGN}_shard_<k>/,
# merged catalog back in data/output/$CAMPAIGN/. Two campaigns therefore cannot
# blend, and the merge is scoped to the manifest's own 0..K-1 rather than to a
# glob. The name is passed down the chain with --export, so B/C/D inherit it;
# every manual resubmit must set it explicitly (they refuse to guess).
#
# Job IDs are echoed to this job's log. There is no state file: every job is
# named swissiso_full_*, so the campaign is recoverable from Slurm alone —
#   squeue -u $USER                                    # queued / running now
#   sacct -u $USER --name=swissiso_full_annot -X       # history (PENDING jobs
#                                                      #   need squeue; they have
#                                                      #   no start time)
#
# RESUME: resubmit the array with the campaign set. Every stage is idempotent, so
# re-running the FULL range is always safe and only redoes what is missing:
#   annotate — a shard whose all_paired.parquet exists exits in <1s (see below)
#       sbatch --partition=24 --export=ALL,SWISSISO_CAMPAIGN=<c> \
#              --array=0-<K-1>%25 $HERE/annotate_array.sbatch
#   GPU     — embed/fold are hash-keyed; finished proteins cache-skip
#       sbatch --partition=nvidia-A6000-20 --export=ALL,SWISSISO_CAMPAIGN=<c> \
#              --array=0-<C-1>%3 $HERE/gpu_*_array.sbatch
#   merge   — idempotent; re-run after refilling shards. It exits nonzero and
#             withholds all_paired.parquet while any shard is missing or stale,
#             so a partial catalog can never be mistaken for a complete one.
#       sbatch --partition=24 --export=ALL,SWISSISO_CAMPAIGN=<c> $HERE/merge.sbatch
# To force a shard to recompute, delete its all_paired.parquet first — that file
# IS the "done" marker. K and C are in $OUT/split_manifest.txt.
#
# REFILL (GPU precompute holes): the inverse of RESUME. Because Phase C depends
# on B with afterany, a failed embed/fold chunk still lets annotation run for its
# proteins — StructureModule/PLM/SAE read the cache miss, record
# status="no_cache", and the shard writes a perfectly valid all_paired.parquet.
# Such a shard is "done" forever: it passes both the [[ -f ]] check and the PAR1
# validation, so resubmitting the array skips it. Merge REPORTS these but still
# publishes (holes are refillable, and some proteins legitimately fail to fold),
# so the exit code will not flag it — read merge_report.tsv.
#   1. merge_report.tsv names the affected shards:
#      frac_no_cache_structure / frac_no_cache_plm / frac_no_cache_sae > 0.
#      That is the whole input — no chunk-to-shard mapping needed.
#   2. Resubmit the FULL GPU array range. Embed/fold are hash-keyed, so finished
#      proteins cache-skip and only the dead chunk's proteins recompute; which
#      chunk failed never has to be identified.
#        sbatch --partition=nvidia-A6000-20 --export=ALL,SWISSISO_CAMPAIGN=<c> \
#               --array=0-<C-1>%3 $HERE/gpu_embed_array.sbatch   # and gpu_fold_array
#   3. Clear the done-marker on each holed shard — BOTH files, so the output dir
#      is never left half-marked (same pair annotate_array.sbatch removes when it
#      finds a truncated parquet):
#        rm data/output/<c>_shard_<k>/all_paired.parquet \
#           data/output/<c>_shard_<k>/shard_meta.json
#   4. Resubmit the annotate array — holed + missing shards recompute, the rest
#      still exit in <1s (same command as RESUME above).
#   5. Re-merge, and confirm the frac_no_cache_* columns are back to zero.
# Pick the partition by contention (`sinfo`), not habit: partition 20 caps at
# 100 cpu/user (12 shards at 8 cpu) and has been badly oversubscribed; 24 caps
# at 200 (25 shards).
#
# afterany (not afterok) is deliberate: the chain always advances to a terminal
# state instead of stalling on a single failed GPU chunk or slow shard —
# coverage is verified and gaps are refilled by resubmitting (above).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

eval "$(conda shell.bash hook)"
conda activate swissisoform-v2

HERE=scripts/slurm/full_run
# One campaign = one output namespace. Dated by default so a fresh run cannot
# land on top of an old one; set SWISSISO_CAMPAIGN to resume a named campaign.
CAMPAIGN="${SWISSISO_CAMPAIGN:-full_catalog_$(date -u +%Y%m%d)}"
export SWISSISO_CAMPAIGN="$CAMPAIGN"
OUT="data/output/$CAMPAIGN"
FASTA="$OUT/proteins.fa"
COMBINED=data/output/filtered/all_samples_combined.parquet
CHUNK_SIZE=1500          # proteins per GPU chunk
SHARD_SIZE=100           # genes per annotate shard
GPU_CONCURRENCY=3        # nvidia-A6000-20 per-user GPU cap
ANNOT_CONCURRENCY=12     # 12 shards x 8 cpus = 96 <= partition-20 cap (100 cpus);
                         # 8 cpus/shard feed PepQuery's -cpu (CPU-bound traversal)
mkdir -p "$OUT" logs

_stamp() { echo "[$(date -u +%FT%TZ)] $*"; }

_stamp "campaign: $CAMPAIGN  (outputs under $OUT and ${OUT}_shard_<k>)"

# ── Phase A ──────────────────────────────────────────────────────────────────
# --drop-unsupported-tis defines the campaign's population: long-read-supported
# TIS only. Only HeLa has IsoQuant, so every TIS called solely in K562/U2OS/RPE1
# is dropped — about half the catalog. Deliberate: long-read data for the other
# lines is expected, and as it lands this narrowing shrinks on its own, with no
# change here. Stamped into the run's population.json (rows_dropped_by_flag) so
# anything calibrated on this substrate can cite the gap it had. run.py
# refuses to start if the combined catalog lacks the verdict columns the flag
# needs (rebuild it with --rebuild-combined), rather than silently keeping
# everything.
_stamp "PHASE A: emit full-catalog FASTA"
python scripts/run.py --all --run-name "$CAMPAIGN" --drop-unsupported-tis --emit-fasta --fasta-out "$FASTA"

_stamp "PHASE A: split into chunks + shards"
python "$HERE/split.py" --campaign "$CAMPAIGN" --proteins "$FASTA" --combined "$COMBINED" \
    --outdir "$OUT" --chunk-size "$CHUNK_SIZE" --shard-size "$SHARD_SIZE"

# split.py is the single source of truth for the counts AND the embed split, so
# the array ranges cannot drift from what it actually wrote.
_manifest() { awk -v k="$1" -F'\t' '$1 == k { print $2 }' "$OUT/split_manifest.txt"; }
C=$(_manifest n_chunks)
K=$(_manifest n_shards)
SPLIT=$(_manifest embed_split)
[[ "${C:-0}" -gt 0 && "${K:-0}" -gt 0 && -n "$SPLIT" ]] \
    || { echo "split produced no chunks/shards (or an old manifest)" >&2; exit 1; }
_stamp "split → C=$C chunks, K=$K shards"

# ── Phase B: GPU precompute arrays (resumable via hash cache) ────────────────
# Embed (the LLR-sweep bottleneck) is split disjointly across BOTH GPU queues:
# A6000 (3 GPUs) takes chunks [0,SPLIT), A100 (2 GPUs, ~2x/GPU on the 6B model)
# takes [SPLIT,C) — ~3/7/~4/7, balanced by effective throughput, so ~5 GPUs run
# embed with no duplicate compute and no cache race (disjoint ranges). The array
# script reads chunk_${SLURM_ARRAY_TASK_ID}.fa, and --partition on the command
# line overrides its #SBATCH default. Fold (cheap) stays on A6000.
#
# SPLIT is clamped to [1,C] by split.embed_split, so the A100 range is empty for
# a single-chunk run and that submission is skipped: one chunk cannot be divided
# across two queues, and the old clamp produced --array=0--1, which sbatch
# rejects — killing Phase A under set -e before anything else was submitted.
_stamp "embed split: A6000 chunks [0,$SPLIT)  A100 chunks [$SPLIT,$C)"

gpu_jobs=()
embed_a6000=$(sbatch --parsable --export=ALL,SWISSISO_CAMPAIGN="$CAMPAIGN" --partition=nvidia-A6000-20 \
    --array=0-$((SPLIT-1))%3 "$HERE/gpu_embed_array.sbatch")
gpu_jobs+=("$embed_a6000")
if [[ "$SPLIT" -lt "$C" ]]; then
    embed_a100=$(sbatch --parsable --export=ALL,SWISSISO_CAMPAIGN="$CAMPAIGN" --partition=nvidia-A100-20 \
        --array=${SPLIT}-$((C-1))%2 "$HERE/gpu_embed_array.sbatch")
    gpu_jobs+=("$embed_a100")
else
    embed_a100="skipped (single chunk — nothing to split)"
fi
fold=$(sbatch --parsable --export=ALL,SWISSISO_CAMPAIGN="$CAMPAIGN" --partition=nvidia-A6000-20 \
    --array=0-$((C-1))%3 "$HERE/gpu_fold_array.sbatch")
gpu_jobs+=("$fold")

# ── Phase C: annotate array, after ALL GPU arrays reach terminal state ───────
annot=$(sbatch --parsable --export=ALL,SWISSISO_CAMPAIGN="$CAMPAIGN" \
    --dependency=afterany:"$(IFS=:; echo "${gpu_jobs[*]}")" \
    --array=0-$((K-1))%${ANNOT_CONCURRENCY} "$HERE/annotate_array.sbatch")

# ── Phase D: merge, after the annotate array ─────────────────────────────────
merge=$(sbatch --parsable --export=ALL,SWISSISO_CAMPAIGN="$CAMPAIGN" \
    --dependency=afterany:"$annot" "$HERE/merge.sbatch")

_stamp "campaign=$CAMPAIGN n_chunks=$C n_shards=$K embed_split: A6000 [0,$SPLIT) / A100 [$SPLIT,$C)"
_stamp "chained: embed_a6000=$embed_a6000 embed_a100=$embed_a100 fold=$fold annotate=$annot merge=$merge"
_stamp "monitor: squeue -u \$USER | sacct -u \$USER --name=swissiso_full_annot -X"
_stamp "resume:  sbatch --partition=24 --export=ALL,SWISSISO_CAMPAIGN=$CAMPAIGN --array=0-$((K-1))%25 $HERE/annotate_array.sbatch"
