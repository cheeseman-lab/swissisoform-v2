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
# Job IDs are echoed to this job's log. There is no state file: every job is
# named swissiso_full_*, so the campaign is recoverable from Slurm alone —
#   squeue -u $USER                                    # queued / running now
#   sacct -u $USER --name=swissiso_full_annot -X       # history (PENDING jobs
#                                                      #   need squeue; they have
#                                                      #   no start time)
#
# RESUME: resubmit the array. Every stage is idempotent, so re-running the FULL
# range is always safe and only redoes what is missing:
#   annotate — a shard whose all_paired.parquet exists exits in <1s (see below)
#       sbatch --partition=24 --array=0-<K-1>%25 $HERE/annotate_array.sbatch
#   GPU     — embed/fold are hash-keyed; finished proteins cache-skip
#       sbatch --partition=nvidia-A6000-20 --array=0-<C-1>%3 $HERE/gpu_*_array.sbatch
#   merge   — idempotent; rebuilds from whatever shard outputs exist
#       sbatch --partition=24 --dependency=afterany:<annot_id> $HERE/merge.sbatch
# To force a shard to recompute, delete its all_paired.parquet first — that file
# IS the "done" marker. K and C are in $OUT/split_manifest.txt.
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
OUT=data/output/full_catalog
FASTA="$OUT/proteins.fa"
COMBINED=data/output/filtered/all_samples_combined.parquet
CHUNK_SIZE=1500          # proteins per GPU chunk
SHARD_SIZE=100           # genes per annotate shard
GPU_CONCURRENCY=3        # nvidia-A6000-20 per-user GPU cap
ANNOT_CONCURRENCY=12     # 12 shards x 8 cpus = 96 <= partition-20 cap (100 cpus);
                         # 8 cpus/shard feed PepQuery's -cpu (CPU-bound traversal)
mkdir -p "$OUT" logs

_stamp() { echo "[$(date -u +%FT%TZ)] $*"; }

# ── Phase A ──────────────────────────────────────────────────────────────────
_stamp "PHASE A: emit full-catalog FASTA"
python scripts/run.py --all --run-name full_catalog --drop-unsupported-tis --emit-fasta --fasta-out "$FASTA"

_stamp "PHASE A: split into chunks + shards"
python "$HERE/split.py" --proteins "$FASTA" --combined "$COMBINED" --outdir "$OUT" \
    --chunk-size "$CHUNK_SIZE" --shard-size "$SHARD_SIZE"

C=$(ls "$OUT"/chunks/chunk_*.fa 2>/dev/null | wc -l)
K=$(ls "$OUT"/shards/shard_*.txt 2>/dev/null | wc -l)
[[ "$C" -gt 0 && "$K" -gt 0 ]] || { echo "split produced no chunks/shards" >&2; exit 1; }
_stamp "split → C=$C chunks, K=$K shards"

# ── Phase B: GPU precompute arrays (resumable via hash cache) ────────────────
# Embed (the LLR-sweep bottleneck) is split disjointly across BOTH GPU queues:
# A6000 (3 GPUs) takes the lower ~3/7 of chunks, A100 (2 GPUs, ~2x/GPU on the 6B
# model) the upper ~4/7 — balanced by effective throughput, so ~5 GPUs run embed
# with no duplicate compute and no cache race (disjoint chunk ranges). The array
# script reads chunk_${SLURM_ARRAY_TASK_ID}.fa, and --partition on the command
# line overrides its #SBATCH default. Fold (cheap) stays on A6000.
SPLIT=$(( C * 3 / 7 ))
[[ "$SPLIT" -lt 1 ]] && SPLIT=1
[[ "$SPLIT" -gt $((C-1)) ]] && SPLIT=$((C-1))
_stamp "embed split: A6000 chunks [0,$SPLIT)  A100 chunks [$SPLIT,$C)"

embed_a6000=$(sbatch --parsable --partition=nvidia-A6000-20 \
    --array=0-$((SPLIT-1))%3 "$HERE/gpu_embed_array.sbatch")
embed_a100=$(sbatch --parsable --partition=nvidia-A100-20 \
    --array=${SPLIT}-$((C-1))%2 "$HERE/gpu_embed_array.sbatch")
fold=$(sbatch --parsable --partition=nvidia-A6000-20 \
    --array=0-$((C-1))%3 "$HERE/gpu_fold_array.sbatch")

# ── Phase C: annotate array, after ALL GPU arrays reach terminal state ───────
annot=$(sbatch --parsable \
    --dependency=afterany:"$embed_a6000":"$embed_a100":"$fold" \
    --array=0-$((K-1))%${ANNOT_CONCURRENCY} "$HERE/annotate_array.sbatch")

# ── Phase D: merge, after the annotate array ─────────────────────────────────
merge=$(sbatch --parsable --dependency=afterany:"$annot" "$HERE/merge.sbatch")

_stamp "n_chunks=$C n_shards=$K embed_split: A6000 [0,$SPLIT) / A100 [$SPLIT,$C)"
_stamp "chained: embed_a6000=$embed_a6000 embed_a100=$embed_a100 fold=$fold annotate=$annot merge=$merge"
_stamp "monitor: squeue -u \$USER | sacct -u \$USER --name=swissiso_full_annot -X"
_stamp "resume:  sbatch --partition=24 --array=0-$((K-1))%25 $HERE/annotate_array.sbatch"
