#!/bin/bash
#SBATCH --job-name=swissiso_full_merge
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=256G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Phase D of the full genome-wide run: concatenate the campaign's per-shard
# paired parquets into data/output/$SWISSISO_CAMPAIGN/all_paired.parquet. Depends
# (afterany) on the annotate array, so it runs once every shard has reached a
# terminal state — which means some shards may have failed.
#
# merge.py accounts every shard against split_manifest.txt and exits nonzero on
# any missing/stale/unreadable shard, writing the partial result to
# all_paired.partial.parquet instead of publishing it. So a nonzero exit here is
# the expected signal to refill shards and re-run; it is not a crash.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
eval "$(conda shell.bash hook)"
conda activate swissisoform-v2

CAMPAIGN="${SWISSISO_CAMPAIGN:?SWISSISO_CAMPAIGN is not set — resubmit with \
    sbatch --export=ALL,SWISSISO_CAMPAIGN=<campaign> ... (see 00_prepare.sbatch)}"

echo "[$(date -u +%FT%TZ)] merging shard outputs for campaign $CAMPAIGN"
# Keep the incomplete-merge exit code (set -e would drop the guidance below).
rc=0
python scripts/slurm/full_run/merge.py --campaign "$CAMPAIGN" || rc=$?
if [[ "$rc" -ne 0 ]]; then
    echo "[$(date -u +%FT%TZ)] merge INCOMPLETE (rc=$rc) — all_paired.parquet not published."
    echo "  refill the shards listed above, then resubmit this job:"
    echo "  sbatch --partition=24 --export=ALL,SWISSISO_CAMPAIGN=$CAMPAIGN $0"
    exit "$rc"
fi
# --skip-llm is the right default for a genome-wide campaign, not a nicety. The
# LLM stage is idempotent (run_llm_interpretation.py reuses an isoform whose
# output already exists), but a fresh campaign has nothing to reuse, so the pass
# bills every isoform: 6 category calls (C,D,L,M,P,S) + 1 synthesis each, with M
# and P running as multi-turn tool loops — five figures' worth of calls over a
# ~6.5k-isoform catalog. Drop the flag only for a gene-scoped run (a preset), or
# after deliberately choosing to pay for a full pass — and then prefer --batch
# (Message Batches API, 50% token price).
#
# Also note the structures stage resolves from $PRESET, which has no genome-wide
# value: it rebuilds that preset's own structures into the preset's run dir, not
# this campaign's. Harmless but wasted; there is no --skip-structures flag.
echo "[$(date -u +%FT%TZ)] merge done. Build the site with:"
echo "  RUN=$CAMPAIGN bash scripts/build_website.sh --skip-llm"
echo "  (--skip-llm is deliberate: a fresh campaign's LLM pass is ~7+ paid calls"
echo "   per isoform over ~6.5k isoforms. Omit it only for a gene-scoped preset"
echo "   run, or with --batch when a full pass is genuinely wanted.)"
