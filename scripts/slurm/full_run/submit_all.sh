#!/bin/bash
# Launch the full genome-wide run (all 11,657 genes, HeLa long-read filtering,
# every module on). Submits only Phase A (00_prepare.sbatch); that job runs on a
# compute node and self-chains Phases B (GPU arrays) → C (annotate array) →
# D (merge), so the whole campaign is detached from this shell and from Claude
# Code. Safe to close Claude Code after this returns.
#
# Every path is scoped to one campaign name, which defaults to the UTC date so a
# new run can never land on top of an old one's shard outputs. Pass a name to
# resume or to label a campaign:
#
# Usage (from anywhere):  bash scripts/slurm/full_run/submit_all.sh [campaign]
# Monitor: squeue -u $USER   /   sacct -u $USER --name=swissiso_full_annot -X
# Resume:  see the header of 00_prepare.sbatch (every resubmit needs the campaign)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../../.."   # repo root

CAMPAIGN="${1:-${SWISSISO_CAMPAIGN:-full_catalog_$(date -u +%Y%m%d)}}"
mkdir -p logs "data/output/$CAMPAIGN"

jobA=$(sbatch --parsable --export=ALL,SWISSISO_CAMPAIGN="$CAMPAIGN" \
    scripts/slurm/full_run/00_prepare.sbatch)

echo "campaign: $CAMPAIGN"
echo "  inputs + merged catalog: data/output/$CAMPAIGN/"
echo "  shard outputs:           data/output/${CAMPAIGN}_shard_<k>/"
echo "submitted Phase A prepare job: $jobA"
echo "downstream jobs (B GPU arrays -> C annotate array -> D merge) self-chain on the cluster"
echo "monitor: squeue -u \$USER   |   sacct -u \$USER --name=swissiso_full_annot -X"
echo "resume:  sbatch --partition=24 --export=ALL,SWISSISO_CAMPAIGN=$CAMPAIGN \\"
echo "                --array=0-<K-1>%25 scripts/slurm/full_run/annotate_array.sbatch"
echo "         (done shards exit immediately, so resubmitting the full range is safe)"
