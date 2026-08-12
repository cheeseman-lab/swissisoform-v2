#!/bin/bash
# Launch the full genome-wide run (all 11,657 genes, HeLa long-read filtering,
# every module on). Submits only Phase A (00_prepare.sbatch); that job runs on a
# compute node and self-chains Phases B (GPU arrays) → C (annotate array) →
# D (merge), so the whole campaign is detached from this shell and from Claude
# Code. Safe to close Claude Code after this returns.
#
# Usage (from anywhere):  bash scripts/slurm/full_run/submit_all.sh
# Monitor: squeue -u $USER   /   sacct -u $USER --name=swissiso_full_annot -X
# Resume:  see the header of 00_prepare.sbatch
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../../.."   # repo root

mkdir -p logs data/output/full_catalog

jobA=$(sbatch --parsable scripts/slurm/full_run/00_prepare.sbatch)

echo "submitted Phase A prepare job: $jobA"
echo "downstream jobs (B GPU arrays -> C annotate array -> D merge) self-chain on the cluster"
echo "monitor: squeue -u \$USER   |   sacct -u \$USER --name=swissiso_full_annot -X"
echo "resume:  sbatch --partition=24 --array=0-<K-1>%25 scripts/slurm/full_run/annotate_array.sbatch"
echo "         (done shards exit immediately, so resubmitting the full range is safe)"
