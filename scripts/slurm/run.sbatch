#!/bin/bash
#SBATCH --job-name=mdiberna_swissiso_run
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Unified entry point (orchestrator). One command runs the whole pipeline end
# to end, spawning the GPU precompute jobs in the middle and waiting on them:
#
#   Stage A  python scripts/run.py <args> --emit-fasta  (CPU: write proteins.fa)
#   Stage B  sbatch --wait run_plm_embed + run_fold      (GPU: populate caches)
#   Stage C  python scripts/run.py <args>                (CPU: annotate + score + paired parquet)
#
# All arguments after the script name are forwarded verbatim to run.py.
#
# Usage (from repo root, after `mkdir -p logs`):
#   sbatch scripts/slurm/run.sbatch --preset 5gene
#   sbatch scripts/slurm/run.sbatch --genes TP53 EIF4G1 --run-name my_run
#   sbatch scripts/slurm/run.sbatch --preset 5gene --no-gpu   # skip Stage B
#
# Tradeoff: this orchestrator holds its CPU allocation while the GPU jobs run,
# and Stages A and C both load references + assemble (the upstream CPU work runs
# twice). Fine for diagnostics; graduate to a Slurm dependency chain
# (run.py --emit-fasta -> afterok -> run.py annotate-only) if Boltz wait-time on
# a held node becomes a courtesy problem.

set -euo pipefail

cd /lab/barcheese01/mdiberna/swissisoform-v2

eval "$(conda shell.bash hook)"
conda activate swissisoform-v2

# Nextflow (InterProScan) needs the conda env's Java 17 + repo-local caches.
export JAVA_HOME="$CONDA_PREFIX"
export JAVA_CMD="$CONDA_PREFIX/bin/java"
export NXF_HOME="$PWD/.nextflow"
export NXF_SINGULARITY_CACHEDIR="$PWD/.singularity_cache"

RUN_ARGS=("$@")
# Per-job FASTA so concurrent orchestrator runs never clobber one shared file.
FASTA="data/cache/run_${SLURM_JOB_ID:-local}/proteins.fa"

# GPU caches are needed unless the run disables both GPU modules — either via
# --no-gpu or via --skip-modules listing both plm_vep and structure.
skip_gpu=0
if [[ " ${RUN_ARGS[*]} " == *" --no-gpu "* ]]; then
    skip_gpu=1
elif [[ " ${RUN_ARGS[*]} " == *"--skip-modules"* \
        && " ${RUN_ARGS[*]} " == *"plm_vep"* && " ${RUN_ARGS[*]} " == *"structure"* ]]; then
    skip_gpu=1
fi

echo "[$(date -u +%FT%TZ)] Stage A — emit $FASTA for: ${RUN_ARGS[*]:-<defaults>}"
python scripts/run.py "${RUN_ARGS[@]}" --fasta-out "$FASTA" --emit-fasta

if (( skip_gpu )); then
    echo "[$(date -u +%FT%TZ)] Stage B — skipped (GPU modules disabled)"
else
    echo "[$(date -u +%FT%TZ)] Stage B — GPU precompute"
    plm_id=$(sbatch --parsable scripts/slurm/run_plm_embed.sbatch "$FASTA")
    fold_id=$(sbatch --parsable scripts/slurm/run_fold.sbatch "$FASTA" boltz)
    echo "[$(date -u +%FT%TZ)]   submitted plm=$plm_id fold=$fold_id"
    # If this orchestrator dies (failure / timeout / scancel), cancel both GPU
    # jobs so neither is left orphaned on the cluster.
    trap 'scancel "$plm_id" "$fold_id" 2>/dev/null || true' EXIT
    # Block until both jobs leave the queue (failing squeue in the condition is
    # fine — set -e does not fire on a loop test).
    while squeue -h -j "${plm_id},${fold_id}" -o '%i' 2>/dev/null | grep -q '[0-9]'; do
        sleep 30
    done
    sleep 10   # let sacct accounting settle to a terminal state
    # Require BOTH jobs COMPLETED — otherwise Stage C would annotate on partial caches.
    gpu_ok=1
    for jid in "$plm_id" "$fold_id"; do
        st=$(sacct -n -X -j "$jid" -o State 2>/dev/null | head -1 | tr -d ' ')
        echo "[$(date -u +%FT%TZ)]   job $jid: ${st:-UNKNOWN}"
        [[ "$st" == "COMPLETED" ]] || gpu_ok=0
    done
    trap - EXIT
    if (( ! gpu_ok )); then
        echo "GPU precompute did not complete cleanly; aborting before annotation." >&2
        scancel "$plm_id" "$fold_id" 2>/dev/null || true
        exit 1
    fi
fi

echo "[$(date -u +%FT%TZ)] Stage C — full annotation pipeline"
python scripts/run.py "${RUN_ARGS[@]}" --fasta-out "$FASTA"
echo "[$(date -u +%FT%TZ)] done."
