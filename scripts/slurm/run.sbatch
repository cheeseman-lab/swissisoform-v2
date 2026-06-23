#!/bin/bash
#SBATCH --job-name=swissiso_run
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Canonical pipeline orchestrator. Runs the whole pipeline end to end, spawning
# the GPU precompute jobs in the middle and waiting on them, with per-stage
# wall-clock timing printed at the end:
#
#   prepare   python scripts/run.py <args> --emit-fasta   (CPU: write proteins.fa)
#   gpu       sbatch --wait run_plm_embed + run_fold       (GPU: populate fold/PLM caches)
#   annotate  python scripts/run.py <args>                 (CPU: annotate + score + paired parquet)
#
# Modes (consumed here; everything else forwards verbatim to run.py):
#   --clear-cache     wipe data/cache/{plm_esmc,structure,pepquery,tmp} first
#                     (forces full regeneration — use for a clean, confident rebuild)
#   --annotate-only   skip the GPU stage and reuse the WARM fold/PLM caches
#                     (fast surgical re-run; no re-folding). Equivalent caches must exist.
#
# With no run.py args, defaults to the single canonical run: --preset cheeseman13.
#
# Usage (from repo root, after `mkdir -p logs`):
#   sbatch scripts/slurm/run.sbatch                              # full cheeseman13 (GPU)
#   sbatch scripts/slurm/run.sbatch --clear-cache               # clean rebuild from scratch
#   sbatch scripts/slurm/run.sbatch --annotate-only             # reuse caches, no folding
#   sbatch scripts/slurm/run.sbatch --annotate-only --skip-modules clinical   # surgical
#
# After this finishes, build the site with: bash scripts/build_website.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

eval "$(conda shell.bash hook)"
conda activate swissisoform-v2

# Nextflow (InterProScan) needs the conda env's Java 17 + repo-local caches.
export JAVA_HOME="$CONDA_PREFIX"
export JAVA_CMD="$CONDA_PREFIX/bin/java"
export NXF_HOME="$PWD/.nextflow"
export NXF_SINGULARITY_CACHEDIR="$PWD/.singularity_cache"

# ── Parse orchestrator flags out of the argument list ───────────────────────
clear_cache=0
annotate_only=0
RUN_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --clear-cache)   clear_cache=1 ;;
        --annotate-only) annotate_only=1 ;;
        *)               RUN_ARGS+=("$arg") ;;
    esac
done
# Default to the single canonical run when no input mode is given.
if [[ ${#RUN_ARGS[@]} -eq 0 ]]; then
    RUN_ARGS=(--preset cheeseman13)
fi

# --no-gpu / both-GPU-modules-skipped also implies no GPU stage.
skip_gpu=$annotate_only
if [[ " ${RUN_ARGS[*]} " == *" --no-gpu "* ]]; then
    skip_gpu=1
elif [[ " ${RUN_ARGS[*]} " == *"--skip-modules"* \
        && " ${RUN_ARGS[*]} " == *"plm_vep"* && " ${RUN_ARGS[*]} " == *"structure"* ]]; then
    skip_gpu=1
fi

FASTA="data/cache/run_${SLURM_JOB_ID:-local}/proteins.fa"

# ── Per-stage timing ────────────────────────────────────────────────────────
declare -A STAGE_SECS
_stamp() { echo "[$(date -u +%FT%TZ)] $*"; }
run_stage() {  # run_stage <name> <command...>
    local name="$1"; shift
    local t0=$SECONDS
    _stamp "STAGE START: $name"
    "$@"
    local took=$((SECONDS - t0))
    STAGE_SECS[$name]=$took
    _stamp "STAGE END:   $name  (${took}s)"
}

_stamp "run: args='${RUN_ARGS[*]}'  clear_cache=$clear_cache  annotate_only=$annotate_only  skip_gpu=$skip_gpu"

# ── (optional) clear caches for a clean, confident rebuild ──────────────────
if (( clear_cache )); then
    run_stage clear-cache bash -c '
        rm -rf data/cache/plm_esmc data/cache/structure data/cache/pepquery data/cache/tmp
        echo "  wiped data/cache/{plm_esmc,structure,pepquery,tmp}"'
fi

# ── prepare (CPU): emit the deduped proteins FASTA ──────────────────────────
run_stage prepare python scripts/run.py "${RUN_ARGS[@]}" --fasta-out "$FASTA" --emit-fasta

# ── gpu (populate fold + PLM caches), unless skipped ────────────────────────
if (( skip_gpu )); then
    _stamp "STAGE SKIP:  gpu (annotate-only / --no-gpu / both GPU modules skipped — reusing warm caches)"
else
    gpu_stage() {
        # The embed job also runs the SAE encode (sae_esmc/) unless SAE is being
        # skipped in annotate — propagate that as an env var to the embed job.
        local plm_export=()
        if [[ " ${RUN_ARGS[*]} " == *"--skip-modules"* && " ${RUN_ARGS[*]} " == *"sae"* ]]; then
            plm_export=(--export=ALL,SWISSISO_SKIP_SAE=1)
            _stamp "  --skip-modules sae → SAE encode disabled in embed job"
        fi
        plm_id=$(sbatch --parsable "${plm_export[@]+"${plm_export[@]}"}" \
            scripts/slurm/run_plm_embed.sbatch "$FASTA")
        fold_id=$(sbatch --parsable scripts/slurm/run_fold.sbatch "$FASTA" boltz)
        _stamp "  submitted plm=$plm_id fold=$fold_id"
        # Cancel both GPU jobs if this orchestrator dies, so neither is orphaned.
        trap 'scancel "$plm_id" "$fold_id" 2>/dev/null || true' EXIT
        while squeue -h -j "${plm_id},${fold_id}" -o '%i' 2>/dev/null | grep -q '[0-9]'; do
            sleep 30
        done
        sleep 10   # let sacct settle to a terminal state
        local gpu_ok=1 jid st
        for jid in "$plm_id" "$fold_id"; do
            st=$(sacct -n -X -j "$jid" -o State 2>/dev/null | head -1 | tr -d ' ')
            _stamp "  job $jid: ${st:-UNKNOWN}"
            [[ "$st" == "COMPLETED" ]] || gpu_ok=0
        done
        trap - EXIT
        if (( ! gpu_ok )); then
            echo "GPU precompute did not complete cleanly; aborting before annotation." >&2
            scancel "$plm_id" "$fold_id" 2>/dev/null || true
            exit 1
        fi
    }
    run_stage gpu gpu_stage
fi

# ── annotate (CPU): full pipeline → paired parquet ──────────────────────────
run_stage annotate python scripts/run.py "${RUN_ARGS[@]}" --fasta-out "$FASTA"

# ── timing summary ──────────────────────────────────────────────────────────
echo
_stamp "TIMING SUMMARY (wall-clock per stage):"
total=0
for name in clear-cache prepare gpu annotate; do
    secs=${STAGE_SECS[$name]:-}
    [[ -z "$secs" ]] && continue
    printf '  %-12s %6ds  (%dm%02ds)\n' "$name" "$secs" $((secs/60)) $((secs%60))
    total=$((total + secs))
done
printf '  %-12s %6ds  (%dm%02ds)\n' "TOTAL" "$total" $((total/60)) $((total%60))
_stamp "done. Next: bash scripts/build_website.sh   (rebuild website/data from data/output/cheeseman_13gene/)"
