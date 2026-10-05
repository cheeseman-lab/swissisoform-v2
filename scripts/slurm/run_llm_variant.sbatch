#!/bin/bash
#SBATCH --job-name=swissiso_llm_arm
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# One arm of the prompt-variant experiment. The work is API-bound, not CPU-bound,
# so this asks for almost nothing — but it is long (the M and P tool loops are
# multi-turn and unbatchable), which is why it wants a real time limit and why
# each arm gets its own job: eight in parallel finish in one arm's wall clock
# rather than eight.
#
# Compute nodes reach api.anthropic.com (verified), and the key comes from .env
# the same way scripts/build_website.sh sources it.
#
# Usage (from the repo root, after `mkdir -p logs`):
#   sbatch --job-name=${USER}_llm_tags_hint scripts/slurm/run_llm_variant.sbatch tags_hint
#
#   # all eight, one job each
#   for a in criteria_hint criteria_nohint raw_hint raw_nohint \
#            tags_hint tags_nohint dist_hint dist_nohint; do
#     sbatch --job-name=${USER}_llm_$a scripts/slurm/run_llm_variant.sbatch "$a"
#   done
#
# Extra args after the arm id are forwarded verbatim (e.g. --gene CBX1).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

if [[ $# -lt 1 ]]; then
    echo "usage: sbatch $0 <arm_id> [extra run_llm_variants.py args...]" >&2
    exit 2
fi
ARM="$1"; shift

eval "$(conda shell.bash hook)"
conda activate swissisoform-v2

# ANTHROPIC_API_KEY. Fail loudly rather than burning the allocation on a run that
# cannot make a single call.
[[ -f .env ]] && set -a && . ./.env && set +a
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ANTHROPIC_API_KEY is not set and .env did not supply it" >&2
    exit 1
fi

echo "[$(date -u +%FT%TZ)] arm=$ARM extra='$*'"
t0=$SECONDS
python scripts/site/run_llm_variants.py --arm "$ARM" --force "$@"
took=$((SECONDS - t0))
printf '[%s] arm=%s done in %ds (%dm%02ds)\n' \
    "$(date -u +%FT%TZ)" "$ARM" "$took" $((took / 60)) $((took % 60))

# The usage report is where the real cost lands — print it so the log carries it.
usage="data/output/cheeseman50_${ARM}/llm/_usage_category.json"
[[ -f "$usage" ]] && python - "$usage" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))["total"]
print(f"  single-shot: {d['calls']} calls, {d['input']:,} in / {d['output']:,} out, "
      f"${d['cost_usd']:.2f}")
PY
usage_tools="data/output/cheeseman50_${ARM}/llm/_usage_category_tools.json"
[[ -f "$usage_tools" ]] && python - "$usage_tools" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))["total"]
print(f"  tool loops:  {d['calls']} calls, {d['input']:,} in / {d['output']:,} out, "
      f"${d['cost_usd']:.2f}")
PY
