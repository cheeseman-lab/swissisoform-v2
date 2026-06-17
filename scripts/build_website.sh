#!/bin/bash
# Build website/data from a completed pipeline run (data/output/cheeseman_13gene/).
# Decoupled from the pipeline so the site can be rebuilt surgically — without
# re-running annotation, GPU folding, or (with --skip-llm) the paid LLM pass.
#
# Stages (each timed):
#   skeletons   transcript exon skeletons for the IGV-style transcript view
#   evidence    per-gene LLM evidence JSON + flat variants_long parquet
#   llm         Anthropic per-criterion interpretation (SKIP with --skip-llm)
#   structures  re-assemble folded CIFs from the structure cache
#   stage       website/prepare_deploy.sh — copy artifacts into website/data/
#
# CPU/network only (no GPU). The LLM stage needs ANTHROPIC_API_KEY (read from .env).
#
# Usage (from repo root):
#   bash scripts/build_website.sh                 # full rebuild incl. LLM
#   bash scripts/build_website.sh --skip-llm      # reuse existing llm/, no API calls
#   RUN=cheeseman_13gene bash scripts/build_website.sh
#
# Deploy is a separate manual step: cd website && railway up --no-gitignore --service swissisoform-viewer

set -euo pipefail
cd /lab/barcheese01/ating/swissisoform-v2

eval "$(conda shell.bash hook)"
conda activate swissisoform-v2
[[ -f .env ]] && set -a && . ./.env && set +a   # ANTHROPIC_API_KEY for the LLM stage

skip_llm=0
for arg in "$@"; do
    [[ "$arg" == "--skip-llm" ]] && skip_llm=1
done

RUN="${RUN:-cheeseman_13gene}"
PRESET="cheeseman13"
OUT="data/output/${RUN}"
GTF="data/reference/gencode.v49.primary_assembly.annotation.gtf"
PAIRED="${OUT}/all_paired.parquet"

[[ -f "$PAIRED" ]] || { echo "ERROR: $PAIRED not found — run the pipeline first (sbatch scripts/slurm/run.sbatch)" >&2; exit 1; }

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

run_stage skeletons python scripts/site/build_transcript_skeletons.py \
    --gtf "$GTF" --parquet "$PAIRED" --out "${OUT}/transcript_skeletons.parquet"

run_stage evidence python scripts/site/build_evidence_records.py \
    --parquet "$PAIRED" --out "${OUT}/llm_evidence/" \
    --variants-long-out "${OUT}/variants_long.parquet"

if (( skip_llm )); then
    _stamp "STAGE SKIP:  llm (--skip-llm — reusing existing ${OUT}/llm/)"
else
    # Two-pass flow the site consumes: per-(isoform,criterion) reads, then a
    # per-isoform synthesis over those reads. Each call uses a COMPACT
    # slice_criterion payload (~7k tokens worst case) — never the full _raw
    # record (which is megabytes of raw variant hits and blows past the API
    # context limit). Writes {tis_slug}/criteria.json and {tis_slug}/synthesis.json.
    run_stage llm_criteria python scripts/site/run_llm_interpretation.py \
        --records "${OUT}/llm_evidence/" --out "${OUT}/llm/" --pass criteria
    run_stage llm_synthesis python scripts/site/run_llm_interpretation.py \
        --records "${OUT}/llm_evidence/" --out "${OUT}/llm/" --pass synthesis
fi

run_stage structures python scripts/export/export_structures.py --preset "$PRESET"

run_stage stage bash -c 'cd website && bash prepare_deploy.sh'

echo
_stamp "TIMING SUMMARY (wall-clock per stage):"
total=0
for name in skeletons evidence llm_criteria llm_synthesis structures stage; do
    secs=${STAGE_SECS[$name]:-}
    [[ -z "$secs" ]] && continue
    printf '  %-12s %6ds  (%dm%02ds)\n' "$name" "$secs" $((secs/60)) $((secs%60))
    total=$((total + secs))
done
printf '  %-12s %6ds  (%dm%02ds)\n' "TOTAL" "$total" $((total/60)) $((total%60))
_stamp "done. website/data/ rebuilt. Deploy: cd website && railway up --no-gitignore --service swissisoform-viewer"
