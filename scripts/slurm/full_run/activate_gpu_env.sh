#!/bin/bash
# Activate the first conda env that can actually run a GPU workload.
#
# Sourced by the Phase B array scripts. These used to chain
# `plm || fold || base` on env *existence*, which masks a provisioning failure:
# an env that exists but lacks the deps is activated anyway and the task dies
# later, or differently per node. Existence is also the wrong question — the
# ESM-C migration left `swissisoform-v2-plm` holding the legacy fair-esm stack
# with no `transformers` at all, so the name that reads like the embed env
# cannot run the embed. Probe for the modules the workload imports instead, and
# fail loudly when nothing satisfies them.
#
# Usage:  activate_gpu_env <label> <env1,env2,...> <module> [module...]

activate_gpu_env() {
    local label="$1" candidates="$2"
    shift 2
    local mods=("$@") env
    for env in ${candidates//,/ }; do
        conda activate "$env" 2>/dev/null || continue
        if python - "${mods[@]}" <<'PY'
import importlib.util, sys


def available(name):
    # find_spec raises rather than returning None when a PARENT package is
    # absent (no `transformers` at all, vs. no `transformers.models.esmc`).
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


sys.exit(0 if all(available(m) for m in sys.argv[1:]) else 1)
PY
        then
            echo "[$(date -u +%FT%TZ)] $label: using conda env $env"
            return 0
        fi
        echo "[$(date -u +%FT%TZ)] $label: $env cannot import ${mods[*]} — trying next" >&2
    done
    {
        echo "ERROR: no conda env can run the $label workload."
        echo "  tried:    ${candidates//,/ }"
        echo "  requires: ${mods[*]}"
        echo "  build the GPU envs first:  sbatch scripts/setup/setup_envs.sbatch"
    } >&2
    exit 1
}
