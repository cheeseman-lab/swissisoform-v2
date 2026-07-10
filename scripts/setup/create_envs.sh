#!/bin/bash
# Reproducibly create the SwissIsoform v2 conda environments from pyproject extras.
#
# One tracked builder for every env (replaces the ad-hoc README recipes and the
# old install_plm_fused_kernels.sh — the fused-attention kernel is now just the
# `xformers` pin in the [plm] extra, so `uv pip install -e ".[plm]"` installs it).
# Each env is a fresh conda env + editable install of the matching extra:
#
#   base -> swissisoform-v2       .[dev]    CPU annotation pipeline + tests
#   plm  -> swissisoform-v2-plm   .[plm]    ESM-C embed (torch + fair-esm + xformers)
#   fold -> swissisoform-v2-fold  .[fold]   ESMFold2 / Boltz-2 / Chai-1 structures
#
# Usage:
#   bash scripts/setup/create_envs.sh                 # all three
#   bash scripts/setup/create_envs.sh --plm --fold    # just the GPU envs
#   bash scripts/setup/create_envs.sh --plm --flash   # + build flash-attn (GPU node!)
#   bash scripts/setup/create_envs.sh --plm --rebuild # remove + recreate -plm cleanly
#
# NOTES
#  - Installing the GPU envs is fine on a login node: xformers ships a prebuilt
#    cu124 wheel matching torch 2.6, so nothing compiles (no GPU needed to install).
#  - --flash builds flash-attn FROM SOURCE (needs nvcc + ninja) -> run on a GPU node.
#    It is a small marginal win on top of xformers; skip it unless wall-clock matters.
#  - Do NOT --rebuild an env while a job is using it: fused-kernel numerics shift the
#    unnormalized residual stream the SAE reads, so swapping kernels mid-run makes SAE
#    features inconsistent across chunks. Build/rebuild between runs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DO_BASE=0 DO_PLM=0 DO_FOLD=0 WITH_FLASH=0 REBUILD=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base)    DO_BASE=1 ;;
    --plm)     DO_PLM=1 ;;
    --fold)    DO_FOLD=1 ;;
    --flash)   WITH_FLASH=1 ;;
    --rebuild) REBUILD=1 ;;
    -h|--help) sed -n '2,29p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
  shift
done
# No env flag -> build all three.
if [[ $DO_BASE -eq 0 && $DO_PLM -eq 0 && $DO_FOLD -eq 0 ]]; then
  DO_BASE=1; DO_PLM=1; DO_FOLD=1
fi

eval "$(conda shell.bash hook)"

# Create (or update) one env from a pyproject extra. --rebuild forces a clean
# recreate; otherwise an existing env is updated in place.
make_env() {
  local name="$1" extra="$2"
  if conda env list | grep -qE "^${name}[[:space:]]"; then
    if [[ $REBUILD -eq 1 ]]; then
      echo ">> [$name] exists — removing for a clean rebuild"
      conda env remove -n "$name" -y
    else
      echo ">> [$name] exists — updating editable install (.[$extra]); --rebuild for clean"
      conda activate "$name"
      uv pip install -e ".[$extra]"
      return
    fi
  fi
  echo ">> [$name] creating from .[$extra]"
  conda create -n "$name" -c conda-forge python=3.12 uv pip -y
  conda activate "$name"
  uv pip install -e ".[$extra]"
}

[[ $DO_BASE -eq 1 ]] && make_env swissisoform-v2      dev
[[ $DO_FOLD -eq 1 ]] && make_env swissisoform-v2-fold fold
[[ $DO_PLM  -eq 1 ]] && make_env swissisoform-v2-plm  plm

# Optional flash-attn (opt-in; from-source compile, wants a GPU node).
if [[ $WITH_FLASH -eq 1 ]]; then
  [[ $DO_PLM -eq 1 ]] || { echo "--flash requires --plm" >&2; exit 1; }
  echo ">> [swissisoform-v2-plm] building flash-attn (needs nvcc + ninja; GPU node)"
  conda activate swissisoform-v2-plm
  pip install flash-attn --no-build-isolation
fi

# Sanity: torch version + which fused kernels each GPU env can import.
for e in swissisoform-v2-plm swissisoform-v2-fold; do
  conda env list | grep -qE "^${e}[[:space:]]" || continue
  echo "== $e =="
  conda activate "$e"
  python - <<'PY'
import importlib
try:
    import torch
    print("  torch:", torch.__version__, "cuda:", torch.version.cuda)
except Exception as exc:  # noqa: BLE001
    print("  torch: NOT importable —", exc)
for m in ("xformers", "flash_attn"):
    try:
        importlib.import_module(m)
        print(f"  {m}: OK")
    except Exception:
        print(f"  {m}: not installed")
PY
done
echo "Done."
