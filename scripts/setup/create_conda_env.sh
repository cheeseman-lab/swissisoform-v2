#!/usr/bin/env bash
# Reproducibly create the conda env(s) for SwissIsoform v2.
#
# `swissisoform-v2` is derived from Matteo's env (see environment.yml) so
# collaborators share one base; this script installs miniforge into lab space
# (home dirs are often quota-limited), builds the env, and editable-installs the
# package. `isoquant` is an optional separate env (environment.isoquant.yml).
#
# Usage:
#   bash scripts/setup/create_conda_env.sh                 # create main env if absent
#   bash scripts/setup/create_conda_env.sh --rebuild       # remove + recreate main env (exact match)
#   bash scripts/setup/create_conda_env.sh --with-isoquant # also build the isoquant env
set -euo pipefail

REBUILD=false
WITH_ISOQUANT=false
for arg in "$@"; do
  case "$arg" in
    --rebuild) REBUILD=true ;;
    --with-isoquant) WITH_ISOQUANT=true ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

# Repo root = two levels up from this script; pip `-e .` is resolved from cwd.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PREFIX="/lab/barcheese01/${USER}/miniforge3"   # conda in lab space, not $HOME
ENV_NAME="swissisoform-v2"

# 1) Install miniforge if absent.
if [ ! -x "${PREFIX}/bin/conda" ]; then
  echo ">> Installing miniforge to ${PREFIX}"
  TMP="/tmp/${USER}"; mkdir -p "${TMP}"
  curl -fsSL -o "${TMP}/miniforge.sh" \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash "${TMP}/miniforge.sh" -b -p "${PREFIX}"
  rm -f "${TMP}/miniforge.sh"
fi
CONDA="${PREFIX}/bin/conda"
MAMBA="${PREFIX}/bin/mamba"

env_exists() { "${CONDA}" env list | grep -qE "^${1}\s"; }

# 2) Build the main env (clean create on --rebuild so it exactly matches the spec).
if env_exists "${ENV_NAME}"; then
  if [ "${REBUILD}" = true ]; then
    echo ">> Removing existing env ${ENV_NAME} for a clean rebuild"
    "${CONDA}" env remove -n "${ENV_NAME}" -y
    "${MAMBA}" env create -f environment.yml
  else
    echo ">> ${ENV_NAME} exists; updating (use --rebuild for an exact match)"
    "${MAMBA}" env update -n "${ENV_NAME}" -f environment.yml
  fi
else
  echo ">> Creating env ${ENV_NAME}"
  "${MAMBA}" env create -f environment.yml
fi

# 3) Optional IsoQuant env (long-read quant; isolated from the pip-pinned stack).
if [ "${WITH_ISOQUANT}" = true ]; then
  if env_exists isoquant; then
    echo ">> isoquant env already exists"
  else
    echo ">> Creating env isoquant"
    "${MAMBA}" env create -f environment.isoquant.yml
  fi
fi

# 4) Smoke-test the main env.
"${PREFIX}/envs/${ENV_NAME}/bin/python" - <<'PY'
import swissisoform, pandas, pysam, numpy
print(f"env OK: swissisoform={swissisoform.__file__}")
print(f"        pandas={pandas.__version__} numpy={numpy.__version__} pysam={pysam.__version__}")
PY
echo ">> Done. Activate with:"
echo "   source ${PREFIX}/etc/profile.d/conda.sh && conda activate ${ENV_NAME}"
