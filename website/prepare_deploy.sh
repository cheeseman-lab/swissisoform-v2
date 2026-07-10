#!/bin/bash
# Stage a self-contained Railway build context under website/.
#
# Two things the repo keeps as references but Docker can't follow:
#   1. website/data/* are symlinks into ../data/output/$RUN/ (default
#      cheeseman_13gene; override with the RUN env var) — dereference into real
#      files so COPY data/ bakes the actual parquet/llm.
#   2. The viewer imports swissisoform.site.evidence (a light presentation
#      module — numpy + pandas only) from the main package — copy just it + the
#      package __init__s into the build context so the image has it on
#      PYTHONPATH=/app/src (set in the Dockerfile).
#
# Both targets are gitignored build artifacts (see website/.gitignore); rerun
# this before every `railway up` so the deploy reflects the latest pipeline run
# and LLM summaries.
set -euo pipefail
cd "$(dirname "$0")"                      # website/
SRC="../data/output/${RUN:-cheeseman_13gene}"

echo "[prepare_deploy] dereferencing website/data/ from $SRC"
rm -rf data/all_paired.parquet data/variants_long.parquet \
       data/transcript_skeletons.parquet data/llm data/structures
cp -L "$SRC/all_paired.parquet"          data/all_paired.parquet
cp -L "$SRC/variants_long.parquet"       data/variants_long.parquet
cp -L "$SRC/transcript_skeletons.parquet" data/transcript_skeletons.parquet
# llm/ + structures/ are optional — the site degrades to placeholders without
# them (e.g. a --skip-llm build, or before GPU folding). Stage an empty llm/ so
# the app's data dir shape is consistent.
if [[ -d "$SRC/llm" ]]; then cp -rL "$SRC/llm" data/llm; else mkdir -p data/llm; fi
if [[ -d "$SRC/structures" ]]; then cp -rL "$SRC/structures" data/structures; fi

echo "[prepare_deploy] staging swissisoform.site.evidence into the build context"
rm -rf src/swissisoform scripts
mkdir -p src/swissisoform/site scripts
touch scripts/__init__.py                # keep scripts/ a package so COPY scripts/ resolves
cp ../src/swissisoform/__init__.py        src/swissisoform/__init__.py
cp ../src/swissisoform/site/__init__.py   src/swissisoform/site/__init__.py
cp ../src/swissisoform/site/evidence.py   src/swissisoform/site/evidence.py

# Purge bytecode from the whole build context. --no-gitignore (needed to
# upload the gitignored data/ + scripts/) also uploads __pycache__, and stale
# .pyc files (full of null bytes) crash the image build with
# "source code string cannot contain null bytes". Delete them physically.
echo "[prepare_deploy] purging __pycache__/*.pyc from the build context"
find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find . -name '*.pyc' -delete 2>/dev/null || true

echo "[prepare_deploy] done. Build context size:"
du -sh data src/swissisoform
echo "[prepare_deploy] next: railway up --no-gitignore --service swissisoform-viewer"
