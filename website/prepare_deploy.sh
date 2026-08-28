#!/bin/bash
# Stage a self-contained Railway build context under website/.
#
# Two things the repo keeps as references but Docker can't follow:
#   1. website/data/* are symlinks into ../data/output/$RUN/ (default
#      cheeseman_test; override with the RUN env var) — dereference into real
#      files so COPY data/ bakes the actual parquet/llm.
#      Not cheeseman_13gene: that run predates a column rename and carries
#      generef_uniprot_function with no generef_keywords, while data.py reads
#      generef_function/generef_keywords — so deploying it yields a site with no
#      gene function text and an empty keyword facet.
#   2. The viewer imports swissisoform.site.evidence (a light presentation
#      module — numpy + pandas only) from the main package — copy just it + the
#      package __init__s into the build context so the image has it on
#      PYTHONPATH=/app/src (set in the Dockerfile). config.py rides along
#      because evidence.py reads its scoring thresholds (e.g. the P3 SSE
#      cutoffs the viewer renders against) from ScoringConfig rather than
#      restating them; it is dataclasses + pathlib only, so it costs nothing.
#      swissisoform.variantquery rides along the same way for the VCF scan
#      endpoint, and with it clinical/validate.py + contract.py, because the scan
#      classifies variants with the pipeline's own ConsequenceValidator rather
#      than a second implementation. That is what puts biopython in the image
#      (see Dockerfile); pysam is not needed, since the coding sequence ships in
#      orf_index.parquet and the genome readers are never called.
#
# ORF_INDEX_RUN is deliberately independent of RUN: the site *displays* one
# small run but should *scan* the whole catalogue. orf_index.parquet is ~1.4 MB
# for all 3,371 genes, whereas that run's all_paired.parquet is 2.09 GB.
#
# Both targets are gitignored build artifacts (see website/.gitignore); rerun
# this before every `railway up` so the deploy reflects the latest pipeline run
# and LLM summaries.
set -euo pipefail
cd "$(dirname "$0")"                      # website/
SRC="../data/output/${RUN:-cheeseman_test}"
INDEX_SRC="../data/output/${ORF_INDEX_RUN:-full_catalog}"

# Check every required input BEFORE touching data/ — the first thing this script
# does is delete the previous staging, so a late failure would leave the build
# context worse than when it started.
for required in "$SRC/all_paired.parquet" "$SRC/variants_long.parquet" \
                "$SRC/transcript_skeletons.parquet"; do
  if [[ ! -f "$required" ]]; then
    echo "[prepare_deploy] ERROR: $required is missing (RUN=${RUN:-cheeseman_test})." >&2
    exit 1
  fi
done
# A missing ORF index would leave the scan endpoint returning 503 with no clue why.
if [[ ! -f "$INDEX_SRC/orf_index.parquet" ]]; then
  echo "[prepare_deploy] ERROR: $INDEX_SRC/orf_index.parquet is missing." >&2
  echo "  Build it first:  python scripts/export/build_orf_index.py --run ${ORF_INDEX_RUN:-full_catalog}" >&2
  exit 1
fi

echo "[prepare_deploy] dereferencing website/data/ from $SRC"
rm -rf data/all_paired.parquet data/variants_long.parquet \
       data/transcript_skeletons.parquet data/orf_index.parquet data/llm data/structures
cp -L "$SRC/all_paired.parquet"          data/all_paired.parquet
cp -L "$SRC/variants_long.parquet"       data/variants_long.parquet
cp -L "$SRC/transcript_skeletons.parquet" data/transcript_skeletons.parquet
# Thresholds the parquet was scored with; without it the deployed site falls back
# to library defaults and its P3 language can contradict the verdicts it renders.
if [[ -f "$SRC/scoring_config.json" ]]; then
    cp -L "$SRC/scoring_config.json"     data/scoring_config.json
else
    rm -f data/scoring_config.json
    echo "  note: no scoring_config.json in $SRC (pre-sidecar run) — site uses defaults"
fi

echo "[prepare_deploy] staging ORF index from $INDEX_SRC"
cp -L "$INDEX_SRC/orf_index.parquet"     data/orf_index.parquet
# llm/ + structures/ are optional — the site degrades to placeholders without
# them (e.g. a --skip-llm build, or before GPU folding). Stage an empty llm/ so
# the app's data dir shape is consistent.
if [[ -d "$SRC/llm" ]]; then cp -rL "$SRC/llm" data/llm; else mkdir -p data/llm; fi
if [[ -d "$SRC/structures" ]]; then cp -rL "$SRC/structures" data/structures; fi

echo "[prepare_deploy] staging swissisoform.site.evidence + clinical.validate + variantquery"
rm -rf src/swissisoform scripts
mkdir -p src/swissisoform/site src/swissisoform/variantquery src/swissisoform/clinical scripts
touch scripts/__init__.py                # keep scripts/ a package so COPY scripts/ resolves
cp ../src/swissisoform/__init__.py        src/swissisoform/__init__.py
cp ../src/swissisoform/config.py          src/swissisoform/config.py
cp ../src/swissisoform/site/__init__.py   src/swissisoform/site/__init__.py
cp ../src/swissisoform/site/evidence.py   src/swissisoform/site/evidence.py
# The scan classifies variants with the pipeline's own ConsequenceValidator rather
# than a second implementation, so validate.py + the contract it reads come along.
# It needs pysam only for the genome-reading functions, which the container never
# calls: orf_index.parquet carries the coding sequence.
cp ../src/swissisoform/contract.py          src/swissisoform/contract.py
# coords.py holds the shared genomic primitives (the exon walk, interval
# membership, revcomp, chromosome naming) that variantquery and the pipeline both
# use. Kept stdlib-only for exactly this reason.
cp ../src/swissisoform/coords.py            src/swissisoform/coords.py
cp ../src/swissisoform/clinical/__init__.py src/swissisoform/clinical/__init__.py
cp ../src/swissisoform/clinical/validate.py src/swissisoform/clinical/validate.py
# variantquery: copy the whole package rather than a hand-listed set of modules.
# A hardcoded list silently omits any module added later — adding consequence.py
# without updating the list shipped an image whose scan.py could not import, so
# gunicorn had no workers and the deploy died on the healthcheck. __main__.py is
# excluded because it is a CLI with no place in a web image.
for f in ../src/swissisoform/variantquery/*.py; do
  [[ "$(basename "$f")" == "__main__.py" ]] && continue
  cp "$f" "src/swissisoform/variantquery/$(basename "$f")"
done

# Import the staged tree the way the container will: ONLY ./src on the path, so a
# module that exists in the repo but was never copied fails here instead of in
# production. Without this check a missing vendored module produces a green build,
# a dead gunicorn, and a healthcheck failure with no hint as to why.
# Log beside the build context, not in /tmp: this script runs on the SHARED head
# node, where a fixed /tmp filename is owned by whoever ran it first — the next
# person's 2> redirect fails on permissions before python even starts, and the
# failure reads as "the staged tree does not import", which is a lie. website/ is
# gitignored build output (*.log already covered), so the log belongs here with the
# thing it describes. The container's own /tmp default in scanstore.py is fine by
# contrast: that path only ever exists inside one Railway container.
IMPORT_LOG="$PWD/prepare_deploy_import.log"
echo "[prepare_deploy] verifying the staged tree imports on its own"
if ! (cd "$PWD" && env -u PYTHONPATH PYTHONPATH="$PWD/src" SWISSISOFORM_DATA_DIR="$PWD/data" \
      python -c "import swissisoform_site.app" >/dev/null 2>"$IMPORT_LOG"); then
  echo "[prepare_deploy] ERROR: the staged build context does not import." >&2
  echo "  Usually a module added under src/swissisoform/ that this script does not copy." >&2
  tail -20 "$IMPORT_LOG" >&2
  exit 1
fi

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
