#!/bin/bash
#SBATCH --job-name=ating_copy_refdata
#SBATCH --partition=18
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --output=/lab/barcheese01/ating/swissisoform-v2/logs/%x_%j.out
#SBATCH --error=/lab/barcheese01/ating/swissisoform-v2/logs/%x_%j.err

set -euo pipefail

SRC=/lab/barcheese01/ating/swissisoform-v2/data/
DST=/lab/barcheese01/ating/swissisoform-v2/data/

echo "Start: $(date)"
echo "Copying ${SRC} -> ${DST}"

# Additive (no --delete): preserves ating's existing data/output/5gene_e2e.
# --no-o (owner becomes ating), keep perms/times/group. Skip macOS cruft.
#
# Two passes so the 824G reference/zoonomia tree transfers LAST: everything
# else (genome, GTF, predict files, caches, outputs) lands first and is usable
# while the big tree is still copying, and a failure in the cheap pass fails
# fast before committing to the 800G+ move.

# Pass 1: everything except reference/zoonomia.
echo "Pass 1/2: all data except reference/zoonomia ..."
rsync -a --no-o --info=progress2,stats2 \
  --exclude='.DS_Store' \
  --exclude='reference/zoonomia/' \
  "${SRC}" "${DST}"

# Pass 2: reference/zoonomia (the 824G bulk), last.
echo "Pass 2/2: reference/zoonomia (~824G) ..."
rsync -a --no-o --info=progress2,stats2 \
  --exclude='.DS_Store' \
  "${SRC}reference/zoonomia/" "${DST}reference/zoonomia/"

echo "Done: $(date)"
du -sh "${DST}"
