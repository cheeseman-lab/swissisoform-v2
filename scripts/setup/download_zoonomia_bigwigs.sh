#!/usr/bin/env bash
# Download Zoonomia 241-mammal PhyloP BigWig + phylogenetic tree from UCSC.
#
# The PhyloP track is the pre-computed nucleotide-level conservation score
# consumed by Path 3 of the conservation module (docs/reviews/
# conservation_module_spec.md).  ~9 GB, fetched once, reused forever.
#
# Note: UCSC does not ship a PhastCons track for the 241-mammal alignment.
# We fall back to the hg38 100-vertebrate PhastCons track (broader
# clade, still a real selection signal) as the default ``phastcons_bigwig``
# input.  ConservationModule returns None for phastcons_* metrics when no
# track is configured, so skipping the download is safe.
#
# The Newick species tree is downloaded alongside (tens of KB) and used by
# ConservationFrameModule to build phylogenetic-depth rankings without
# requiring halStats + the full HAL on every machine.
#
# Output:
#   data/reference/zoonomia/cactus241way.phyloP.bw
#   data/reference/zoonomia/hg38.cactus241way.nh
#   data/reference/zoonomia/hg38.phastCons100way.bw
#
# Idempotent: skips files that already exist and match the expected size range.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "${SCRIPT_DIR}")")"
DEST_DIR="${REPO_ROOT}/data/reference/zoonomia"

mkdir -p "${DEST_DIR}"

# Confirmed via https://hgdownload.soe.ucsc.edu/goldenpath/hg38/cactus241way/
# as of 2026-04-21 (UCSC may reorganize the goldenpath hierarchy in future —
# update these URLs if they 404).
PHYLOP_URL="https://hgdownload.soe.ucsc.edu/goldenpath/hg38/cactus241way/cactus241way.phyloP.bw"
TREE_URL="https://hgdownload.soe.ucsc.edu/goldenpath/hg38/cactus241way/hg38.cactus241way.nh"
# 100-vertebrate PhastCons — broader clade than the 241-mammal alignment, but
# it's the only hg38 PhastCons track UCSC currently ships.  ~9 GB.
PHASTCONS_URL="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons100way/hg38.phastCons100way.bw"

download_if_missing() {
    local url="$1"
    local dest="$2"
    local min_bytes="$3"

    if [[ -f "${dest}" ]]; then
        local size
        size=$(stat -c '%s' "${dest}")
        if (( size >= min_bytes )); then
            echo "SKIP  ${dest} (${size} bytes)"
            return 0
        fi
        echo "REDO  ${dest} (${size} bytes, below ${min_bytes})"
        rm -f "${dest}"
    fi

    echo "FETCH ${url}"
    echo "   -> ${dest}"
    curl --fail --location --retry 3 --output "${dest}" "${url}"
}

# PhyloP BigWig expected ~9 GB; 500 MB minimum as a truncation sanity check.
download_if_missing "${PHYLOP_URL}"    "${DEST_DIR}/cactus241way.phyloP.bw"    $((500 * 1024 * 1024))
download_if_missing "${PHASTCONS_URL}" "${DEST_DIR}/hg38.phastCons100way.bw"   $((500 * 1024 * 1024))
# Tree is ~11 KB — just redownload if missing or smaller than 1 KB.
download_if_missing "${TREE_URL}"      "${DEST_DIR}/hg38.cactus241way.nh"      1024

echo "Done.  Point ConservationConfig at:"
echo "  phylop_bigwig    = ${DEST_DIR}/cactus241way.phyloP.bw"
echo "  phastcons_bigwig = ${DEST_DIR}/hg38.phastCons100way.bw"
echo "  hal_tree_newick  = \$(cat ${DEST_DIR}/hg38.cactus241way.nh)"
