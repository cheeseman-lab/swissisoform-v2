#!/usr/bin/env bash
# Download Zoonomia 241-mammal PhyloP + PhastCons BigWig tracks from UCSC.
#
# These are the pre-computed nucleotide-level conservation scores used by the
# conservation module (Path 3 in docs/reviews/conservation_module_spec.md).
# Combined size ~13 GB — fetched once, reused forever.
#
# Output:
#   data/reference/zoonomia/241-mammalian-2020v2.phyloP.bw
#   data/reference/zoonomia/241-mammalian-2020v2.phastCons.bw
#
# Idempotent: skips files that already exist and match the expected size range.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
DEST_DIR="${REPO_ROOT}/data/reference/zoonomia"

mkdir -p "${DEST_DIR}"

# UCSC Zoonomia 241-mammal track hub (Christmas et al. 2023, Science).
# Paths confirmed from the Cactus hub README; if UCSC reorganizes the hub,
# update these URLs.  See docs/reviews/conservation_module_spec.md.
PHYLOP_URL="https://hgdownload.soe.ucsc.edu/gbdb/hg38/cactus241way/cactus241way.phyloP.bw"
PHASTCONS_URL="https://hgdownload.soe.ucsc.edu/gbdb/hg38/cactus241way/cactus241way.phastCons.bw"

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

# PhyloP BigWig expected ~8 GB, PhastCons ~5 GB.  Require at least 500 MB as a
# "probably complete" sanity check — if the file is clearly truncated, redo it.
download_if_missing "${PHYLOP_URL}"    "${DEST_DIR}/cactus241way.phyloP.bw"    $((500 * 1024 * 1024))
download_if_missing "${PHASTCONS_URL}" "${DEST_DIR}/cactus241way.phastCons.bw" $((500 * 1024 * 1024))

echo "Done.  Point ConservationConfig at:"
echo "  phylop_bigwig    = ${DEST_DIR}/cactus241way.phyloP.bw"
echo "  phastcons_bigwig = ${DEST_DIR}/cactus241way.phastCons.bw"
