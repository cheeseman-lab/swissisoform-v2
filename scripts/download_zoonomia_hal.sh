#!/usr/bin/env bash
# Download the Zoonomia 241-mammal Cactus HAL alignment.
#
# Backs Path 1/2 (primate + mammalian reading-frame integrity) in
# swissisoform.modules.conservation_frame. The HAL is large
# (~200-600 GB); only download if you actually plan to run Path 1/2.
#
# Usage: bash scripts/download_zoonomia_hal.sh [dest_dir]
#
# Defaults to data/reference/zoonomia/. Skips the download if the file
# already exists. Writes a sidecar .provenance.txt next to the HAL so
# downstream users can see which release this was.

set -euo pipefail

DEST_DIR="${1:-data/reference/zoonomia}"
HAL_FILENAME="241-mammalian-2020v2.hal"
HAL_URL="https://cglgenomics.ucsc.edu/data/cactus/241-mammalian-2020v2.hal"

mkdir -p "$DEST_DIR"
HAL_PATH="$DEST_DIR/$HAL_FILENAME"

if [[ -f "$HAL_PATH" ]]; then
    echo "HAL already present: $HAL_PATH"
    echo "Size: $(du -h "$HAL_PATH" | cut -f1)"
    exit 0
fi

echo "Downloading $HAL_URL"
echo "  -> $HAL_PATH"
echo "  (This is 200-600 GB; plan accordingly.)"

# --retry handles transient network hiccups; --continue-at - resumes if interrupted.
curl --fail --retry 5 --continue-at - --output "$HAL_PATH" "$HAL_URL"

cat >"$HAL_PATH.provenance.txt" <<EOF
source: $HAL_URL
downloaded: $(date -u +%Y-%m-%dT%H:%M:%SZ)
reference: Christmas MJ et al. Science 380:eabn3943 (2023)
consumed_by: swissisoform.modules.conservation_frame
EOF

echo "Done. Verify with: halStats --genomes $HAL_PATH"
