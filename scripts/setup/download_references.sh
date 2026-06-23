#!/bin/bash

################################################################################
# download_references.sh
#
# Downloads GENCODE v49 reference files from the GENCODE FTP server.
# Extracts gzipped files and verifies completion.
#
# Usage:
#   scripts/setup/download_references.sh [--force]
#
# Options:
#   --force    Re-download files even if they already exist
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
REFERENCE_DIR="${REPO_ROOT}/data/reference"
GENCODE_FTP="https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49"

FORCE=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE=true
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--force]" >&2
            exit 1
            ;;
    esac
done

# Ensure reference directory exists
if [ ! -d "$REFERENCE_DIR" ]; then
    echo "Creating $REFERENCE_DIR..."
    mkdir -p "$REFERENCE_DIR"
fi

echo "========================================"
echo "GENCODE v49 Reference Download"
echo "========================================"
echo "Target directory: $REFERENCE_DIR"
echo "Force re-download: $FORCE"
echo ""

################################################################################
# Function to download and extract a file
################################################################################
download_and_extract() {
    local filename=$1
    local output_name=$2
    local output_path="${REFERENCE_DIR}/${output_name}"
    local temp_gz="${REFERENCE_DIR}/${filename}"

    echo "Processing: $filename"

    # Check if output file already exists
    if [ -f "$output_path" ]; then
        if [ "$FORCE" = false ]; then
            echo "  → Already exists (use --force to re-download)"
            return 0
        else
            echo "  → Removing existing file (--force enabled)"
            rm -f "$output_path"
        fi
    fi

    # Remove temp file if it exists (from interrupted download)
    if [ -f "$temp_gz" ]; then
        echo "  → Removing incomplete temp file"
        rm -f "$temp_gz"
    fi

    # Download the file
    echo "  → Downloading from GENCODE FTP..."
    if wget -q --show-progress "${GENCODE_FTP}/${filename}" -O "$temp_gz"; then
        echo "  → Download complete"
    else
        echo "  ✗ Download failed!" >&2
        exit 1
    fi

    # Extract the gzipped file
    echo "  → Extracting to $output_name..."
    if gunzip -f "$temp_gz"; then
        echo "  ✓ Successfully processed"
    else
        echo "  ✗ Extraction failed!" >&2
        exit 1
    fi
    echo ""
}

################################################################################
# Download and extract the three reference files
################################################################################

download_and_extract \
    "GRCh38.primary_assembly.genome.fa.gz" \
    "GRCh38.primary_assembly.genome.fa"

download_and_extract \
    "gencode.v49.pc_translations.fa.gz" \
    "gencode.v49.pc_translations.fa"

download_and_extract \
    "gencode.v49.primary_assembly.annotation.gtf.gz" \
    "gencode.v49.primary_assembly.annotation.gtf"

################################################################################
# Verify all files exist
################################################################################
echo "========================================"
echo "Verification"
echo "========================================"

REQUIRED_FILES=(
    "GRCh38.primary_assembly.genome.fa"
    "gencode.v49.pc_translations.fa"
    "gencode.v49.primary_assembly.annotation.gtf"
)

all_present=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "${REFERENCE_DIR}/${file}" ]; then
        size=$(du -h "${REFERENCE_DIR}/${file}" | cut -f1)
        echo "✓ ${file} (${size})"
    else
        echo "✗ ${file} (MISSING)" >&2
        all_present=false
    fi
done
echo ""

################################################################################
# Check for Ribo-TISH predict_all files
################################################################################
echo "========================================"
echo "Ribo-TISH predict_all Files Status"
echo "========================================"
echo "(These must be copied manually from Ribo-TISH output)"
echo ""

RIBOTISH_FILES=(
    "HeLa_TIS_predict_all.txt"
    "K562_TIS_predict_all.txt"
    "U2OS_TIS_predict_all.txt"
    "RPE1_Async_TIS_predict_all.txt"
    "RPE1_Que_TIS_predict_all.txt"
    "RPE1_Sen_TIS_predict_all.txt"
)

ribotish_present=true
for file in "${RIBOTISH_FILES[@]}"; do
    if [ -f "${REFERENCE_DIR}/${file}" ]; then
        size=$(du -h "${REFERENCE_DIR}/${file}" | cut -f1)
        echo "✓ ${file} (${size})"
    else
        echo "✗ ${file} (MISSING)" >&2
        ribotish_present=false
    fi
done
echo ""

################################################################################
# Final summary
################################################################################
echo "========================================"
if [ "$all_present" = true ]; then
    echo "✓ All GENCODE reference files ready!"
else
    echo "✗ Some reference files are missing!" >&2
    exit 1
fi

if [ "$ribotish_present" = false ]; then
    echo "⚠ Note: Some Ribo-TISH predict_all files are missing."
    echo "  Copy them manually to data/reference/"
fi
echo "========================================"
