#!/bin/bash
# Lightweight reachability test: can a partition-20 compute node reach SRA?
# Resolves + HEAD-requests the archive URL instead of downloading it, so this
# finishes in seconds rather than pulling tens of GB.
#SBATCH --job-name=eblack_sra_conntest
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail   # NOT -e: we want every probe to run and report

source /lab/barcheese01/${USER}/miniforge3/etc/profile.d/conda.sh
conda activate swissisoform-v2

ACC=SRX26164613
echo "=== node: $(hostname) ==="

echo "=== probe 1: outbound HTTPS to NCBI ==="
curl -sS -o /dev/null -w "HTTP %{http_code}  (%{time_total}s)\n" -I https://www.ncbi.nlm.nih.gov/ \
  || echo "FAILED: no outbound HTTPS to ncbi.nlm.nih.gov"

echo "=== probe 2: SRA name resolution (srapath ${ACC}) ==="
URLS=$(srapath "${ACC}" 2>&1) && echo "${URLS}" || echo "FAILED: srapath could not resolve ${ACC}"

echo "=== probe 3: HEAD the resolved archive URL (no download) ==="
FIRST_URL=$(echo "${URLS}" | grep -Eo 'https?://[^ ]+' | head -1)
if [ -n "${FIRST_URL}" ]; then
  echo "resolved -> ${FIRST_URL}"
  curl -sS -o /dev/null -w "HTTP %{http_code}  size=%{size_download}  (%{time_total}s)\n" -I "${FIRST_URL}" \
    || echo "FAILED: could not HEAD the SRA archive URL"
else
  echo "SKIPPED: no http(s) URL resolved to HEAD"
fi

echo "=== probe 4: ENA fallback reachability ==="
curl -sS -o /dev/null -w "HTTP %{http_code}  (%{time_total}s)\n" \
  -I "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=${ACC}&result=read_run" \
  || echo "FAILED: no outbound HTTPS to ENA"

echo "=== done ==="
