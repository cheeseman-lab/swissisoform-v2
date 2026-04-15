"""Fetch and standardize clinical variants from gnomAD and ClinVar.

Synchronous wrappers using the ``requests`` library. No async dependencies.

Important: ``protein_pos`` in hit dicts is a *hint* derived from the database's
HGVSP annotation when available.  For alternative TIS sites (5' extensions,
non-canonical start codons) the database HGVSP is relative to the **canonical**
protein and will not capture mutations in the extension region. The clinical
module must re-map genomic positions to isoform-specific protein coordinates
using the TIS site's transcript model before downstream analysis.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from swissisoform.modules.clinical import parse_hgvsp_position

logger = logging.getLogger(__name__)

GNOMAD_QUERY = """
query VariantsInGene($geneSymbol: String!, $referenceGenome: ReferenceGenomeId!) {
  gene(gene_symbol: $geneSymbol, reference_genome: $referenceGenome) {
    variants(dataset: gnomad_r4) {
      variant_id
      pos
      ref
      alt
      consequence
      hgvsp
      hgvsc
      exome {
        ac
        an
        af
        ac_hom
      }
      genome {
        ac
        an
        af
        ac_hom
      }
    }
  }
}
"""


class VariantFetcher:
    """Fetch and standardize variants from gnomAD and ClinVar.

    Usage::

        fetcher = VariantFetcher(gnomad_url="https://gnomad.broadinstitute.org/api")
        variants = fetcher.fetch_gene("TP53")
        # Returns list of variant dicts in the clinical module's hit format
    """

    def __init__(
        self,
        gnomad_url: str = "https://gnomad.broadinstitute.org/api",
        clinvar_email: str = "",
        clinvar_api_key: str = "",
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.gnomad_url = gnomad_url
        self.clinvar_email = clinvar_email
        self.clinvar_api_key = clinvar_api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_gene(
        self, gene_name: str, sources: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Fetch variants from specified sources for a gene.

        Args:
            gene_name: Gene symbol (e.g. ``"TP53"``).
            sources: List of sources to query. Default: ``["gnomad", "clinvar"]``.

        Returns:
            List of variant dicts in the clinical module's standard hit format.
        """
        if sources is None:
            sources = ["gnomad", "clinvar"]
        all_variants: list[dict[str, Any]] = []
        for source in sources:
            if source == "gnomad":
                all_variants.extend(self._fetch_gnomad(gene_name))
            elif source == "clinvar":
                all_variants.extend(self._fetch_clinvar(gene_name))
        return all_variants

    # ------------------------------------------------------------------
    # gnomAD
    # ------------------------------------------------------------------

    def _fetch_gnomad(self, gene_name: str) -> list[dict[str, Any]]:
        """Fetch variants from gnomAD GraphQL API."""
        payload = {
            "query": GNOMAD_QUERY,
            "variables": {
                "geneSymbol": gene_name,
                "referenceGenome": "GRCh38",
            },
        }

        data = self._post_with_retry(self.gnomad_url, json_payload=payload)
        if data is None:
            return []

        gene_data = (data.get("data") or {}).get("gene")
        if gene_data is None:
            return []

        raw_variants = gene_data.get("variants") or []
        hits: list[dict[str, Any]] = []
        for v in raw_variants:
            freq = v.get("exome") or v.get("genome")
            if freq is None or freq.get("af", 0) == 0:
                continue

            # protein_pos from hgvsp is a *hint* — only valid for the
            # canonical isoform.  For 5' extensions / non-canonical starts
            # this will be None and the caller must remap using
            # genomic_pos + the isoform's transcript model.
            protein_pos = parse_hgvsp_position(v.get("hgvsp"))

            hits.append(
                {
                    "source": "gnomAD",
                    "variant_id": v.get("variant_id", ""),
                    "chrom": f"chr{v.get('variant_id', '').split('-')[0]}"
                    if v.get("variant_id")
                    else "",
                    "genomic_pos": v.get("pos", 0),
                    "ref": v.get("ref", ""),
                    "alt": v.get("alt", ""),
                    "consequence": v.get("consequence", ""),
                    "protein_pos": protein_pos,
                    "hgvsp": v.get("hgvsp"),
                    "allele_frequency": freq.get("af"),
                    "clinical_significance": None,
                    "metadata": {
                        "allele_count": freq.get("ac"),
                        "allele_number": freq.get("an"),
                        "homozygote_count": freq.get("ac_hom"),
                        "hgvsc": v.get("hgvsc"),
                    },
                }
            )
        return hits

    # ------------------------------------------------------------------
    # ClinVar
    # ------------------------------------------------------------------

    def _fetch_clinvar(self, gene_name: str) -> list[dict[str, Any]]:
        """Fetch variants from ClinVar via NCBI E-utilities."""
        # Step 1: esearch
        esearch_params: dict[str, Any] = {
            "db": "clinvar",
            "term": f"{gene_name}[gene]",
            "retmax": 500,
            "retmode": "json",
        }
        if self.clinvar_email:
            esearch_params["email"] = self.clinvar_email
        if self.clinvar_api_key:
            esearch_params["api_key"] = self.clinvar_api_key

        esearch_data = self._get_with_retry(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params=esearch_params,
        )
        if esearch_data is None:
            return []

        id_list = (esearch_data.get("esearchresult") or {}).get("idlist") or []
        if not id_list:
            return []

        # Step 2: esummary in batches of 50
        hits: list[dict[str, Any]] = []
        for i in range(0, len(id_list), 50):
            batch_ids = id_list[i : i + 50]
            time.sleep(0.35)  # NCBI rate limiting

            esummary_params: dict[str, Any] = {
                "db": "clinvar",
                "id": ",".join(batch_ids),
                "retmode": "json",
            }
            if self.clinvar_email:
                esummary_params["email"] = self.clinvar_email
            if self.clinvar_api_key:
                esummary_params["api_key"] = self.clinvar_api_key

            esummary_data = self._get_with_retry(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params=esummary_params,
            )
            if esummary_data is None:
                continue

            result = esummary_data.get("result") or {}
            uids = result.get("uids") or []
            for uid in uids:
                record = result.get(uid)
                if record is None:
                    continue
                hits.append(self._parse_clinvar_record(record))

        return hits

    def _parse_clinvar_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Parse a single ClinVar esummary record into a hit dict."""
        uid = record.get("uid", "")
        title = record.get("title", "")
        clin_sig = (record.get("clinical_significance") or {}).get("description", "")

        # Extract genomic location (GRCh38)
        chrom = ""
        genomic_pos = 0
        variation_sets = record.get("variation_set") or []
        if variation_sets:
            locs = variation_sets[0].get("variation_loc") or []
            for loc in locs:
                if loc.get("assembly_name") == "GRCh38":
                    chrom = f"chr{loc.get('chr', '')}"
                    genomic_pos = loc.get("start", 0)
                    break

        # Extract protein change from title
        hgvsp = _extract_protein_from_title(title)
        protein_pos = parse_hgvsp_position(hgvsp)

        return {
            "source": "ClinVar",
            "variant_id": f"ClinVar:{uid}",
            "chrom": chrom,
            "genomic_pos": genomic_pos,
            "ref": "",
            "alt": "",
            "consequence": record.get("obj_type", ""),
            "protein_pos": protein_pos,
            "hgvsp": hgvsp,
            "allele_frequency": None,
            "clinical_significance": clin_sig or None,
            "metadata": {
                "title": title,
                "review_status": (record.get("clinical_significance") or {}).get(
                    "review_status", ""
                ),
            },
        }

    # ------------------------------------------------------------------
    # HTTP helpers with retry
    # ------------------------------------------------------------------

    def _post_with_retry(
        self, url: str, json_payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        """POST with retry and exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, json=json_payload, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                if attempt < self.max_retries - 1:
                    wait = self.retry_delay * (2**attempt)
                    logger.warning(
                        "gnomAD request failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1,
                        self.max_retries,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "gnomAD request failed after %d attempts: %s",
                        self.max_retries,
                        exc,
                    )
        return None

    def _get_with_retry(
        self, url: str, params: dict[str, Any]
    ) -> dict[str, Any] | None:
        """GET with retry and exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                resp = requests.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                if attempt < self.max_retries - 1:
                    wait = self.retry_delay * (2**attempt)
                    logger.warning(
                        "ClinVar request failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1,
                        self.max_retries,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "ClinVar request failed after %d attempts: %s",
                        self.max_retries,
                        exc,
                    )
        return None


def _extract_protein_from_title(title: str) -> str | None:
    """Extract protein change notation from a ClinVar title.

    Looks for patterns like ``(p.Pro72Arg)`` in the title string.
    """
    import re

    match = re.search(r"\(p\.[^)]+\)", title)
    if match:
        return match.group().strip("()")
    return None
