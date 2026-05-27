"""Fetch and standardize clinical variants from gnomAD, ClinVar, and COSMIC.

Synchronous wrappers using the ``requests`` library. No async dependencies.

**protein_pos policy:** hit dicts always have ``protein_pos=None`` from the
fetcher.  Database HGVSP strings are relative to the **canonical** transcript
and would be wrong for alternative TIS sites (5' extensions, non-canonical
starts).  The raw HGVSP string is preserved in ``hit["hgvsp"]`` and
``hit["metadata"]["hgvsp_canonical_hint"]`` so downstream code can parse it
if it has reason to trust canonical-frame coordinates, but the authoritative
source of ``protein_pos`` is ``ConsequenceValidator``, which re-maps genomic
positions to isoform-specific protein coordinates using the transcript model.

Until a validator runs, ``ClinicalModule.annotate`` filters variants by
``protein_pos is not None`` and will return zero hits — this is the correct
behavior when we genuinely don't know where the variant falls in the isoform
protein.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import pandas as pd
import requests

from swissisoform.modules.clinical import parse_hgvsp_position

logger = logging.getLogger(__name__)


def _nz(val: Any) -> Any:
    """Normalize pandas nan / empty-string / None → None."""
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    if isinstance(val, str) and val == "":
        return None
    return val


def _str_or_empty(val: Any) -> str:
    """Like ``_nz`` but returns ``""`` instead of ``None``."""
    return "" if _nz(val) is None else str(val)


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
    """Fetch and standardize variants from gnomAD, ClinVar, and COSMIC.

    Usage::

        fetcher = VariantFetcher()
        variants = fetcher.fetch_gene("TP53")
        # Returns list of variant dicts in the clinical module's hit format

        # With COSMIC (local parquet):
        fetcher = VariantFetcher(cosmic_db="/path/to/cosmic_variants_combined.parquet")
        variants = fetcher.fetch_gene("TP53", sources=["gnomad", "clinvar", "cosmic"])
    """

    def __init__(
        self,
        gnomad_url: str = "https://gnomad.broadinstitute.org/api",
        clinvar_email: str = "",
        clinvar_api_key: str = "",
        cosmic_db: str | None = None,
        gnomad_db: str | None = None,
        clinvar_db: str | None = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """Initialize a VariantFetcher.

        When the bulk-parquet paths (*gnomad_db*, *clinvar_db*, *cosmic_db*)
        are provided, ``fetch_gene`` reads from local files via PyArrow
        filter pushdown.  When omitted, it falls back to the live HTTP
        sources (gnomAD GraphQL, NCBI E-utilities).  The HTTP fallbacks
        are intended for tests and first-run situations where the setup
        script hasn't populated the bulk artifacts yet.

        Args:
            gnomad_url: gnomAD GraphQL endpoint (used only when
                *gnomad_db* is None).
            clinvar_email: NCBI E-utilities email (used only when *clinvar_db* is None).
            clinvar_api_key: NCBI E-utilities API key (used only when *clinvar_db* is None).
            cosmic_db: Path to the standardized COSMIC parquet built by
                ``scripts/setup/setup_databases.py cosmic``.
            gnomad_db: Path to the gene-indexed gnomAD parquet built by
                ``scripts/setup/setup_databases.py gnomad``.
            clinvar_db: Path to the ClinVar variant_summary parquet
                built by ``scripts/setup/setup_databases.py clinvar``.
            timeout: HTTP timeout in seconds.
            max_retries: Max HTTP retry attempts.
            retry_delay: Base delay between retries in seconds.
        """
        self.gnomad_url = gnomad_url
        self.clinvar_email = clinvar_email
        self.clinvar_api_key = clinvar_api_key
        self.cosmic_db = cosmic_db
        self.gnomad_db = gnomad_db
        self.clinvar_db = clinvar_db
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_gene(self, gene_name: str, sources: list[str] | None = None) -> list[dict[str, Any]]:
        """Fetch variants from specified sources for a gene.

        Args:
            gene_name: Gene symbol (e.g. ``"TP53"``).
            sources: List of sources to query. Default: ``["gnomad", "clinvar"]``.

        Returns:
            List of variant dicts in the clinical module's standard hit format.
        """
        if sources is None:
            sources = ["gnomad", "clinvar", "cosmic"]
        all_variants: list[dict[str, Any]] = []
        for source in sources:
            if source == "gnomad":
                all_variants.extend(self._fetch_gnomad(gene_name))
            elif source == "clinvar":
                all_variants.extend(self._fetch_clinvar(gene_name))
            elif source == "cosmic":
                all_variants.extend(self._fetch_cosmic(gene_name))
        return all_variants

    # ------------------------------------------------------------------
    # gnomAD
    # ------------------------------------------------------------------

    def _fetch_gnomad(self, gene_name: str) -> list[dict[str, Any]]:
        """Fetch gnomAD variants for *gene_name*.

        Prefers the local bulk parquet (``self.gnomad_db``) when configured;
        falls back to the live GraphQL API otherwise.
        """
        if self.gnomad_db:
            return self._fetch_gnomad_local(gene_name)
        return self._fetch_gnomad_api(gene_name)

    def _fetch_gnomad_local(self, gene_name: str) -> list[dict[str, Any]]:
        """Read gnomAD variants for *gene_name* from the bulk parquet."""
        if not os.path.exists(self.gnomad_db):
            logger.warning("gnomAD parquet not found at %s — returning empty", self.gnomad_db)
            return []
        try:
            import pyarrow.dataset as ds

            dataset = ds.dataset(self.gnomad_db, format="parquet")
            table = dataset.to_table(filter=ds.field("gene_symbol") == gene_name)
            df = table.to_pandas()
        except Exception as exc:
            logger.error("gnomAD parquet query for %s failed: %s", gene_name, exc)
            return []

        hits: list[dict[str, Any]] = []
        for _, v in df.iterrows():
            hgvsp = _nz(v.get("hgvsp"))
            hgvsc = _nz(v.get("hgvsc"))
            hits.append(
                {
                    "source": "gnomAD",
                    "variant_id": str(v["variant_id"]),
                    "chrom": f"chr{v['chrom']}"
                    if not str(v["chrom"]).startswith("chr")
                    else str(v["chrom"]),
                    "genomic_pos": int(v["pos"]),
                    "ref": str(v["ref"]),
                    "alt": str(v["alt"]),
                    "consequence": str(v["consequence"]),
                    "protein_pos": None,  # ConsequenceValidator is authoritative
                    "hgvsp": hgvsp,
                    "allele_frequency": (
                        float(v["allele_frequency"])
                        if _nz(v["allele_frequency"]) is not None
                        else None
                    ),
                    "clinical_significance": None,
                    "metadata": {
                        "hgvsc": hgvsc,
                        "hgvsp_canonical_hint": parse_hgvsp_position(hgvsp),
                        "protein_position_vep": _nz(v.get("protein_position")),
                    },
                }
            )
        return hits

    def _fetch_gnomad_api(self, gene_name: str) -> list[dict[str, Any]]:
        """Fetch variants from gnomAD GraphQL API (fallback when no parquet)."""
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

            # protein_pos is intentionally None here — the HGVSP string is
            # canonical-coords and would be wrong for alternative TIS.
            # The canonical-frame hint is preserved in metadata for downstream
            # validators / debugging.
            hgvsp = v.get("hgvsp")
            canonical_hint = parse_hgvsp_position(hgvsp)

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
                    "protein_pos": None,  # set only by ConsequenceValidator
                    "hgvsp": hgvsp,
                    "allele_frequency": freq.get("af"),
                    "clinical_significance": None,
                    "metadata": {
                        "allele_count": freq.get("ac"),
                        "allele_number": freq.get("an"),
                        "homozygote_count": freq.get("ac_hom"),
                        "hgvsc": v.get("hgvsc"),
                        "hgvsp_canonical_hint": canonical_hint,
                    },
                }
            )
        return hits

    # ------------------------------------------------------------------
    # ClinVar
    # ------------------------------------------------------------------

    def _fetch_clinvar(self, gene_name: str) -> list[dict[str, Any]]:
        """Fetch ClinVar variants for *gene_name*.

        Prefers the local bulk parquet (``self.clinvar_db``) when configured;
        falls back to the live NCBI E-utilities otherwise.
        """
        if self.clinvar_db:
            return self._fetch_clinvar_local(gene_name)
        return self._fetch_clinvar_api(gene_name)

    def _fetch_clinvar_local(self, gene_name: str) -> list[dict[str, Any]]:
        """Read ClinVar variants for *gene_name* from the bulk parquet."""
        if not os.path.exists(self.clinvar_db):
            logger.warning("ClinVar parquet not found at %s — returning empty", self.clinvar_db)
            return []
        try:
            import pyarrow.dataset as ds

            dataset = ds.dataset(self.clinvar_db, format="parquet")
            table = dataset.to_table(
                filter=(ds.field("GeneSymbol") == gene_name) & (ds.field("Assembly") == "GRCh38")
            )
            df = table.to_pandas()
        except Exception as exc:
            logger.error("ClinVar parquet query for %s failed: %s", gene_name, exc)
            return []

        hits: list[dict[str, Any]] = []
        for _, r in df.iterrows():
            title = _str_or_empty(r.get("Name"))
            hgvsp = _extract_protein_from_title(title) if title else None
            # Prefer VCF-format alleles: guaranteed genomic +strand orientation.
            # `ReferenceAllele`/`AlternateAllele` (non-VCF) are usually empty in
            # modern ClinVar dumps, and the HGVSc title parse returns
            # transcript-direction bases — wrong for minus-strand genes. The
            # VCF columns are the correct genomic-direction source.
            ref_vcf = _str_or_empty(r.get("ReferenceAlleleVCF"))
            alt_vcf = _str_or_empty(r.get("AlternateAlleleVCF"))
            pos_vcf = r.get("PositionVCF")
            if ref_vcf and alt_vcf and pos_vcf:
                ref = ref_vcf
                alt = alt_vcf
                genomic_pos = int(pos_vcf)
            else:
                ref = _str_or_empty(r.get("ReferenceAllele"))
                alt = _str_or_empty(r.get("AlternateAllele"))
                if not ref or not alt:
                    parsed_ref, parsed_alt = (
                        _extract_ref_alt_from_title(title) if title else ("", "")
                    )
                    ref = ref or parsed_ref
                    alt = alt or parsed_alt
                genomic_pos = int(r["Start"]) if r.get("Start") else 0
            chrom_raw = _str_or_empty(r.get("Chromosome"))
            chrom = (
                f"chr{chrom_raw}" if chrom_raw and not chrom_raw.startswith("chr") else chrom_raw
            )
            clin_sig = _str_or_empty(r.get("ClinicalSignificance")) or None

            hits.append(
                {
                    "source": "ClinVar",
                    "variant_id": f"ClinVar:{r.get('VariationID', '')}",
                    "chrom": chrom,
                    "genomic_pos": genomic_pos,
                    "ref": ref,
                    "alt": alt,
                    "consequence": str(r.get("Type") or ""),
                    "protein_pos": None,  # ConsequenceValidator is authoritative
                    "hgvsp": hgvsp,
                    "allele_frequency": None,
                    "clinical_significance": clin_sig,
                    "metadata": {
                        "title": title,
                        "rs_id": r.get("RS# (dbSNP)") or None,
                        "hgvsp_canonical_hint": parse_hgvsp_position(hgvsp),
                    },
                }
            )
        return hits

    def _fetch_clinvar_api(self, gene_name: str) -> list[dict[str, Any]]:
        """Fetch variants from ClinVar via NCBI E-utilities (fallback)."""
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

        # Parse HGVSp (protein change) and HGVSc (coding change) from title
        hgvsp = _extract_protein_from_title(title)
        ref, alt = _extract_ref_alt_from_title(title)
        canonical_hint = parse_hgvsp_position(hgvsp)

        return {
            "source": "ClinVar",
            "variant_id": f"ClinVar:{uid}",
            "chrom": chrom,
            "genomic_pos": genomic_pos,
            "ref": ref,
            "alt": alt,
            "consequence": record.get("obj_type", ""),
            # protein_pos left None — HGVSp is canonical-frame, not isoform.
            # ConsequenceValidator is the authoritative source.
            "protein_pos": None,
            "hgvsp": hgvsp,
            "allele_frequency": None,
            "clinical_significance": clin_sig or None,
            "metadata": {
                "title": title,
                "review_status": (record.get("clinical_significance") or {}).get(
                    "review_status", ""
                ),
                "hgvsp_canonical_hint": canonical_hint,
            },
        }

    # ------------------------------------------------------------------
    # COSMIC
    # ------------------------------------------------------------------

    def _fetch_cosmic(self, gene_name: str) -> list[dict[str, Any]]:
        """Fetch variants from a local COSMIC parquet database.

        Uses PyArrow filter pushdown for efficient per-gene queries without
        loading the entire file into memory.
        """
        if not self.cosmic_db:
            logger.info("No COSMIC database path configured — skipping COSMIC")
            return []

        import os

        if not os.path.exists(self.cosmic_db):
            logger.warning("COSMIC database not found at %s", self.cosmic_db)
            return []

        try:
            import pyarrow.dataset as ds

            dataset = ds.dataset(self.cosmic_db, format="parquet")
            table = dataset.to_table(
                filter=(ds.field("GENE_SYMBOL") == gene_name)
                | (ds.field("GENE_SYMBOL") == gene_name.upper())
            )
            cosmic_df = table.to_pandas()
        except Exception as exc:
            logger.error("Failed to query COSMIC for %s: %s", gene_name, exc)
            return []

        if cosmic_df.empty:
            return []

        hits: list[dict[str, Any]] = []
        seen: set[tuple[str, int, str, str]] = set()

        for _, row in cosmic_df.iterrows():
            genomic_pos = row.get("GENOME_START")
            if genomic_pos is None or (
                hasattr(genomic_pos, "__class__") and str(genomic_pos) == "nan"
            ):
                continue
            genomic_pos = int(genomic_pos)

            chrom = str(row.get("CHROMOSOME", ""))
            ref = str(row.get("GENOMIC_WT_ALLELE", "") or "")
            alt = str(row.get("GENOMIC_MUT_ALLELE", "") or "")

            # Deduplicate by genomic identity
            dedup_key = (chrom, genomic_pos, ref, alt)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            hgvsp = str(row.get("HGVSP", "") or "") or None
            canonical_hint = parse_hgvsp_position(hgvsp)

            hits.append(
                {
                    "source": "COSMIC",
                    "variant_id": str(row.get("GENOMIC_MUTATION_ID", "")),
                    "chrom": f"chr{chrom}" if chrom and not chrom.startswith("chr") else chrom,
                    "genomic_pos": genomic_pos,
                    "ref": ref,
                    "alt": alt,
                    "consequence": str(row.get("MUTATION_DESCRIPTION", "") or ""),
                    # protein_pos left None — HGVSp is canonical-frame.
                    # ConsequenceValidator is the authoritative source.
                    "protein_pos": None,
                    "hgvsp": hgvsp,
                    "allele_frequency": None,
                    "clinical_significance": None,
                    "metadata": {
                        "cosmic_sample_count": int(row.get("GENOME_SCREEN_SAMPLE_COUNT", 0) or 0),
                        "somatic_status": str(row.get("MUTATION_SOMATIC_STATUS", "") or ""),
                        "mutation_aa": str(row.get("MUTATION_AA", "") or ""),
                        "mutation_cds": str(row.get("MUTATION_CDS", "") or ""),
                        "hgvsp_canonical_hint": canonical_hint,
                    },
                }
            )

        logger.info("COSMIC: %d variants for %s", len(hits), gene_name)
        return hits

    # ------------------------------------------------------------------
    # HTTP helpers with retry
    # ------------------------------------------------------------------

    def _post_with_retry(self, url: str, json_payload: dict[str, Any]) -> dict[str, Any] | None:
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

    def _get_with_retry(self, url: str, params: dict[str, Any]) -> dict[str, Any] | None:
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


def _extract_ref_alt_from_title(title: str) -> tuple[str, str]:
    """Extract ref/alt bases from a ClinVar title's HGVSc notation.

    Examples:
        ``"NM_000546.6(TP53):c.215C>G (p.Pro72Arg)"`` -> ``("C", "G")``
        ``"NM_xxx.1:c.100delA"`` -> ``("A", "")`` (deletion)
        ``"NM_xxx.1:c.100insGT"`` -> ``("", "GT")`` (insertion)

    Returns:
        Tuple of (ref, alt). Either or both may be empty strings when the
        title is a structural variant or the pattern isn't recognized.
    """
    import re

    # Standard SNV: c.215C>G
    snv = re.search(r"c\.\d+([ACGT])>([ACGT])", title)
    if snv:
        return snv.group(1), snv.group(2)

    # Deletion: c.100delA or c.100_102del
    delete = re.search(r"c\.\d+(?:_\d+)?del([ACGT]*)", title)
    if delete:
        return delete.group(1), ""

    # Insertion: c.100insGT or c.100_101insGT
    insert = re.search(r"c\.\d+(?:_\d+)?ins([ACGT]+)", title)
    if insert:
        return "", insert.group(1)

    # Duplication: c.100dupA
    dup = re.search(r"c\.\d+(?:_\d+)?dup([ACGT]*)", title)
    if dup:
        return "", dup.group(1)

    return "", ""
