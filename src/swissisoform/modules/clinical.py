"""Module: Clinical — variant annotation from gnomAD, ClinVar, COSMIC.

Annotates proteins with clinical variant data. Returns positional hits
with protein-space coordinates for downstream differential region subsetting.

Can operate in two modes:
- Lookup: load pre-computed variant data from a per-gene cache
- Inline: fetch from gnomAD API directly (for small runs / testing)
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from swissisoform.clinical.validate import ConsequenceValidator
from swissisoform.config import PipelineConfig
from swissisoform.models import TranslationInitiationSite


def parse_hgvsp_position(hgvsp: str | None) -> int | None:
    """Parse protein position from HGVS protein notation.

    Extracts the first integer from the HGVS string and converts to 0-indexed.

    Args:
        hgvsp: HGVS protein notation string (e.g. "p.Arg42Gly", "p.R42G").

    Returns:
        0-indexed protein position, or None if parsing fails.

    Examples:
        >>> parse_hgvsp_position("p.Arg42Gly")
        41
        >>> parse_hgvsp_position("p.R42G")
        41
        >>> parse_hgvsp_position("p.Ter100Arg")
        99
        >>> parse_hgvsp_position("p.Met1?")
        0
        >>> parse_hgvsp_position(None)
    """
    if not hgvsp:
        return None
    match = re.search(r"\d+", hgvsp)
    if match is None:
        return None
    return int(match.group()) - 1


def _build_summary(hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Build summary dict from a list of variant hit dicts.

    Args:
        hits: List of variant dicts, each with 'source', 'consequence',
            and 'clinical_significance' keys.

    Returns:
        Summary dict with total_variants, by_source, by_consequence,
        and pathogenic_count.
    """
    by_source: dict[str, int] = dict(Counter(h["source"] for h in hits))
    by_consequence: dict[str, int] = dict(Counter(h["consequence"] for h in hits))
    pathogenic_count = sum(
        1
        for h in hits
        if h.get("clinical_significance")
        and "pathogenic" in h["clinical_significance"].lower()
    )
    return {
        "total_variants": len(hits),
        "by_source": by_source,
        "by_consequence": by_consequence,
        "pathogenic_count": pathogenic_count,
    }


_EMPTY_SUMMARY: dict[str, Any] = {
    "total_variants": 0,
    "by_source": {},
    "by_consequence": {},
    "pathogenic_count": 0,
}


class ClinicalModule:
    """Clinical variant annotation module.

    Annotates TIS sites with variant data from gnomAD, ClinVar, and COSMIC.
    Operates via a pre-loaded variant cache for fast lookup.

    Attributes:
        MODULE_NAME: Unique module identifier.
        OUTPUT_COLUMNS: Column names produced by this module.
        SCOPE: Module scope ('C' for classification).
    """

    MODULE_NAME: str = "clinical"
    OUTPUT_COLUMNS: list[str] = ["clinical_hits", "clinical_summary"]
    SCOPE: str = "C"

    def __init__(
        self,
        config: PipelineConfig,
        variant_cache: dict[str, list[dict[str, Any]]] | None = None,
        validator: ConsequenceValidator | None = None,
    ) -> None:
        """Initialize with pipeline configuration and optional variant cache.

        Args:
            config: Pipeline configuration (uses config.clinical for API settings).
            variant_cache: Optional pre-loaded variant data. Dict mapping
                gene_name -> list of variant dicts. Each variant dict must have:
                - source: str ("gnomAD", "ClinVar", "COSMIC")
                - variant_id: str
                - chrom: str
                - genomic_pos: int
                - ref: str
                - alt: str
                - consequence: str (e.g. "missense_variant", "stop_gained")
                - protein_pos: int | None (0-indexed position in protein)
                - hgvsp: str | None (HGVS protein notation)
                - allele_frequency: float | None
                - clinical_significance: str | None (for ClinVar)
                - metadata: dict (source-specific extras)
            validator: Optional ``ConsequenceValidator`` for codon-level
                consequence validation. When provided, fetched variants are
                validated against the coding sequence.
        """
        self._config = config
        self._variant_cache = variant_cache or {}
        self._validator = validator

    # Keep public aliases for backward compatibility
    @property
    def config(self) -> PipelineConfig:
        return self._config

    @property
    def variant_cache(self) -> dict[str, list[dict[str, Any]]]:
        return self._variant_cache

    def fetch_variants(
        self,
        gene_name: str,
        sources: list[str] | None = None,
        transcript_id: str = "",
    ) -> list[dict[str, Any]]:
        """Fetch variants from external databases for a gene.

        This is the inline compute path -- fetches from gnomAD/ClinVar directly.
        For batch processing of many genes, use ``VariantFetcher`` directly and
        pass results via *variant_cache*.

        Args:
            gene_name: Gene symbol (e.g. ``"TP53"``).
            sources: List of sources to query. Default: ``["gnomad", "clinvar"]``.
            transcript_id: Optional GENCODE transcript ID. When provided along
                with a ``ConsequenceValidator``, fetched variants are validated
                via codon-level translation analysis.

        Returns:
            List of variant dicts in the standard hit format.
        """
        from swissisoform.clinical.fetch import VariantFetcher

        clinical_cfg = self._config.clinical
        fetcher = VariantFetcher(
            gnomad_url=clinical_cfg.gnomad_api_url
            if clinical_cfg
            else "https://gnomad.broadinstitute.org/api",
            clinvar_email=clinical_cfg.clinvar_email if clinical_cfg else "",
            clinvar_api_key=clinical_cfg.clinvar_api_key if clinical_cfg else "",
            timeout=clinical_cfg.fetch_timeout if clinical_cfg else 30,
            max_retries=clinical_cfg.max_retries if clinical_cfg else 3,
            retry_delay=clinical_cfg.retry_delay if clinical_cfg else 1.0,
        )
        variants = fetcher.fetch_gene(gene_name, sources=sources)

        # Validate if validator available and transcript known
        if self._validator and transcript_id:
            variants = self._validator.validate_variants(variants, transcript_id)

        return variants

    def annotate(
        self,
        protein: str,
        gene_name: str = "",
        fetch_if_missing: bool = False,
    ) -> dict[str, Any]:
        """Annotate a protein with clinical variant data.

        Looks up variants for the given gene in the variant cache, filters to
        those with protein-space positions, and builds a summary.

        Args:
            protein: Protein sequence (unused in cache-lookup mode, but part
                of the ProteinModule interface).
            gene_name: Gene symbol to look up in the variant cache.
            fetch_if_missing: If ``True`` and the gene is not in the cache,
                fetch variants from external APIs inline.

        Returns:
            Dict with 'hits' (list of variant dicts with protein_pos != None)
            and 'summary' (counts by source, consequence, pathogenic).
        """
        if gene_name and gene_name in self._variant_cache:
            raw_variants = self._variant_cache[gene_name]
        elif gene_name and fetch_if_missing:
            raw_variants = self.fetch_variants(gene_name)
            self._variant_cache[gene_name] = raw_variants
        else:
            raw_variants = []

        if not raw_variants:
            return {"hits": [], "summary": dict(_EMPTY_SUMMARY)}

        # Filter to variants that map to protein space
        hits = [v for v in raw_variants if v.get("protein_pos") is not None]

        return {
            "hits": hits,
            "summary": _build_summary(hits),
        }

    def annotate_by_key(self, gene_name: str) -> dict[str, Any]:
        """Convenience method for gene-level lookup without a protein string.

        Args:
            gene_name: Gene symbol to look up.

        Returns:
            Same as annotate().
        """
        return self.annotate("", gene_name)

    def run(
        self, tis_sites: list[TranslationInitiationSite]
    ) -> list[TranslationInitiationSite]:
        """Annotate all TIS sites with clinical variant data.

        Args:
            tis_sites: Input TIS sites.

        Returns:
            The same sites with isoform_annotations["clinical"] populated.
        """
        for site in tis_sites:
            site.isoform_annotations[self.MODULE_NAME] = self.annotate(
                site.isoform_protein, site.gene_name
            )
        return tis_sites
