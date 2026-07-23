"""Gene Reference module — external database annotations per gene.

Annotates each gene with reference fields (function, subcellular location,
functional keywords) looked up by gene name. Function/localization/keywords come
from the Affinage API and the UniProt accession from a minimal UniProtKB lookup
(see scripts/setup/fetch_generef.py). Field keys keep the legacy ``uniprot_``
prefix so downstream consumers are unchanged; this module maps gene names to the
pre-loaded annotation dicts. HPA / DepMap / OMIM are a future extension pending
their data sources.
"""

from __future__ import annotations

from typing import Any

from swissisoform.config import PipelineConfig
from swissisoform.models import Gene, TranslationInitiationSite

# Annotation keys produced by this module (without the MODULE_NAME prefix).
# Scoped to the fields we can reliably populate from a no-auth source
# (UniProt REST). HPA / DepMap / OMIM are future fields pending their data
# infra — adding them here without data would emit blank columns.
_ANNOTATION_KEYS: list[str] = [
    "uniprot_id",
    "function",
    "subcellular_location",
    "keywords",
]


class GeneRefModule:
    """Gene-level reference annotations from external databases.

    Attributes:
        MODULE_NAME: Unique module identifier.
        OUTPUT_COLUMNS: Column names produced (prefixed with MODULE_NAME_).
        SCOPE: Module scope ('G' for gene-level).
        gene_annotations: Mapping of gene_name -> dict of annotation values.
    """

    MODULE_NAME: str = "generef"
    OUTPUT_COLUMNS: list[str] = [f"generef_{k}" for k in _ANNOTATION_KEYS]
    SCOPE: str = "G"

    def __init__(
        self,
        config: PipelineConfig,
        gene_annotations: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize with pipeline configuration and optional gene annotations.

        Args:
            config: Pipeline configuration (unused, kept for protocol compliance).
            gene_annotations: Mapping of gene_name to a dict whose keys match
                ``_ANNOTATION_KEYS``. Missing keys default to ``None``.
        """
        self.gene_annotations: dict[str, dict[str, Any]] = gene_annotations or {}

    def annotate_by_gene(self, gene_name: str) -> dict[str, Any]:
        """Return the gene-reference annotation dict for a given gene name.

        Args:
            gene_name: Gene symbol to look up.

        Returns:
            Dict with all keys in ``_ANNOTATION_KEYS``. Missing genes produce
            a dict of all-None values.
        """
        gene_data = self.gene_annotations.get(gene_name, {})
        return {key: gene_data.get(key) for key in _ANNOTATION_KEYS}

    def annotate_gene(self, gene: Gene) -> dict[str, Any]:
        """GeneModule protocol method — annotate a Gene object.

        Used by ``AnnotationPipeline`` when GeneRefModule is registered as a
        ``gene_module``.  Result is written to ``gene.gene_annotations``.

        Args:
            gene: A Gene object (uses ``gene.gene_name`` for the lookup).

        Returns:
            Same shape as ``annotate_by_gene``.
        """
        return self.annotate_by_gene(gene.gene_name)

    def run(self, tis_sites: list[TranslationInitiationSite]) -> list[TranslationInitiationSite]:
        """Annotate each TIS site with gene-level reference data.

        Backward-compatible wrapper: for standalone use (outside the pipeline
        wiring), writes gene-level reference data to each TIS's
        ``isoform_annotations["generef"]``.  Inside the pipeline wiring, this
        should NOT be used — register the module as ``gene_modules=[...]``
        so results land in ``gene.gene_annotations``.

        For each site, writes to ``site.isoform_annotations["generef"]`` with keys
        matching ``_ANNOTATION_KEYS``. Values come from ``gene_annotations``
        if the gene is present; otherwise all values are ``None``.

        Args:
            tis_sites: Input TIS sites.

        Returns:
            The same sites with ``isoform_annotations["generef"]`` populated.
        """
        for site in tis_sites:
            site.isoform_annotations[self.MODULE_NAME] = self.annotate_by_gene(site.gene_name)
        return tis_sites
