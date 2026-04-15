"""Gene Reference module — external database annotations per gene.

Annotates each TIS site with reference information (UniProt, HPA, DepMap, OMIM)
looked up by gene name. The caller is responsible for loading reference data;
this module simply maps gene names to pre-loaded annotation dicts.
"""

from __future__ import annotations

from typing import Any

from swissisoform.config import PipelineConfig
from swissisoform.models import TranslationInitiationSite

# Annotation keys produced by this module (without the MODULE_NAME prefix).
_ANNOTATION_KEYS: list[str] = [
    "uniprot_id",
    "uniprot_function",
    "subcellular_location",
    "hpa_protein_class",
    "hpa_subcellular_location",
    "depmap_mean_effect",
    "omim_phenotypes",
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

    def run(
        self, tis_sites: list[TranslationInitiationSite]
    ) -> list[TranslationInitiationSite]:
        """Annotate each TIS site with gene-level reference data.

        For each site, writes to ``site.isoform_annotations["generef"]`` with keys
        matching ``_ANNOTATION_KEYS``. Values come from ``gene_annotations``
        if the gene is present; otherwise all values are ``None``.

        Args:
            tis_sites: Input TIS sites.

        Returns:
            The same sites with ``isoform_annotations["generef"]`` populated.
        """
        for site in tis_sites:
            gene_data = self.gene_annotations.get(site.gene_name, {})
            site.isoform_annotations[self.MODULE_NAME] = {
                key: gene_data.get(key) for key in _ANNOTATION_KEYS
            }
        return tis_sites
