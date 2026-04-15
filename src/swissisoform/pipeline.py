"""Pipeline orchestration for SwissIsoform v2.

Wires annotation modules to Gene objects, running ProteinModules symmetrically
on canonical and isoform proteins, SiteModules on each TIS, and gene-level
modules once per gene.
"""

from __future__ import annotations

import logging
from typing import Any

from swissisoform.models import Gene, TranslationInitiationSite

logger = logging.getLogger(__name__)


class AnnotationPipeline:
    """Orchestrate annotation modules over a list of genes.

    The pipeline runs three kinds of modules:
    - ProteinModules: called with `annotate(protein)`, run on both canonical
      and each isoform protein. Must have MODULE_NAME attribute.
    - SiteModules: called with `annotate_site(site)`, run once per TIS.
      Must have MODULE_NAME attribute.
    - GeneModules: called with `annotate_gene(gene)`, run once per gene.
      Must have MODULE_NAME attribute. Results stored in gene.gene_annotations.

    Attributes:
        protein_modules: Modules implementing annotate(protein: str) -> dict.
        site_modules: Modules implementing annotate_site(site: TIS) -> dict.
        gene_modules: Modules implementing annotate_gene(gene: Gene) -> dict.
    """

    def __init__(
        self,
        protein_modules: list[Any] | None = None,
        site_modules: list[Any] | None = None,
        gene_modules: list[Any] | None = None,
    ) -> None:
        self.protein_modules = protein_modules or []
        self.site_modules = site_modules or []
        self.gene_modules = gene_modules or []

    def run(self, genes: list[Gene]) -> list[Gene]:
        """Run all annotation modules over a list of genes.

        Processing order per gene:
        1. ProteinModules on canonical protein -> gene.canonical_annotations
        2. ProteinModules on each isoform protein -> tis.isoform_annotations
        3. SiteModules on each TIS -> tis.isoform_annotations
        4. GeneModules on the gene -> gene.gene_annotations

        Args:
            genes: List of Gene objects with canonical_protein and tis_sites
                populated. Each TIS should have isoform_protein set.

        Returns:
            The same Gene objects with annotations populated.
        """
        logger.info(
            "Running annotation pipeline: %d genes, %d protein modules, "
            "%d site modules, %d gene modules",
            len(genes),
            len(self.protein_modules),
            len(self.site_modules),
            len(self.gene_modules),
        )

        for gene in genes:
            self._annotate_gene(gene)

        total_tis = sum(len(g.tis_sites) for g in genes)
        logger.info("Annotation complete: %d genes, %d TIS sites", len(genes), total_tis)
        return genes

    def _annotate_gene(self, gene: Gene) -> None:
        """Run all modules for a single gene."""
        # 1. ProteinModules on canonical protein (once per gene)
        for mod in self.protein_modules:
            name = mod.MODULE_NAME
            gene.canonical_annotations[name] = mod.annotate(gene.canonical_protein)

        # 2-3. Per-TIS annotations
        for site in gene.tis_sites:
            self._annotate_site(site)

        # 4. Gene-level modules
        for mod in self.gene_modules:
            name = mod.MODULE_NAME
            gene.gene_annotations[name] = mod.annotate_gene(gene)

    def _annotate_site(self, site: TranslationInitiationSite) -> None:
        """Run ProteinModules on isoform + SiteModules on a single TIS."""
        # ProteinModules on isoform protein
        for mod in self.protein_modules:
            name = mod.MODULE_NAME
            site.isoform_annotations[name] = mod.annotate(site.isoform_protein)

        # SiteModules
        for mod in self.site_modules:
            name = mod.MODULE_NAME
            site.isoform_annotations[name] = mod.annotate_site(site)
