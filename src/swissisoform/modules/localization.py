"""Module: Localization — subcellular localization prediction consumer.

Attaches pre-computed subcellular localization predictions (DeepLoc, WoLF PSORT)
to TIS sites. Predictions are looked up by an external key (tis_id for isoforms,
gene_name for canonical proteins).
"""

from __future__ import annotations

from typing import Any

from swissisoform.config import PipelineConfig
from swissisoform.models import TranslationInitiationSite


class LocalizationModule:
    """Localization annotation module.

    Consumes pre-computed subcellular localization predictions and attaches
    them to TIS sites. The actual prediction tools (DeepLoc, WoLF PSORT)
    are run separately as GPU jobs; this module only loads and merges results.

    Attributes:
        MODULE_NAME: Unique module identifier.
        OUTPUT_COLUMNS: Column names produced (prefixed with MODULE_NAME_).
        SCOPE: Module scope ('C' for per-site classification).
    """

    MODULE_NAME: str = "localization"
    OUTPUT_COLUMNS: list[str] = [
        "localization_deeploc_prediction",
        "localization_deeploc_signals",
        "localization_deeploc_membrane",
        "localization_wolfpsort_prediction",
    ]
    SCOPE: str = "C"

    def __init__(
        self,
        config: PipelineConfig,
        predictions: dict[str, dict[str, str | None]] | None = None,
    ) -> None:
        """Initialize with pipeline configuration and pre-computed predictions.

        Args:
            config: Pipeline configuration.
            predictions: Mapping of key (e.g. tis_id) to prediction dict with keys:
                deeploc, deeploc_signals, deeploc_membrane, wolfpsort.
        """
        self.config = config
        self.predictions = predictions or {}

    def annotate_by_key(self, key: str) -> dict[str, Any]:
        """Look up pre-computed predictions by an arbitrary key.

        Args:
            key: Lookup key (tis_id for isoforms, gene_name for canonical).

        Returns:
            Annotation dict with unprefixed keys. All values are None if the
            key is not found in predictions.
        """
        pred = self.predictions.get(key, {})
        return {
            "deeploc_prediction": pred.get("deeploc"),
            "deeploc_signals": pred.get("deeploc_signals"),
            "deeploc_membrane": pred.get("deeploc_membrane"),
            "wolfpsort_prediction": pred.get("wolfpsort"),
        }

    def run(
        self, tis_sites: list[TranslationInitiationSite]
    ) -> list[TranslationInitiationSite]:
        """Attach localization annotations to each TIS site.

        For each site, writes to site.isoform_annotations["localization"] with keys:
        deeploc_prediction, deeploc_signals, deeploc_membrane, wolfpsort_prediction.

        Args:
            tis_sites: Input TIS sites.

        Returns:
            The same sites with isoform_annotations["localization"] populated.
        """
        for site in tis_sites:
            site.isoform_annotations[self.MODULE_NAME] = self.annotate_by_key(site.tis_id)
        return tis_sites
