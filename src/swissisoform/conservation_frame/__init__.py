"""Path 1/2 conservation: primate + mammalian reading-frame integrity.

Pure-Python MAF parsing + frame analysis (``maf``, ``frame``), plus a
``hal2maf`` subprocess wrapper (``hal``) that gracefully returns
``None`` when the binary or the Zoonomia HAL aren't available. Wired
into a SiteModule by ``swissisoform.modules.conservation_frame``.

Spec: ``docs/reviews/conservation_path12_spec.md``.
"""

from swissisoform.conservation_frame.frame import (
    SpeciesFrameResult,
    aggregate_species_results,
    analyze_species,
)
from swissisoform.conservation_frame.maf import MafBlock, MafRow, parse_maf
from swissisoform.conservation_frame.species import (
    MAMMALIAN_SPECIES,
    PRIMATE_SPECIES,
)

__all__ = [
    "MAMMALIAN_SPECIES",
    "MafBlock",
    "MafRow",
    "PRIMATE_SPECIES",
    "SpeciesFrameResult",
    "aggregate_species_results",
    "analyze_species",
    "parse_maf",
]
