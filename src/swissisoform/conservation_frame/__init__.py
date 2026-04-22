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
from swissisoform.conservation_frame.tree import (
    Node,
    depth_from_reference,
    mrca_depth,
    parse_newick,
)

__all__ = [
    "MAMMALIAN_SPECIES",
    "MafBlock",
    "MafRow",
    "Node",
    "PRIMATE_SPECIES",
    "SpeciesFrameResult",
    "aggregate_species_results",
    "analyze_species",
    "depth_from_reference",
    "mrca_depth",
    "parse_maf",
    "parse_newick",
]
