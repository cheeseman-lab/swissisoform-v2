"""Evidence-scoring buckets — dual-axis existence (E1–E6) + functional (F1–F7).

Each bucket is a subpackage exposing a ``score(site, cfg) -> CriterionResult``
function. ``EXISTENCE_CRITERIA`` / ``FUNCTIONAL_CRITERIA`` collect them in the
canonical order consumed by ``EvidenceScoringModule``.

Bucket-specific plumbing (the annotator that produces the evidence) lives in the
bucket folder; genuinely shared plumbing stays as infrastructure packages::

    Bucket                      Plumbing
    E1 primate conservation     conservation_frame            (shared E1/E2)
    E2 mammalian conservation   conservation_frame            (shared E1/E2)
    E3 phylop selection         modules.conservation
    E4 reproducibility          site.expression               (shared E4/E5)
    E5 initiation efficiency    site.expression               (shared E4/E5)
    E6 mass spec                evidence/e6_mass_spec          (owns it)
    F1 structure                structure + modules.biophysics
    F2 localization             evidence/f2_localization       (owns it)
    F3 domains                  evidence/f3_domains            (owns it)
    F4 targeting                evidence/f4_targeting           (owns it)
    F5 germline constraint      modules.varianteffect + clinical (shared F5/F6)
    F6 disease enrichment       modules.variant_intersection + clinical (shared F5/F6)
    F7 shared structural change structure (shared-region Cα RMSD)
"""

from __future__ import annotations

from swissisoform.evidence import (
    e1_primate_conservation as e1,
)
from swissisoform.evidence import (
    e2_mammalian_conservation as e2,
)
from swissisoform.evidence import (
    e3_phylop_selection as e3,
)
from swissisoform.evidence import (
    e4_reproducibility as e4,
)
from swissisoform.evidence import (
    e5_initiation_efficiency as e5,
)
from swissisoform.evidence import (
    e6_mass_spec as e6,
)
from swissisoform.evidence import (
    f1_structure as f1,
)
from swissisoform.evidence import (
    f2_localization as f2,
)
from swissisoform.evidence import (
    f3_domains as f3,
)
from swissisoform.evidence import (
    f4_targeting as f4,
)
from swissisoform.evidence import (
    f5_germline_constraint as f5,
)
from swissisoform.evidence import (
    f6_disease_enrichment as f6,
)
from swissisoform.evidence import (
    f7_shared_rmsd as f7,
)
from swissisoform.evidence.common import Criterion, CriterionResult

EXISTENCE_CRITERIA: list[Criterion] = [
    e1.score,
    e2.score,
    e3.score,
    e4.score,
    e5.score,
    e6.score,
]

FUNCTIONAL_CRITERIA: list[Criterion] = [
    f1.score,
    f2.score,
    f3.score,
    f4.score,
    f5.score,
    f6.score,
    f7.score,
]

__all__ = [
    "EXISTENCE_CRITERIA",
    "FUNCTIONAL_CRITERIA",
    "Criterion",
    "CriterionResult",
]
