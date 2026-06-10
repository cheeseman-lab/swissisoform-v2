"""Evidence-scoring buckets — dual-axis existence (E1–E6) + functional (F1–F6).

Each bucket is a self-contained subpackage exposing a ``score(site, cfg)``
function. ``EXISTENCE_CRITERIA`` / ``FUNCTIONAL_CRITERIA`` collect them in the
canonical order consumed by ``EvidenceScoringModule``.
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
]

__all__ = [
    "EXISTENCE_CRITERIA",
    "FUNCTIONAL_CRITERIA",
    "Criterion",
    "CriterionResult",
]
