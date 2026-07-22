"""Evidence-scoring buckets — grouped by the six CDLMPS categories.

Each bucket is a subpackage exposing a ``score(site, cfg) -> CriterionResult``
function. ``CATEGORY_CRITERIA`` maps each category letter to its criteria in the
canonical order; ``EXISTENCE_CRITERIA`` (C+D) / ``FUNCTIONAL_CRITERIA``
(L+M+P+S) are derived from it for the two-axis roll-up consumed by
``EvidenceScoringModule``.

Categories
----------

    C  Conservation          C1 primate, C2 mammalian, C3 phylop selection
    D  Detection             D1 multi-cell-line, D2 initiation eff., D3 mass-spec
    L  Localization          L1 localization change, L2 targeting change
    M  Mutation Landscape    M1 germline constraint, M2 disease enrichment
    P  Predicted Structure   P1 structured extension, P2 shared-region RMSD
    S  Structural Chars.     S1 domain change, S2 biophysics, S3 SAE features

Bucket-specific plumbing (the annotator that produces the evidence) lives in the
bucket folder; genuinely shared plumbing stays as infrastructure packages::

    Bucket                      Plumbing
    C1 primate conservation     conservation_frame            (shared C1/C2)
    C2 mammalian conservation   conservation_frame            (shared C1/C2)
    C3 phylop selection         modules.conservation
    D1 reproducibility          site.expression               (shared D1/D2)
    D2 initiation efficiency    site.expression               (shared D1/D2)
    D3 mass spec                evidence/d3_mass_spec          (owns it)
    L1 localization             evidence/l1_localization       (owns it)
    L2 targeting                evidence/l2_targeting           (owns it)
    M1 germline constraint      modules.varianteffect + clinical (shared M1/M2)
    M2 disease enrichment       modules.variant_intersection + clinical (shared M1/M2)
    P1 structure                structure (diff-region pLDDT)
    P2 shared structural change structure (shared-region Cα RMSD)
    S1 domains                  evidence/s1_domains            (owns it)
    S2 biophysics               modules.biophysics (whole-protein deltas)
    S3 sae                      plm.sae_module
"""

from __future__ import annotations

from swissisoform.evidence import (
    c1_primate_conservation as c1,
)
from swissisoform.evidence import (
    c2_mammalian_conservation as c2,
)
from swissisoform.evidence import (
    c3_phylop_selection as c3,
)
from swissisoform.evidence import (
    d1_reproducibility as d1,
)
from swissisoform.evidence import (
    d2_initiation_efficiency as d2,
)
from swissisoform.evidence import (
    d3_mass_spec as d3,
)
from swissisoform.evidence import (
    l1_localization as l1,
)
from swissisoform.evidence import (
    l2_targeting as l2,
)
from swissisoform.evidence import (
    m1_germline_constraint as m1,
)
from swissisoform.evidence import (
    m2_disease_enrichment as m2,
)
from swissisoform.evidence import (
    p1_structure as p1,
)
from swissisoform.evidence import (
    p2_shared_rmsd as p2,
)
from swissisoform.evidence import (
    s1_domains as s1,
)
from swissisoform.evidence import (
    s2_biophysics as s2,
)
from swissisoform.evidence import (
    s3_sae as s3,
)
from swissisoform.evidence.common import Criterion, CriterionResult

# Category-keyed registration — the primary structure. Order within each list
# is the canonical display / evaluation order.
CATEGORY_CRITERIA: dict[str, list[Criterion]] = {
    "C": [c1.score, c2.score, c3.score],
    "D": [d1.score, d2.score, d3.score],
    "L": [l1.score, l2.score],
    "M": [m1.score, m2.score],
    "P": [p1.score, p2.score],
    "S": [s1.score, s2.score, s3.score],
}

# Two-axis roll-up (back-compat): existence = Conservation + Detection,
# functional = Localization + Mutation + Predicted structure + Structural.
EXISTENCE_CRITERIA: list[Criterion] = [
    *CATEGORY_CRITERIA["C"],
    *CATEGORY_CRITERIA["D"],
]

FUNCTIONAL_CRITERIA: list[Criterion] = [
    *CATEGORY_CRITERIA["L"],
    *CATEGORY_CRITERIA["M"],
    *CATEGORY_CRITERIA["P"],
    *CATEGORY_CRITERIA["S"],
]

__all__ = [
    "CATEGORY_CRITERIA",
    "EXISTENCE_CRITERIA",
    "FUNCTIONAL_CRITERIA",
    "Criterion",
    "CriterionResult",
]
