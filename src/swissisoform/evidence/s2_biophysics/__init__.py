"""S2 — whole-protein biophysical shift (isoform vs canonical). Plumbing: modules.biophysics."""

from __future__ import annotations

from swissisoform.config import ScoringConfig
from swissisoform.evidence.common import CriterionResult
from swissisoform.models import TranslationInitiationSite

__all__ = ["score"]


def score(site: TranslationInitiationSite, cfg: ScoringConfig) -> CriterionResult:
    """S2: whole-protein biophysical shift — isoform vs canonical, 3 levers.

    Asks whether the isoform, taken as a whole, is biophysically different from
    the canonical protein taken as a whole. Reads the comparator's whole-protein
    ``<feat>_delta`` keys on ``site.comparison['biophysics']`` (each = isoform
    mean − canonical mean over the entire protein) for three descriptors:

    * ``gravy`` (mean hydropathy),
    * ``fraction_charged``,
    * ``disorder`` (mean Top-IDP propensity).

    Each lever fires when ``|<feat>_delta| >= cutoff``. ``True`` when *any* lever
    fires. This is the frame the old P1 distinctness half used (folding, P1's
    other half, now lives in P1); unlike the region-vs-core contrast it also
    scores uORF/altORF isoforms, which have no shared region.

    Returns ``None`` when the biophysics comparison is missing or none of the
    three ``*_delta`` descriptors are numerically present (nothing to compare).
    Because the deltas are whole-protein averages, a genuine but short
    differential region is diluted by length — by design, S2 fires only when the
    change is large enough to move the whole-protein average.

    Thresholds are PROVISIONAL — set in threshold discussion.
    """
    bio = site.comparison.get("biophysics")
    if not isinstance(bio, dict):
        return CriterionResult(
            "S2_biophysics", None, "biophysics comparison missing"
        )

    levers = (
        ("gravy", cfg.s2_gravy_delta_min),
        ("fraction_charged", cfg.s2_fraction_charged_delta_min),
        ("disorder", cfg.s2_disorder_delta_min),
    )
    fired: list[str] = []
    n_evaluable = 0
    for feat, cutoff in levers:
        delta = bio.get(f"{feat}_delta")
        if not isinstance(delta, (int, float)):
            continue
        n_evaluable += 1
        if abs(delta) >= cutoff:
            fired.append(feat)

    if n_evaluable == 0:
        return CriterionResult(
            "S2_biophysics", None, "biophysics whole-protein deltas unavailable"
        )

    distinct_str = ",".join(fired) if fired else "none"
    return CriterionResult(
        "S2_biophysics",
        len(fired) > 0,
        f"whole-protein |delta| distinct={distinct_str} ({n_evaluable}/3 descriptors evaluable)",
    )
