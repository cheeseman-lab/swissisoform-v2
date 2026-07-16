"""S2 — biophysical distinctness of the differential region. Plumbing: modules.biophysics."""

from __future__ import annotations

from swissisoform.config import ScoringConfig
from swissisoform.evidence.common import CriterionResult
from swissisoform.models import TranslationInitiationSite

__all__ = ["score"]


def score(site: TranslationInitiationSite, cfg: ScoringConfig) -> CriterionResult:
    """S2: biophysically distinct differential region — region-vs-core, 3 levers.

    Asks whether the isoform's differential region is biophysically distinct
    from the shared canonical core. Reads the Scope-A region-vs-core keys the
    comparator emits on ``site.comparison['biophysics']`` — for each descriptor
    ``<feat>_unique`` (diff region) and ``<feat>_shared`` (retained core):

    * ``gravy`` (mean hydropathy),
    * ``fraction_charged``,
    * ``disorder`` (mean Top-IDP propensity).

    Each lever fires when ``|<feat>_unique − <feat>_shared| >= cutoff``. ``True``
    when *any* lever fires — the added/lost segment differs biophysically from
    the core, not merely in length. This is sharper than the whole-protein
    isoform-vs-canonical ``_delta`` the old F1 half used, because it contrasts
    the region against the retained core directly.

    Returns ``None`` when the biophysics comparison is missing or none of the
    three descriptor pairs are numerically present (nothing to compare).

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
        unique = bio.get(f"{feat}_unique")
        shared = bio.get(f"{feat}_shared")
        if not isinstance(unique, (int, float)) or not isinstance(shared, (int, float)):
            continue
        n_evaluable += 1
        if abs(unique - shared) >= cutoff:
            fired.append(feat)

    if n_evaluable == 0:
        return CriterionResult(
            "S2_biophysics", None, "biophysics region-vs-core descriptors unavailable"
        )

    distinct_str = ",".join(fired) if fired else "none"
    return CriterionResult(
        "S2_biophysics",
        len(fired) > 0,
        f"region-vs-core distinct={distinct_str} ({n_evaluable}/3 descriptors evaluable)",
    )
