"""P1 — structured differential region (folding only). Plumbing: swissisoform.structure."""

from __future__ import annotations

from swissisoform.config import ScoringConfig
from swissisoform.evidence.common import CriterionResult, _annotation
from swissisoform.models import TranslationInitiationSite


def score(site: TranslationInitiationSite, cfg: ScoringConfig) -> CriterionResult:
    """P1: structured differential region — diff-region pLDDT exceeds threshold.

    The metric is symmetric across ORF types — it asks "is the differential
    region structured?" — and the interpretation differs by direction:

    * extension/uORF/altORF — the diff region is the NEW N-terminal sequence
      the isoform adds; a high diff-region pLDDT indicates a well-folded
      addition (possibly a new functional domain).
    * truncation — the diff region is the canonical N-terminal sequence the
      isoform LOSES; a high diff-region pLDDT indicates that what was lost
      was structured (potentially functional). Same measurement, opposite
      narrative.

    **Folding only.** P1 asks solely whether the differential region is
    well-folded (mean pLDDT >= ``cfg.f1_plddt_threshold``). The biophysical-
    distinctness half that this criterion used to carry now lives in its own
    criterion (``S2_biophysics``), so the two signals are scored independently.

    Reads ``site.isoform_annotations['structure']`` written by
    ``StructureModule``.  Returns ``None`` when:

    - structure cache is empty (``status ∈ {no_cache, too_long, failed}``)
    - the backend produced only a scalar complex pLDDT, not per-residue
      (``status='uniform_plddt'``) — region-level statistics from a
      uniform fill aren't a real per-region measurement.
    - ``plddt_diffregion_mean`` is unavailable.

    The threshold's scale must match the backend (Boltz-2 emits 0–1,
    AlphaFold-style emits 0–100).
    """
    ann = _annotation(site, "structure")
    if ann is None:
        return CriterionResult(
            "P1_structured_extension", None, "structure annotation missing"
        )
    status = ann.get("status")
    if status in ("no_cache", "too_long", "failed", "uniform_plddt"):
        return CriterionResult(
            "P1_structured_extension", None, f"structure status={status}"
        )
    plddt = ann.get("plddt_diffregion_mean")
    if plddt is None:
        return CriterionResult(
            "P1_structured_extension", None, "plddt_diffregion_mean unavailable"
        )

    threshold = cfg.f1_plddt_threshold
    folded = plddt >= threshold
    folded_str = "folded" if folded else "unfolded"
    return CriterionResult(
        "P1_structured_extension",
        folded,
        f"plddt_diffregion_mean={plddt:.3f} ({folded_str}, threshold {threshold})",
    )
