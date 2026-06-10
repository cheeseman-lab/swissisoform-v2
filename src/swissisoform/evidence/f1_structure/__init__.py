"""F1 — structured differential region. Plumbing: swissisoform.structure + biophysics."""

from __future__ import annotations

from swissisoform.config import ScoringConfig
from swissisoform.evidence.common import CriterionResult, _annotation
from swissisoform.models import TranslationInitiationSite


def score(site: TranslationInitiationSite, cfg: ScoringConfig) -> CriterionResult:
    """F1: structured differential region — diff-region pLDDT exceeds threshold.

    The metric is symmetric across ORF types — it asks "is the differential
    region structured?" — and the interpretation differs by direction:

    * extension/uORF/altORF — the diff region is the NEW N-terminal sequence
      the isoform adds; a high diff-region pLDDT indicates a well-folded
      addition (possibly a new functional domain).
    * truncation — the diff region is the canonical N-terminal sequence the
      isoform LOSES; a high diff-region pLDDT indicates that what was lost
      was structured (potentially functional). Same measurement, opposite
      narrative.

    Reads ``site.isoform_annotations['structure']`` written by
    ``StructureModule``.  Returns ``None`` when:

    - structure cache is empty (``status ∈ {no_cache, too_long, failed}``)
    - the backend produced only a scalar complex pLDDT, not per-residue
      (``status='uniform_plddt'``) — region-level statistics from a
      uniform fill aren't a real per-region measurement.

    ``True`` requires BOTH halves: the diff region is well-folded
    (mean pLDDT >= ``cfg.f1_plddt_threshold``) AND it is biophysically
    distinct from the shared core. Distinctness is read from the biophysics
    comparator deltas — any of ``|gravy_delta|``, ``|fraction_charged_delta|``,
    ``|disorder_delta|`` exceeding its provisional cutoff. A structured-but-
    biophysically-identical addition is not counted as a functional change.

    Returns ``None`` when structure status is not ``ok`` (cache empty /
    too long / failed / uniform fill) OR the biophysics comparison is missing.
    The threshold's scale must match the backend (Boltz-2 emits 0–1,
    AlphaFold-style emits 0–100).
    """
    ann = _annotation(site, "structure")
    if ann is None:
        return CriterionResult(
            "F1_structured_extension", None, "structure annotation missing"
        )
    status = ann.get("status")
    if status in ("no_cache", "too_long", "failed", "uniform_plddt"):
        return CriterionResult(
            "F1_structured_extension", None, f"structure status={status}"
        )
    plddt = ann.get("plddt_diffregion_mean")
    if plddt is None:
        return CriterionResult(
            "F1_structured_extension", None, "plddt_diffregion_mean unavailable"
        )

    bio = site.comparison.get("biophysics")
    if not isinstance(bio, dict):
        return CriterionResult(
            "F1_structured_extension", None, "biophysics comparison missing"
        )
    # Provisional distinctness cutoffs — CALIBRATE ON GENOME-WIDE RUN.
    gravy_d = bio.get("gravy_delta")
    charged_d = bio.get("fraction_charged_delta")
    disorder_d = bio.get("disorder_delta")
    distinct_flags = []
    if isinstance(gravy_d, (int, float)) and abs(gravy_d) >= 0.3:
        distinct_flags.append("gravy")
    if isinstance(charged_d, (int, float)) and abs(charged_d) >= 0.05:
        distinct_flags.append("charged")
    if isinstance(disorder_d, (int, float)) and abs(disorder_d) >= 0.05:
        distinct_flags.append("disorder")
    distinct = len(distinct_flags) > 0

    threshold = cfg.f1_plddt_threshold
    folded = plddt >= threshold
    passed = folded and distinct
    folded_str = "folded" if folded else "unfolded"
    distinct_str = ",".join(distinct_flags) if distinct_flags else "none"
    return CriterionResult(
        "F1_structured_extension",
        passed,
        f"plddt_diffregion_mean={plddt:.3f} ({folded_str}, threshold {threshold}); "
        f"biophysically_distinct={distinct_str}",
    )
