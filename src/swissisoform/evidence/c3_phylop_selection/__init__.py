"""C3 — phylop coding selection. Plumbing: swissisoform.conservation."""

from __future__ import annotations

from swissisoform.config import ScoringConfig
from swissisoform.evidence.common import CriterionResult, _annotation
from swissisoform.models import TranslationInitiationSite


def score(site: TranslationInitiationSite, cfg: ScoringConfig) -> CriterionResult:
    """C3: absolute PhyloP over the unique region indicates purifying selection.

    Scored on the *absolute* mean PhyloP of the unique region exceeding
    ``cfg.c3_phylop_min`` (strong purifying selection at coding level). This is
    NOT a unique-vs-shared comparison — the claim is that the unique region is
    itself under coding-level constraint. ``phylop_enrichment`` (unique/shared)
    is reported as context only.
    """
    ann = _annotation(site, "conservation")
    if ann is None:
        return CriterionResult("C3_phylop_coding_selection", None, "conservation not run")
    summary = ann.get("summary") if isinstance(ann, dict) else None
    status = summary.get("region_status") if isinstance(summary, dict) else None
    if status != "ok":
        return CriterionResult(
            "C3_phylop_coding_selection", None, f"region_status={status}"
        )
    val = ann.get("phylop_unique_region_mean")
    if val is None:
        return CriterionResult(
            "C3_phylop_coding_selection", None, "phylop_unique_region_mean unavailable"
        )
    passed = val >= cfg.c3_phylop_min
    enrich = ann.get("phylop_enrichment")
    enrich_str = f"{enrich:.2f}" if isinstance(enrich, (int, float)) else "n/a"
    return CriterionResult(
        "C3_phylop_coding_selection",
        passed,
        f"phylop_unique={val:.2f} (threshold {cfg.c3_phylop_min}); "
        f"enrichment={enrich_str} (context)",
    )
