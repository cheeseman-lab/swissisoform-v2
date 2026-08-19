"""D3 — mass spec. Plumbing: swissisoform.massspec."""

from __future__ import annotations

from swissisoform.config import ScoringConfig
from swissisoform.evidence.common import CriterionResult, _annotation
from swissisoform.evidence.d3_mass_spec.massspec import (
    MassSpecModule,
    collect_unique_peptides,
    precompute_pepquery,
)
from swissisoform.models import TranslationInitiationSite

__all__ = [
    "score",
    "MassSpecModule",
    "collect_unique_peptides",
    "precompute_pepquery",
]


def score(site: TranslationInitiationSite, cfg: ScoringConfig) -> CriterionResult:
    """D3: PepQuery2-validated unique peptide(s) match public MS spectra.

    Requires the massspec module to have been initialised with a
    precomputed ``validated_peptides`` cache covering the gene.  When
    PepQuery2 has NOT been run (``summary.pepquery_run = False``) the
    criterion reports ``None`` — the isoform cannot be scored on mass
    spec until the precompute lands.  In-silico tryptic detectability
    alone is not evidence of proteomic observation.
    """
    ann = _annotation(site, "massspec")
    if ann is None:
        return CriterionResult("D3_mass_spec", None, "massspec not run")
    summary = ann.get("summary") if isinstance(ann, dict) else None
    if not isinstance(summary, dict) or not summary.get("pepquery_run"):
        return CriterionResult("D3_mass_spec", None, "pepquery2 not precomputed")
    hits = ann.get("hits")
    if not isinstance(hits, list):
        return CriterionResult("D3_mass_spec", None, "no hits field")
    n_validated_unique = sum(
        1 for h in hits
        if h.get("unique_to_isoform") is True and h.get("validated") is True
    )
    passed = n_validated_unique >= cfg.massspec_unique_peptides_min
    return CriterionResult(
        "D3_mass_spec",
        passed,
        f"n_validated_unique_peptides={n_validated_unique} "
        f"(threshold {cfg.massspec_unique_peptides_min})",
    )
