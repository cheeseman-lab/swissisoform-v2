"""ClinVar clinical-significance families — the one definition of "pathogenic".

``clinical_significance`` is a ClinVar-only free-text field: gnomAD and COSMIC
rows carry no value at all, and ClinVar spells the same call several ways
("Pathogenic", "Pathogenic/Likely pathogenic", "Likely pathogenic"). Anything
selecting on it must match by family rather than by equality, or it silently
undercounts; and it must not match by bare substring, because "Conflicting
classifications of pathogenicity" contains "pathogenic". The pipeline counts
(``variant_intersection``, ``clinical``), the LLM tool readers
(``swissisoform.site.tools``) and the website all classify through here.

Stdlib-only: ``website/prepare_deploy.sh`` vendors this file into the web image.
"""

from __future__ import annotations

from typing import Any

CLINSIG_FAMILIES = ("pathogenic", "benign", "uncertain", "conflicting", "none")


def clinsig_family(value: Any) -> str:
    """Bucket a ClinVar ``clinical_significance`` string into a coarse family.

    Families are the ones a caller actually filters on:
    ``pathogenic`` (Pathogenic + Pathogenic/Likely pathogenic + Likely
    pathogenic), ``benign`` (Benign + Benign/Likely benign + Likely benign),
    ``uncertain``, ``conflicting``, and ``none`` for an absent value (every
    gnomAD/COSMIC row, plus ClinVar rows with no assertion).

    ``conflicting`` is tested first because "Conflicting classifications of
    pathogenicity" contains the substring "pathogenic" and would otherwise be
    counted as a pathogenic call.
    """
    sig = str(value or "").strip().lower()
    if not sig or sig == "nan" or sig == "none":
        return "none"
    if "conflicting" in sig:
        return "conflicting"
    if "pathogenic" in sig:
        return "pathogenic"
    if "uncertain" in sig:
        return "uncertain"
    if "benign" in sig:
        return "benign"
    return "none"


def is_pathogenic(value: Any) -> bool:
    """True for a Pathogenic / Likely pathogenic call (and their combined forms)."""
    return clinsig_family(value) == "pathogenic"
