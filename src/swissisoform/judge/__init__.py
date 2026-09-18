"""Weighing the prompt-variant arms with a local open-weight judge.

The corpus under test is 9 arms x 50 isoforms x 7 output units (the six CDLMPS
category verdicts plus the synthesis) = 3,150 outputs, produced by
``scripts/site/run_llm_variants.py``. This package scores them.

Two ideas carry the whole design:

**A cell is (isoform, unit), and judging never leaves it.** Comparing verdicts
across isoforms grades biology -- a truncation in a conserved gene and a uORF in
a variant-poor one have incomparable evidence. Comparing across categories grades
vocabulary depth -- ``tags`` carries 24 tags in S against 3 in D. Arms combine
only in :mod:`.weigh`, where the isoform is a blocking factor.

**Anything arithmetic is taken away from the judge.** Prometheus has no isoform
biology; it grades whether a verdict is supported by the text in front of it.
Number fabrication and not-evaluable violations are decidable in Python, so
:mod:`.checks` decides them exactly and for free, and no rubric asks.
"""

from __future__ import annotations

# Output units of one isoform: the six category letters plus the synthesis pass.
CATEGORY_LETTERS: tuple[str, ...] = ("C", "D", "L", "M", "P", "S")
SYNTHESIS_UNIT = "synthesis"
UNITS: tuple[str, ...] = CATEGORY_LETTERS + (SYNTHESIS_UNIT,)

# Letter <-> the category name used as a key inside categories.json.
CATEGORY_NAMES: dict[str, str] = {
    "C": "Conservation",
    "D": "Detection",
    "L": "Localization",
    "M": "Mutation Landscape",
    "P": "Predicted Structure",
    "S": "Structural Characteristics",
}
LETTER_BY_NAME: dict[str, str] = {v: k for k, v in CATEGORY_NAMES.items()}

# The eight arms of the matrix, then the replicate. The replicate is NOT a cell of
# the design: it exists only to set the noise floor, and including it in the
# factorial would invent a fifth grounding.
ARMS: tuple[str, ...] = (
    "criteria_hint",
    "criteria_nohint",
    "raw_hint",
    "raw_nohint",
    "tags_hint",
    "tags_nohint",
    "dist_hint",
    "dist_nohint",
)
REPLICATE = "criteria_hint_rep"
BASELINE = "criteria_hint"  # pinned at strength 0 in the Bradley-Terry fit
ALL_ARMS: tuple[str, ...] = ARMS + (REPLICATE,)

DEFAULT_CORPUS = "cheeseman50"

__all__ = [
    "ALL_ARMS",
    "ARMS",
    "BASELINE",
    "CATEGORY_LETTERS",
    "CATEGORY_NAMES",
    "DEFAULT_CORPUS",
    "LETTER_BY_NAME",
    "REPLICATE",
    "SYNTHESIS_UNIT",
    "UNITS",
]
