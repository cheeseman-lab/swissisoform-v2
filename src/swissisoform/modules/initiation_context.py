"""Module 2 — Initiation context: Kozak Hamming distance and GC content.

Computes Kozak sequence context features for each TIS using the kozak_context
field (a 13-nt string around the start codon). Ported from coTISja
analysis_pipeline_helpers.py.
"""

from __future__ import annotations

from swissisoform.config import PipelineConfig
from swissisoform.models import TranslationInitiationSite

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KOZAK_CONSENSUS = "gccgccRccATGG"  # 13-mer, positions -9 to +4

FULL_WEIGHTS = [1] * 13
MAJOR_WEIGHTS = [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1]
PARTIAL_WEIGHTS = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 0.1, 0.1, 1, 1, 1, 1]

AMBIGUITY_DICT: dict[str, set[str]] = {
    "A": {"A"}, "C": {"C"}, "G": {"G"}, "T": {"T"}, "U": {"T"},
    "R": {"A", "G"}, "Y": {"C", "T"}, "M": {"A", "C"},
    "K": {"G", "T"}, "S": {"C", "G"}, "W": {"A", "T"},
    "B": {"C", "G", "T"}, "D": {"A", "G", "T"}, "H": {"A", "C", "T"},
    "V": {"A", "C", "G"}, "N": {"A", "C", "G", "T"},
    # Lowercase versions
    "a": {"A"}, "c": {"C"}, "g": {"G"}, "t": {"T"}, "u": {"T"},
    "r": {"A", "G"}, "y": {"C", "T"}, "m": {"A", "C"},
    "k": {"G", "T"}, "s": {"C", "G"}, "w": {"A", "T"},
    "b": {"C", "G", "T"}, "d": {"A", "G", "T"}, "h": {"A", "C", "T"},
    "v": {"A", "C", "G"}, "n": {"A", "C", "G", "T"},
}

# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------


def hamming_distance_ambiguous(
    s1: str,
    s2: str,
    weights: list[float] | None = None,
) -> float:
    """Weighted Hamming distance with IUPAC ambiguity code support.

    For each position the IUPAC-expanded nucleotide sets of *s1* and *s2* are
    compared. If their intersection is empty the position counts as a mismatch
    and the corresponding weight is added to the distance.

    Args:
        s1: First sequence (may contain IUPAC codes).
        s2: Second sequence (may contain IUPAC codes).
        weights: Per-position weights.  ``None`` means all-ones.

    Returns:
        Weighted Hamming distance (float).

    Raises:
        ValueError: If *s1* and *s2* differ in length.
    """
    if len(s1) != len(s2):
        raise ValueError("Sequences must be of equal length")

    if weights is None:
        weights = [1] * len(s1)

    distance: float = 0
    for i, (a, b) in enumerate(zip(s1, s2)):
        set_a = AMBIGUITY_DICT.get(a, {a.upper()})
        set_b = AMBIGUITY_DICT.get(b, {b.upper()})
        if not set_a & set_b:
            distance += weights[i]
    return distance


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class InitiationContextModule:
    """Annotates each TIS with Kozak context features.

    Attributes:
        MODULE_NAME: ``"initiation_context"``
        OUTPUT_COLUMNS: Nine prefixed column names.
        SCOPE: ``"C"`` (per-candidate site).
    """

    MODULE_NAME: str = "initiation_context"
    OUTPUT_COLUMNS: list[str] = [
        "initiation_context_kozak_context",
        "initiation_context_kozak_hamming_major",
        "initiation_context_kozak_hamming_partial",
        "initiation_context_kozak_hamming_full",
        "initiation_context_utr5_gc_content",
        "initiation_context_upstream_aug_count",
        "initiation_context_upstream_non_aug_count",
        "initiation_context_gc_window_50bp",
        "initiation_context_gc_window_250bp",
    ]
    SCOPE: str = "C"

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize with pipeline configuration."""
        self._config = config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gc_content(seq: str) -> float | None:
        """Return GC fraction of *seq*, or ``None`` if empty."""
        if not seq:
            return None
        upper = seq.upper()
        gc = sum(1 for c in upper if c in ("G", "C"))
        return gc / len(upper)

    @staticmethod
    def _kozak_hamming(
        kozak_context: str | None,
    ) -> tuple[float | None, float | None, float | None]:
        """Return (major, partial, full) Hamming distances to consensus.

        Returns a triple of ``None`` when *kozak_context* is missing or the
        wrong length.
        """
        if kozak_context is None or len(kozak_context) != 13:
            return None, None, None
        return (
            hamming_distance_ambiguous(kozak_context, KOZAK_CONSENSUS, MAJOR_WEIGHTS),
            hamming_distance_ambiguous(kozak_context, KOZAK_CONSENSUS, PARTIAL_WEIGHTS),
            hamming_distance_ambiguous(kozak_context, KOZAK_CONSENSUS, FULL_WEIGHTS),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self, tis_sites: list[TranslationInitiationSite]
    ) -> list[TranslationInitiationSite]:
        """Annotate each TIS with Kozak context features.

        Args:
            tis_sites: Input sites to annotate.

        Returns:
            The same sites with ``annotations["initiation_context"]`` populated.
        """
        for site in tis_sites:
            major, partial, full = self._kozak_hamming(site.kozak_context)
            gc = self._gc_content(site.kozak_context) if site.kozak_context else None

            site.annotations[self.MODULE_NAME] = {
                "kozak_context": site.kozak_context,
                "kozak_hamming_major": major,
                "kozak_hamming_partial": partial,
                "kozak_hamming_full": full,
                "utr5_gc_content": gc,
                "upstream_aug_count": None,
                "upstream_non_aug_count": None,
                "gc_window_50bp": None,
                "gc_window_250bp": None,
            }

        return tis_sites
