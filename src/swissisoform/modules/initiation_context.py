"""Module 2 — Initiation context: Kozak Hamming distance and GC content.

Computes Kozak sequence context features for each TIS using the kozak_context
field (a 13-nt string around the start codon).

Implements the ``SiteModule`` protocol: per-site annotation logic lives in
``annotate_site()`` and ``run()`` is a thin loop wrapper.
"""

from __future__ import annotations

from typing import Any

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
    "A": {"A"},
    "C": {"C"},
    "G": {"G"},
    "T": {"T"},
    "U": {"T"},
    "R": {"A", "G"},
    "Y": {"C", "T"},
    "M": {"A", "C"},
    "K": {"G", "T"},
    "S": {"C", "G"},
    "W": {"A", "T"},
    "B": {"C", "G", "T"},
    "D": {"A", "G", "T"},
    "H": {"A", "C", "T"},
    "V": {"A", "C", "G"},
    "N": {"A", "C", "G", "T"},
    # Lowercase versions
    "a": {"A"},
    "c": {"C"},
    "g": {"G"},
    "t": {"T"},
    "u": {"T"},
    "r": {"A", "G"},
    "y": {"C", "T"},
    "m": {"A", "C"},
    "k": {"G", "T"},
    "s": {"C", "G"},
    "w": {"A", "T"},
    "b": {"C", "G", "T"},
    "d": {"A", "G", "T"},
    "h": {"A", "C", "T"},
    "v": {"A", "C", "G"},
    "n": {"A", "C", "G", "T"},
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

    Implements the ``SiteModule`` protocol: ``annotate_site()`` computes
    annotations for a single TIS, and ``run()`` loops over all sites.

    Current outputs are derived entirely from ``site.kozak_context`` (a 13-nt
    string around the start codon).  Additional upstream-sequence features
    (``upstream_aug_count``, ``gc_window_50bp`` etc.) are NOT implemented in
    this module; they require 5'UTR sequence from the genome FASTA, which is
    not available here.  See future work to wire genome access through the
    assembly layer.

    Attributes:
        MODULE_NAME: ``"initiation_context"``
        OUTPUT_COLUMNS: Five prefixed column names (Kozak-derived only).
        SCOPE: ``"C"`` (per-candidate site).
    """

    MODULE_NAME: str = "initiation_context"
    OUTPUT_COLUMNS: list[str] = [
        "initiation_context_kozak_context",
        "initiation_context_kozak_hamming_major",
        "initiation_context_kozak_hamming_partial",
        "initiation_context_kozak_hamming_full",
        "initiation_context_utr5_gc_content",
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
    # Public API (SiteModule protocol)
    # ------------------------------------------------------------------

    def annotate_site(self, site: TranslationInitiationSite) -> dict[str, Any]:
        """Compute initiation-context annotations for a single TIS.

        This is the core per-site logic required by the ``SiteModule`` protocol.
        It returns the annotation dict but does **not** write to
        ``site.isoform_annotations``.

        Args:
            site: A single translation initiation site.

        Returns:
            Dict of annotation key-value pairs for this site.
        """
        major, partial, full = self._kozak_hamming(site.kozak_context)
        gc = self._gc_content(site.kozak_context) if site.kozak_context else None

        return {
            "kozak_context": site.kozak_context,
            "kozak_hamming_major": major,
            "kozak_hamming_partial": partial,
            "kozak_hamming_full": full,
            "utr5_gc_content": gc,
        }

    def run(self, tis_sites: list[TranslationInitiationSite]) -> list[TranslationInitiationSite]:
        """Annotate all TIS sites with Kozak context features.

        Thin wrapper that calls ``annotate_site()`` for each site and stores
        the result in ``site.isoform_annotations[MODULE_NAME]``.

        Args:
            tis_sites: Input sites to annotate.

        Returns:
            The same sites with ``isoform_annotations["initiation_context"]``
            populated.
        """
        for site in tis_sites:
            site.isoform_annotations[self.MODULE_NAME] = self.annotate_site(site)
        return tis_sites
