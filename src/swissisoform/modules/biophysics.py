"""Module: Biophysics — biophysical properties of a protein sequence.

Implements ``ProteinModule``: the core logic lives in ``annotate(protein)`` which
accepts any raw protein sequence and returns a dict of 18 biophysical properties
(isoelectric point, hydropathy, disorder propensity, sequence complexity, LLPS
scores, etc.).  The ``run()`` method is a convenience wrapper that calls
``annotate()`` for each TIS site's differential region.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from swissisoform.config import PipelineConfig
from swissisoform.models import TranslationInitiationSite

try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    _HAS_BIOPYTHON = True
except ImportError:
    _HAS_BIOPYTHON = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHARGED_AA = set("DEKRH")
DISORDER_PROMOTING = set("AESGQKPRD")
AROMATIC_AA = set("FWY")
PIPI_AA = set("FYWRHQN")
PRIONLIKE_AA = set("QNGSY")

KD_HYDROPATHY: dict[str, float] = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "E": -3.5,
    "Q": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

PK_VALUES: dict[str, float] = {
    "N_term": 9.69,
    "C_term": 2.34,
    "D": 3.9,
    "E": 4.07,
    "C": 8.18,
    "Y": 10.46,
    "H": 6.04,
    "K": 10.54,
    "R": 12.48,
}

TOP_IDP: dict[str, float] = {
    "A": 0.06,
    "R": 0.18,
    "N": 0.007,
    "D": 0.192,
    "C": -0.02,
    "E": 0.736,
    "Q": 0.318,
    "G": 0.166,
    "H": 0.303,
    "I": -0.486,
    "L": -0.326,
    "K": 0.586,
    "M": -0.397,
    "F": -0.697,
    "P": 0.987,
    "S": 0.341,
    "T": 0.059,
    "W": -0.884,
    "Y": -0.510,
    "V": -0.121,
}

_MIN_SEQ_LEN = 3

# ---------------------------------------------------------------------------
# Private helper functions
# ---------------------------------------------------------------------------


def _clean_seq(seq: str) -> str:
    """Strip trailing stop codon."""
    return seq.rstrip("*")


def _charge_at_ph(seq: str, ph: float) -> float:
    """Net charge at a given pH via Henderson-Hasselbalch."""
    counts = Counter(seq)
    # Positive contributions: N-term, K, R, H
    charge = 1.0 / (1.0 + 10 ** (ph - PK_VALUES["N_term"]))
    for aa, pk in (("K", PK_VALUES["K"]), ("R", PK_VALUES["R"]), ("H", PK_VALUES["H"])):
        charge += counts.get(aa, 0) / (1.0 + 10 ** (ph - pk))
    # Negative contributions: C-term, D, E, C, Y
    charge -= 1.0 / (1.0 + 10 ** (PK_VALUES["C_term"] - ph))
    for aa, pk in (
        ("D", PK_VALUES["D"]),
        ("E", PK_VALUES["E"]),
        ("C", PK_VALUES["C"]),
        ("Y", PK_VALUES["Y"]),
    ):
        charge -= counts.get(aa, 0) / (1.0 + 10 ** (pk - ph))
    return charge


def _compute_pi(seq: str) -> float:
    """Isoelectric point via binary search."""
    lo, hi = 0.0, 14.0
    while hi - lo > 0.01:
        mid = (lo + hi) / 2.0
        if _charge_at_ph(seq, mid) > 0:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2.0, 4)


def _compute_gravy(seq: str) -> float:
    """Grand average of hydropathy (Kyte-Doolittle)."""
    values = [KD_HYDROPATHY[aa] for aa in seq if aa in KD_HYDROPATHY]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _compute_disorder(seq: str) -> float:
    """Mean Top-IDP disorder propensity."""
    values = [TOP_IDP[aa] for aa in seq if aa in TOP_IDP]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _fraction_in_set(seq: str, aa_set: set[str]) -> float:
    """Fraction of residues belonging to *aa_set*."""
    if not seq:
        return 0.0
    count = sum(1 for aa in seq if aa in aa_set)
    return round(count / len(seq), 4)


def _instability_index(seq: str) -> float | None:
    """Instability index via BioPython, or None if unavailable."""
    if not _HAS_BIOPYTHON:
        return None
    try:
        pa = ProteinAnalysis(seq)
        return round(pa.instability_index(), 4)
    except Exception:  # noqa: BLE001
        return None


def _shannon_entropy(seq: str) -> float:
    """Shannon entropy over amino acid frequencies."""
    n = len(seq)
    if n == 0:
        return 0.0
    counts = Counter(seq)
    entropy = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            entropy -= p * math.log2(p)
    return round(entropy, 4)


def _normalized_complexity(seq: str) -> float:
    """Shannon entropy normalized by log2(min(len, 20))."""
    n = len(seq)
    if n < 2:
        return 0.0
    se = _shannon_entropy(seq)
    denom = math.log2(min(n, 20))
    if denom == 0:
        return 0.0
    return round(se / denom, 4)


def _mean_window_entropy(seq: str, window: int = 20) -> float:
    """Average Shannon entropy over sliding windows."""
    n = len(seq)
    if n < window:
        return _shannon_entropy(seq)
    entropies = [_shannon_entropy(seq[i : i + window]) for i in range(n - window + 1)]
    return round(sum(entropies) / len(entropies), 4)


def _fraction_in_lcr(seq: str, window: int = 12, threshold: float = 2.2) -> float:
    """Fraction of positions falling in low-complexity regions (SEG-like)."""
    n = len(seq)
    if n < window:
        return 0.0
    in_lcr = [False] * n
    for i in range(n - window + 1):
        w = seq[i : i + window]
        if _shannon_entropy(w) < threshold:
            for j in range(i, i + window):
                in_lcr[j] = True
    return round(sum(in_lcr) / n, 4)


def _aa_diversity(seq: str) -> int:
    """Number of distinct amino acid types."""
    return len(set(seq))


def _longest_homopolymer(seq: str) -> int:
    """Length of the longest run of a single residue."""
    if not seq:
        return 0
    max_run = 1
    current_run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 1
    return max_run


def _rg_fg_density(seq: str) -> float:
    """RG/FG motif density per 100 amino acids."""
    if not seq:
        return 0.0
    count = 0
    for i in range(len(seq) - 1):
        pair = seq[i : i + 2]
        if pair in ("RG", "FG"):
            count += 1
    return round(count / len(seq) * 100, 4)


def _composite_llps(
    disorder: float,
    prionlike: float,
    pipi: float,
    rg_fg: float,
    lcr_frac: float,
) -> float:
    """Composite LLPS propensity score."""
    score = (
        0.25 * disorder
        + 0.25 * prionlike
        + 0.20 * pipi
        + 0.15 * min(rg_fg / 10.0, 1.0)
        + 0.15 * lcr_frac
    )
    return round(score, 4)


# ---------------------------------------------------------------------------
# Module class
# ---------------------------------------------------------------------------


class BiophysicsModule:
    """Biophysics module implementing the ``ProteinModule`` protocol.

    The core logic lives in :meth:`annotate`, which accepts any raw protein
    sequence string and returns a dict of 18 biophysical properties.  The
    :meth:`run` method is a thin wrapper that calls ``annotate()`` for each
    TIS site's differential region.

    Attributes:
        MODULE_NAME: Unique module identifier.
        OUTPUT_COLUMNS: Column names produced (prefixed with MODULE_NAME_).
        SCOPE: Module scope ('C' for per-site classification).
    """

    MODULE_NAME: str = "biophysics"
    OUTPUT_COLUMNS: list[str] = [
        "biophysics_pI",
        "biophysics_gravy",
        "biophysics_disorder",
        "biophysics_fraction_charged",
        "biophysics_fraction_disorder_promoting",
        "biophysics_aromaticity",
        "biophysics_instability_index",
        "biophysics_length",
        "biophysics_shannon_entropy",
        "biophysics_normalized_complexity",
        "biophysics_mean_window_entropy",
        "biophysics_fraction_lcr",
        "biophysics_aa_diversity",
        "biophysics_longest_homopolymer",
        "biophysics_pipi_propensity",
        "biophysics_prionlike_fraction",
        "biophysics_rg_fg_density",
        "biophysics_llps_score",
    ]
    SCOPE: str = "C"

    # Keys in the annotation dict (without the module prefix).
    _ANNOTATION_KEYS: list[str] = [
        "pI",
        "gravy",
        "disorder",
        "fraction_charged",
        "fraction_disorder_promoting",
        "aromaticity",
        "instability_index",
        "length",
        "shannon_entropy",
        "normalized_complexity",
        "mean_window_entropy",
        "fraction_lcr",
        "aa_diversity",
        "longest_homopolymer",
        "pipi_propensity",
        "prionlike_fraction",
        "rg_fg_density",
        "llps_score",
    ]

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize with pipeline configuration.

        Args:
            config: Pipeline configuration (no biophysics-specific settings needed).
        """
        self._config = config

    # --------------------------------------------------------------------- #
    # ProteinModule interface
    # --------------------------------------------------------------------- #

    def annotate(self, protein: str) -> dict[str, Any]:
        """Compute biophysical properties for a raw protein sequence.

        This is the core, pure-function logic of the module. It accepts any
        protein sequence (not just differential regions) and returns a dict
        of 18 biophysical annotations.

        Args:
            protein: Raw amino-acid sequence (may contain trailing ``*``).

        Returns:
            Dict mapping annotation keys to computed values, or to ``None``
            when the (cleaned) sequence is shorter than 3 residues.
        """
        seq = _clean_seq(protein.upper())

        if len(seq) < _MIN_SEQ_LEN:
            return {key: None for key in self._ANNOTATION_KEYS}

        pi = _compute_pi(seq)
        gravy = _compute_gravy(seq)
        disorder = _compute_disorder(seq)
        frac_charged = _fraction_in_set(seq, CHARGED_AA)
        frac_disorder = _fraction_in_set(seq, DISORDER_PROMOTING)
        aromaticity = _fraction_in_set(seq, AROMATIC_AA)
        instability = _instability_index(seq)
        length = len(seq)
        entropy = _shannon_entropy(seq)
        norm_complexity = _normalized_complexity(seq)
        mean_win_ent = _mean_window_entropy(seq)
        frac_lcr = _fraction_in_lcr(seq)
        diversity = _aa_diversity(seq)
        homo = _longest_homopolymer(seq)
        pipi = _fraction_in_set(seq, PIPI_AA)
        prionlike = _fraction_in_set(seq, PRIONLIKE_AA)
        rg_fg = _rg_fg_density(seq)
        llps = _composite_llps(disorder, prionlike, pipi, rg_fg, frac_lcr)

        return {
            "pI": pi,
            "gravy": gravy,
            "disorder": disorder,
            "fraction_charged": frac_charged,
            "fraction_disorder_promoting": frac_disorder,
            "aromaticity": aromaticity,
            "instability_index": instability,
            "length": length,
            "shannon_entropy": entropy,
            "normalized_complexity": norm_complexity,
            "mean_window_entropy": mean_win_ent,
            "fraction_lcr": frac_lcr,
            "aa_diversity": diversity,
            "longest_homopolymer": homo,
            "pipi_propensity": pipi,
            "prionlike_fraction": prionlike,
            "rg_fg_density": rg_fg,
            "llps_score": llps,
        }

    # --------------------------------------------------------------------- #
    # Convenience wrapper (backward-compatible)
    # --------------------------------------------------------------------- #

    def run(self, tis_sites: list[TranslationInitiationSite]) -> list[TranslationInitiationSite]:
        """Compute biophysical annotations for each TIS site.

        Thin wrapper around :meth:`annotate` that extracts the differential
        region sequence from each site.

        Args:
            tis_sites: Input TIS sites with diff_region populated.

        Returns:
            The same sites with isoform_annotations["biophysics"] populated.
        """
        for site in tis_sites:
            seq = site.diff_region.sequence if site.diff_region else ""
            site.isoform_annotations[self.MODULE_NAME] = self.annotate(seq)

        return tis_sites
