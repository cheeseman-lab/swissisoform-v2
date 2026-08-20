"""Module: Conservation — Zoonomia PhyloP / PhastCons BigWig lookups.

Implementation of Path 3 from ``docs/reviews/conservation_module_spec.md``:
nucleotide-level conservation metrics queried as pre-computed BigWig tracks.
PhyloP comes from the 241-mammal Cactus alignment (``cactus241way``, Christmas
et al. 2023); PhastCons is the 100-vertebrate ``hg38.phastCons100way`` track.
Start-codon, Kozak-window, and region (unique /
shared / enrichment) metrics are all computed here.  Region metrics consume
``TIS.orf_exons`` and ``TIS.canonical_orf_exons`` produced by the assembly
layer's Layer-2 walker over the GTF transcript skeletons.

Module type
-----------
``SiteModule``.  Conservation is a property of *genomic* coordinates, not of
a protein sequence — two distinct proteins can land on the same codon and
share the same PhyloP score at the TIS.  ``annotate_site(site)`` reads
``chrom`` / ``position`` / ``strand`` directly.

Outputs distinguish ``status="ok"``, ``status="not_run"`` (no BigWig / no
config), and ``status="no_skeleton"`` (ORF exons unavailable for region
metrics).  Missing values never silently become zero.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from swissisoform.config import PipelineConfig
from swissisoform.coords import (
    interval_difference,
    interval_intersection,
    interval_length,
    normalize_chrom,
)
from swissisoform.models import TranslationInitiationSite

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Kozak window: 13 nt, ATG at positions 9–11 (matches assembly.py Kozak
# extraction — 9 nt of 5' context + ATG + 1 nt of 3' context).
KOZAK_UPSTREAM_NT: int = 9
KOZAK_DOWNSTREAM_NT: int = 1
KOZAK_CODON_LEN: int = 3

START_CODON_LEN: int = 3


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class ConservationModule:
    """Annotate TIS sites with PhyloP / PhastCons conservation scores.

    Implements the ``SiteModule`` protocol.  Reads genomic coordinates from
    the TIS and looks up PhyloP and PhastCons means over:

    - the start codon itself (3 nt window at the TIS)
    - the Kozak window (13 nt: ATG + 9 nt upstream + 1 nt downstream)

    Unique / shared / enrichment region means are computed from
    ``site.orf_exons`` and ``site.canonical_orf_exons`` (Layer-2 ORF exon
    intervals populated by the assembly layer).  When the skeleton is
    unavailable for a TIS, region means fall through to ``None`` with
    ``region_status="no_skeleton"``.

    Attributes:
        MODULE_NAME: ``"conservation"``
        OUTPUT_COLUMNS: Prefixed column names produced by this module.
        SCOPE: ``"C"`` (per-candidate).
    """

    MODULE_NAME: str = "conservation"
    OUTPUT_COLUMNS: list[str] = [
        "conservation_phylop_at_tis",
        "conservation_phylop_kozak_mean",
        "conservation_phylop_unique_region_mean",
        "conservation_phylop_shared_region_mean",
        "conservation_phylop_enrichment",
        "conservation_phastcons_at_tis",
        "conservation_phastcons_kozak_mean",
        "conservation_phastcons_unique_region_mean",
        "conservation_phastcons_shared_region_mean",
        "conservation_summary",
    ]
    SCOPE: str = "C"

    def __init__(self, config: PipelineConfig) -> None:
        """Resolve BigWig paths and open handles if available.

        Args:
            config: Pipeline config.  Reads ``config.conservation.phylop_bigwig``
                and ``config.conservation.phastcons_bigwig``.  Either or both
                may be absent — each track is looked up independently.
        """
        self._config = config
        self._phylop_path: Path | None = None
        self._phastcons_path: Path | None = None
        self._phylop_bw = None
        self._phastcons_bw = None

        if config.conservation is not None:
            self._phylop_path = _resolve_existing(config.conservation.phylop_bigwig)
            self._phastcons_path = _resolve_existing(config.conservation.phastcons_bigwig)

        self._phylop_bw = _open_bigwig(self._phylop_path, "PhyloP")
        self._phastcons_bw = _open_bigwig(self._phastcons_path, "PhastCons")

    # ------------------------------------------------------------------
    # BigWig helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_chrom(chrom: str) -> str:
        """Return *chrom* as stored in UCSC BigWig (prefer ``chrN``)."""
        if not chrom:
            return chrom
        return normalize_chrom(chrom)

    @staticmethod
    def _mean_from_bw(bw, chrom: str, start: int, end: int) -> float | None:
        """Return the mean BigWig value over a half-open interval, or None.

        Uses pyBigWig's native ``stats(..., type="mean")``.  Returns ``None``
        when the interval is empty, the chromosome is absent, or all
        positions are NaN.  Tries both ``chrN`` and ``N`` naming.
        """
        if bw is None or end <= start:
            return None
        chroms = bw.chroms()
        candidates = [chrom, chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"]
        for name in candidates:
            if name in chroms and 0 <= start < chroms[name]:
                end_clamped = min(end, chroms[name])
                try:
                    values = bw.stats(name, start, end_clamped, type="mean")
                except (RuntimeError, OverflowError) as exc:
                    logger.warning("BigWig stats failed for %s:%d-%d: %s", name, start, end, exc)
                    return None
                if values and values[0] is not None:
                    return float(values[0])
                return None
        return None

    @classmethod
    def _mean_over_intervals(
        cls,
        bw,
        chrom: str,
        intervals: list[tuple[int, int]],
    ) -> float | None:
        """Length-weighted mean BigWig value across a list of intervals.

        Returns ``None`` if the intervals list is empty or every interval
        returned ``None`` (chromosome absent / all-NaN region).
        """
        if bw is None or not intervals:
            return None
        total_weight = 0
        weighted_sum = 0.0
        for start, end in intervals:
            if end <= start:
                continue
            mean = cls._mean_from_bw(bw, chrom, start, end)
            if mean is None:
                continue
            width = end - start
            weighted_sum += mean * width
            total_weight += width
        if total_weight == 0:
            return None
        return weighted_sum / total_weight

    # ------------------------------------------------------------------
    # Coordinate windows around the TIS
    # ------------------------------------------------------------------

    @staticmethod
    def _tis_codon_window(position: int, strand: str) -> tuple[int, int]:
        """Return a half-open 0-based reference interval for the 3 nt start codon.

        Matches the coordinate convention documented on
        ``assembly.extract_kozak_context``: ``TIS.position`` is 0-based.  On
        ``+`` strand it is the A of ATG — codon spans ``[position, position+3)``.
        On ``-`` strand it is the exclusive end of the ORF, so the A of ATG on
        mRNA sits at plus-strand ``position - 1`` and the codon spans
        ``[position - 3, position)`` in reference coordinates.
        """
        if strand == "-":
            start = position - START_CODON_LEN
            end = position
        else:
            start = position
            end = position + START_CODON_LEN
        return max(start, 0), end

    @staticmethod
    def _kozak_window(position: int, strand: str) -> tuple[int, int]:
        """Return a half-open 0-based reference interval for the Kozak window.

        13 nt (mRNA −9..+4) with ATG at mRNA indices 9–11.  In reference
        coordinates this is ``[position-9, position+4)`` on ``+`` strand and
        ``[position-4, position+9)`` on ``-`` strand — identical to the
        genomic fetch windows in ``assembly.extract_kozak_context``.
        """
        if strand == "-":
            start = position - 4
            end = position + KOZAK_UPSTREAM_NT
        else:
            start = position - KOZAK_UPSTREAM_NT
            end = position + 4
        return max(start, 0), end

    # ------------------------------------------------------------------
    # Public API (SiteModule protocol)
    # ------------------------------------------------------------------

    def annotate_site(self, site: TranslationInitiationSite) -> dict[str, Any]:
        """Compute PhyloP / PhastCons lookups at the start codon and Kozak window.

        Args:
            site: A TIS with ``chrom`` / ``position`` / ``strand`` populated.

        Returns:
            Dict with scalar ``conservation_*`` metrics plus a ``summary``
            dict recording the status of each track and of the region
            analysis (stubbed — see module docstring).  Columns with no
            available track return ``None``.
        """
        chrom = self._clean_chrom(site.chrom)
        codon_start, codon_end = self._tis_codon_window(site.position, site.strand)
        kozak_start, kozak_end = self._kozak_window(site.position, site.strand)

        phylop_at_tis = self._mean_from_bw(self._phylop_bw, chrom, codon_start, codon_end)
        phylop_kozak = self._mean_from_bw(self._phylop_bw, chrom, kozak_start, kozak_end)
        phastcons_at_tis = self._mean_from_bw(self._phastcons_bw, chrom, codon_start, codon_end)
        phastcons_kozak = self._mean_from_bw(self._phastcons_bw, chrom, kozak_start, kozak_end)

        phylop_status = "ok" if self._phylop_bw is not None else "not_run"
        phastcons_status = "ok" if self._phastcons_bw is not None else "not_run"

        # ── Region metrics (unique vs shared) from Layer-2 ORF exons ──────
        # Unique region is ORF-type-aware: extensions/uORFs/altORFs contribute
        # new sequence (isoform_orf \ canonical_orf); truncations *lose* sequence
        # so their unique region is canonical_orf \ isoform_orf. Without this
        # split, truncations would always show unique_region_nt == 0 and the
        # lost canonical N-terminus would silently get no conservation signal.
        from swissisoform.models import ORFType
        region_status: str
        unique_space: str
        if not site.orf_exons or not site.canonical_orf_exons:
            unique_intervals: list[tuple[int, int]] = []
            shared_intervals: list[tuple[int, int]] = []
            region_status = "no_skeleton"
            unique_space = "none"
        else:
            if site.orf_type == ORFType.TRUNCATED:
                unique_intervals = interval_difference(
                    site.canonical_orf_exons, site.orf_exons
                )
                unique_space = "canonical"
            else:
                unique_intervals = interval_difference(
                    site.orf_exons, site.canonical_orf_exons
                )
                unique_space = "isoform"
            shared_intervals = interval_intersection(site.orf_exons, site.canonical_orf_exons)
            region_status = "ok"

        phylop_unique = self._mean_over_intervals(self._phylop_bw, chrom, unique_intervals)
        phylop_shared = self._mean_over_intervals(self._phylop_bw, chrom, shared_intervals)
        phastcons_unique = self._mean_over_intervals(self._phastcons_bw, chrom, unique_intervals)
        phastcons_shared = self._mean_over_intervals(self._phastcons_bw, chrom, shared_intervals)

        phylop_enrichment: float | None = None
        if (
            phylop_unique is not None
            and phylop_shared is not None
            and phylop_shared != 0.0
        ):
            phylop_enrichment = phylop_unique / phylop_shared

        return {
            "phylop_at_tis": phylop_at_tis,
            "phylop_kozak_mean": phylop_kozak,
            "phylop_unique_region_mean": phylop_unique,
            "phylop_shared_region_mean": phylop_shared,
            "phylop_enrichment": phylop_enrichment,
            "phastcons_at_tis": phastcons_at_tis,
            "phastcons_kozak_mean": phastcons_kozak,
            "phastcons_unique_region_mean": phastcons_unique,
            "phastcons_shared_region_mean": phastcons_shared,
            "summary": {
                "phylop_status": phylop_status,
                "phastcons_status": phastcons_status,
                "region_status": region_status,
                "unique_space": unique_space,
                "unique_region_nt": interval_length(unique_intervals),
                "shared_region_nt": interval_length(shared_intervals),
                "phylop_bigwig": str(self._phylop_path) if self._phylop_path else None,
                "phastcons_bigwig": str(self._phastcons_path) if self._phastcons_path else None,
            },
        }

    def annotate_canonical_site(self, site: TranslationInitiationSite) -> dict[str, Any] | None:
        """Point conservation at the canonical (Annotated) start.

        The symmetric twin of the at-TIS / Kozak-window point values.
        Returns the four ``phylop_*`` / ``phastcons_*`` point values keyed
        without a prefix (serialized as ``canonical_conservation_*``), or
        ``None`` when the canonical start position can't be derived.
        """
        from swissisoform.assembly import canonical_start_position

        pos = canonical_start_position(site.canonical_orf_exons, site.strand)
        if pos is None:
            return None
        chrom = self._clean_chrom(site.chrom)
        codon_start, codon_end = self._tis_codon_window(pos, site.strand)
        kozak_start, kozak_end = self._kozak_window(pos, site.strand)
        return {
            "phylop_at_tis": self._mean_from_bw(self._phylop_bw, chrom, codon_start, codon_end),
            "phylop_kozak_mean": self._mean_from_bw(self._phylop_bw, chrom, kozak_start, kozak_end),
            "phastcons_at_tis": self._mean_from_bw(
                self._phastcons_bw, chrom, codon_start, codon_end
            ),
            "phastcons_kozak_mean": self._mean_from_bw(
                self._phastcons_bw, chrom, kozak_start, kozak_end
            ),
        }

    def run(
        self,
        tis_sites: list[TranslationInitiationSite],
    ) -> list[TranslationInitiationSite]:
        """Annotate every TIS and write results to ``isoform_annotations``.

        Backward-compatible wrapper matching the module contract: never drops
        sites, always writes ``isoform_annotations[MODULE_NAME]``.
        """
        for site in tis_sites:
            site.isoform_annotations[self.MODULE_NAME] = self.annotate_site(site)
        return tis_sites

    def close(self) -> None:
        """Release BigWig handles.  Safe to call more than once."""
        for attr in ("_phylop_bw", "_phastcons_bw"):
            bw = getattr(self, attr, None)
            if bw is not None:
                try:
                    bw.close()
                except Exception:  # noqa: BLE001
                    pass
                setattr(self, attr, None)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _resolve_existing(path: Path | None) -> Path | None:
    """Return *path* if it exists, else ``None`` (with a warning)."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        logger.warning("Conservation BigWig not found: %s", p)
        return None
    return p


def _open_bigwig(path: Path | None, label: str):
    """Open a BigWig for reading, returning ``None`` if unavailable."""
    if path is None:
        return None
    try:
        import pyBigWig
    except ImportError:
        logger.warning("pyBigWig not installed; %s track disabled", label)
        return None
    try:
        return pyBigWig.open(str(path))
    except (RuntimeError, OSError) as exc:
        logger.warning("Failed to open %s BigWig %s: %s", label, path, exc)
        return None
