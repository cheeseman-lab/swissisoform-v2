"""Module: Conservation — Zoonomia PhyloP / PhastCons BigWig lookups.

Implementation of Path 3 from ``docs/reviews/conservation_module_spec.md``:
nucleotide-level conservation metrics derived from the 241-mammal Cactus
alignment (Christmas et al. 2023), queried as pre-computed PhyloP and
PhastCons BigWig tracks.  Start-codon and Kozak-window metrics are computed
here; region (unique / shared / enrichment) metrics are stubbed — they
require a protein→genomic coordinate mapper that doesn't exist yet and is
tracked separately.

Module type
-----------
``SiteModule``.  Conservation is a property of *genomic* coordinates, not of
a protein sequence — two distinct proteins can land on the same codon and
share the same PhyloP score at the TIS.  ``annotate_site(site)`` reads
``chrom`` / ``position`` / ``strand`` directly.

Outputs distinguish ``status="ok"``, ``status="not_run"`` (no BigWig / no
config), and ``status="region_map_not_implemented"`` for the stubbed region
metrics.  Missing values never silently become zero.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from swissisoform.config import PipelineConfig
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

    Unique / shared / enrichment region means are declared as outputs but
    stubbed to ``None`` with ``status="region_map_not_implemented"``.  They
    require mapping protein-space diff regions to genomic coordinates, which
    needs the isoform transcript's exon structure — a separate workstream.

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
        return chrom if chrom.startswith("chr") else f"chr{chrom}"

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

        return {
            "phylop_at_tis": phylop_at_tis,
            "phylop_kozak_mean": phylop_kozak,
            # Region metrics require protein→genomic mapping on the isoform
            # transcript (exon-aware).  Stubbed here so downstream consumers
            # can tell "not computed" from "computed and zero".
            "phylop_unique_region_mean": None,
            "phylop_shared_region_mean": None,
            "phylop_enrichment": None,
            "phastcons_at_tis": phastcons_at_tis,
            "phastcons_kozak_mean": phastcons_kozak,
            "phastcons_unique_region_mean": None,
            "phastcons_shared_region_mean": None,
            "summary": {
                "phylop_status": phylop_status,
                "phastcons_status": phastcons_status,
                "region_status": "region_map_not_implemented",
                "phylop_bigwig": str(self._phylop_path) if self._phylop_path else None,
                "phastcons_bigwig": str(self._phastcons_path) if self._phastcons_path else None,
            },
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
