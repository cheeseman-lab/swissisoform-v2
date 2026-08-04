"""Module: SLiM Scanning — Short Linear Motif detection in differential sequences.

Scans the differential sequence for Short Linear Motifs (SLiMs) using
regex patterns derived from the ELM database. Ported from TIAP
``tiap.modules.motifs.MotifScanner``.

Implements the ProteinModule protocol: ``annotate(protein)`` returns
positional hit lists and summary counts; ``run()`` is a convenience wrapper.
"""

from __future__ import annotations

import re
from typing import Any

from swissisoform.config import PipelineConfig
from swissisoform.models import TranslationInitiationSite

# ---------------------------------------------------------------------------
# Motif definitions: (name, regex_pattern, description)
# ---------------------------------------------------------------------------

MOTIF_DEFS: list[tuple[str, str, str]] = [
    ("CDK_Sites_SP_TP", r"[ST]P", "CDK substrate consensus"),
    ("ATM_Sites_SQ_TQ", r"[ST]Q", "ATM/ATR substrate consensus"),
    ("Mode1_1433", r"R[SFYW][^P][ST][LVIMF]P", "14-3-3 binding Mode I"),
    ("Mode2_1433", r"R[^E][^P][ST][LEVM][^P][^P]", "14-3-3 binding Mode II"),
    ("EB1_SxIP", r"[ST].[IL]P", "EB1/microtubule plus-end tracking"),
    ("RGG_Sites", r"RGG", "RGG methylation / RNA binding"),
    ("SH3_ClassI", r"[RK].{2}P.{2}P", "SH3 domain binding Class I"),
    ("SH3_ClassII", r"P.{2}P.[RK]", "SH3 domain binding Class II"),
    ("Heme_CP_HRM", r"C[^C].{2}C[^C]H", "Heme regulatory motif"),
    ("Heme_CXXCH", r"C.{2}CH", "Cytochrome c heme binding"),
    ("PIP_box", r"Q.{2}[ILMV].{2}[FY][FY]", "PCNA-interacting peptide box"),
    ("APIM", r"[KR][FYW][ILMV][ILMV][KR]", "PCNA-interacting motif"),
    ("ZnF_C2H2", r"C.{2}C.{12}H.{3,5}H", "C2H2 zinc finger"),
    (
        "RING_finger",
        r"C.{2}C.{9,39}C.{1,3}H.{2,3}C.{2}C.{4,48}C.{2}C",
        "RING finger E3 ligase",
    ),
]

# All motif names in definition order (used for summary key ordering)
_MOTIF_NAMES: list[str] = [name for name, _, _ in MOTIF_DEFS]

# Precompiled RGG cluster and RG patterns
_RGG_CLUSTER_RE = re.compile(r"(RGG.{0,5}){2,}")
_RG_RE = re.compile(r"RG")


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class MotifsModule:
    """Scan differential sequences for Short Linear Motifs (SLiMs).

    Implements the ProteinModule protocol with ``annotate()`` returning
    positional hit lists and summary counts.

    Attributes:
        MODULE_NAME: ``"motifs"``
        OUTPUT_COLUMNS: ``["motifs_hits", "motifs_summary"]``
        SCOPE: ``"C"`` (per-candidate site).
    """

    MODULE_NAME: str = "motifs"
    OUTPUT_COLUMNS: list[str] = ["motifs_hits", "motifs_summary"]
    SCOPE: str = "C"

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize with pipeline configuration.

        Args:
            config: Pipeline configuration (no motif-specific fields required).
        """
        self._config = config
        # Precompile motif regexes for performance
        self._compiled: list[tuple[str, re.Pattern[str], str]] = [
            (name, re.compile(pattern), desc) for name, pattern, desc in MOTIF_DEFS
        ]

    # ------------------------------------------------------------------
    # Public API — ProteinModule protocol
    # ------------------------------------------------------------------

    def annotate(self, protein: str) -> dict[str, Any]:
        """Annotate a protein sequence with SLiM positional hits and summary.

        Args:
            protein: Protein amino-acid sequence (may include trailing ``*``).

        Returns:
            Dict with ``"hits"`` (list of positional hit dicts) and
            ``"summary"`` (counts, densities, aggregates).
        """
        # Clean: uppercase, strip trailing stop codon
        seq = protein.upper().rstrip("*") if protein else ""
        seq_len = len(seq)

        # Build zero-valued summary template
        summary: dict[str, Any] = {}
        for name in _MOTIF_NAMES:
            summary[name] = 0
            summary[f"{name}_density_per100aa"] = 0.0

        summary["total_1433"] = 0
        summary["density_1433_per100aa"] = 0.0
        summary["RGG_clusters"] = 0
        summary["RG_rich"] = 0

        if seq_len == 0:
            return {"hits": [], "summary": summary}

        # Scan all motifs and collect positional hits
        hits: list[dict[str, Any]] = []
        for name, pattern, _desc in self._compiled:
            matches = list(pattern.finditer(seq))
            count = len(matches)

            summary[name] = count
            summary[f"{name}_density_per100aa"] = round(count * 100.0 / seq_len, 4)

            for m in matches:
                hits.append(
                    {
                        "name": name,
                        "pos": m.start(),
                        "end": m.end(),
                        "match": m.group(),
                    }
                )

        # Aggregate: 14-3-3 totals
        total_1433 = summary["Mode1_1433"] + summary["Mode2_1433"]
        summary["total_1433"] = total_1433
        summary["density_1433_per100aa"] = round(total_1433 * 100.0 / seq_len, 4)

        # RGG clusters
        summary["RGG_clusters"] = len(_RGG_CLUSTER_RE.findall(seq))

        # RG-rich flag
        rg_count = len(_RG_RE.findall(seq))
        summary["RG_rich"] = 1 if rg_count >= 3 else 0

        return {"hits": hits, "summary": summary}

    def run(self, tis_sites: list[TranslationInitiationSite]) -> list[TranslationInitiationSite]:
        """Scan differential sequences for SLiMs and annotate each site.

        Args:
            tis_sites: Input sites to annotate.

        Returns:
            The same sites with ``isoform_annotations["motifs"]`` populated.
        """
        for site in tis_sites:
            seq = site.diff_region.sequence if site.diff_region else ""
            site.isoform_annotations[self.MODULE_NAME] = self.annotate(seq)

        return tis_sites
