"""Module: Structure — predicted-structure consumer.

SiteModule that looks up cached Boltz-2 / Chai-1 fold results for both
``canonical_protein`` and ``isoform_protein``, slices diff-region pLDDT,
and (when ``tmtools`` + ``biotite`` are available) computes TM-score,
shared-region RMSD, and extension-to-canonical-body contact count.

Activates F1 (structured extension) when ``plddt_diffregion_mean`` exceeds
``ScoringConfig.f1_plddt_threshold`` (default 0.70 on Boltz-2's 0-1 scale;
AlphaFold-style backends emit 0-100 and would use 70.0). Pure lookup + numpy
at pipeline runtime; GPU folding happens out-of-band via
``scripts/slurm/run_fold.sbatch``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from swissisoform.config import PipelineConfig
from swissisoform.models import TranslationInitiationSite
from swissisoform.structure.compare import compare_confidence, compare_structures
from swissisoform.structure.fold import (
    DEFAULT_BACKEND,
    DEFAULT_CACHE_DIR,
    load_cache,
    protein_hash,
)

logger = logging.getLogger(__name__)


class StructureModule:
    """Predicted-structure annotation module (SiteModule)."""

    MODULE_NAME: str = "structure"
    OUTPUT_COLUMNS: list[str] = [
        "structure_status",
        "structure_backend",
        "structure_plddt_canonical_mean",
        "structure_plddt_isoform_mean",
        "structure_plddt_diffregion_mean",
        "structure_plddt_diffregion_std",
        "structure_plddt_delta_shared",
        "structure_tm_score",
        "structure_rmsd_global",
        "structure_extension_contacts",
    ]
    SCOPE: str = "C"

    def __init__(
        self,
        config: PipelineConfig,
        *,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        backend: str = DEFAULT_BACKEND,
    ) -> None:
        """Initialize the module.

        Args:
            config: Pipeline configuration.
            cache_dir: Root cache directory (per-backend subdirs underneath).
            backend: Which backend's cache to read (``"esmfold2"`` default,
                ``"boltz"`` or ``"chai"``).
        """
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.backend = backend

    def _load(self, protein: str) -> dict[str, Any] | None:
        if not protein:
            return None
        return load_cache(protein_hash(protein), self.cache_dir, self.backend)

    @staticmethod
    def _empty(reason: str = "no_cache") -> dict[str, Any]:
        return {
            "status": reason,
            "backend": None,
            "plddt_canonical_mean": None,
            "plddt_isoform_mean": None,
            "plddt_diffregion_mean": None,
            "plddt_diffregion_std": None,
            "plddt_delta_shared": None,
            "tm_score": None,
            "rmsd_global": None,
            "extension_contacts": None,
        }

    def annotate_site(self, site: TranslationInitiationSite) -> dict[str, Any]:
        """Compute structure-derived comparison metrics for a single TIS."""
        can = self._load(site.canonical_protein)
        iso = self._load(site.isoform_protein)
        if can is None and iso is None:
            return self._empty("no_cache")

        can_metrics = (can or {}).get("metrics") or {}
        iso_metrics = (iso or {}).get("metrics") or {}
        can_status = can_metrics.get("status")
        iso_status = iso_metrics.get("status")

        # Surface the worst non-ok status if either side failed; otherwise "ok".
        # Order: too_long > failed > uniform_plddt > ok > partial.
        # uniform_plddt = backend only produced complex_plddt (uniform fill),
        # not per-residue. Downstream criteria (F1) should opt out rather
        # than score against the planted scalar.
        if can_status == "too_long" or iso_status == "too_long":
            status = "too_long"
        elif can_status in ("failed", "oom") or iso_status in ("failed", "oom"):
            status = "failed"
        elif can_status == "uniform_plddt" or iso_status == "uniform_plddt":
            status = "uniform_plddt"
        elif can_metrics and iso_metrics:
            status = "ok"
        else:
            status = "partial"

        out = self._empty(status)
        out["backend"] = self.backend

        can_plddt = (can or {}).get("confidence", {}).get("plddt") if can else None
        iso_plddt = (iso or {}).get("confidence", {}).get("plddt") if iso else None

        dr = site.diff_region
        conf_metrics = compare_confidence(
            can_plddt,
            iso_plddt,
            diff_isoform_start=getattr(dr, "isoform_start", None) if dr else None,
            diff_isoform_end=getattr(dr, "isoform_end", None) if dr else None,
            diff_canonical_start=getattr(dr, "canonical_start", None) if dr else None,
            diff_canonical_end=getattr(dr, "canonical_end", None) if dr else None,
            orf_type=site.orf_type.value if site.orf_type else None,
        )
        for k, v in conf_metrics.items():
            out[k] = v

        # Structural metrics (TM/RMSD/contacts) — lazy, optional.
        struct_metrics = compare_structures(
            (can or {}).get("cif_path"),
            (iso or {}).get("cif_path"),
            diff_isoform_start=getattr(dr, "isoform_start", None) if dr else None,
            diff_isoform_end=getattr(dr, "isoform_end", None) if dr else None,
        )
        for k, v in struct_metrics.items():
            out[k] = v

        return out

    def run(self, tis_sites: list[TranslationInitiationSite]) -> list[TranslationInitiationSite]:
        """Attach structure annotations to each TIS."""
        for site in tis_sites:
            site.isoform_annotations[self.MODULE_NAME] = self.annotate_site(site)
        return tis_sites
