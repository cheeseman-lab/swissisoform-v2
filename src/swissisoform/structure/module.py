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
from swissisoform.structure.compare import (
    compare_confidence,
    compare_structures,
    load_pae,
    pae_region_blocks,
)
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
        "structure_plddt_shared_mean_isoform",
        "structure_plddt_shared_mean_canonical",
        "structure_tm_score",
        "structure_rmsd_global",
        "structure_extension_contacts",
        "structure_rmsd_shared",
        "structure_tm_score_shared",
        "structure_shared_region_len",
        "structure_rmsd_shared_status",
        "structure_ptm_isoform",
        "structure_ptm_canonical",
        "structure_pae_diff_vs_diff",
        "structure_pae_body_vs_body",
        "structure_pae_diff_vs_body",
        "structure_pae_status",
        # Cache addresses. The fold cache is keyed by sha1 of the protein
        # sequence, but the sequences are not carried downstream (the parquet
        # holds only lengths), so anything reading the cache after the pipeline
        # — the P-category LLM readers, ad-hoc analysis — has no way back to a
        # cache entry from a tis_id. Emitting the hashes here is that bridge;
        # combined with ``structure_backend`` they reconstruct the full path via
        # ``fold.cache_path``. Populated from the sequence alone, so they are
        # present even when nothing has been folded yet.
        "structure_canonical_hash",
        "structure_isoform_hash",
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
            "plddt_shared_mean_isoform": None,
            "plddt_shared_mean_canonical": None,
            "tm_score": None,
            "rmsd_global": None,
            "extension_contacts": None,
            "rmsd_shared": None,
            "tm_score_shared": None,
            "shared_region_len": None,
            "rmsd_shared_status": reason,
            "ptm_isoform": None,
            "ptm_canonical": None,
            "pae_diff_vs_diff": None,
            "pae_body_vs_body": None,
            "pae_diff_vs_body": None,
            "pae_status": reason,
            "canonical_hash": None,
            "isoform_hash": None,
        }

    @staticmethod
    def _hashes(site: TranslationInitiationSite) -> dict[str, Any]:
        """Fold-cache addresses for this site's two proteins.

        Derived from the sequences alone, so they are emitted whether or not a
        fold exists — a hash with no cache entry is the honest way to say "this
        protein was never folded", which a null hash could not distinguish from
        "no sequence".
        """
        return {
            "canonical_hash": (
                protein_hash(site.canonical_protein) if site.canonical_protein else None
            ),
            "isoform_hash": (
                protein_hash(site.isoform_protein) if site.isoform_protein else None
            ),
        }

    def annotate_site(self, site: TranslationInitiationSite) -> dict[str, Any]:
        """Compute structure-derived comparison metrics for a single TIS."""
        hashes = self._hashes(site)
        can = self._load(site.canonical_protein)
        iso = self._load(site.isoform_protein)
        if can is None and iso is None:
            return {**self._empty("no_cache"), **hashes}

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
        out.update(hashes)
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
            diff_canonical_start=getattr(dr, "canonical_start", None) if dr else None,
            diff_canonical_end=getattr(dr, "canonical_end", None) if dr else None,
            orf_type=site.orf_type.value if site.orf_type else None,
            diff_region_confidence=getattr(dr, "confidence", None) if dr else None,
        )
        for k, v in struct_metrics.items():
            out[k] = v

        # Global pTM — the single best "is this fold trustworthy at all" scalar,
        # already cached in metrics.json; qualifies every pLDDT/RMSD/PAE claim.
        out["ptm_isoform"] = iso_metrics.get("ptm") if iso_metrics else None
        out["ptm_canonical"] = can_metrics.get("ptm") if can_metrics else None

        # PAE region blocks — how confidently the diff region is positioned
        # relative to the rest of the fold. Computed on whichever structure
        # CONTAINS the diff region (isoform for extensions/uORFs, canonical for
        # truncations), mirroring the pLDDT diff-region logic.
        for k, v in self._pae_blocks(site, can, iso).items():
            out[k] = v

        return out

    def _pae_blocks(
        self,
        site: TranslationInitiationSite,
        can: dict[str, Any] | None,
        iso: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Compute PAE block means over the structure carrying the diff region."""
        dr = site.diff_region
        orf = site.orf_type.value if site.orf_type else None
        iso_start = getattr(dr, "isoform_start", None) if dr else None
        can_start = getattr(dr, "canonical_start", None) if dr else None
        truncated = (orf == "truncated") or (iso_start is None and can_start is not None)

        if truncated:
            entry, start, end = (
                can,
                can_start,
                getattr(dr, "canonical_end", None) if dr else None,
            )
        else:
            entry, start, end = (
                iso,
                iso_start,
                getattr(dr, "isoform_end", None) if dr else None,
            )

        pae = load_pae((entry or {}).get("pae_path")) if entry else None
        return pae_region_blocks(pae, start, end)

    def run(self, tis_sites: list[TranslationInitiationSite]) -> list[TranslationInitiationSite]:
        """Attach structure annotations to each TIS."""
        for site in tis_sites:
            site.isoform_annotations[self.MODULE_NAME] = self.annotate_site(site)
        return tis_sites
