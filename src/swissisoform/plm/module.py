"""Module: PLM VEP — variant-effect scores from ESM-2 masked-marginal LLR.

Consumes per-residue log-likelihood scores (computed offline by
``swissisoform.plm.embed.precompute_plm_esm2``) and emits TIS-level
constraint metrics:

- mean LLR over the isoform-unique region,
- mean LLR over the shared region (canonical-frame body),
- enrichment ratio (unique / shared),
- count of strongly-constrained positions (LLR < threshold) in the unique region.

Why a SiteModule, not a ProteinModule: LLR is context-dependent — running
ESM-2 on ``diff_region.sequence`` alone (Scope-A re-run) gives different
scores than slicing the same positions out of the full-protein forward
pass. So we compute LLR on the full canonical and isoform proteins, then
slice to unique/shared regions per the diff_region coordinates.

Activates F5 (pathogenic variant enrichment) once paired with the clinical
module's variant positions in a follow-up; for now, F5 needs the
constraint-enrichment column emitted here as input.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from swissisoform.config import PipelineConfig
from swissisoform.models import ORFType, TranslationInitiationSite
from swissisoform.plm.embed import DEFAULT_CACHE_DIR, load_cache, protein_hash

logger = logging.getLogger(__name__)

# Empirical threshold: ESM-2 650M log-probs below ~-5 mark strongly-constrained
# positions in human proteomes (top decile of constraint).
DEFAULT_CONSTRAINT_THRESHOLD = -5.0


def _safe_mean(arr: Any) -> float | None:
    if arr is None:
        return None
    if hasattr(arr, "size") and arr.size == 0:
        return None
    if len(arr) == 0:
        return None
    return float(sum(arr) / len(arr))


class PLMVEPModule:
    """ESM-2 masked-marginal variant-effect predictor (SiteModule)."""

    MODULE_NAME: str = "plm_vep"
    OUTPUT_COLUMNS: list[str] = [
        "plm_vep_status",
        "plm_vep_mean_llr_isoform",
        "plm_vep_mean_llr_canonical",
        "plm_vep_mean_llr_unique_region",
        "plm_vep_mean_llr_shared_region",
        "plm_vep_constraint_enrichment",
        "plm_vep_n_constrained_positions_unique",
        "plm_vep_n_constrained_positions_shared",
    ]
    SCOPE: str = "C"

    def __init__(
        self,
        config: PipelineConfig,
        *,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        constraint_threshold: float = DEFAULT_CONSTRAINT_THRESHOLD,
    ) -> None:
        """Initialize the module.

        Args:
            config: Pipeline configuration.
            cache_dir: Directory holding the ``<hash>.npz`` LLR cache files.
            constraint_threshold: LLR threshold below which a position counts as
                "constrained". Default -5.0 (top decile in ESM-2 650M).
        """
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.constraint_threshold = constraint_threshold

    def _load_llr(self, protein: str) -> Any | None:
        if not protein:
            return None
        cached = load_cache(protein_hash(protein), self.cache_dir)
        if cached is None:
            return None
        return cached.get("llr")

    @staticmethod
    def _unique_indices(site: TranslationInitiationSite) -> tuple[list[int], list[int]] | None:
        """Return (unique, shared) isoform-protein 0-based indices for the diff region.

        ``None`` when the diff region has no isoform-coords interpretation
        (truncations live in canonical coords).
        """
        dr = site.diff_region
        if dr is None or dr.isoform_start is None or dr.isoform_end is None:
            return None
        L = len(site.isoform_protein.rstrip("*"))
        if L <= 0:
            return None
        u_lo = max(0, int(dr.isoform_start))
        u_hi = min(L, int(dr.isoform_end))
        unique = list(range(u_lo, u_hi))
        unique_set = set(unique)
        shared = [i for i in range(L) if i not in unique_set]
        return unique, shared

    @staticmethod
    def _canonical_unique_indices(
        site: TranslationInitiationSite,
    ) -> tuple[list[int], list[int]] | None:
        """Return (unique, shared) canonical-protein indices for truncations."""
        dr = site.diff_region
        if dr is None or dr.canonical_start is None or dr.canonical_end is None:
            return None
        L = len(site.canonical_protein.rstrip("*"))
        if L <= 0:
            return None
        u_lo = max(0, int(dr.canonical_start))
        u_hi = min(L, int(dr.canonical_end))
        unique = list(range(u_lo, u_hi))
        unique_set = set(unique)
        shared = [i for i in range(L) if i not in unique_set]
        return unique, shared

    def annotate_site(self, site: TranslationInitiationSite) -> dict[str, Any]:
        """Compute LLR-derived constraint metrics for a single TIS."""
        empty = {
            "status": "not_run",
            "mean_llr_isoform": None,
            "mean_llr_canonical": None,
            "mean_llr_unique_region": None,
            "mean_llr_shared_region": None,
            "constraint_enrichment": None,
            "n_constrained_positions_unique": None,
            "n_constrained_positions_shared": None,
        }

        iso_llr = self._load_llr(site.isoform_protein)
        can_llr = self._load_llr(site.canonical_protein)

        if iso_llr is None and can_llr is None:
            empty["status"] = "no_cache"
            return empty

        out = dict(empty)
        out["status"] = "ok"
        out["mean_llr_isoform"] = _safe_mean(iso_llr) if iso_llr is not None else None
        out["mean_llr_canonical"] = _safe_mean(can_llr) if can_llr is not None else None

        # Pick the protein space matching the diff region.
        unique_idx: list[int] = []
        shared_idx: list[int] = []
        scores: Any | None = None
        space = "none"

        if site.orf_type == ORFType.TRUNCATED and can_llr is not None:
            res = self._canonical_unique_indices(site)
            if res is not None:
                unique_idx, shared_idx = res
                scores = can_llr
                space = "canonical"
        else:
            res = self._unique_indices(site)
            if res is not None and iso_llr is not None:
                unique_idx, shared_idx = res
                scores = iso_llr
                space = "isoform"

        if scores is None or len(unique_idx) == 0:
            out["status"] = "no_diff_region"
            return out

        # Guard length mismatch (shouldn't happen if cache built from same
        # canonical/isoform proteins, but be defensive).
        if len(scores) < max(unique_idx + shared_idx + [0]) + 1:
            logger.warning(
                "plm_vep: cached LLR length %d shorter than %s protein indices for %s",
                len(scores),
                space,
                site.tis_id,
            )
            out["status"] = "length_mismatch"
            return out

        unique_vals = [float(scores[i]) for i in unique_idx]
        shared_vals = [float(scores[i]) for i in shared_idx]

        unique_mean = (sum(unique_vals) / len(unique_vals)) if unique_vals else None
        shared_mean = (sum(shared_vals) / len(shared_vals)) if shared_vals else None
        out["mean_llr_unique_region"] = unique_mean
        out["mean_llr_shared_region"] = shared_mean

        if unique_mean is not None and shared_mean is not None and shared_mean != 0:
            out["constraint_enrichment"] = unique_mean / shared_mean
        elif unique_mean is not None and shared_mean == 0:
            out["constraint_enrichment"] = math.inf if unique_mean != 0 else 0.0

        thr = self.constraint_threshold
        out["n_constrained_positions_unique"] = sum(1 for v in unique_vals if v < thr)
        out["n_constrained_positions_shared"] = sum(1 for v in shared_vals if v < thr)
        return out

    def run(self, tis_sites: list[TranslationInitiationSite]) -> list[TranslationInitiationSite]:
        """Attach LLR-derived constraint annotations to each TIS."""
        for site in tis_sites:
            site.isoform_annotations[self.MODULE_NAME] = self.annotate_site(site)
        return tis_sites
