"""Module: SAE features — interpretable sparse-feature differential (SiteModule).

Consumes the per-residue sparse SAE feature codes precomputed by
:func:`swissisoform.plm.sae.precompute_sae` (cached as ``idx``/``val`` Top-K
arrays) and asks: *which learned features are differentially active in the
isoform's unique region vs its shared core?*

Mirrors :class:`~swissisoform.plm.module.PLMVEPModule`'s region split: for
truncations the diff region is the lost canonical sequence (canonical protein
space + canonical features); for extensions / uORFs / altORFs it is the new
isoform sequence (isoform space + isoform features). Per feature it aggregates
over each region — max activation, mean activation, prevalence (how many region
residues it fires on) — and ranks features by how much more they cover the
unique region than the shared core (``delta_density``).

Feature *indices* are emitted; human-readable labels are attached separately
(Step 4, ESM-Atlas — a placeholder dictionary for the 600M SAE). Surfacing is
descriptive: this is not a scored E/F criterion.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from swissisoform.config import PipelineConfig
from swissisoform.models import ORFType, TranslationInitiationSite
from swissisoform.plm.atlas import ATLAS_PROVENANCE, load_atlas
from swissisoform.plm.embed import protein_hash
from swissisoform.plm.sae import DEFAULT_SAE_CACHE_DIR, load_sae_cache

logger = logging.getLogger(__name__)

# Number of top differentially-active features to emit per isoform.
DEFAULT_TOP_K = 10
# Density pseudocount so the enrichment ratio is finite for pure-unique features
# (shared_density == 0). Ranking uses delta_density and is unaffected by it.
_EPS = 1e-3


def _region_indices(
    site: TranslationInitiationSite,
) -> tuple[str, list[int], list[int]] | None:
    """Return ``(space, unique_idx, shared_idx)`` for the diff region.

    ``space`` is ``"canonical"`` for truncations (diff region = lost canonical
    sequence) and ``"isoform"`` otherwise. ``None`` when the diff region has no
    coordinates in the relevant protein space.
    """
    dr = site.diff_region
    if dr is None:
        return None

    if site.orf_type == ORFType.TRUNCATED:
        if dr.canonical_start is None or dr.canonical_end is None:
            return None
        L = len(site.canonical_protein.rstrip("*"))
        lo, hi = int(dr.canonical_start), int(dr.canonical_end)
        space = "canonical"
    else:
        if dr.isoform_start is None or dr.isoform_end is None:
            return None
        L = len(site.isoform_protein.rstrip("*"))
        lo, hi = int(dr.isoform_start), int(dr.isoform_end)
        space = "isoform"

    if L <= 0:
        return None
    unique = list(range(max(0, lo), min(L, hi)))
    unique_set = set(unique)
    shared = [i for i in range(L) if i not in unique_set]
    return space, unique, shared


def _region_feature_stats(
    idx: Any, val: Any, residues: list[int]
) -> dict[int, dict[str, float]]:
    """Aggregate sparse features over a set of residue rows.

    Returns ``{feature_index: {"max", "sum", "count"}}`` accumulated across the
    given residues' Top-K activations.
    """
    stats: dict[int, dict[str, float]] = defaultdict(
        lambda: {"max": 0.0, "sum": 0.0, "count": 0.0}
    )
    for r in residues:
        for f, v in zip(idx[r], val[r]):
            fi = int(f)
            fv = float(v)
            s = stats[fi]
            if fv > s["max"]:
                s["max"] = fv
            s["sum"] += fv
            s["count"] += 1.0
    return stats


class SAEFeatureModule:
    """ESM-C SAE differential-feature module (SiteModule)."""

    MODULE_NAME: str = "sae"
    OUTPUT_COLUMNS: list[str] = [
        "sae_status",
        "sae_region_space",
        "sae_n_unique_residues",
        "sae_n_shared_residues",
        "sae_n_active_features_unique",
        "sae_n_features_enriched",
        "sae_top_feature_index",
        "sae_top_feature_label",
        "sae_top_feature_delta_density",
        "sae_top_features",
    ]
    SCOPE: str = "C"

    def __init__(
        self,
        config: PipelineConfig,
        *,
        cache_dir: Path | str = DEFAULT_SAE_CACHE_DIR,
        top_k: int = DEFAULT_TOP_K,
        atlas: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the module.

        Args:
            config: Pipeline configuration.
            cache_dir: Directory holding the sparse-feature ``<hash>.npz`` files.
            top_k: How many top differentially-active features to emit per TIS.
            atlas: ``{feature_index: {label, category, ...}}`` term dictionary
                used to attach human-readable labels. Defaults to the local
                ESM-Atlas cache (empty if not fetched → index-only output).
                NOTE: the Atlas describes the 6B-layer60 dictionary — labels are
                a PLACEHOLDER for the 600M features (see ``ATLAS_PROVENANCE``).
        """
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.top_k = top_k
        self.atlas = load_atlas() if atlas is None else atlas

    def _label(self, feature_index: int) -> tuple[str | None, str | None, str | None]:
        """Return ``(label, description, label_source)`` for a feature index."""
        term = self.atlas.get(feature_index)
        if not term:
            return None, None, None
        return term.get("label"), term.get("description"), ATLAS_PROVENANCE

    def _load_features(self, protein: str) -> dict[str, Any] | None:
        if not protein:
            return None
        return load_sae_cache(protein_hash(protein), self.cache_dir)

    def annotate_site(self, site: TranslationInitiationSite) -> dict[str, Any]:
        """Compute the unique-vs-shared SAE feature differential for one TIS."""
        empty: dict[str, Any] = {
            "status": "not_run",
            "region_space": None,
            "n_unique_residues": None,
            "n_shared_residues": None,
            "n_active_features_unique": None,
            "n_features_enriched": None,
            "top_feature_index": None,
            "top_feature_label": None,
            "top_feature_delta_density": None,
            "top_features": [],
        }

        region = _region_indices(site)
        if region is None:
            empty["status"] = "no_diff_region"
            return empty
        space, unique_idx, shared_idx = region
        if not unique_idx:
            empty["status"] = "no_diff_region"
            return empty

        protein = site.canonical_protein if space == "canonical" else site.isoform_protein
        feats = self._load_features(protein)
        if feats is None or feats.get("idx") is None:
            empty["status"] = "no_cache"
            return empty

        idx, val = feats["idx"], feats["val"]
        L = int(idx.shape[0])
        # Defensive: cached features must cover the indexed residues.
        if L < (max(unique_idx + shared_idx) + 1 if (unique_idx or shared_idx) else 0):
            logger.warning(
                "sae: cached features length %d shorter than %s protein indices for %s",
                L, space, site.tis_id,
            )
            empty["status"] = "length_mismatch"
            return empty

        n_unique = len(unique_idx)
        n_shared = len(shared_idx)
        u_stats = _region_feature_stats(idx, val, unique_idx)
        s_stats = _region_feature_stats(idx, val, shared_idx)

        rows: list[dict[str, Any]] = []
        for f, u in u_stats.items():
            u_count = u["count"]
            if u_count == 0:
                continue
            u_density = u_count / n_unique if n_unique else 0.0
            s = s_stats.get(f)
            s_density = (s["count"] / n_shared) if (s and n_shared) else 0.0
            label, description, label_source = self._label(int(f))
            rows.append(
                {
                    "feature_index": int(f),
                    "label": label,
                    "description": description,
                    "label_source": label_source,
                    "unique_max": round(u["max"], 4),
                    "unique_mean": round(u["sum"] / u_count, 4),
                    "unique_prevalence": int(u_count),
                    "unique_density": round(u_density, 4),
                    "shared_density": round(s_density, 4),
                    "delta_density": round(u_density - s_density, 4),
                    "enrichment": round((u_density + _EPS) / (s_density + _EPS), 3),
                }
            )

        # Rank by how much more the feature covers the unique region than the
        # shared core; tie-break by peak activation in the unique region.
        rows.sort(key=lambda d: (d["delta_density"], d["unique_max"]), reverse=True)
        top = rows[: self.top_k]

        return {
            "status": "ok",
            "region_space": space,
            "n_unique_residues": n_unique,
            "n_shared_residues": n_shared,
            "n_active_features_unique": len(rows),
            "n_features_enriched": sum(1 for r in rows if r["delta_density"] > 0),
            "top_feature_index": top[0]["feature_index"] if top else None,
            "top_feature_label": top[0]["label"] if top else None,
            "top_feature_delta_density": top[0]["delta_density"] if top else None,
            "top_features": top,
        }

    def run(
        self, tis_sites: list[TranslationInitiationSite]
    ) -> list[TranslationInitiationSite]:
        """Attach SAE differential-feature annotations to each TIS."""
        for site in tis_sites:
            site.isoform_annotations[self.MODULE_NAME] = self.annotate_site(site)
        return tis_sites
