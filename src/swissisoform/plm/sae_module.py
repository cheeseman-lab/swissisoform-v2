"""Module: SAE features — isoform↔canonical feature comparison (SiteModule).

Consumes the per-residue sparse SAE feature codes precomputed by
:func:`swissisoform.plm.sae.precompute_sae` (cached as ``idx``/``val`` Top-K
arrays) for **both** the isoform and its canonical protein, and asks two
questions — the SAE analogue of :class:`~swissisoform.plm.module.PLMVEPModule`'s
isoform-vs-canonical framing:

1. **Categorical changes** — which learned features are *present in one protein
   but absent in the other* (``features_isoform_only`` /
   ``features_canonical_only``). A feature is "present" when it fires (is in the
   residue Top-K) on at least ``min_prevalence`` residues.
2. **Activation changes** — for features present in *both* (shared), the change in
   activation going isoform-vs-canonical (``delta_max = isoform.max −
   canonical.max``, codebase ``isoform − canonical`` convention; positive =
   stronger in the isoform). The shared deltas reuse
   :func:`swissisoform.compare.comparator._scalar_deltas`.

This is a whole-protein comparison (no positional alignment needed — features are
compared as protein-level activation profiles). Feature *indices* are emitted;
human-readable labels come from the ESM-Atlas dictionary, which describes the
6B-layer60 SAE — so for the default 6B SAE the labels are correct, and only a
placeholder for other sizes (``model_size``). Surfacing is descriptive: this is
not a scored E/F criterion.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from swissisoform.compare.comparator import _scalar_deltas
from swissisoform.config import PipelineConfig
from swissisoform.models import TranslationInitiationSite
from swissisoform.plm.atlas import atlas_provenance, load_atlas
from swissisoform.plm.embed import DEFAULT_MODEL_SIZE, protein_hash
from swissisoform.plm.sae import DEFAULT_SAE_CACHE_DIR, load_sae_cache

logger = logging.getLogger(__name__)

# A feature counts as "present" in a protein when it fires on at least this many
# residues (i.e. is in the per-residue Top-K). 1 = fires anywhere.
DEFAULT_MIN_PREVALENCE = 1


def _feature_profile(idx: Any, val: Any, min_prevalence: int) -> dict[int, dict[str, float]]:
    """Reduce a protein's ``(L, K)`` sparse features to a per-feature profile.

    Returns ``{feature_index: {"max", "mean", "prevalence"}}`` aggregated over
    every residue where the feature fires, keeping only features present on at
    least ``min_prevalence`` residues.
    """
    acc: dict[int, dict[str, float]] = defaultdict(
        lambda: {"max": 0.0, "sum": 0.0, "count": 0.0}
    )
    n = int(idx.shape[0])
    for r in range(n):
        for f, v in zip(idx[r], val[r]):
            fi = int(f)
            fv = float(v)
            s = acc[fi]
            if fv > s["max"]:
                s["max"] = fv
            s["sum"] += fv
            s["count"] += 1.0
    return {
        fi: {
            "max": round(s["max"], 4),
            "mean": round(s["sum"] / s["count"], 4),
            "prevalence": int(s["count"]),
        }
        for fi, s in acc.items()
        if s["count"] >= min_prevalence
    }


class SAEFeatureModule:
    """ESM-C SAE isoform↔canonical feature-comparison module (SiteModule)."""

    MODULE_NAME: str = "sae"
    OUTPUT_COLUMNS: list[str] = [
        "sae_status",
        "sae_n_isoform_total",
        "sae_n_canonical_total",
        "sae_n_isoform_only",
        "sae_n_canonical_only",
        "sae_n_shared",
        "sae_mean_abs_delta_shared",
        "sae_top_gained_feature_index",
        "sae_top_gained_feature_label",
        "sae_top_gained_delta_max",
        "sae_top_lost_feature_index",
        "sae_top_lost_feature_label",
        "sae_top_lost_delta_max",
        "sae_features_isoform_only",
        "sae_features_canonical_only",
        "sae_shared_feature_deltas",
    ]
    SCOPE: str = "C"

    def __init__(
        self,
        config: PipelineConfig,
        *,
        cache_dir: Path | str = DEFAULT_SAE_CACHE_DIR,
        atlas: dict[int, dict[str, Any]] | None = None,
        model_size: str = DEFAULT_MODEL_SIZE,
        min_prevalence: int = DEFAULT_MIN_PREVALENCE,
        top_k: int | None = None,
    ) -> None:
        """Initialize the module.

        Args:
            config: Pipeline configuration.
            cache_dir: Directory holding the sparse-feature ``<hash>.npz`` files
                (default the per-size SAE cache, e.g. ``data/cache/sae_esmc/6B``).
            atlas: ``{feature_index: {label, description, ...}}`` term dictionary
                for human-readable labels. Defaults to the local ESM-Atlas cache
                (empty if not fetched → index-only output).
            model_size: ESM-C size of the SAE that produced the features. The
                Atlas describes the 6B-layer60 dictionary, so for ``6b`` the
                labels are correct; for any other size they are a PLACEHOLDER
                (see :func:`~swissisoform.plm.atlas.atlas_provenance`).
            min_prevalence: Residue-count threshold for a feature to count as
                "present" in a protein (default 1 = fires anywhere).
            top_k: If set, truncate the three emitted feature lists to this many
                entries (counts/summary still reflect the full sets). Used when
                wiring into the pipeline so ``all_paired.parquet`` list columns
                stay bounded; the standalone export uses ``None`` (full lists).
        """
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.atlas = load_atlas() if atlas is None else atlas
        self.model_size = model_size
        self.min_prevalence = min_prevalence
        self.top_k = top_k
        self._provenance = atlas_provenance(model_size)

    def _label(self, feature_index: int) -> tuple[str | None, str | None, str | None]:
        """Return ``(label, description, label_source)`` for a feature index."""
        term = self.atlas.get(feature_index)
        if not term:
            return None, None, None
        return term.get("label"), term.get("description"), self._provenance

    def _load_features(self, protein: str) -> dict[str, Any] | None:
        if not protein:
            return None
        return load_sae_cache(protein_hash(protein), self.cache_dir)

    def _unique_entry(self, f: int, prof: dict[str, float]) -> dict[str, Any]:
        """Build a record for a feature present in only one protein."""
        label, description, source = self._label(f)
        return {
            "feature_index": f,
            "label": label,
            "description": description,
            "label_source": source,
            "max": prof["max"],
            "mean": prof["mean"],
            "prevalence": prof["prevalence"],
        }

    def annotate_site(self, site: TranslationInitiationSite) -> dict[str, Any]:
        """Compute the isoform-vs-canonical SAE feature comparison for one TIS."""
        empty: dict[str, Any] = {
            "status": "not_run",
            "n_isoform_total": None,
            "n_canonical_total": None,
            "n_isoform_only": None,
            "n_canonical_only": None,
            "n_shared": None,
            "mean_abs_delta_shared": None,
            "top_gained_feature_index": None,
            "top_gained_feature_label": None,
            "top_gained_delta_max": None,
            "top_lost_feature_index": None,
            "top_lost_feature_label": None,
            "top_lost_delta_max": None,
            "features_isoform_only": [],
            "features_canonical_only": [],
            "shared_feature_deltas": [],
        }

        iso = self._load_features(site.isoform_protein)
        can = self._load_features(site.canonical_protein)
        if (
            iso is None or can is None
            or iso.get("idx") is None or can.get("idx") is None
        ):
            empty["status"] = "no_cache"
            return empty

        iso_prof = _feature_profile(iso["idx"], iso["val"], self.min_prevalence)
        can_prof = _feature_profile(can["idx"], can["val"], self.min_prevalence)
        iso_keys, can_keys = set(iso_prof), set(can_prof)

        # ── 1. Categorical: features present in only one protein ─────────────
        features_isoform_only = sorted(
            (self._unique_entry(f, iso_prof[f]) for f in iso_keys - can_keys),
            key=lambda d: d["max"],
            reverse=True,
        )
        features_canonical_only = sorted(
            (self._unique_entry(f, can_prof[f]) for f in can_keys - iso_keys),
            key=lambda d: d["max"],
            reverse=True,
        )

        # ── 2. Activation deltas for shared features (isoform − canonical) ───
        shared = iso_keys & can_keys
        # Reuse the comparator's isoform−canonical delta over the per-feature
        # peak-activation profiles (keys must be str for _scalar_deltas).
        delta_max = _scalar_deltas(
            {str(f): can_prof[f]["max"] for f in shared},
            {str(f): iso_prof[f]["max"] for f in shared},
        )
        delta_mean = _scalar_deltas(
            {str(f): can_prof[f]["mean"] for f in shared},
            {str(f): iso_prof[f]["mean"] for f in shared},
        )
        shared_feature_deltas: list[dict[str, Any]] = []
        for f in shared:
            label, description, source = self._label(f)
            shared_feature_deltas.append(
                {
                    "feature_index": f,
                    "label": label,
                    "description": description,
                    "label_source": source,
                    "isoform_max": iso_prof[f]["max"],
                    "canonical_max": can_prof[f]["max"],
                    "delta_max": round(delta_max[f"{f}_delta"], 4),
                    "isoform_mean": iso_prof[f]["mean"],
                    "canonical_mean": can_prof[f]["mean"],
                    "delta_mean": round(delta_mean[f"{f}_delta"], 4),
                    "isoform_prevalence": iso_prof[f]["prevalence"],
                    "canonical_prevalence": can_prof[f]["prevalence"],
                }
            )
        shared_feature_deltas.sort(key=lambda d: abs(d["delta_max"]), reverse=True)

        gained = max(shared_feature_deltas, key=lambda d: d["delta_max"], default=None)
        lost = min(shared_feature_deltas, key=lambda d: d["delta_max"], default=None)
        mean_abs_delta = (
            round(
                sum(abs(d["delta_max"]) for d in shared_feature_deltas)
                / len(shared_feature_deltas),
                4,
            )
            if shared_feature_deltas
            else None
        )

        return {
            "status": "ok",
            "n_isoform_total": len(features_isoform_only) + len(shared),
            "n_canonical_total": len(features_canonical_only) + len(shared),
            "n_isoform_only": len(features_isoform_only),
            "n_canonical_only": len(features_canonical_only),
            "n_shared": len(shared),
            "mean_abs_delta_shared": mean_abs_delta,
            "top_gained_feature_index": gained["feature_index"] if gained else None,
            "top_gained_feature_label": gained["label"] if gained else None,
            "top_gained_delta_max": gained["delta_max"] if gained else None,
            "top_lost_feature_index": lost["feature_index"] if lost else None,
            "top_lost_feature_label": lost["label"] if lost else None,
            "top_lost_delta_max": lost["delta_max"] if lost else None,
            # Counts above reflect the full sets; lists may be truncated for the
            # pipeline (top_k) — the full inventory lives in the export.
            "features_isoform_only": features_isoform_only[: self.top_k] if self.top_k
            else features_isoform_only,
            "features_canonical_only": features_canonical_only[: self.top_k] if self.top_k
            else features_canonical_only,
            "shared_feature_deltas": shared_feature_deltas[: self.top_k] if self.top_k
            else shared_feature_deltas,
        }

    def run(
        self, tis_sites: list[TranslationInitiationSite]
    ) -> list[TranslationInitiationSite]:
        """Attach SAE isoform↔canonical comparison annotations to each TIS."""
        for site in tis_sites:
            site.isoform_annotations[self.MODULE_NAME] = self.annotate_site(site)
        return tis_sites
