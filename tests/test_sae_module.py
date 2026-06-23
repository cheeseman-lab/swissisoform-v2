"""Tests for the cross-protein (isoform↔canonical) SAE feature comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from swissisoform.models import ORFType, TranslationInitiationSite
from swissisoform.plm.embed import protein_hash
from swissisoform.plm.sae import _save_sae_cache
from swissisoform.plm.sae_module import SAEFeatureModule


def _seed(cache_dir: Path, seq: str, idx: np.ndarray, val: np.ndarray) -> None:
    """Write a synthetic sparse-feature cache entry for *seq*."""
    _save_sae_cache(protein_hash(seq), cache_dir, idx=idx, val=val, recon_loss=0.1)


def _site(iso: str, can: str) -> TranslationInitiationSite:
    return TranslationInitiationSite(
        tis_id="t1", gene_name="G", transcript_id="x", chrom="1", position=1,
        strand="+", start_codon="ATG", orf_type=ORFType.EXTENDED,
        isoform_protein=iso, canonical_protein=can,
    )


def _module(cache_dir: Path) -> SAEFeatureModule:
    # config is stored but unused by annotate_site; atlas={} → index-only labels.
    return SAEFeatureModule(None, cache_dir=cache_dir, atlas={}, model_size="6b")


class TestCrossProteinComparison:
    """Whole-protein isoform-vs-canonical feature diff."""

    def test_unique_and_shared_sets(self, tmp_path):
        iso_seq, can_seq = "MAAA", "MCCC"
        # isoform fires features {1,2,3}; canonical fires {2,3,4}
        _seed(tmp_path, iso_seq,
              np.array([[1, 2], [1, 3]]), np.array([[0.9, 0.4], [0.8, 0.6]]))
        _seed(tmp_path, can_seq,
              np.array([[2, 4], [3, 4]]), np.array([[0.2, 0.5], [0.3, 0.7]]))

        ann = _module(tmp_path).annotate_site(_site(iso_seq, can_seq))

        assert ann["status"] == "ok"
        assert {e["feature_index"] for e in ann["features_isoform_only"]} == {1}
        assert {e["feature_index"] for e in ann["features_canonical_only"]} == {4}
        assert {e["feature_index"] for e in ann["shared_feature_deltas"]} == {2, 3}
        assert ann["n_isoform_only"] == 1
        assert ann["n_canonical_only"] == 1
        assert ann["n_shared"] == 2

    def test_shared_activation_delta_sign_and_value(self, tmp_path):
        iso_seq, can_seq = "MAAA", "MCCC"
        _seed(tmp_path, iso_seq,
              np.array([[1, 2], [1, 3]]), np.array([[0.9, 0.4], [0.8, 0.6]]))
        _seed(tmp_path, can_seq,
              np.array([[2, 4], [3, 4]]), np.array([[0.2, 0.5], [0.3, 0.7]]))

        ann = _module(tmp_path).annotate_site(_site(iso_seq, can_seq))
        d2 = next(e for e in ann["shared_feature_deltas"] if e["feature_index"] == 2)
        # feature 2: isoform peak 0.4, canonical peak 0.2 → delta = +0.2 (stronger in
        # isoform). Activations round-trip through fp16 storage, so allow tolerance.
        assert abs(d2["isoform_max"] - 0.4) < 1e-2
        assert abs(d2["canonical_max"] - 0.2) < 1e-2
        assert abs(d2["delta_max"] - 0.2) < 1e-2
        # feature 3: 0.6 vs 0.3 → +0.3 ; it is the top gained (largest positive delta)
        assert ann["top_gained_feature_index"] == 3
        assert abs(ann["top_gained_delta_max"] - 0.3) < 1e-2

    def test_missing_cache_is_no_cache(self, tmp_path):
        # Only the isoform is seeded; canonical cache absent.
        _seed(tmp_path, "MAAA", np.array([[1, 2]]), np.array([[0.9, 0.4]]))
        ann = _module(tmp_path).annotate_site(_site("MAAA", "MCCC"))
        assert ann["status"] == "no_cache"
        assert ann["features_isoform_only"] == []

    def test_min_prevalence_filters_features(self, tmp_path):
        iso_seq, can_seq = "MAAA", "MCCC"
        # feature 2 fires once in isoform; with min_prevalence=2 it should drop out.
        _seed(tmp_path, iso_seq,
              np.array([[1, 2], [1, 3]]), np.array([[0.9, 0.4], [0.8, 0.6]]))
        _seed(tmp_path, can_seq,
              np.array([[1, 4], [1, 4]]), np.array([[0.2, 0.5], [0.3, 0.7]]))
        mod = SAEFeatureModule(
            None, cache_dir=tmp_path, atlas={}, model_size="6b", min_prevalence=2
        )
        ann = mod.annotate_site(_site(iso_seq, can_seq))
        present_iso = {e["feature_index"] for e in ann["features_isoform_only"]}
        present_shared = {e["feature_index"] for e in ann["shared_feature_deltas"]}
        # feature 1 fires twice in both → shared; singletons (2,3,4) filtered out.
        assert present_shared == {1}
        assert 2 not in present_iso  # prevalence 1 < 2
