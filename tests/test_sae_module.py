"""Tests for the cross-protein (isoform↔canonical) SAE feature comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from swissisoform.models import DifferentialRegion, ORFType, TranslationInitiationSite
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


def _dr_site(
    iso: str,
    can: str,
    orf_type: ORFType,
    dr: DifferentialRegion | None,
) -> TranslationInitiationSite:
    return TranslationInitiationSite(
        tis_id="t1", gene_name="G", transcript_id="x", chrom="1", position=1,
        strand="+", start_codon="ATG", orf_type=orf_type,
        isoform_protein=iso, canonical_protein=can, diff_region=dr,
    )


class TestUniqueRegionTopFeatures:
    """Top features ranked over only the isoform-unique region."""

    def test_extension_uses_isoform_space(self, tmp_path):
        # Extension: isoform longer; unique region = isoform residues [0, 2).
        iso_seq, can_seq = "MAAAA", "MAAA"
        _seed(tmp_path, iso_seq, np.array(
            [[10, 11], [10, 12], [20, 21], [20, 22], [23, 24]]),
            np.array([[0.9, 0.5], [0.8, 0.4], [0.7, 0.3], [0.6, 0.2], [0.1, 0.1]]))
        _seed(tmp_path, can_seq, np.array(
            [[20, 21], [20, 22], [23, 24], [25, 26]]),
            np.array([[0.7, 0.3], [0.6, 0.2], [0.1, 0.1], [0.2, 0.2]]))
        dr = DifferentialRegion(isoform_start=0, isoform_end=2, sequence="MA")

        ann = _module(tmp_path).annotate_site(
            _dr_site(iso_seq, can_seq, ORFType.EXTENDED, dr))

        assert ann["status"] == "ok"
        assert ann["unique_region_space"] == "isoform"
        # Only features firing on residues 0,1 (the extension): {10, 11, 12}.
        idxs = [e["feature_index"] for e in ann["unique_region_top_features"]]
        assert set(idxs) == {10, 11, 12}
        assert idxs[0] == 10  # ranked by peak activation (0.9 is highest)
        assert ann["n_unique_region_features"] == 3

    def test_truncation_uses_canonical_space(self, tmp_path):
        # Truncation: isoform shorter; unique region = canonical residues [0, 2).
        iso_seq, can_seq = "MAA", "MAAAA"
        _seed(tmp_path, iso_seq, np.array(
            [[40, 41], [42, 43], [44, 45]]),
            np.array([[0.5, 0.4], [0.3, 0.2], [0.1, 0.1]]))
        _seed(tmp_path, can_seq, np.array(
            [[30, 31], [30, 32], [40, 41], [42, 43], [44, 45]]),
            np.array([[0.9, 0.5], [0.8, 0.4], [0.5, 0.4], [0.3, 0.2], [0.1, 0.1]]))
        dr = DifferentialRegion(canonical_start=0, canonical_end=2, sequence="MA")

        ann = _module(tmp_path).annotate_site(
            _dr_site(iso_seq, can_seq, ORFType.TRUNCATED, dr))

        assert ann["unique_region_space"] == "canonical"
        idxs = {e["feature_index"] for e in ann["unique_region_top_features"]}
        assert idxs == {30, 31, 32}  # features over canonical residues 0,1
        assert ann["n_unique_region_features"] == 3

    def test_no_diff_region_is_empty(self, tmp_path):
        iso_seq, can_seq = "MAAA", "MAAA"
        _seed(tmp_path, iso_seq, np.array([[1, 2]]), np.array([[0.9, 0.4]]))
        _seed(tmp_path, can_seq, np.array([[1, 2]]), np.array([[0.9, 0.4]]))

        ann = _module(tmp_path).annotate_site(
            _dr_site(iso_seq, can_seq, ORFType.ANNOTATED, None))

        assert ann["unique_region_space"] == "none"
        assert ann["n_unique_region_features"] == 0
        assert ann["unique_region_top_features"] == []

    def test_no_cache_has_empty_unique_region(self, tmp_path):
        _seed(tmp_path, "MAAAA", np.array([[1, 2]]), np.array([[0.9, 0.4]]))
        dr = DifferentialRegion(isoform_start=0, isoform_end=2, sequence="MA")
        ann = _module(tmp_path).annotate_site(
            _dr_site("MAAAA", "MAAA", ORFType.EXTENDED, dr))
        assert ann["status"] == "no_cache"
        assert ann["unique_region_space"] == "none"
        assert ann["unique_region_top_features"] == []

    def test_unique_top_n_caps_list(self, tmp_path):
        # 4 distinct features over the unique region, cap at 2.
        iso_seq, can_seq = "MAA", "MA"
        _seed(tmp_path, iso_seq, np.array(
            [[1, 2], [3, 4], [9, 9]]),
            np.array([[0.9, 0.8], [0.7, 0.6], [0.1, 0.1]]))
        _seed(tmp_path, can_seq, np.array([[9, 9], [9, 9]]),
              np.array([[0.1, 0.1], [0.1, 0.1]]))
        dr = DifferentialRegion(isoform_start=0, isoform_end=2, sequence="MA")
        mod = SAEFeatureModule(
            None, cache_dir=tmp_path, atlas={}, model_size="6b", unique_top_n=2
        )
        ann = mod.annotate_site(_dr_site(iso_seq, can_seq, ORFType.EXTENDED, dr))
        assert ann["n_unique_region_features"] == 4  # full count, not capped
        assert len(ann["unique_region_top_features"]) == 2  # list capped
        # Top 2 by peak activation: feature 1 (0.9) and feature 2 (0.8).
        assert [e["feature_index"] for e in ann["unique_region_top_features"]] == [1, 2]
