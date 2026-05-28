"""Tests for VariantEffectModule — per-variant ESM-2 ΔLLR + AlphaMissense."""

from __future__ import annotations

import numpy as np
import pytest

from swissisoform.config import PipelineConfig
from swissisoform.models import DifferentialRegion, ORFType, TranslationInitiationSite
from swissisoform.modules.varianteffect import VariantEffectModule
from swissisoform.plm.embed import AA_TO_COL, STANDARD_AA, protein_hash

# Canonical protein: index 3 is 'A', index 5 is 'I'.
CANON = "MKTAYIAKQR*"


def _seed_aa_logprobs(cache_dir, seq, overrides):
    """Write a cache .npz with an aa_logprobs matrix; overrides set (pos, aa)->logprob."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    L = len(seq.rstrip("*"))
    mat = np.full((L, len(STANDARD_AA)), -1.0, dtype=np.float32)
    for (pos, aa), val in overrides.items():
        mat[pos, AA_TO_COL[aa]] = val
    h = protein_hash(seq)
    np.savez_compressed(cache_dir / f"{h}.npz", llr=mat.max(axis=1), aa_logprobs=mat)
    return h


def _tis(**kw):
    base = dict(
        tis_id="chr1:100:+:ATG",
        gene_name="GENE",
        transcript_id="ENST_TEST.1",
        chrom="chr1",
        position=100,
        strand="+",
        start_codon="ATG",
        orf_type=ORFType.TRUNCATED,
        gene_id="ENSG_TEST",
        aa_len=len(CANON) - 1,
        canonical_protein=CANON,
        isoform_protein=CANON,
        diff_region=DifferentialRegion(sequence=""),
    )
    base.update(kw)
    return TranslationInitiationSite(**base)


class _StubAM:
    """Minimal AlphaMissenseLookup stand-in keyed by (chrom,pos,ref,alt)."""

    def __init__(self, table):
        self._t = table

    @property
    def available(self):
        return True

    def lookup(self, chrom, pos, ref, alt, *, transcript_id=None):
        return self._t.get((chrom, pos, ref, alt))


def _vi_hit(**kw):
    h = dict(
        source="gnomAD",
        variant_id="v",
        chrom="chr1",
        genomic_pos=100,
        ref="C",
        alt="T",
        consequence="missense_variant",
        protein_pos=3,
        aa_ref="A",
        aa_alt="V",
        clinical_significance=None,
        in_isoform_unique=True,
        in_isoform_shared=False,
    )
    h.update(kw)
    return h


def _with_intersection(site, hits):
    site.isoform_annotations["variant_intersection"] = {"summary": {"status": "ok"}, "hits": hits}
    return site


class TestDeltaLLR:
    def test_missense_delta_llr(self, tmp_path):
        # wt 'A' logprob -1.0, alt 'V' -9.0 → ΔLLR -8.0 (≤ -7.5 → damaging).
        _seed_aa_logprobs(tmp_path, CANON, {(3, "A"): -1.0, (3, "V"): -9.0})
        site = _with_intersection(_tis(), [_vi_hit()])
        mod = VariantEffectModule(PipelineConfig(), plm_cache_dir=tmp_path)
        out = mod.annotate_site(site)
        hit = out["hits"][0]
        assert hit["plm_status"] == "ok"
        assert hit["plm_llr_wt"] == pytest.approx(-1.0)
        assert hit["plm_llr_alt"] == pytest.approx(-9.0)
        assert hit["plm_delta_llr"] == pytest.approx(-8.0)
        assert hit["effect_damaging"] is True
        assert out["n_scored_plm"] == 1
        assert out["n_scorable_in_unique"] == 1
        assert out["n_damaging_in_unique"] == 1
        assert out["mean_delta_llr_unique"] == pytest.approx(-8.0)

    def test_aa_ref_mismatch_skipped(self, tmp_path):
        # Canonical[3] is 'A' but the hit claims aa_ref='K' → mismatch → None.
        _seed_aa_logprobs(tmp_path, CANON, {(3, "K"): -1.0, (3, "V"): -9.0})
        site = _with_intersection(_tis(), [_vi_hit(aa_ref="K")])
        out = VariantEffectModule(PipelineConfig(), plm_cache_dir=tmp_path).annotate_site(site)
        hit = out["hits"][0]
        assert hit["plm_status"] == "aa_ref_mismatch"
        assert hit["plm_delta_llr"] is None
        assert out["n_scored_plm"] == 0

    def test_indel_not_missense(self, tmp_path):
        _seed_aa_logprobs(tmp_path, CANON, {})
        hit = _vi_hit(ref="CTT", alt="C", aa_ref=None, aa_alt=None, consequence="frameshift")
        site = _with_intersection(_tis(), [hit])
        out = VariantEffectModule(PipelineConfig(), plm_cache_dir=tmp_path).annotate_site(site)
        assert out["hits"][0]["plm_status"] == "not_missense"
        assert out["hits"][0]["plm_delta_llr"] is None

    def test_no_cache_status(self, tmp_path):
        site = _with_intersection(_tis(), [_vi_hit()])
        out = VariantEffectModule(PipelineConfig(), plm_cache_dir=tmp_path).annotate_site(site)
        assert out["hits"][0]["plm_status"] == "no_aa_logprobs"

    def test_non_unique_variant_not_aggregated(self, tmp_path):
        _seed_aa_logprobs(tmp_path, CANON, {(3, "A"): -1.0, (3, "V"): -9.0})
        site = _with_intersection(_tis(), [_vi_hit(in_isoform_unique=False)])
        out = VariantEffectModule(PipelineConfig(), plm_cache_dir=tmp_path).annotate_site(site)
        # Scored, but not counted toward the unique-region aggregates.
        assert out["n_scored_plm"] == 1
        assert out["n_scorable_in_unique"] == 0
        assert out["n_damaging_in_unique"] == 0
        assert out["mean_delta_llr_unique"] is None


class TestAlphaMissense:
    def test_am_pathogenic_counts(self, tmp_path):
        # ΔLLR -0.5 (not damaging by LLR); damaging must come from AlphaMissense.
        _seed_aa_logprobs(tmp_path, CANON, {(3, "A"): -1.0, (3, "V"): -1.5})
        am = _StubAM(
            {("chr1", 100, "C", "T"): {"am_pathogenicity": 0.92, "am_class": "likely_pathogenic"}}
        )
        site = _with_intersection(_tis(), [_vi_hit()])
        mod = VariantEffectModule(PipelineConfig(), plm_cache_dir=tmp_path, alphamissense=am)
        out = mod.annotate_site(site)
        hit = out["hits"][0]
        assert hit["am_pathogenicity"] == pytest.approx(0.92)
        assert hit["am_class"] == "likely_pathogenic"
        assert hit["effect_damaging"] is True  # damaging via AlphaMissense
        assert out["n_scored_am"] == 1
        assert out["n_am_pathogenic_in_unique"] == 1
        assert out["n_damaging_in_unique"] == 1
        assert out["max_am_pathogenicity_unique"] == pytest.approx(0.92)


class TestSiteModuleContract:
    def test_run_does_not_drop_sites(self, synthetic_tis, tmp_path):
        mod = VariantEffectModule(PipelineConfig(), plm_cache_dir=tmp_path)
        out = mod.run(synthetic_tis)
        assert len(out) == len(synthetic_tis)
        for s in out:
            assert "varianteffect" in s.isoform_annotations

    def test_no_intersection_status(self, tmp_path):
        # No variant_intersection and no clinical → empty hits, no_intersection.
        site = _tis()
        out = VariantEffectModule(PipelineConfig(), plm_cache_dir=tmp_path).annotate_site(site)
        assert out["summary"]["status"] == "no_intersection"
        assert out["hits"] == []
