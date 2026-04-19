"""Tests for clinical variant fetching."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

from swissisoform.clinical.fetch import VariantFetcher
from swissisoform.config import PipelineConfig
from swissisoform.modules.clinical import ClinicalModule

# ---------------------------------------------------------------------------
# Mock response data
# ---------------------------------------------------------------------------

MOCK_GNOMAD_RESPONSE = {
    "data": {
        "gene": {
            "variants": [
                {
                    "variant_id": "1-12345-A-G",
                    "pos": 12345,
                    "ref": "A",
                    "alt": "G",
                    "consequence": "missense_variant",
                    "hgvsp": "p.Arg42Gly",
                    "hgvsc": "c.124A>G",
                    "exome": {"ac": 5, "an": 50000, "af": 0.0001, "ac_hom": 0},
                    "genome": None,
                },
                {
                    "variant_id": "1-12400-C-T",
                    "pos": 12400,
                    "ref": "C",
                    "alt": "T",
                    "consequence": "synonymous_variant",
                    "hgvsp": "p.Ala60=",
                    "hgvsc": "c.180C>T",
                    "exome": {"ac": 100, "an": 50000, "af": 0.002, "ac_hom": 1},
                    "genome": None,
                },
            ]
        }
    }
}

MOCK_GNOMAD_EMPTY = {"data": {"gene": {"variants": []}}}

MOCK_GNOMAD_ZERO_AF = {
    "data": {
        "gene": {
            "variants": [
                {
                    "variant_id": "1-99999-A-T",
                    "pos": 99999,
                    "ref": "A",
                    "alt": "T",
                    "consequence": "missense_variant",
                    "hgvsp": "p.Met1?",
                    "hgvsc": "c.1A>T",
                    "exome": {"ac": 0, "an": 50000, "af": 0, "ac_hom": 0},
                    "genome": None,
                },
            ]
        }
    }
}

MOCK_GNOMAD_NO_HGVSP = {
    "data": {
        "gene": {
            "variants": [
                {
                    "variant_id": "1-10000-G-A",
                    "pos": 10000,
                    "ref": "G",
                    "alt": "A",
                    "consequence": "5_prime_UTR_variant",
                    "hgvsp": None,
                    "hgvsc": "c.-50G>A",
                    "exome": {"ac": 3, "an": 50000, "af": 0.00006, "ac_hom": 0},
                    "genome": None,
                },
            ]
        }
    }
}

MOCK_CLINVAR_ESEARCH = {
    "esearchresult": {
        "idlist": ["12345", "12346"],
        "count": "2",
    }
}

MOCK_CLINVAR_ESEARCH_EMPTY = {
    "esearchresult": {
        "idlist": [],
        "count": "0",
    }
}

MOCK_CLINVAR_ESUMMARY = {
    "result": {
        "uids": ["12345", "12346"],
        "12345": {
            "uid": "12345",
            "title": "NM_000546.6(TP53):c.215C>G (p.Pro72Arg)",
            "clinical_significance": {"description": "Benign", "review_status": "reviewed"},
            "variation_set": [
                {
                    "variation_loc": [
                        {
                            "assembly_name": "GRCh38",
                            "chr": "17",
                            "start": 7675088,
                            "stop": 7675088,
                        }
                    ]
                }
            ],
            "genes": [{"symbol": "TP53"}],
            "obj_type": "single nucleotide variant",
        },
        "12346": {
            "uid": "12346",
            "title": "NM_000546.6(TP53):c.743G>A (p.Arg248Gln)",
            "clinical_significance": {
                "description": "Pathogenic",
                "review_status": "reviewed",
            },
            "variation_set": [
                {
                    "variation_loc": [
                        {
                            "assembly_name": "GRCh38",
                            "chr": "17",
                            "start": 7674220,
                            "stop": 7674220,
                        }
                    ]
                }
            ],
            "genes": [{"symbol": "TP53"}],
            "obj_type": "single nucleotide variant",
        },
    }
}


def _mock_response(data: dict, status_code: int = 200) -> MagicMock:
    """Build a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.text = json.dumps(data)
    resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# TestVariantFetcher
# ---------------------------------------------------------------------------


class TestVariantFetcher:
    """Tests for the VariantFetcher class."""

    @patch("swissisoform.clinical.fetch.requests.post")
    def test_fetch_gene_gnomad_mock(self, mock_post: MagicMock) -> None:
        """gnomAD returns 2 variants with correct format."""
        mock_post.return_value = _mock_response(MOCK_GNOMAD_RESPONSE)
        fetcher = VariantFetcher(max_retries=1)
        hits = fetcher.fetch_gene("TP53", sources=["gnomad"])

        assert len(hits) == 2
        assert hits[0]["source"] == "gnomAD"
        assert hits[0]["variant_id"] == "1-12345-A-G"
        assert hits[0]["consequence"] == "missense_variant"
        assert hits[0]["allele_frequency"] == 0.0001
        assert hits[0]["hgvsp"] == "p.Arg42Gly"

    @patch("swissisoform.clinical.fetch.requests.post")
    def test_fetch_gene_gnomad_empty_response(self, mock_post: MagicMock) -> None:
        """Empty gnomAD response returns empty list."""
        mock_post.return_value = _mock_response(MOCK_GNOMAD_EMPTY)
        fetcher = VariantFetcher(max_retries=1)
        hits = fetcher.fetch_gene("FAKEGENE", sources=["gnomad"])

        assert hits == []

    @patch("swissisoform.clinical.fetch.requests.post")
    def test_fetch_gene_gnomad_api_error(self, mock_post: MagicMock) -> None:
        """gnomAD request failure returns empty list, does not crash."""
        import requests as _requests

        mock_post.side_effect = _requests.RequestException("API down")
        fetcher = VariantFetcher(max_retries=1, retry_delay=0.0)
        hits = fetcher.fetch_gene("TP53", sources=["gnomad"])

        assert hits == []

    @patch("swissisoform.clinical.fetch.requests.get")
    def test_fetch_gene_clinvar_mock(self, mock_get: MagicMock) -> None:
        """ClinVar returns hits from esearch + esummary."""
        mock_get.side_effect = [
            _mock_response(MOCK_CLINVAR_ESEARCH),
            _mock_response(MOCK_CLINVAR_ESUMMARY),
        ]
        fetcher = VariantFetcher(max_retries=1)
        hits = fetcher.fetch_gene("TP53", sources=["clinvar"])

        assert len(hits) == 2
        assert hits[0]["source"] == "ClinVar"
        assert hits[0]["variant_id"] == "ClinVar:12345"
        assert hits[1]["clinical_significance"] == "Pathogenic"

    @patch("swissisoform.clinical.fetch.requests.get")
    def test_fetch_gene_clinvar_no_ids(self, mock_get: MagicMock) -> None:
        """Empty esearch result returns empty list."""
        mock_get.return_value = _mock_response(MOCK_CLINVAR_ESEARCH_EMPTY)
        fetcher = VariantFetcher(max_retries=1)
        hits = fetcher.fetch_gene("FAKEGENE", sources=["clinvar"])

        assert hits == []

    @patch("swissisoform.clinical.fetch.requests.get")
    @patch("swissisoform.clinical.fetch.requests.post")
    def test_fetch_gene_both_sources(self, mock_post: MagicMock, mock_get: MagicMock) -> None:
        """Both gnomAD and ClinVar results are combined."""
        mock_post.return_value = _mock_response(MOCK_GNOMAD_RESPONSE)
        mock_get.side_effect = [
            _mock_response(MOCK_CLINVAR_ESEARCH),
            _mock_response(MOCK_CLINVAR_ESUMMARY),
        ]
        fetcher = VariantFetcher(max_retries=1)
        hits = fetcher.fetch_gene("TP53")

        sources = {h["source"] for h in hits}
        assert "gnomAD" in sources
        assert "ClinVar" in sources
        assert len(hits) == 4  # 2 gnomAD + 2 ClinVar

    @patch("swissisoform.clinical.fetch.requests.post")
    def test_gnomad_protein_pos_none_canonical_hint_in_metadata(self, mock_post: MagicMock) -> None:
        """protein_pos is None from the fetcher; the canonical-frame HGVSp
        position is preserved in metadata as ``hgvsp_canonical_hint``.

        Rationale: HGVSp strings are canonical-transcript-relative and would
        be wrong for alternative TIS. The authoritative source of
        ``protein_pos`` is ``ConsequenceValidator``, which re-maps genomic
        positions to the isoform's protein coordinates.
        """
        mock_post.return_value = _mock_response(MOCK_GNOMAD_RESPONSE)
        fetcher = VariantFetcher(max_retries=1)
        hits = fetcher.fetch_gene("TP53", sources=["gnomad"])

        # protein_pos is intentionally None — caller must validate
        assert hits[0]["protein_pos"] is None
        assert hits[1]["protein_pos"] is None
        # Canonical-frame hint preserved in metadata
        assert hits[0]["metadata"]["hgvsp_canonical_hint"] == 41
        assert hits[1]["metadata"]["hgvsp_canonical_hint"] == 59

    @patch("swissisoform.clinical.fetch.requests.post")
    def test_gnomad_skips_zero_af(self, mock_post: MagicMock) -> None:
        """Variants with af=0 are skipped."""
        mock_post.return_value = _mock_response(MOCK_GNOMAD_ZERO_AF)
        fetcher = VariantFetcher(max_retries=1)
        hits = fetcher.fetch_gene("TP53", sources=["gnomad"])

        assert hits == []

    @patch("swissisoform.clinical.fetch.requests.post")
    def test_gnomad_no_hgvsp_returns_variant(self, mock_post: MagicMock) -> None:
        """5' UTR variants without hgvsp are still returned with protein_pos=None.

        These variants need genomic-coordinate remapping by the clinical module
        using the isoform's transcript model.
        """
        mock_post.return_value = _mock_response(MOCK_GNOMAD_NO_HGVSP)
        fetcher = VariantFetcher(max_retries=1)
        hits = fetcher.fetch_gene("TP53", sources=["gnomad"])

        assert len(hits) == 1
        assert hits[0]["protein_pos"] is None
        assert hits[0]["genomic_pos"] == 10000
        assert hits[0]["consequence"] == "5_prime_UTR_variant"

    def test_fetch_cosmic_from_parquet(self, tmp_path: Any) -> None:
        """COSMIC fetcher reads from local parquet and returns hits."""
        import pandas as pd

        # Create a minimal COSMIC-like parquet file
        cosmic_df = pd.DataFrame(
            {
                "GENE_SYMBOL": ["TP53", "TP53", "BRCA1"],
                "GENOME_START": [7675088, 7674220, 1000000],
                "CHROMOSOME": ["17", "17", "17"],
                "GENOMIC_WT_ALLELE": ["C", "G", "A"],
                "GENOMIC_MUT_ALLELE": ["T", "A", "G"],
                "HGVSP": ["p.Pro72Arg", "p.Arg248Gln", "p.Met1Ile"],
                "MUTATION_DESCRIPTION": ["Substitution - Missense"] * 3,
                "GENOMIC_MUTATION_ID": ["COSV1", "COSV2", "COSV3"],
                "GENOME_SCREEN_SAMPLE_COUNT": [100, 50, 10],
                "MUTATION_SOMATIC_STATUS": ["Confirmed somatic"] * 3,
                "MUTATION_AA": ["p.P72R", "p.R248Q", "p.M1I"],
                "MUTATION_CDS": ["c.215C>T", "c.743G>A", "c.3G>A"],
            }
        )
        parquet_path = tmp_path / "cosmic_variants_combined.parquet"
        cosmic_df.to_parquet(str(parquet_path))

        fetcher = VariantFetcher(cosmic_db=str(parquet_path))
        hits = fetcher.fetch_gene("TP53", sources=["cosmic"])

        assert len(hits) == 2  # only TP53, not BRCA1
        assert all(h["source"] == "COSMIC" for h in hits)
        # protein_pos from fetcher is None (HGVSp is canonical-frame only);
        # authoritative value comes from ConsequenceValidator.
        assert hits[0]["protein_pos"] is None
        # Canonical-frame hint preserved in metadata
        assert hits[0]["metadata"]["hgvsp_canonical_hint"] == 71
        assert hits[0]["genomic_pos"] == 7675088
        assert hits[0]["metadata"]["cosmic_sample_count"] == 100

    def test_fetch_cosmic_no_db(self) -> None:
        """COSMIC with no db path returns empty list."""
        fetcher = VariantFetcher(cosmic_db=None)
        hits = fetcher.fetch_gene("TP53", sources=["cosmic"])
        assert hits == []

    def test_fetch_cosmic_missing_file(self, tmp_path: Any) -> None:
        """COSMIC with nonexistent path returns empty list."""
        fetcher = VariantFetcher(cosmic_db=str(tmp_path / "nonexistent.parquet"))
        hits = fetcher.fetch_gene("TP53", sources=["cosmic"])
        assert hits == []


# ---------------------------------------------------------------------------
# TestClinicalModuleFetch
# ---------------------------------------------------------------------------


class TestClinicalModuleFetch:
    """Tests for ClinicalModule.fetch_variants integration."""

    def test_fetch_variants_method_exists(self) -> None:
        """ClinicalModule exposes a fetch_variants method."""
        mod = ClinicalModule(PipelineConfig())
        assert hasattr(mod, "fetch_variants")
        assert callable(mod.fetch_variants)

    @patch("swissisoform.clinical.fetch.requests.get")
    @patch("swissisoform.clinical.fetch.requests.post")
    def test_annotate_with_fetch_if_missing_no_validator(
        self, mock_post: MagicMock, mock_get: MagicMock
    ) -> None:
        """Without a ConsequenceValidator, fetched variants have protein_pos=None
        and annotate() filters them out.

        This is the correct, honest behavior: the fetcher cannot know the
        isoform-frame protein position from HGVSp alone (HGVSp is canonical-
        frame), so without a validator we refuse to claim positions. The
        raw variants are still cached for when a validator is attached.
        """
        mock_post.return_value = _mock_response(MOCK_GNOMAD_RESPONSE)
        mock_get.side_effect = [
            _mock_response(MOCK_CLINVAR_ESEARCH),
            _mock_response(MOCK_CLINVAR_ESUMMARY),
        ]
        mod = ClinicalModule(PipelineConfig())

        result = mod.annotate("", gene_name="TP53", fetch_if_missing=True)
        # All variants filtered out because protein_pos is None without validator
        assert result["summary"]["total_variants"] == 0
        assert result["hits"] == []

        # But raw variants ARE cached (4 total: 2 gnomAD + 2 ClinVar)
        assert "TP53" in mod.variant_cache
        assert len(mod.variant_cache["TP53"]) == 4
        # And all have protein_pos=None (honest output)
        for v in mod.variant_cache["TP53"]:
            assert v["protein_pos"] is None
        # The canonical-frame hints are preserved in metadata
        canonical_hints = [
            v["metadata"].get("hgvsp_canonical_hint") for v in mod.variant_cache["TP53"]
        ]
        assert any(h is not None for h in canonical_hints)
