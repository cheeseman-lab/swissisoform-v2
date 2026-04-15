"""Synthetic TIS fixtures for SwissIsoform v2 tests.

Provides 8 TIS across 3 genes covering all major ORF types, both strands,
AUG and non-AUG start codons, and two cell lines (HeLa, K562).
"""

from __future__ import annotations

import copy

import pytest

from swissisoform.config import PipelineConfig
from swissisoform.models import CellLineExpression, ORFType, TranslationInitiationSite

# Canonical proteins
CANONICAL_POS = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPS*"
CANONICAL_NEG = "MDLSALREVELIQNMHRKEVTHPVFD*"
CANONICAL_MULTI = "MSSGNAKIGHPAPNFKATAVMHNEFIASK*"

# Reusable expression dicts
_EXPR_HELA = CellLineExpression(raw_count=150, cpm=12.5, p_value=1e-4)
_EXPR_K562 = CellLineExpression(raw_count=90, cpm=8.3, p_value=5e-3)


def _make_expression() -> dict[str, CellLineExpression]:
    return {
        "HeLa": CellLineExpression(
            raw_count=_EXPR_HELA.raw_count,
            cpm=_EXPR_HELA.cpm,
            p_value=_EXPR_HELA.p_value,
        ),
        "K562": CellLineExpression(
            raw_count=_EXPR_K562.raw_count,
            cpm=_EXPR_K562.cpm,
            p_value=_EXPR_K562.p_value,
        ),
    }


SYNTHETIC_TIS: list[TranslationInitiationSite] = [
    # 1. Annotated (canonical == isoform)
    TranslationInitiationSite(
        tis_id="chr1:1000:+:ATG",
        gene_name="TESTGENE_POS",
        transcript_id="ENST_POS_001",
        chrom="chr1",
        position=1000,
        strand="+",
        start_codon="ATG",
        orf_type=ORFType.ANNOTATED,
        gene_id="ENSG_POS",
        aa_len=len(CANONICAL_POS) - 1,
        tis_pvalue=1e-6,
        ribo_pvalue=1e-5,
        fisher_qvalue=1e-4,
        expression=_make_expression(),
        canonical_protein=CANONICAL_POS,
        isoform_protein=CANONICAL_POS,
        differential_sequence="",
        shared_sequence=CANONICAL_POS,
        kozak_context="GCCGCCATGGCGA",
    ),
    # 2. Extension +strand AUG
    TranslationInitiationSite(
        tis_id="chr1:950:+:ATG",
        gene_name="TESTGENE_POS",
        transcript_id="ENST_POS_001",
        chrom="chr1",
        position=950,
        strand="+",
        start_codon="ATG",
        orf_type=ORFType.EXTENDED,
        gene_id="ENSG_POS",
        aa_len=len(CANONICAL_POS) - 1 + 11,
        tis_pvalue=5e-5,
        ribo_pvalue=3e-4,
        fisher_qvalue=2e-3,
        expression=_make_expression(),
        canonical_protein=CANONICAL_POS,
        isoform_protein="MRGSHHHHHGS" + CANONICAL_POS,
        differential_sequence="MRGSHHHHHGS",
        shared_sequence=CANONICAL_POS,
        kozak_context="TCCACCATGAGGA",
    ),
    # 3. Extension +strand CTG (non-AUG)
    TranslationInitiationSite(
        tis_id="chr1:940:+:CTG",
        gene_name="TESTGENE_POS",
        transcript_id="ENST_POS_001",
        chrom="chr1",
        position=940,
        strand="+",
        start_codon="CTG",
        orf_type=ORFType.EXTENDED,
        gene_id="ENSG_POS",
        aa_len=len(CANONICAL_POS) - 1 + 8,
        tis_pvalue=1e-3,
        ribo_pvalue=2e-3,
        fisher_qvalue=5e-2,
        expression=_make_expression(),
        canonical_protein=CANONICAL_POS,
        isoform_protein="LRRPPAGA" + CANONICAL_POS,
        differential_sequence="LRRPPAGA",
        shared_sequence=CANONICAL_POS,
        kozak_context="AGCGTCCTGACTG",
    ),
    # 4. Extension -strand AUG
    TranslationInitiationSite(
        tis_id="chr2:5000:-:ATG",
        gene_name="TESTGENE_NEG",
        transcript_id="ENST_NEG_001",
        chrom="chr2",
        position=5000,
        strand="-",
        start_codon="ATG",
        orf_type=ORFType.EXTENDED,
        gene_id="ENSG_NEG",
        aa_len=len(CANONICAL_NEG) - 1 + 7,
        tis_pvalue=2e-5,
        ribo_pvalue=1e-4,
        fisher_qvalue=8e-4,
        expression=_make_expression(),
        canonical_protein=CANONICAL_NEG,
        isoform_protein="MAGTKLM" + CANONICAL_NEG,
        differential_sequence="MAGTKLM",
        shared_sequence=CANONICAL_NEG,
        kozak_context="GCCACCATGGCTG",
    ),
    # 5. Truncation +strand
    TranslationInitiationSite(
        tis_id="chr1:1030:+:ATG",
        gene_name="TESTGENE_POS",
        transcript_id="ENST_POS_001",
        chrom="chr1",
        position=1030,
        strand="+",
        start_codon="ATG",
        orf_type=ORFType.TRUNCATED,
        gene_id="ENSG_POS",
        aa_len=len(CANONICAL_POS) - 1 - 9,
        tis_pvalue=8e-4,
        ribo_pvalue=6e-3,
        fisher_qvalue=1e-2,
        expression=_make_expression(),
        canonical_protein=CANONICAL_POS,
        isoform_protein=CANONICAL_POS[9:],
        differential_sequence="EEPQSDPSV",
        shared_sequence=CANONICAL_POS[9:],
        kozak_context="ACCGCCATGGTCA",
    ),
    # 6. Truncation -strand
    TranslationInitiationSite(
        tis_id="chr2:4970:-:ATG",
        gene_name="TESTGENE_NEG",
        transcript_id="ENST_NEG_001",
        chrom="chr2",
        position=4970,
        strand="-",
        start_codon="ATG",
        orf_type=ORFType.TRUNCATED,
        gene_id="ENSG_NEG",
        aa_len=len(CANONICAL_NEG) - 1 - 6,
        tis_pvalue=3e-4,
        ribo_pvalue=2e-3,
        fisher_qvalue=5e-3,
        expression=_make_expression(),
        canonical_protein=CANONICAL_NEG,
        isoform_protein=CANONICAL_NEG[6:],
        differential_sequence="DLSALR",
        shared_sequence=CANONICAL_NEG[6:],
        kozak_context="TGCACCATGCAGT",
    ),
    # 7. uORF (entirely unique product)
    TranslationInitiationSite(
        tis_id="chr3:800:+:ATG",
        gene_name="TESTGENE_MULTI",
        transcript_id="ENST_MULTI_001",
        chrom="chr3",
        position=800,
        strand="+",
        start_codon="ATG",
        orf_type=ORFType.UORF,
        gene_id="ENSG_MULTI",
        aa_len=8,
        tis_pvalue=1e-4,
        ribo_pvalue=5e-4,
        fisher_qvalue=3e-3,
        expression=_make_expression(),
        canonical_protein=CANONICAL_MULTI,
        isoform_protein="MPKLQRST*",
        differential_sequence="MPKLQRST*",
        shared_sequence="",
        kozak_context="CCCGCCATGCCCA",
    ),
    # 8. uoORF (overlapping CDS)
    TranslationInitiationSite(
        tis_id="chr3:850:+:ATG",
        gene_name="TESTGENE_MULTI",
        transcript_id="ENST_MULTI_001",
        chrom="chr3",
        position=850,
        strand="+",
        start_codon="ATG",
        orf_type=ORFType.UOORF,
        gene_id="ENSG_MULTI",
        aa_len=len(CANONICAL_MULTI) - 1 + 6,
        tis_pvalue=7e-5,
        ribo_pvalue=2e-4,
        fisher_qvalue=1e-3,
        expression=_make_expression(),
        canonical_protein=CANONICAL_MULTI,
        isoform_protein="MFLGRT" + CANONICAL_MULTI,
        differential_sequence="MFLGRT",
        shared_sequence=CANONICAL_MULTI,
        kozak_context="GCTGCCATGTTCG",
    ),
]


@pytest.fixture()
def synthetic_tis() -> list[TranslationInitiationSite]:
    """Return a deep copy of the 8 synthetic TIS sites."""
    return copy.deepcopy(SYNTHETIC_TIS)


@pytest.fixture()
def config() -> PipelineConfig:
    """Return a default PipelineConfig."""
    return PipelineConfig()
