"""Unit tests for PepQuery2 precompute helpers (pure logic only).

The actual ``pepquery`` subprocess is not exercised here.  These tests
cover the three pure helpers: peptide collection, cache key stability,
and output parsing.
"""

from __future__ import annotations

from swissisoform.models import Gene, ORFType, TranslationInitiationSite
from swissisoform.evidence.e6_mass_spec.massspec import (
    _parse_pepquery_output,
    _pepquery_cache_key,
    _regroup_by_gene,
    collect_unique_peptides,
)


def _gene_with_isoform(gene_name: str, canonical: str, isoform: str) -> Gene:
    """Build a single-TIS gene for peptide-collection tests."""
    site = TranslationInitiationSite(
        tis_id=f"chr1:100:+:ATG:{gene_name}",
        gene_name=gene_name,
        transcript_id="ENST_T",
        chrom="chr1",
        position=100,
        strand="+",
        start_codon="ATG",
        orf_type=ORFType.EXTENDED,
        canonical_protein=canonical,
        isoform_protein=isoform,
    )
    return Gene(
        gene_name=gene_name,
        gene_id="ENSG_T",
        canonical_transcript_id="ENST_T",
        canonical_protein=canonical,
        tis_sites=[site],
    )


class TestCollectUniquePeptides:
    def test_isoform_only_peptides_included(self):
        # Isoform has a 7-aa extension that introduces one new tryptic peptide
        gene = _gene_with_isoform(
            "TESTGENE",
            canonical="MAAAAAAKVVVVVVVR",
            isoform="MNNNNNNKMAAAAAAKVVVVVVVR",
        )
        peptides = collect_unique_peptides([gene])
        assert "TESTGENE" in peptides
        # The extension introduces at least one peptide not in canonical
        assert any(len(p) >= 7 for p in peptides["TESTGENE"])

    def test_gene_skipped_when_no_isoform(self):
        gene = _gene_with_isoform("TESTGENE", canonical="MAAAKVVVR", isoform="")
        assert collect_unique_peptides([gene]) == {}


class TestCacheKey:
    def test_same_inputs_stable(self):
        k1 = _pepquery_cache_key("w", "gencode:human", ["AAAR", "BBBR"])
        k2 = _pepquery_cache_key("w", "gencode:human", ["AAAR", "BBBR"])
        assert k1 == k2
        assert len(k1) == 16

    def test_dataset_change_changes_key(self):
        k_w = _pepquery_cache_key("w", "gencode:human", ["AAAR"])
        k_all = _pepquery_cache_key("all", "gencode:human", ["AAAR"])
        assert k_w != k_all


class TestParsePepqueryOutput:
    def test_no_file_returns_empty(self, tmp_path):
        assert _parse_pepquery_output(tmp_path) == set()

    def test_confident_yes_extracted_in_dataset_subdir(self, tmp_path):
        # PepQuery2 writes output/<dataset>/psm_rank.txt, not output/psm_rank.txt
        dataset_dir = tmp_path / "CPTAC_LUAD_Discovery_Study_Proteome_PDC000153"
        dataset_dir.mkdir()
        (dataset_dir / "psm_rank.txt").write_text(
            "peptide\tconfident\trank\n"
            "VALIDHIT\tYes\t1\n"
            "REJECTED\tNo\t1\n"
            "ALSOHIT\tyes\t1\n"
        )
        assert _parse_pepquery_output(tmp_path) == {"VALIDHIT", "ALSOHIT"}

    def test_multiple_datasets_union(self, tmp_path):
        for ds, peps in (
            ("GTEx_32_Tissues_Proteome_PXD016999", "A\tYes\nB\tNo\n"),
            ("Deep_29_healthy_human_tissues_PXD010154", "C\tYes\nA\tYes\n"),
        ):
            d = tmp_path / ds
            d.mkdir()
            (d / "psm_rank.txt").write_text("peptide\tconfident\n" + peps)
        assert _parse_pepquery_output(tmp_path) == {"A", "C"}

    def test_ignores_database_subdir(self, tmp_path):
        # The FMIndex build leaves output/database/ — no psm_rank.txt but
        # glob must not trip over it either.
        (tmp_path / "database").mkdir()
        (tmp_path / "database" / "some_index.fmi").write_text("binary")
        assert _parse_pepquery_output(tmp_path) == set()

    def test_falls_back_to_rank_when_no_confident_col(self, tmp_path):
        ds = tmp_path / "any_dataset"
        ds.mkdir()
        (ds / "psm_rank.txt").write_text(
            "peptide\trank\n"
            "R1HIT\t1\n"
            "R2MISS\t2\n"
        )
        assert _parse_pepquery_output(tmp_path) == {"R1HIT"}


class TestRegroupByGene:
    def test_regroup_spreads_shared_peptide_to_all_genes(self):
        peptide_to_genes = {
            "SHARED": {"GENE_A", "GENE_B"},
            "UNIQUE": {"GENE_A"},
        }
        out = _regroup_by_gene({"SHARED", "UNIQUE"}, peptide_to_genes)
        assert out["GENE_A"] == {"SHARED", "UNIQUE"}
        assert out["GENE_B"] == {"SHARED"}

    def test_unvalidated_peptide_absent(self):
        # The unvalidated peptide must not appear in any gene's set, but the
        # queried gene itself stays present with an empty set — that is how
        # MassSpecModule tells "queried, no evidence" (E6=False) apart from
        # "never queried" (E6=None). See _regroup_by_gene's docstring.
        peptide_to_genes = {"UNVALIDATED": {"GENE_A"}}
        assert _regroup_by_gene(set(), peptide_to_genes) == {"GENE_A": set()}
