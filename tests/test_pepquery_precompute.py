"""Unit tests for PepQuery2 precompute helpers (pure logic only).

The actual ``pepquery`` subprocess is not exercised here.  These tests
cover the three pure helpers: peptide collection, cache key stability,
and output parsing.
"""

from __future__ import annotations

from swissisoform.models import (
    DifferentialRegion,
    Gene,
    ORFType,
    TranslationInitiationSite,
)
from swissisoform.evidence.d3_mass_spec.massspec import (
    _parse_pepquery_output,
    _pepquery_cache_key,
    _regroup_by_gene,
    collect_unique_peptides,
)


def _gene_with_isoform(
    gene_name: str,
    canonical: str,
    isoform: str,
    *,
    orf_type: ORFType = ORFType.EXTENDED,
    diff_region: DifferentialRegion | None = None,
) -> Gene:
    """Build a single-TIS gene for peptide-collection tests."""
    site = TranslationInitiationSite(
        tis_id=f"chr1:100:+:ATG:{gene_name}",
        gene_name=gene_name,
        transcript_id="ENST_T",
        chrom="chr1",
        position=100,
        strand="+",
        start_codon="ATG",
        orf_type=orf_type,
        canonical_protein=canonical,
        isoform_protein=isoform,
        diff_region=diff_region,
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

    def test_extension_scoped_to_diff_region(self):
        # Extension = isoform[0:8] ("MNNNNNNK"). Only peptides overlapping the
        # extension are collected; the shared core peptides are excluded by
        # scope (and would be dropped by the set-difference anyway).
        gene = _gene_with_isoform(
            "EXTGENE",
            canonical="MAAAAAAKVVVVVVVR",
            isoform="MNNNNNNKMAAAAAAKVVVVVVVR",
            orf_type=ORFType.EXTENDED,
            diff_region=DifferentialRegion(isoform_start=0, isoform_end=8),
        )
        peptides = collect_unique_peptides([gene])
        assert "MNNNNNNK" in peptides["EXTGENE"]
        # The shared-core peptide is outside the extension scope.
        assert "VVVVVVVR" not in peptides["EXTGENE"]

    def test_truncation_collects_only_start_peptide(self):
        # Truncation: isoform is the canonical C-terminus with a new Met start.
        # diff_region is canonical-space, so scope falls to pos==0 start peptides.
        # Second residue G → an NME (Met-excised) variant is also collected.
        # Canonical's residue at the new start is L (the near-cognate codon
        # Ribo-TISH would translate); the isoform installs M there, so the
        # Met-started peptide is genuinely absent from the canonical digest.
        gene = _gene_with_isoform(
            "TRUNCGENE",
            canonical="MZZZZZZZKLGAAAAAAKVVVVVVVR",
            isoform="MGAAAAAAKVVVVVVVR",
            orf_type=ORFType.TRUNCATED,
            diff_region=DifferentialRegion(canonical_start=0, canonical_end=9),
        )
        peptides = collect_unique_peptides([gene])
        peps = peptides.get("TRUNCGENE", set())
        # Met-retained start peptide is unique vs canonical (canonical's twin
        # lacks the new Met context); the Met-excised twin GAAAAAAK is also present.
        assert "MGAAAAAAK" in peps
        assert "GAAAAAAK" in peps
        # Downstream shared peptide is out of scope.
        assert "VVVVVVVR" not in peps


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

    def test_mod_change_changes_key(self):
        # A different modification signature must bust the cache — otherwise a
        # varMod change would serve a stale hit from the unmodified search.
        base = _pepquery_cache_key("w", "swissprot:human", ["AAAR"], "1|2|3|")
        acetyl = _pepquery_cache_key("w", "swissprot:human", ["AAAR"], "1|2,5|3|")
        assert base != acetyl
        # Default empty mods_sig stays backward-stable across calls.
        assert _pepquery_cache_key("w", "swissprot:human", ["AAAR"]) == _pepquery_cache_key(
            "w", "swissprot:human", ["AAAR"]
        )


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
