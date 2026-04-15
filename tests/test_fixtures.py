"""Smoke tests for the synthetic TIS fixtures."""

from __future__ import annotations

from swissisoform.models import ORFType, TranslationInitiationSite


def test_fixture_count(synthetic_tis: list[TranslationInitiationSite]) -> None:
    """Exactly 8 TIS sites."""
    assert len(synthetic_tis) == 8


def test_all_orf_types_covered(synthetic_tis: list[TranslationInitiationSite]) -> None:
    """At least ANNOTATED, EXTENDED, TRUNCATED, UORF, UOORF are present."""
    types = {s.orf_type for s in synthetic_tis}
    for expected in (
        ORFType.ANNOTATED,
        ORFType.EXTENDED,
        ORFType.TRUNCATED,
        ORFType.UORF,
        ORFType.UOORF,
    ):
        assert expected in types, f"Missing ORF type: {expected}"


def test_both_strands(synthetic_tis: list[TranslationInitiationSite]) -> None:
    """Both + and - strands are represented."""
    strands = {s.strand for s in synthetic_tis}
    assert strands == {"+", "-"}


def test_ctg_codon_present(synthetic_tis: list[TranslationInitiationSite]) -> None:
    """At least one site uses a non-AUG (CTG) start codon."""
    codons = {s.start_codon for s in synthetic_tis}
    assert "CTG" in codons


def test_kozak_contexts_are_13mers(synthetic_tis: list[TranslationInitiationSite]) -> None:
    """All kozak_context strings are exactly 13 characters (5 upstream + 3 codon + 5 downstream)."""
    for site in synthetic_tis:
        assert site.kozak_context is not None
        assert len(site.kozak_context) == 13, (
            f"{site.tis_id}: kozak_context length {len(site.kozak_context)}, expected 13"
        )


def test_expression_has_two_cell_lines(
    synthetic_tis: list[TranslationInitiationSite],
) -> None:
    """Every TIS has HeLa and K562 expression data."""
    for site in synthetic_tis:
        assert "HeLa" in site.expression, f"{site.tis_id} missing HeLa"
        assert "K562" in site.expression, f"{site.tis_id} missing K562"


def test_config_has_six_cell_lines(config) -> None:  # noqa: ANN001
    """Default PipelineConfig has 6 cell lines."""
    assert len(config.cell_lines) == 6


def test_three_genes(synthetic_tis: list[TranslationInitiationSite]) -> None:
    """Sites span exactly 3 gene names."""
    genes = {s.gene_name for s in synthetic_tis}
    assert len(genes) == 3


def test_deepcopy_independence(synthetic_tis: list[TranslationInitiationSite]) -> None:
    """Modifying the fixture does not affect other calls."""
    synthetic_tis[0].gene_name = "MUTATED"
    # Re-import to check original is untouched
    from tests.conftest import SYNTHETIC_TIS

    assert SYNTHETIC_TIS[0].gene_name == "TESTGENE_POS"
