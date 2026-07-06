"""Tests for the annotation pipeline wiring layer."""

from __future__ import annotations

import pandas as pd

from swissisoform.modules.biophysics import BiophysicsModule
from swissisoform.modules.core_identity import CoreIdentityModule
from swissisoform.modules.initiation_context import InitiationContextModule
from swissisoform.modules.motifs import MotifsModule
from swissisoform.pipeline import AnnotationPipeline, normalize_canonical_initiator


class TestNormalizeCanonicalInitiator:
    """Detected near-cognate canonical starts get an installed initiator Met."""

    def test_annotated_near_cognate_normalized_to_met(self):
        # GTG→V and CTG→L on Annotated rows become M, matching the imputed
        # (pc_translations, M-start) path; ATG Annotated rows are unchanged, and
        # non-Annotated near-cognate isoforms keep Ribo-TISH's literal residue.
        df = pd.DataFrame(
            {
                "RecatTISType": ["Annotated", "Annotated", "Annotated", "Extended", "Annotated"],
                "AASeq": ["VESAIAEG", "LDFFRVV", "MPLNVSF", "VDFSSVV", None],
            }
        )
        out = normalize_canonical_initiator(df)
        assert out.loc[0, "AASeq"] == "MESAIAEG"  # GTG→V normalized
        assert out.loc[1, "AASeq"] == "MDFFRVV"  # CTG→L normalized
        assert out.loc[2, "AASeq"] == "MPLNVSF"  # ATG already M — unchanged
        assert out.loc[3, "AASeq"] == "VDFSSVV"  # non-Annotated — untouched
        assert pd.isna(out.loc[4, "AASeq"])  # NaN AASeq tolerated
        # Input frame not mutated in place.
        assert df.loc[0, "AASeq"] == "VESAIAEG"

    def test_missing_aaseq_column_is_noop(self):
        df = pd.DataFrame({"RecatTISType": ["Annotated", "Extended"]})
        out = normalize_canonical_initiator(df)
        assert out.equals(df)


class TestAnnotationPipeline:
    """Tests for AnnotationPipeline orchestration."""

    def test_empty_genes_list(self):
        """Pipeline handles empty gene list."""
        pipeline = AnnotationPipeline()
        result = pipeline.run([])
        assert result == []

    def test_empty_modules(self, synthetic_genes):
        """Pipeline with no modules leaves annotations empty."""
        pipeline = AnnotationPipeline()
        result = pipeline.run(synthetic_genes)
        assert len(result) == 3
        for gene in result:
            assert gene.canonical_annotations == {}

    def test_protein_module_on_canonical(self, synthetic_genes, config):
        """ProteinModule annotates canonical protein on Gene."""
        bio = BiophysicsModule(config)
        pipeline = AnnotationPipeline(protein_modules=[bio])
        result = pipeline.run(synthetic_genes)

        for gene in result:
            assert "biophysics" in gene.canonical_annotations
            ann = gene.canonical_annotations["biophysics"]
            # Canonical proteins are all > 3 AA, so values should be non-None
            assert ann["pI"] is not None
            assert ann["gravy"] is not None

    def test_protein_module_on_isoform(self, synthetic_genes, config):
        """ProteinModule annotates isoform protein on each TIS."""
        bio = BiophysicsModule(config)
        pipeline = AnnotationPipeline(protein_modules=[bio])
        result = pipeline.run(synthetic_genes)

        for gene in result:
            for site in gene.tis_sites:
                assert "biophysics" in site.isoform_annotations
                ann = site.isoform_annotations["biophysics"]
                # Isoform proteins are all > 3 AA
                assert ann["pI"] is not None

    def test_canonical_annotated_once(self, synthetic_genes, config):
        """Canonical annotations are the same object for all TIS in a gene."""
        bio = BiophysicsModule(config)
        pipeline = AnnotationPipeline(protein_modules=[bio])
        result = pipeline.run(synthetic_genes)

        for gene in result:
            canonical_ann = gene.canonical_annotations["biophysics"]
            # All TIS sites should have isoform annotations but they differ
            # from canonical (different proteins)
            for site in gene.tis_sites:
                iso_ann = site.isoform_annotations["biophysics"]
                # Extension isoforms have different proteins than canonical
                if site.isoform_protein != gene.canonical_protein:
                    # At least some values should differ
                    assert iso_ann != canonical_ann or iso_ann["length"] != canonical_ann["length"]

    def test_site_module_runs(self, synthetic_genes, config):
        """SiteModule annotates each TIS."""
        core = CoreIdentityModule(config)
        pipeline = AnnotationPipeline(site_modules=[core])
        result = pipeline.run(synthetic_genes)

        for gene in result:
            for site in gene.tis_sites:
                assert "core_identity" in site.isoform_annotations
                ann = site.isoform_annotations["core_identity"]
                assert "orf_type" in ann
                assert "in_frame" in ann

    def test_multiple_protein_modules(self, synthetic_genes, config):
        """Multiple ProteinModules all run."""
        bio = BiophysicsModule(config)
        mot = MotifsModule(config)
        pipeline = AnnotationPipeline(protein_modules=[bio, mot])
        result = pipeline.run(synthetic_genes)

        for gene in result:
            assert "biophysics" in gene.canonical_annotations
            assert "motifs" in gene.canonical_annotations
            for site in gene.tis_sites:
                assert "biophysics" in site.isoform_annotations
                assert "motifs" in site.isoform_annotations

    def test_mixed_module_types(self, synthetic_genes, config):
        """ProteinModules + SiteModules together."""
        bio = BiophysicsModule(config)
        core = CoreIdentityModule(config)
        init = InitiationContextModule(config)
        pipeline = AnnotationPipeline(
            protein_modules=[bio],
            site_modules=[core, init],
        )
        result = pipeline.run(synthetic_genes)

        for gene in result:
            assert "biophysics" in gene.canonical_annotations
            for site in gene.tis_sites:
                assert "biophysics" in site.isoform_annotations
                assert "core_identity" in site.isoform_annotations
                assert "initiation_context" in site.isoform_annotations

    def test_site_modules_not_on_canonical(self, synthetic_genes, config):
        """SiteModules should NOT appear in canonical_annotations."""
        core = CoreIdentityModule(config)
        pipeline = AnnotationPipeline(site_modules=[core])
        result = pipeline.run(synthetic_genes)

        for gene in result:
            assert "core_identity" not in gene.canonical_annotations

    def test_protein_modules_not_in_gene_annotations(self, synthetic_genes, config):
        """ProteinModules should NOT appear in gene_annotations."""
        bio = BiophysicsModule(config)
        pipeline = AnnotationPipeline(protein_modules=[bio])
        result = pipeline.run(synthetic_genes)

        for gene in result:
            assert "biophysics" not in gene.gene_annotations

    def test_motifs_positional_on_canonical(self, synthetic_genes, config):
        """Motifs module produces positional hits for canonical protein."""
        mot = MotifsModule(config)
        pipeline = AnnotationPipeline(protein_modules=[mot])
        result = pipeline.run(synthetic_genes)

        for gene in result:
            ann = gene.canonical_annotations["motifs"]
            assert "hits" in ann
            assert "summary" in ann
            assert isinstance(ann["hits"], list)

    def test_genes_returned_are_same_objects(self, synthetic_genes, config):
        """Pipeline mutates and returns the same Gene objects."""
        bio = BiophysicsModule(config)
        pipeline = AnnotationPipeline(protein_modules=[bio])
        result = pipeline.run(synthetic_genes)
        for orig, ret in zip(synthetic_genes, result):
            assert orig is ret

    def test_no_tis_sites_dropped(self, synthetic_genes, config):
        """Pipeline does not drop any TIS sites."""
        original_counts = [len(g.tis_sites) for g in synthetic_genes]
        bio = BiophysicsModule(config)
        core = CoreIdentityModule(config)
        pipeline = AnnotationPipeline(protein_modules=[bio], site_modules=[core])
        result = pipeline.run(synthetic_genes)
        result_counts = [len(g.tis_sites) for g in result]
        assert original_counts == result_counts
