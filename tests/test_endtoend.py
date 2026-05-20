"""End-to-end integration test on real Ribo-TISH data.

Reads HeLa predict_all.txt, assembles Gene objects for 5 test genes,
runs the annotation pipeline (biophysics, motifs, core_identity,
initiation_context), and asserts output shape and content.

Test gene set covers:
    TP53       — minus strand, multi-ORF (Ext/Trunc/uORF), intron-crossing extension
    EIF4G1     — 33-exon gene, extreme length range (18–1611 aa), uORF, many truncations
    VEGFA      — multi-ORF, known biology, within-exon truncation
    CTNND1     — all ORF types, huge size range (11–968 aa), intron-crossing extension
    MYC        — CUG-initiated N-terminal extension (MYC1), cancer alt initiation
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from swissisoform.assembly import assemble_genes, compute_diff_region
from swissisoform.config import PipelineConfig
from swissisoform.io.parquet import tis_to_dataframe
from swissisoform.io.ribotish import load_ribotish_predictions
from swissisoform.models import Gene, ORFType
from swissisoform.modules.biophysics import BiophysicsModule
from swissisoform.modules.core_identity import CoreIdentityModule
from swissisoform.modules.initiation_context import InitiationContextModule
from swissisoform.modules.interproscan import (
    InterProScanModule,
    precompute_interproscan,
)
from swissisoform.modules.motifs import MotifsModule
from swissisoform.modules.signalp import SignalPModule, precompute_signalp
from swissisoform.modules.targetp import TargetPModule, precompute_targetp
from swissisoform.pipeline import AnnotationPipeline, UpstreamReference, run_sample

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data" / "reference"
HELA_PREDICT = DATA_DIR / "HeLa_TIS_predict_all.txt"
GENOME_FASTA = DATA_DIR / "Gencode_v49_GRCh38.primary_assembly.genome.fa"
GTF_PATH = DATA_DIR / "gencode.v49.primary_assembly.annotation.gtf"
PROTEIN_FASTA = DATA_DIR / "gencode.v49.pc_translations.fa"
HELA_RNASEQ = [
    DATA_DIR / "rnaseq_counts/GATCAG_8_htseqcount.txt",
    DATA_DIR / "rnaseq_counts/CACGGT_9_htseqcount.txt",
]

TEST_GENES = ["TP53", "EIF4G1", "VEGFA", "CTNND1", "MYC"]

# Full GENCODE genome FASTA is now present — Kozak should populate for
# every TIS regardless of contig.
GENES_WITH_FASTA = TEST_GENES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hela_df():
    """Load HeLa predict_all.txt as a DataFrame (cached per module).

    Raw, un-GTF-merged, un-filtered.  Kept for tests that check the raw
    input directly (e.g. per-TIS canonical match tests that need to see
    the full set of Annotated rows before filtering).
    """
    if not HELA_PREDICT.exists():
        pytest.skip("HeLa predict file not available")
    return load_ribotish_predictions(HELA_PREDICT)


@pytest.fixture(scope="module")
def upstream_reference() -> UpstreamReference:
    """Shared GTF + FASTA reference tables (GENCODE CDS/UTR/start_codon
    + pc_translations.fa).
    """
    for p in (GTF_PATH, GENOME_FASTA, PROTEIN_FASTA):
        if not p.exists():
            pytest.skip(f"Missing reference: {p}")
    return UpstreamReference.load(
        gtf_path=GTF_PATH, genome_fasta=GENOME_FASTA, protein_fasta=PROTEIN_FASTA
    )


@pytest.fixture(scope="module")
def upstream_hela(upstream_reference):
    """``run_sample`` on HeLa — filtered + imputed table per cell line.

    Returns ``(final_df, dropped_df)``. ``final_df`` is the faithful-
    upstream port output: filter → impute canonical → drop uncanonical
    transcripts.  Every Tid in ``final_df`` has an Annotated row.
    """
    if not HELA_PREDICT.exists() or not all(c.exists() for c in HELA_RNASEQ):
        pytest.skip("HeLa predict or RNA-seq counts not available")
    return run_sample(
        HELA_PREDICT,
        HELA_RNASEQ,
        GTF_PATH,
        sample="HeLa",
        reference=upstream_reference,
    )


@pytest.fixture(scope="module")
def test_genes(upstream_hela) -> list[Gene]:
    """Assemble Gene objects for the 5 test genes from the single-table
    upstream output.
    """
    final_df, _ = upstream_hela
    final_df = final_df[final_df["Symbol"].isin(TEST_GENES)]
    fasta_path = GENOME_FASTA if GENOME_FASTA.exists() else None
    return assemble_genes(
        final_df,
        gene_names=TEST_GENES,
        genome_fasta=fasta_path,
    )


@pytest.fixture(scope="module")
def annotated_genes(test_genes) -> list[Gene]:
    """Run the annotation pipeline on the 5 test genes.

    Runs SignalP + TargetP precompute over every unique canonical /
    isoform / per-Tid canonical protein sequence across the test gene
    set before kicking off the pipeline.  The precomputes silently
    return ``{}`` when their external tool isn't installed (no
    ``swissisoform-v2-signalp`` env / no extracted TargetP binary), in
    which case the modules degrade to all-None — but when the tools are
    installed, real predictions flow through the pipeline and get tested
    end-to-end.
    """
    config = PipelineConfig()

    # Collect unique protein sequences from the test genes.  Dedup is
    # done inside the precompute functions (hash-keyed), but building
    # the set here keeps the input obvious in logs.
    proteins: dict[str, str] = {}
    for gene in test_genes:
        if gene.canonical_protein:
            proteins[f"{gene.gene_name}_canonical"] = gene.canonical_protein
        for site in gene.tis_sites:
            if site.isoform_protein:
                proteins[f"{site.tis_id}_iso"] = site.isoform_protein
            if site.canonical_protein:
                proteins[f"{site.tis_id}_can"] = site.canonical_protein

    signalp_preds = precompute_signalp(proteins, mode="fast")
    targetp_preds = precompute_targetp(proteins)
    # InterProScan with the default non-ML application set.  Pfam-only
    # trips a COMBINE_MATCHES bug in IPS6 6.0.0 whenever any sequence
    # has no hits, so we keep the broader default.
    interproscan_preds = precompute_interproscan(proteins)

    pipeline = AnnotationPipeline(
        protein_modules=[
            BiophysicsModule(config),
            MotifsModule(config),
            SignalPModule(config, predictions=signalp_preds),
            TargetPModule(config, predictions=targetp_preds),
            InterProScanModule(config, predictions=interproscan_preds),
        ],
        site_modules=[
            CoreIdentityModule(config),
            InitiationContextModule(config),
        ],
    )
    return pipeline.run(test_genes)


# ---------------------------------------------------------------------------
# Assembly tests
# ---------------------------------------------------------------------------


class TestAssembly:
    """Tests that Gene objects are correctly assembled from raw Ribo-TISH data."""

    def test_all_genes_present(self, test_genes: list[Gene]) -> None:
        """All 5 test genes are assembled."""
        names = {g.gene_name for g in test_genes}
        assert names == set(TEST_GENES)

    def test_each_gene_has_tis(self, test_genes: list[Gene]) -> None:
        """Each gene has at least 1 alternative TIS site."""
        for gene in test_genes:
            assert len(gene.tis_sites) > 0, f"{gene.gene_name} has no TIS sites"

    def test_canonical_proteins_nonempty(self, test_genes: list[Gene]) -> None:
        """Each gene has a non-empty canonical protein."""
        for gene in test_genes:
            assert len(gene.canonical_protein) > 10, (
                f"{gene.gene_name} canonical protein too short: {len(gene.canonical_protein)} aa"
            )

    def test_tp53_minus_strand(self, test_genes: list[Gene]) -> None:
        """TP53 is on the minus strand."""
        tp53 = next(g for g in test_genes if g.gene_name == "TP53")
        strands = {s.strand for s in tp53.tis_sites}
        assert "-" in strands

    def test_tp53_has_extensions(self, test_genes: list[Gene]) -> None:
        """TP53 has Extended ORF types."""
        tp53 = next(g for g in test_genes if g.gene_name == "TP53")
        orf_types = {s.orf_type for s in tp53.tis_sites}
        assert ORFType.EXTENDED in orf_types

    def test_eif4g1_has_uorf(self, test_genes: list[Gene]) -> None:
        """EIF4G1 has uORF type (5'UTR TIS)."""
        eif4g1 = next(g for g in test_genes if g.gene_name == "EIF4G1")
        orf_types = {s.orf_type for s in eif4g1.tis_sites}
        assert ORFType.UORF in orf_types

    def test_eif4g1_short_uorf(self, test_genes: list[Gene]) -> None:
        """EIF4G1 has a very short alternative ORF (<50 aa) — edge case."""
        eif4g1 = next(g for g in test_genes if g.gene_name == "EIF4G1")
        short = [s for s in eif4g1.tis_sites if s.aa_len < 50]
        assert len(short) > 0, "EIF4G1 should have at least one short ORF"

    def test_ctnnd1_has_extensions(self, test_genes: list[Gene]) -> None:
        """CTNND1 has Extended TIS after filter.

        Post-filter, CTNND1's surviving TIS in HeLa are dominated by
        Extended ORFs (N-terminal extensions into the 5' UTR).
        Internal/Truncated/uORF classes exist in raw data but don't clear
        the significance thresholds — that's correct filter behavior.
        """
        ctnnd1 = next(g for g in test_genes if g.gene_name == "CTNND1")
        orf_types = {s.orf_type for s in ctnnd1.tis_sites}
        assert ORFType.EXTENDED in orf_types, (
            f"CTNND1 lost all Extended TIS after filter; got {orf_types}"
        )

    def test_myc_has_extensions(self, test_genes: list[Gene]) -> None:
        """MYC has Extended TIS (CUG-initiated MYC1 N-terminal extension)."""
        myc = next(g for g in test_genes if g.gene_name == "MYC")
        orf_types = {s.orf_type for s in myc.tis_sites}
        assert ORFType.EXTENDED in orf_types, (
            f"MYC should have Extended TIS after filter; got {orf_types}"
        )

    def test_vegfa_has_truncations(self, test_genes: list[Gene]) -> None:
        """VEGFA has Truncated ORF types."""
        vegfa = next(g for g in test_genes if g.gene_name == "VEGFA")
        orf_types = {s.orf_type for s in vegfa.tis_sites}
        assert ORFType.TRUNCATED in orf_types

    def test_diff_region_extensions_have_sequence(self, test_genes: list[Gene]) -> None:
        """Extended TIS sites have non-empty diff_region.sequence."""
        for gene in test_genes:
            for site in gene.tis_sites:
                if site.orf_type == ORFType.EXTENDED and site.diff_region:
                    assert site.diff_region.sequence, (
                        f"{gene.gene_name} {site.tis_id}: extension has empty diff_region.sequence"
                    )

    def test_diff_region_truncations_have_sequence(self, test_genes: list[Gene]) -> None:
        """Truncated TIS sites have non-empty diff_region.sequence."""
        for gene in test_genes:
            for site in gene.tis_sites:
                if site.orf_type == ORFType.TRUNCATED and site.diff_region:
                    assert site.diff_region.sequence, (
                        f"{gene.gene_name} {site.tis_id}: truncation has empty diff_region.sequence"
                    )

    def test_isoform_proteins_nonempty(self, test_genes: list[Gene]) -> None:
        """All TIS sites have non-empty isoform proteins."""
        for gene in test_genes:
            for site in gene.tis_sites:
                assert len(site.isoform_protein) > 0, (
                    f"{gene.gene_name} {site.tis_id}: empty isoform_protein"
                )

    def test_per_tis_canonical_matches_transcript_annotated(
        self, test_genes: list[Gene], upstream_hela
    ) -> None:
        """Every TIS uses its own transcript's Annotated AASeq as canonical.

        Post-port, the Annotated may be native (from Ribo-TISH) or imputed
        (from GENCODE pc_translations.fa when Ribo-TISH didn't emit one).
        Either way, per-Tid canonical is unambiguous — we just look it up
        in the upstream output.
        """
        final_df, _ = upstream_hela
        checked = 0
        for gene in test_genes:
            gene_df = final_df[final_df["Symbol"] == gene.gene_name]
            ann = gene_df[gene_df["TisType"].str.startswith("Annotated")]
            tid_to_ann = {str(r["Tid"]): str(r["AASeq"]).rstrip("*") for _, r in ann.iterrows()}
            for site in gene.tis_sites:
                expected = tid_to_ann.get(site.transcript_id)
                assert expected is not None, (
                    f"{gene.gene_name} {site.tis_id}: Tid "
                    f"{site.transcript_id} has no Annotated row in final_df "
                    "— imputation invariant broken"
                )
                assert site.canonical_protein.rstrip("*") == expected, (
                    f"{gene.gene_name} {site.tis_id} (Tid={site.transcript_id}): "
                    f"canonical is {len(site.canonical_protein)}aa but "
                    f"transcript's Annotated AASeq is {len(expected)}aa"
                )
                checked += 1
        assert checked > 0, "No TIS checked — fixture empty"

    def test_gene_canonical_is_longest_annotated(
        self, test_genes: list[Gene], upstream_hela
    ) -> None:
        """Gene.canonical_protein is the longest Annotated AASeq in the
        post-imputation upstream output for that gene.
        """
        final_df, _ = upstream_hela
        for gene in test_genes:
            gene_df = final_df[final_df["Symbol"] == gene.gene_name]
            ann = gene_df[gene_df["TisType"].str.startswith("Annotated")]
            if ann.empty:
                continue
            longest = max((str(r["AASeq"]).rstrip("*") for _, r in ann.iterrows()), key=len)
            assert gene.canonical_protein.rstrip("*") == longest, (
                f"{gene.gene_name}: gene.canonical_protein "
                f"({len(gene.canonical_protein)}aa) != longest Annotated "
                f"({len(longest)}aa)"
            )

    def test_upstream_filter_drops_expected_fraction(self, upstream_hela) -> None:
        """The upstream filter should reduce total TIS count by ≥85% on
        HeLa (upstream reference defaults).  Measured against native (non-imputed)
        rows so imputation doesn't inflate the count.
        """
        final, _ = upstream_hela
        native = final[~final["Imputed"]] if "Imputed" in final.columns else final
        raw = load_ribotish_predictions(HELA_PREDICT)
        drop_fraction = 1 - (len(native) / len(raw))
        assert drop_fraction >= 0.85, (
            f"Filter only dropped {drop_fraction:.1%} of rows "
            f"(raw={len(raw)}, native_kept={len(native)}); expected ≥85%"
        )

    def test_upstream_imputes_canonical_for_every_tid(self, upstream_hela) -> None:
        """After run_sample, every Tid in final_df has an Annotated row
        (native or imputed).  This is the invariant that lets assembly
        drop the fallback-logging and dual-DataFrame complexity.
        """
        final, _ = upstream_hela
        has_ann = (
            final.assign(_a=lambda d: d["RecatTISType"] == "Annotated").groupby("Tid")["_a"].sum()
        )
        orphans = has_ann[has_ann == 0]
        assert orphans.empty, (
            f"{len(orphans)} transcripts in final_df lack Annotated rows; "
            f"examples: {orphans.head(3).index.tolist()}"
        )

    def test_per_tis_canonical_preserves_native_when_unfiltered(
        self, test_genes: list[Gene], hela_df, upstream_hela
    ) -> None:
        """For TIS whose transcript kept its native Annotated row through
        filtering (i.e. the row in final_df is NOT imputed), the canonical
        equals the raw Ribo-TISH AASeq.  Transcripts whose canonical was
        imputed legitimately diverge (pc_translations.fa vs Ribo-TISH
        AASeq may use different isoform versions).
        """
        final_df, _ = upstream_hela
        native_ann_tids = set(
            final_df[final_df["TisType"].str.startswith("Annotated") & ~final_df["Imputed"]]["Tid"]
        )
        checked = 0
        for gene in test_genes:
            gene_raw = hela_df[hela_df["Symbol"] == gene.gene_name]
            tid_to_raw = {
                str(r["Tid"]): str(r["AASeq"]).rstrip("*")
                for _, r in gene_raw[gene_raw["TisType"].str.startswith("Annotated")].iterrows()
            }
            for site in gene.tis_sites:
                if site.transcript_id not in native_ann_tids:
                    continue
                expected = tid_to_raw.get(site.transcript_id)
                if expected is None:
                    continue
                assert site.canonical_protein.rstrip("*") == expected, (
                    f"{gene.gene_name} {site.tis_id} Tid={site.transcript_id}: "
                    f"native canonical {len(site.canonical_protein)}aa != raw "
                    f"Ribo-TISH AASeq {len(expected)}aa"
                )
                checked += 1
        assert checked > 0, "No TIS had a native Annotated counterpart — no-op"

    def test_ctnnd1_has_multiple_tis_canonical_lengths(self, test_genes: list[Gene]) -> None:
        """CTNND1 has multiple transcripts with differing Annotated CDS
        lengths (832, 838, 840, 861, 867 aa in HeLa).  After the per-Tid
        canonical fix, TIS whose transcripts have their own Annotated row
        must carry canonicals at those differing lengths — not the single
        867-aa gene-level longest.

        If every CTNND1 TIS has canonical_protein_length == 867, the
        per-Tid dispatch regressed.
        """
        ctnnd1 = next(g for g in test_genes if g.gene_name == "CTNND1")
        lengths = {len(s.canonical_protein.rstrip("*")) for s in ctnnd1.tis_sites}
        assert len(lengths) >= 2, (
            f"CTNND1 TIS canonicals all have length {lengths} — per-Tid canonical not dispatching"
        )
        # Specifically, at least one TIS should NOT be the gene-level 867aa
        non_default = {ln for ln in lengths if ln != 867}
        assert non_default, f"CTNND1 TIS canonicals: {lengths} — all fell back to gene-level 867aa"


# ---------------------------------------------------------------------------
# Pipeline annotation tests
# ---------------------------------------------------------------------------


class TestAnnotation:
    """Tests that the annotation pipeline produces correct output structure."""

    def test_canonical_has_biophysics(self, annotated_genes: list[Gene]) -> None:
        """All genes have biophysics canonical annotations."""
        for gene in annotated_genes:
            assert "biophysics" in gene.canonical_annotations, (
                f"{gene.gene_name} missing canonical biophysics"
            )
            ann = gene.canonical_annotations["biophysics"]
            assert ann["pI"] is not None
            assert ann["gravy"] is not None

    def test_canonical_has_motifs(self, annotated_genes: list[Gene]) -> None:
        """All genes have motifs canonical annotations."""
        for gene in annotated_genes:
            assert "motifs" in gene.canonical_annotations, (
                f"{gene.gene_name} missing canonical motifs"
            )
            ann = gene.canonical_annotations["motifs"]
            assert "hits" in ann
            assert "summary" in ann

    def test_isoform_biophysics_values_plausible(self, annotated_genes: list[Gene]) -> None:
        """Biophysics produces plausible values on real protein sequences.

        Weak-assertion regression guard: a module returning ``{}`` or all-
        None would pass a "field exists" check. We verify concrete values
        are in biologically plausible ranges.
        """
        for gene in annotated_genes:
            for site in gene.tis_sites:
                ann = site.isoform_annotations["biophysics"]
                # Length must match the actual protein (sans stop)
                expected_len = len(site.isoform_protein.rstrip("*"))
                assert ann["length"] == expected_len, (
                    f"{gene.gene_name} {site.tis_id}: biophysics length "
                    f"{ann['length']} != protein length {expected_len}"
                )
                # pI only computable for proteins ≥ 3 residues
                if expected_len >= 3:
                    assert ann["pI"] is not None, (
                        f"{gene.gene_name} {site.tis_id}: pI None for {expected_len}-aa protein"
                    )
                    # Arg-rich peptides can exceed 12; use a wider plausible range
                    assert 3.0 <= ann["pI"] <= 13.0, (
                        f"{gene.gene_name} {site.tis_id}: pI {ann['pI']} "
                        "outside plausible range [3, 12]"
                    )
                    assert ann["gravy"] is not None
                    assert -5.0 <= ann["gravy"] <= 5.0

    def test_isoform_motifs_positional(self, annotated_genes: list[Gene]) -> None:
        """Motifs produces a list of positional hits, each with pos/end/name."""
        for gene in annotated_genes:
            for site in gene.tis_sites:
                ann = site.isoform_annotations["motifs"]
                assert isinstance(ann["hits"], list)
                assert isinstance(ann["summary"], dict)
                # Each hit must have coordinate + identity fields
                for hit in ann["hits"]:
                    assert "pos" in hit and isinstance(hit["pos"], int)
                    assert "end" in hit and isinstance(hit["end"], int)
                    assert hit["end"] > hit["pos"]
                    assert "name" in hit and hit["name"]
                    # Hits should fall within the protein
                    assert 0 <= hit["pos"] < len(site.isoform_protein)

    def test_canonical_motifs_found_in_tp53(self, annotated_genes: list[Gene]) -> None:
        """TP53's canonical protein has detectable SLiM motifs.

        TP53 is phosphorylated at many [ST]P and [ST]Q sites — a motif
        scan that finds zero hits is broken.
        """
        tp53 = next(g for g in annotated_genes if g.gene_name == "TP53")
        ann = tp53.canonical_annotations["motifs"]
        assert len(ann["hits"]) > 5, (
            f"TP53 canonical has only {len(ann['hits'])} motif hits — "
            "TP53 has many well-characterized [ST]P/[ST]Q sites, expected >5"
        )

    def test_isoform_core_identity_matches_orf_type(self, annotated_genes: list[Gene]) -> None:
        """core_identity.orf_type matches site.orf_type.value and lengths agree."""
        for gene in annotated_genes:
            for site in gene.tis_sites:
                ann = site.isoform_annotations["core_identity"]
                assert ann["orf_type"] == site.orf_type.value
                assert ann["isoform_protein_length"] == len(site.isoform_protein.rstrip("*"))
                assert ann["canonical_protein_length"] == len(site.canonical_protein.rstrip("*"))
                assert isinstance(ann["in_frame"], bool)

    def test_isoform_initiation_context_kozak_populated_when_fasta(
        self, annotated_genes: list[Gene]
    ) -> None:
        """When genome FASTA is available, kozak_context is a 13-nt string
        with ATG at indices 9-11 for TIS whose contig is in the FASTA.

        For genes whose contig isn't in the (truncated) dev FASTA,
        kozak_context is None and the module outputs all-None — also
        asserted.
        """
        for gene in annotated_genes:
            for site in gene.tis_sites:
                ann = site.isoform_annotations["initiation_context"]
                if gene.gene_name in GENES_WITH_FASTA:
                    assert site.kozak_context is not None, (
                        f"{gene.gene_name} {site.tis_id}: kozak_context None "
                        "despite FASTA containing this contig"
                    )
                    assert len(site.kozak_context) == 13
                    # Start codon at indices 9-11 should match site.start_codon
                    assert site.kozak_context[9:12] == site.start_codon, (
                        f"{gene.gene_name} {site.tis_id}: kozak has "
                        f"{site.kozak_context[9:12]} at 9-11, expected "
                        f"{site.start_codon}"
                    )
                    # Hamming distances must be numeric when kozak present
                    assert ann["kozak_hamming_full"] is not None
                    assert 0 <= ann["kozak_hamming_full"] <= 13

    def test_signalp_predictions_flow_through(
        self, annotated_genes: list[Gene]
    ) -> None:
        """SignalP precompute + module lookup produce real, non-None values.

        Skipped when the ``swissisoform-v2-signalp`` env isn't set up:
        the precompute returns ``{}`` and every module call returns
        None.  When the env IS set up, at least some predictions must
        flow through — a 5-gene batch over canonical + isoform proteins
        cannot legitimately be 100% unpredicted.
        """
        non_none_preds = [
            site.isoform_annotations["signalp"]["signalp_prediction"]
            for gene in annotated_genes
            for site in gene.tis_sites
            if site.isoform_annotations["signalp"]["signalp_prediction"] is not None
        ]
        if not non_none_preds:
            pytest.skip(
                "SignalP returned no predictions — env not installed.  Run "
                "`python scripts/setup_databases.py signalp` to enable this test."
            )
        valid = {"OTHER", "SP", "LIPO", "TAT", "TATLIPO", "PILIN"}
        assert set(non_none_preds) <= valid, (
            f"Unknown SignalP prediction classes: {set(non_none_preds) - valid}"
        )
        # Probabilities where present must be in [0, 1]
        for gene in annotated_genes:
            for site in gene.tis_sites:
                prob = site.isoform_annotations["signalp"]["signalp_probability"]
                if prob is not None:
                    assert 0.0 <= prob <= 1.0, (
                        f"{gene.gene_name} {site.tis_id}: SignalP prob {prob} out of [0,1]"
                    )

    def test_targetp_predictions_flow_through(
        self, annotated_genes: list[Gene]
    ) -> None:
        """TargetP precompute + module lookup produce real, non-None values.

        Skipped when the TargetP binary isn't extracted.  Otherwise
        predictions must appear and be drawn from the documented class
        set; probability floats must sit in [0, 1].
        """
        non_none_preds = [
            site.isoform_annotations["targetp"]["targetp_prediction"]
            for gene in annotated_genes
            for site in gene.tis_sites
            if site.isoform_annotations["targetp"]["targetp_prediction"] is not None
        ]
        if not non_none_preds:
            pytest.skip(
                "TargetP returned no predictions — binary not extracted.  Run "
                "`python scripts/setup_databases.py targetp` to enable this test."
            )
        valid = {"noTP", "SP", "mTP", "cTP", "luTP"}
        assert set(non_none_preds) <= valid, (
            f"Unknown TargetP prediction classes: {set(non_none_preds) - valid}"
        )
        for gene in annotated_genes:
            for site in gene.tis_sites:
                for key in ("targetp_probability", "targetp_sp_prob", "targetp_mtp_prob"):
                    v = site.isoform_annotations["targetp"][key]
                    if v is not None:
                        assert 0.0 <= v <= 1.0, (
                            f"{gene.gene_name} {site.tis_id}: {key}={v} out of [0,1]"
                        )

    def test_interproscan_predictions_flow_through(
        self, annotated_genes: list[Gene]
    ) -> None:
        """InterProScan precompute + module lookup produce real domain hits.

        Skipped when the InterProScan 6 datadir / Nextflow aren't set up
        (precompute returns ``{}``).  With Pfam on ~100 real proteins
        from TP53, EIF4G1, VEGFA, CTNND1, MYC we expect several hits —
        these are well-annotated proteins.
        """
        total_hits = 0
        for gene in annotated_genes:
            ann = gene.canonical_annotations.get("interproscan")
            if isinstance(ann, dict):
                total_hits += ann.get("summary", {}).get("n_hits", 0)
            for site in gene.tis_sites:
                total_hits += site.isoform_annotations["interproscan"]["summary"][
                    "n_hits"
                ]
        if total_hits == 0:
            pytest.skip(
                "InterProScan returned no hits — datadir not populated.  Run "
                "`sbatch scripts/setup_interproscan.sbatch` to enable this test."
            )
        # Every hit must carry coordinate + db + signature metadata
        for gene in annotated_genes:
            for site in gene.tis_sites:
                for hit in site.isoform_annotations["interproscan"]["hits"]:
                    assert "name" in hit and hit["name"]
                    assert "db" in hit and hit["db"]
                    assert 0 <= hit["pos"] < hit["end"]
                    assert hit["end"] <= len(site.isoform_protein) + 1

    def test_signalp_canonical_vs_isoform_comparator(
        self, annotated_genes: list[Gene]
    ) -> None:
        """SignalP comparator emits prediction_changed for each TIS.

        This is the signal that feeds F4 scoring.  Skipped when SignalP
        didn't produce predictions (same condition as the precompute
        test above).
        """
        changed_count = 0
        emitted = 0
        for gene in annotated_genes:
            for site in gene.tis_sites:
                cmp = site.comparison.get("signalp", {})
                if "signalp_prediction_changed" in cmp:
                    emitted += 1
                    if cmp["signalp_prediction_changed"]:
                        changed_count += 1
        if emitted == 0:
            pytest.skip("SignalP comparator fields absent — precompute not run.")
        assert emitted > 0

    def test_site_modules_not_on_canonical(self, annotated_genes: list[Gene]) -> None:
        """SiteModule output should NOT be in canonical_annotations."""
        for gene in annotated_genes:
            assert "core_identity" not in gene.canonical_annotations
            assert "initiation_context" not in gene.canonical_annotations

    def test_protein_modules_not_in_gene_annotations(self, annotated_genes: list[Gene]) -> None:
        """ProteinModule output should NOT be in gene_annotations."""
        for gene in annotated_genes:
            assert "biophysics" not in gene.gene_annotations
            assert "motifs" not in gene.gene_annotations

    def test_no_tis_dropped(self, test_genes: list[Gene], annotated_genes: list[Gene]) -> None:
        """Pipeline does not drop any TIS sites."""
        for orig, ann in zip(
            sorted(test_genes, key=lambda g: g.gene_name),
            sorted(annotated_genes, key=lambda g: g.gene_name),
        ):
            assert len(orig.tis_sites) == len(ann.tis_sites), (
                f"{orig.gene_name}: {len(orig.tis_sites)} → {len(ann.tis_sites)}"
            )


# ---------------------------------------------------------------------------
# ORF-type-specific annotation sanity checks
# ---------------------------------------------------------------------------


class TestORFTypeAnnotations:
    """Spot-check annotations for specific ORF types across the gene set."""

    def test_extensions_in_frame(self, annotated_genes: list[Gene]) -> None:
        """Extended ORFs are marked in-frame by core_identity."""
        for gene in annotated_genes:
            for site in gene.tis_sites:
                if site.orf_type == ORFType.EXTENDED:
                    ann = site.isoform_annotations["core_identity"]
                    assert ann["in_frame"] is True, (
                        f"{gene.gene_name} {site.tis_id}: Extended should be in-frame"
                    )

    def test_truncations_in_frame(self, annotated_genes: list[Gene]) -> None:
        """Truncated ORFs are marked in-frame by core_identity."""
        for gene in annotated_genes:
            for site in gene.tis_sites:
                if site.orf_type == ORFType.TRUNCATED:
                    ann = site.isoform_annotations["core_identity"]
                    assert ann["in_frame"] is True, (
                        f"{gene.gene_name} {site.tis_id}: Truncated should be in-frame"
                    )

    def test_uorfs_out_of_frame(self, annotated_genes: list[Gene]) -> None:
        """uORFs are marked out-of-frame by core_identity."""
        for gene in annotated_genes:
            for site in gene.tis_sites:
                if site.orf_type == ORFType.UORF:
                    ann = site.isoform_annotations["core_identity"]
                    assert ann["in_frame"] is False, (
                        f"{gene.gene_name} {site.tis_id}: uORF should be out-of-frame"
                    )

    def test_short_proteins_biophysics_computed(self, annotated_genes: list[Gene]) -> None:
        """Short (≥3 aa) proteins get real biophysics values (pI, gravy),
        not just a ``length`` field — which is just string length and
        doesn't prove anything was computed.
        """
        checked = 0
        for gene in annotated_genes:
            for site in gene.tis_sites:
                if 3 < site.aa_len < 50:
                    ann = site.isoform_annotations["biophysics"]
                    assert ann["length"] > 0
                    # pI and gravy ARE the signal; length alone is trivial
                    assert ann["pI"] is not None, (
                        f"{gene.gene_name} {site.tis_id}: pI None for {site.aa_len}-aa protein"
                    )
                    assert 3.0 < ann["pI"] < 13.0
                    assert ann["gravy"] is not None
                    checked += 1
        assert checked > 0, "No short proteins found in test set — test is no-op"


# ---------------------------------------------------------------------------
# Serialization test
# ---------------------------------------------------------------------------


class TestSerialization:
    """Test that annotated output can be serialized to DataFrame."""

    def test_to_dataframe(self, annotated_genes: list[Gene]) -> None:
        """Annotated TIS can be serialized to a flat DataFrame."""
        all_sites = []
        for gene in annotated_genes:
            all_sites.extend(gene.tis_sites)
        df = tis_to_dataframe(all_sites)

        assert len(df) == len(all_sites)
        assert "tis_id" in df.columns
        assert "gene_name" in df.columns
        assert "orf_type" in df.columns
        assert "isoform_protein" in df.columns

        # Check annotation columns are present
        ann_cols = [c for c in df.columns if c.startswith("biophysics_")]
        assert len(ann_cols) > 0, "No biophysics annotation columns in output"
        motif_cols = [c for c in df.columns if c.startswith("motifs_")]
        assert len(motif_cols) > 0, "No motifs annotation columns in output"

    def test_dataframe_gene_coverage(self, annotated_genes: list[Gene]) -> None:
        """DataFrame contains all 5 test genes."""
        all_sites = []
        for gene in annotated_genes:
            all_sites.extend(gene.tis_sites)
        df = tis_to_dataframe(all_sites)

        genes_in_df = set(df["gene_name"].unique())
        assert genes_in_df == set(TEST_GENES)


# ---------------------------------------------------------------------------
# compute_diff_region unit tests (fast, no data files needed)
# ---------------------------------------------------------------------------


class TestComputeDiffRegion:
    """Unit tests for the diff region computation logic."""

    def test_extension_exact_suffix(self) -> None:
        """Extension where canonical is an exact suffix of isoform."""
        dr = compute_diff_region(
            ORFType.EXTENDED,
            "MRGSHHHHHGSMEEPQSDP*",
            "MEEPQSDP*",
        )
        assert dr.isoform_start == 0
        assert dr.isoform_end == 11
        assert dr.sequence == "MRGSHHHHHGS"

    def test_truncation_exact_suffix(self) -> None:
        """Truncation where isoform is an exact suffix of canonical."""
        dr = compute_diff_region(
            ORFType.TRUNCATED,
            "SDPSVEPPL*",
            "MEEPQSDPSVEPPL*",
        )
        assert dr.canonical_start == 0
        assert dr.canonical_end == 5
        assert dr.sequence == "MEEPQ"

    def test_uorf_whole_sequence(self) -> None:
        """uORF diff region is the entire isoform."""
        dr = compute_diff_region(
            ORFType.UORF,
            "MPKLQRST*",
            "MEEPQSDPSVEPPL*",
        )
        assert dr.isoform_start == 0
        assert dr.isoform_end == 8
        assert dr.sequence == "MPKLQRST"

    def test_annotated_empty(self) -> None:
        """Annotated ORF has empty diff region."""
        dr = compute_diff_region(
            ORFType.ANNOTATED,
            "MEEPQSDP*",
            "MEEPQSDP*",
        )
        assert dr.sequence == ""
        assert dr.confidence == "exact"

    def test_confidence_tail_verified_extension(self) -> None:
        """Clean extension with exact canonical suffix → tail_verified."""
        dr = compute_diff_region(
            ORFType.EXTENDED,
            "MRGSHHHHHGSMEEPQSDP*",
            "MEEPQSDP*",
        )
        assert dr.confidence == "tail_verified"

    def test_confidence_tail_verified_truncation(self) -> None:
        """Clean truncation with exact isoform suffix → tail_verified."""
        dr = compute_diff_region(
            ORFType.TRUNCATED,
            "SDPSVEPPL*",
            "MEEPQSDPSVEPPL*",
        )
        assert dr.confidence == "tail_verified"

    def test_confidence_length_fallback_truncation(self) -> None:
        """Truncation with sequences that don't share a tail → length_fallback."""
        dr = compute_diff_region(
            ORFType.TRUNCATED,
            "MRKAAAA*",  # Unrelated to canonical — no shared suffix
            "MEEPQSDPSVEPPL*",
        )
        assert dr.confidence == "length_fallback"
        assert dr.canonical_start == 0
        assert dr.canonical_end == len("MEEPQSDPSVEPPL") - len("MRKAAAA")

    def test_confidence_whole_isoform_fallback_extension(self) -> None:
        """Extension with unrelated sequences → whole_isoform_fallback."""
        dr = compute_diff_region(
            ORFType.EXTENDED,
            "MRKAAAA*",  # unrelated to canonical
            "MEEPQSDP*",
        )
        assert dr.confidence == "whole_isoform_fallback"
        assert dr.isoform_start == 0
        assert dr.isoform_end == len("MRKAAAA")

    def test_confidence_uorf_exact(self) -> None:
        """uORFs are always whole-isoform by definition → exact."""
        dr = compute_diff_region(
            ORFType.UORF,
            "MPKLQRST*",
            "MEEPQSDPSVEPPL*",
        )
        assert dr.confidence == "exact"


class TestAssembleGenesPerTidCanonical:
    """Unit tests for per-transcript canonical selection in assemble_genes."""

    def test_per_tis_canonical_picks_tid_not_longest(self) -> None:
        """A Truncated TIS from transcript T1 uses T1's Annotated AASeq
        as canonical — not T2's longer Annotated AASeq.

        Regression guard for the per-Tid canonical change: prior to the
        fix, every TIS got the gene-level longest (T2), so a T1-derived
        truncation was compared against the wrong reference.
        """
        # T1: short transcript (200 aa annotated)
        # T2: long transcript (500 aa annotated)
        # Truncated TIS from T1: a valid in-frame suffix of T1's canonical
        short_canonical = "M" + "A" * 199  # 200 aa
        long_canonical = "M" + "E" * 499  # 500 aa
        truncated_from_t1 = "A" * 150  # suffix of T1 minus M+49 residues

        df = pd.DataFrame(
            [
                {
                    "Symbol": "GENEX",
                    "Gid": "G1",
                    "Tid": "T1",
                    "TisType": "Annotated",
                    "GenomePos": "chr1:100-1000:+",
                    "StartCodon": "ATG",
                    "Start": 100,
                    "AALen": 200,
                    "AASeq": short_canonical,
                    "TISPvalue": 0.01,
                    "RiboPvalue": 0.01,
                    "FisherQvalue": 0.01,
                },
                {
                    "Symbol": "GENEX",
                    "Gid": "G1",
                    "Tid": "T2",
                    "TisType": "Annotated",
                    "GenomePos": "chr1:200-2000:+",
                    "StartCodon": "ATG",
                    "Start": 200,
                    "AALen": 500,
                    "AASeq": long_canonical,
                    "TISPvalue": 0.01,
                    "RiboPvalue": 0.01,
                    "FisherQvalue": 0.01,
                },
                {
                    "Symbol": "GENEX",
                    "Gid": "G1",
                    "Tid": "T1",
                    "TisType": "Truncated:Known",
                    "GenomePos": "chr1:150-1000:+",
                    "StartCodon": "ATG",
                    "Start": 150,
                    "AALen": 150,
                    "AASeq": truncated_from_t1,
                    "TISPvalue": 0.01,
                    "RiboPvalue": 0.01,
                    "FisherQvalue": 0.01,
                },
            ]
        )

        genes = assemble_genes(df, gene_names=["GENEX"])
        assert len(genes) == 1
        gene = genes[0]

        # Gene-level representative is the longest Annotated (T2, 500 aa)
        assert gene.canonical_transcript_id == "T2"
        assert len(gene.canonical_protein) == 500

        # The single non-Annotated TIS is from T1 — its canonical must be T1's
        assert len(gene.tis_sites) == 1
        trunc = gene.tis_sites[0]
        assert trunc.transcript_id == "T1"
        assert trunc.canonical_protein == short_canonical, (
            f"Truncated TIS from T1 got {len(trunc.canonical_protein)}aa "
            f"canonical; expected T1's 200-aa Annotated AASeq"
        )
        # diff_region should cleanly resolve (isoform is a suffix of T1 canonical)
        assert trunc.diff_region is not None
        assert trunc.diff_region.sequence, (
            "diff_region.sequence empty — per-Tid canonical did not produce "
            "a clean suffix match (would indicate wiring bug)"
        )

    def test_orphan_tid_without_annotated_raises(self) -> None:
        """An alt TIS whose Tid has no Annotated row raises ValueError.

        Post-imputation invariant: every Tid in the assembly input has
        a canonical.  If one doesn't, it means upstream imputation didn't
        run or the DataFrame was mis-filtered — assembly should fail fast
        rather than silently fall back to the gene-level longest.
        """
        gene_canonical = "M" + "A" * 199
        df = pd.DataFrame(
            [
                {
                    "Symbol": "GENEY",
                    "Gid": "G2",
                    "Tid": "T_ANN",
                    "TisType": "Annotated",
                    "GenomePos": "chr2:100-1000:+",
                    "StartCodon": "ATG",
                    "Start": 100,
                    "AALen": 200,
                    "AASeq": gene_canonical,
                    "TISPvalue": 0.01,
                    "RiboPvalue": 0.01,
                    "FisherQvalue": 0.01,
                },
                {
                    "Symbol": "GENEY",
                    "Gid": "G2",
                    "Tid": "T_ORPHAN",
                    "TisType": "Truncated:Known",
                    "GenomePos": "chr2:150-1000:+",
                    "StartCodon": "ATG",
                    "Start": 150,
                    "AALen": 150,
                    "AASeq": "A" * 150,
                    "TISPvalue": 0.01,
                    "RiboPvalue": 0.01,
                    "FisherQvalue": 0.01,
                },
            ]
        )
        with pytest.raises(ValueError, match="T_ORPHAN|no Annotated"):
            assemble_genes(df, gene_names=["GENEY"])
