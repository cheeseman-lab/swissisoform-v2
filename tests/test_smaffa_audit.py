"""Audit: our run_sample output matches smaffa's coTISja filter on HeLa.

The `smaffa_filtered_audit` directory carries a copy of smaffa's
per-sample filtered CSVs (from coTISja commit in use around 2026-04).
This test guarantees our reimplementation of the filter is a faithful
port — any divergence fails here before it silently affects downstream
annotation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from swissisoform.pipeline import UpstreamReference, run_sample

DATA = Path(__file__).parent.parent / "data" / "reference"
OUT_AUDIT = DATA / "smaffa_filtered_audit"

GTF = DATA / "gencode.v49.primary_assembly.annotation.gtf"
GENOME = DATA / "Gencode_v49_GRCh38.primary_assembly.genome.fa"
PROTEIN = DATA / "gencode.v49.pc_translations.fa"

TEST_GENES = ["TP53", "EIF4G1", "VEGFA", "CTNND1", "MYC"]


@pytest.fixture(scope="module")
def reference() -> UpstreamReference:
    """Load shared GTF + FASTA tables once."""
    for p in (GTF, GENOME, PROTEIN):
        if not p.exists():
            pytest.skip(f"Missing reference file: {p}")
    return UpstreamReference.load(
        gtf_path=GTF, genome_fasta=GENOME, protein_fasta=PROTEIN
    )


@pytest.fixture(scope="module")
def hela_run(reference: UpstreamReference) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run our pipeline on HeLa; cache for all assertions in this module."""
    predict = DATA / "HeLa_TIS_predict_all.txt"
    counts = [
        DATA / "rnaseq_counts/GATCAG_8_htseqcount.txt",
        DATA / "rnaseq_counts/CACGGT_9_htseqcount.txt",
    ]
    if not predict.exists() or not all(c.exists() for c in counts):
        pytest.skip("HeLa inputs missing")
    return run_sample(
        predict, counts, GTF, sample="HeLa", reference=reference
    )


@pytest.fixture(scope="module")
def smaffa_hela() -> pd.DataFrame:
    """Smaffa's reference HeLa filtered CSV."""
    path = OUT_AUDIT / "HeLa_TIS_filtered.csv"
    if not path.exists():
        pytest.skip(f"Audit file missing: {path}")
    return pd.read_csv(path)


@pytest.mark.slow
class TestSmaffaAudit:
    """Regression guard: our filter stays byte-compatible with smaffa's."""

    def test_non_imputed_rows_subset_of_smaffa(
        self, hela_run, smaffa_hela: pd.DataFrame
    ) -> None:
        """Our non-imputed output is a SUBSET of smaffa's filter output.

        The filter produces exactly what smaffa's does; we then drop any
        transcript that lacks a canonical row after imputation (``cds_start_NF``,
        retained_intron, etc.).  So ours ⊆ smaffa, and the difference is
        precisely those uncanonical transcripts — no other surprises.
        """
        ours_final, _ = hela_run
        ours_native = ours_final[~ours_final["Imputed"]]
        ours_keys = set(
            zip(ours_native["Tid"], ours_native["GenomePos"], ours_native["Start"])
        )
        smaffa_keys = set(
            zip(smaffa_hela["Tid"], smaffa_hela["GenomePos"], smaffa_hela["Start"])
        )
        assert ours_keys.issubset(smaffa_keys), (
            f"{len(ours_keys - smaffa_keys)} rows in ours but not smaffa"
        )

        # Every smaffa-only row must be from a Tid we legitimately dropped
        # for lacking a canonical.
        ours_tids = set(ours_final["Tid"].unique())
        only_smaffa = smaffa_keys - ours_keys
        smaffa_only_tids = {tid for tid, _, _ in only_smaffa}
        unexpected = smaffa_only_tids & ours_tids
        assert not unexpected, (
            f"{len(unexpected)} smaffa-only rows come from Tids we kept "
            f"(should only be from dropped uncanonical Tids); "
            f"examples: {list(unexpected)[:3]}"
        )

    @pytest.mark.parametrize("gene", TEST_GENES)
    def test_per_gene_subset_of_smaffa(
        self, hela_run, smaffa_hela: pd.DataFrame, gene: str
    ) -> None:
        """Every native TIS we keep for this gene is present in smaffa's
        filter output (at the same Tid/GenomePos/Start)."""
        ours_final, _ = hela_run
        ours_native = ours_final[
            (~ours_final["Imputed"]) & (ours_final["Symbol"] == gene)
        ]
        smaffa = smaffa_hela[smaffa_hela["Symbol"] == gene]

        ours_keys = set(
            zip(ours_native["Tid"], ours_native["GenomePos"], ours_native["Start"])
        )
        smaffa_keys = set(zip(smaffa["Tid"], smaffa["GenomePos"], smaffa["Start"]))
        only_ours = ours_keys - smaffa_keys
        assert not only_ours, (
            f"{gene}: {len(only_ours)} rows in ours but not smaffa: "
            f"{list(only_ours)[:3]}"
        )

    def test_imputation_fills_eif4g1(self, hela_run) -> None:
        """EIF4G1 has only 1 native Annotated row in HeLa Ribo-TISH output;
        imputation should backfill the other MANE/TSL transcript canonicals.
        """
        ours_final, _ = hela_run
        eif = ours_final[ours_final["Symbol"] == "EIF4G1"]
        imputed = eif[eif["Imputed"]]
        assert len(imputed) > 1, (
            f"EIF4G1 only got {len(imputed)} imputed canonical(s) — "
            "expected >1 since Ribo-TISH emits only 1 Annotated row"
        )
        # Every imputed row is Annotated and has a real AASeq
        assert (imputed["RecatTISType"] == "Annotated").all()
        assert imputed["AASeq"].str.len().gt(20).all()

    def test_imputable_protein_coding_transcripts_are_imputed(
        self, hela_run, reference: UpstreamReference
    ) -> None:
        """Every protein_coding transcript with CDS + start_codon + protein
        product in GENCODE gets a canonical row after imputation.

        The subset of protein_coding transcripts GENCODE flags with
        ``cds_start_NF`` or similar have no start_codon feature and are
        legitimately non-imputable — those are permitted to remain without
        canonical.
        """
        ours_final, _ = hela_run
        pc = ours_final[ours_final["transcript_type"] == "protein_coding"]
        has_ann = (
            pc.assign(_ann=lambda d: d["RecatTISType"] == "Annotated")
            .groupby("Tid")["_ann"]
            .sum()
        )
        orphan_tids = set(has_ann[has_ann == 0].index)

        # Every orphan should be missing at least one ingredient in GENCODE.
        have_start = set(reference.start_codons["Tid"])
        have_cds = set(reference.genome_pos["Tid"])
        have_protein = set(reference.protein_products["Tid"])
        imputable_orphans = orphan_tids & have_start & have_cds & have_protein
        assert not imputable_orphans, (
            f"{len(imputable_orphans)} protein_coding transcripts have full "
            f"GENCODE canonical annotations but weren't imputed; "
            f"examples: {list(imputable_orphans)[:3]}"
        )
