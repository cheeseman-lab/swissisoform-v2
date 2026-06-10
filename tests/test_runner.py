"""Tests for swissisoform.runner public API (prepare / annotate / run).

Exercises the three-function pipeline orchestration layer on the 5-gene
HeLa diagnostic preset.  All three tests share one PreparedRun via a
module-scoped fixture to pay the ~1-minute ref-load + assembly cost once.

Heavy/network-dependent modules are skipped via the ``skip`` set so the
suite runs with no external tools or network access.

Marker note: the project defines only ``gpu`` and ``network`` markers;
there is no ``slow`` marker.  These tests are in the default run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from swissisoform.runner import PreparedRun, RunSpec, annotate, prepare, run

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_GENES = ["TP53", "EIF4G1", "VEGFA", "CTNND1", "MYC"]

# Skip every module that requires network, GPU precompute, or external tools.
# Keeps only biophysics + motifs (pure-Python, inline, ~seconds).
FAST_SKIP: set[str] = {
    "plm_vep",
    "structure",
    "interproscan",
    "signalp",
    "targetp",
    "clinical",
    "conservation",
    "conservation_frame",
    "massspec",
    "variant_intersection",
    "generef",
    "localization",
}


# ---------------------------------------------------------------------------
# Shared module-scoped fixture — pay the ref-load + assembly cost once
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fast_spec(tmp_path_factory) -> RunSpec:
    """RunSpec for the 5-gene HeLa preset with only biophysics+motifs active."""
    out_dir = tmp_path_factory.mktemp("runner_out")
    fasta_out = out_dir / "proteins.fa"
    return RunSpec(
        gene_names=TEST_GENES,
        restricted_df=None,
        cell_lines=["HeLa"],
        single_sample=True,
        min_cell_lines=1,
        skip=FAST_SKIP,
        run_name="test_runner",
        fasta_out=fasta_out,
        emit_fasta=False,
        out_dir=out_dir,
    )


@pytest.fixture(scope="module")
def prepared(fast_spec: RunSpec) -> PreparedRun:
    """PreparedRun from the 5-gene HeLa preset (shared across tests)."""
    return prepare(fast_spec)


# ---------------------------------------------------------------------------
# Test 1: prepare() assembles genes + writes FASTA
# ---------------------------------------------------------------------------


def test_prepare_assembles_genes(prepared: PreparedRun, fast_spec: RunSpec) -> None:
    """prepare() assembles ≥5 genes, collects proteins, and writes the FASTA."""
    assert len(prepared.genes) >= 5, (
        f"Expected ≥5 assembled genes; got {len(prepared.genes)}"
    )
    gene_names = {g.gene_name for g in prepared.genes}
    assert set(TEST_GENES) == gene_names

    assert len(prepared.all_proteins) > 0, "all_proteins should be non-empty after assembly"

    assert fast_spec.fasta_out.exists(), (
        f"FASTA output not written to {fast_spec.fasta_out}"
    )
    fasta_text = fast_spec.fasta_out.read_text()
    # At minimum one FASTA header line should be present
    assert fasta_text.startswith(">"), "FASTA file does not start with a header line"


# ---------------------------------------------------------------------------
# Test 2: annotate() returns a scored paired DataFrame
# ---------------------------------------------------------------------------


def test_annotate_returns_scored_paired(prepared: PreparedRun, fast_spec: RunSpec) -> None:
    """annotate() returns a non-empty DataFrame with identity + scoring columns."""
    df = annotate(prepared, fast_spec)

    assert not df.empty, "annotate() returned an empty DataFrame"
    assert "gene_name" in df.columns, "DataFrame missing 'gene_name' column"
    assert "tis_id" in df.columns, "DataFrame missing 'tis_id' column"

    # EvidenceScoringModule writes into isoform_annotations["scoring"] which
    # _flatten_annotations expands to isoform_scoring_* columns.
    scoring_cols = [c for c in df.columns if c.startswith("isoform_scoring_")]
    assert scoring_cols, (
        "No isoform_scoring_* columns found — EvidenceScoringModule output missing. "
        f"Available columns (first 40): {list(df.columns[:40])}"
    )
    # Specifically assert the primary score column is present and has values
    assert "isoform_scoring_existence_score" in df.columns, (
        "Column 'isoform_scoring_existence_score' not found in DataFrame"
    )
    # Scores must be numeric (int or float), not all-None
    non_null = df["isoform_scoring_existence_score"].dropna()
    assert len(non_null) > 0, (
        "'isoform_scoring_existence_score' is entirely null — scoring did not run"
    )


# ---------------------------------------------------------------------------
# Test 3: run() with emit_fasta=True exits early (no parquet written)
# ---------------------------------------------------------------------------


def test_run_emit_fasta_early_exit(tmp_path: Path) -> None:
    """run() with emit_fasta=True returns 0 and skips annotation (no parquet)."""
    fasta_out = tmp_path / "proteins.fa"
    out_dir = tmp_path / "output"
    spec = RunSpec(
        gene_names=TEST_GENES,
        restricted_df=None,
        cell_lines=["HeLa"],
        single_sample=True,
        min_cell_lines=1,
        skip=FAST_SKIP,
        run_name="test_emit_fasta",
        fasta_out=fasta_out,
        emit_fasta=True,
        out_dir=out_dir,
    )

    exit_code = run(spec)

    assert exit_code == 0, f"run() returned {exit_code}; expected 0 for emit_fasta early exit"
    assert fasta_out.exists(), f"FASTA file not written to {fasta_out}"

    paired_parquet = out_dir / "all_paired.parquet"
    assert not paired_parquet.exists(), (
        "all_paired.parquet was written despite emit_fasta=True "
        "(early-exit path should stop before annotation)"
    )
