"""The ORF index builder's length guard.

``extract_cds`` is the last place a walk/length disagreement can be caught before
it becomes a wrong residue number on the website, so the guard covers *both*
sequences it extracts — and each against the protein it actually encodes.

Loaded by path: ``scripts/export`` is not a package, and the builder is a script.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pyarrow as pa
import pytest

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "build_orf_index", ROOT / "scripts" / "export" / "build_orf_index.py"
)
build_orf_index = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_orf_index)

ORF_EXONS = [[100, 130]]
CANON_EXONS = [[100, 190]]


class _StubValidator:
    """Returns a fixed sequence per exon set, so no genome FASTA is needed."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def build_coding_sequence_from_orf(self, exons, strand, chrom):  # noqa: ARG002
        return "A" * sum(end - start for start, end in exons)


def _table(*, isoform_len: int, per_tid_len: int | None) -> pa.Table:
    cols = {
        "tis_id": pa.array(["chr1:101:+:ATG:T1"]),
        "chrom": pa.array(["chr1"]),
        "strand": pa.array(["+"]),
        "orf_exons": pa.array([ORF_EXONS], pa.list_(pa.list_(pa.int64()))),
        "canonical_orf_exons": pa.array([CANON_EXONS], pa.list_(pa.list_(pa.int64()))),
        "isoform_len": pa.array([isoform_len], pa.int64()),
    }
    if per_tid_len is not None:
        cols["canonical_per_tid_length"] = pa.array([per_tid_len], pa.int64())
    return pa.table(cols)


@pytest.fixture(autouse=True)
def _stub_validator(monkeypatch):
    monkeypatch.setattr(
        "swissisoform.clinical.validate.ConsequenceValidator", _StubValidator
    )


def test_both_sequences_pass_when_the_lengths_agree() -> None:
    # 30 nt / 3 = 10 residues; 90 nt / 3 = 30 residues.
    table, mismatched = build_orf_index.extract_cds(
        _table(isoform_len=10, per_tid_len=30), Path("unused.fa")
    )
    assert mismatched == 0
    assert len(table.column("orf_cds")[0].as_py()) == 30
    assert len(table.column("canonical_cds")[0].as_py()) == 90


def test_canonical_cds_is_checked_against_the_per_tid_canonical() -> None:
    """The gap this closes: canonical_cds used to be appended unvalidated.

    A truncation's lost N-terminus is classified in canonical frame against this
    sequence, so a walk that disagrees with the recorded length mis-numbers exactly
    the variants the canonical branch exists to place.
    """
    _table_, mismatched = build_orf_index.extract_cds(
        _table(isoform_len=10, per_tid_len=29), Path("unused.fa")
    )
    assert mismatched == 1


def test_canonical_cds_is_not_checked_against_the_gene_level_length() -> None:
    """``canonical_orf_exons`` describes the per-Tid canonical, not ``canonical_len``.

    Checking it against the gene-level length would fail 1,670 of 6,462 real ORFs.
    """
    _table_, mismatched = build_orf_index.extract_cds(
        _table(isoform_len=10, per_tid_len=30), Path("unused.fa")
    )
    assert mismatched == 0


def test_orf_cds_is_still_checked() -> None:
    _table_, mismatched = build_orf_index.extract_cds(
        _table(isoform_len=11, per_tid_len=30), Path("unused.fa")
    )
    assert mismatched == 1


def test_missing_per_tid_length_warns_instead_of_silently_skipping(caplog) -> None:
    """An older all_paired has no such column — say so rather than pass quietly."""
    with caplog.at_level("WARNING"):
        _table_, mismatched = build_orf_index.extract_cds(
            _table(isoform_len=10, per_tid_len=None), Path("unused.fa")
        )
    assert mismatched == 0
    assert "canonical_per_tid_length is absent" in caplog.text
