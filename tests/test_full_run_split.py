"""Chunk-to-GPU-queue split for the genome-wide campaign.

``embed_split`` decides the boundary between the two GPU embed queues. Getting
it wrong does not degrade gracefully: an inverted range makes ``sbatch`` reject
the submission, which under ``set -e`` kills the whole Phase A chain before
anything downstream is submitted. The single-chunk case is pinned here because
it is the smoke-test shape, and it is the one the old clamp got wrong.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SPLIT_PY = Path(__file__).resolve().parents[1] / "scripts" / "slurm" / "full_run" / "split.py"


def _load_split():
    spec = importlib.util.spec_from_file_location("full_run_split", SPLIT_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


split = _load_split()


@pytest.mark.parametrize("n_chunks", range(1, 65))
def test_ranges_are_valid_disjoint_and_covering(n_chunks: int) -> None:
    """A6000 [0,k) and A100 [k,C) must tile the chunks exactly once."""
    k = split.embed_split(n_chunks)
    assert 1 <= k <= n_chunks, f"C={n_chunks}: k={k} outside [1, {n_chunks}]"
    a6000 = list(range(0, k))
    a100 = list(range(k, n_chunks))
    assert a6000, "the A6000 queue must never get an empty range"
    assert not set(a6000) & set(a100), "queues must not overlap (duplicate GPU work)"
    assert a6000 + a100 == list(range(n_chunks)), "every chunk must be embedded once"


@pytest.mark.parametrize("n_chunks", range(1, 65))
def test_sbatch_array_expressions_are_well_formed(n_chunks: int) -> None:
    """The literal `--array=lo-hi` strings 00_prepare builds must never invert."""
    k = split.embed_split(n_chunks)
    assert 0 <= k - 1, f"C={n_chunks}: A6000 range would be 0-{k - 1}"
    if k < n_chunks:  # A100 is submitted only when its range is non-empty
        assert k <= n_chunks - 1, f"C={n_chunks}: A100 range would be {k}-{n_chunks - 1}"


def test_single_chunk_puts_everything_on_one_queue() -> None:
    """The regression: C==1 used to clamp to 0 and emit --array=0--1."""
    assert split.embed_split(1) == 1  # A6000 gets chunk 0, A100 range is empty


def test_two_chunks_split_one_each() -> None:
    assert split.embed_split(2) == 1


def test_full_catalog_shape_is_unchanged() -> None:
    """The production run (8 chunks) must keep its historical 3/5 split."""
    assert split.embed_split(8) == 3


def test_large_runs_follow_the_three_sevenths_ratio() -> None:
    for n_chunks in (7, 14, 21, 70):
        assert split.embed_split(n_chunks) == n_chunks * 3 // 7


def test_zero_chunks_is_rejected() -> None:
    with pytest.raises(ValueError):
        split.embed_split(0)
