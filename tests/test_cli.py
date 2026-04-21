"""Tests for the CLI (``python -m swissisoform``)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from swissisoform.cli import (
    _annotate_genes_serial,
    _build_parser,
    _chunk_genes,
    _load_replicate_manifest,
    _load_sample_manifest,
    main,
)
from swissisoform.models import Gene


def test_parser_requires_subcommand():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_run_subcommand_accepts_required_args():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "run",
            "--sample-manifest",
            "m.csv",
            "--replicate-manifest",
            "r.csv",
            "--gtf",
            "x.gtf",
            "--genome",
            "x.fa",
            "--protein-fasta",
            "p.fa",
            "--output",
            "out/",
        ]
    )
    assert args.command == "run"
    assert args.workers == 8  # default
    assert args.sample_manifest == Path("m.csv")


def test_parser_run_subcommand_accepts_optional_flags():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "run",
            "--sample-manifest",
            "m.csv",
            "--replicate-manifest",
            "r.csv",
            "--gtf",
            "x.gtf",
            "--genome",
            "x.fa",
            "--protein-fasta",
            "p.fa",
            "--output",
            "out/",
            "--workers",
            "4",
            "--samples",
            "HeLa",
            "K562",
            "--genes",
            "TP53",
            "MYC",
            "--deeploc",
            "--no-comparator",
            "-v",
        ]
    )
    assert args.workers == 4
    assert args.samples == ["HeLa", "K562"]
    assert args.genes == ["TP53", "MYC"]
    assert args.deeploc is True
    assert args.no_comparator is True
    assert args.verbose is True


def test_sample_manifest_loader(tmp_path, monkeypatch):
    # Avoid CWD-relative resolution affecting the test.
    monkeypatch.chdir(tmp_path)
    manifest = tmp_path / "samples.csv"
    manifest.write_text(
        "sample,predict_file\nHeLa,predicts/HeLa_TIS_predict_all.txt\n"
        "K562,/abs/path/K562_TIS_predict_all.txt\n"
    )
    mapping = _load_sample_manifest(manifest)
    # HeLa path doesn't exist under cwd → falls back to manifest_dir/relative
    assert mapping["HeLa"] == tmp_path / "predicts" / "HeLa_TIS_predict_all.txt"
    assert mapping["K562"] == Path("/abs/path/K562_TIS_predict_all.txt")


def test_replicate_manifest_loader_collects_multiple(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = tmp_path / "reps.csv"
    manifest.write_text(
        "sample,rnaseq_count_file\n"
        "HeLa,reps/hela_1.txt\n"
        "HeLa,reps/hela_2.txt\n"
        "K562,reps/k562_1.txt\n"
    )
    mapping = _load_replicate_manifest(manifest)
    assert len(mapping["HeLa"]) == 2
    assert mapping["K562"] == [tmp_path / "reps" / "k562_1.txt"]


def test_replicate_manifest_loader_accepts_legacy_column(tmp_path, monkeypatch):
    """The legacy 'htseq_count_file' column name must still work."""
    monkeypatch.chdir(tmp_path)
    manifest = tmp_path / "reps.csv"
    manifest.write_text("sample,htseq_count_file\nHeLa,reps/x.txt\n")
    mapping = _load_replicate_manifest(manifest)
    assert mapping["HeLa"] == [tmp_path / "reps" / "x.txt"]


def test_chunk_genes_even_split():
    genes = [
        Gene(gene_name=f"G{i}", gene_id=f"g{i}", canonical_transcript_id="t", canonical_protein="M")
        for i in range(10)
    ]
    chunks = _chunk_genes(genes, 3)
    assert [len(c) for c in chunks] == [4, 3, 3]
    assert sum(len(c) for c in chunks) == 10


def test_chunk_genes_fewer_than_workers():
    genes = [
        Gene(gene_name=f"G{i}", gene_id="g", canonical_transcript_id="t", canonical_protein="M")
        for i in range(2)
    ]
    chunks = _chunk_genes(genes, 4)
    flat = [g for c in chunks for g in c]
    assert len(flat) == 2


def test_annotate_genes_serial_rebuilds_pipeline(config, tmp_path):
    """Smoke test: the worker function should run biophysics + motifs on a
    minimal gene graph without any external data sources.
    """
    gene = Gene(
        gene_name="TEST",
        gene_id="G1",
        canonical_transcript_id="T1",
        canonical_protein="MAAAAPPPPQQQQRRRRR",
        tis_sites=[],
    )
    # Dummy genome path (not read in this path since conservation/clinical
    # don't have DBs configured)
    fake_genome = tmp_path / "genome.fa"
    fake_genome.touch()
    cds_df = pd.DataFrame(
        columns=["chromosome", "start", "end", "strand", "gene_id", "transcript_id", "feature_type"]
    )

    annotated = _annotate_genes_serial(
        [gene],
        config,
        deeploc_lookup={},
        clinical_prefetch={},
        cds_df=cds_df,
        genome_path=fake_genome,
    )

    assert len(annotated) == 1
    bio = annotated[0].canonical_annotations.get("biophysics", {})
    assert "pI" in bio
    assert isinstance(bio["pI"], float)


def test_main_exits_without_subcommand():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
