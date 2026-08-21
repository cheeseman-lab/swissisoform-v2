"""Schema stability for the paired parquets.

The clinical summaries are Counters over open key sets, so per-frame type
inference is unstable in two ways that both corrupt a sharded run: an all-empty
frame infers ``struct<>`` (which Parquet refuses to write), and two frames with
different variant consequences infer incompatible structs for the same column.
``paired_schema`` pins them; these tests pin ``paired_schema``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from swissisoform.io.parquet import CLINICAL_SUMMARY_COLUMNS, paired_schema
from swissisoform.runner import _write_parquet_atomic

EMPTY_SUMMARY = {
    "total_variants": 0,
    "by_source": {},
    "by_consequence": {},
    "pathogenic_count": 0,
}
GNOMAD_SUMMARY = {
    "total_variants": 2,
    "by_source": {"gnomAD": 2},
    "by_consequence": {"missense_variant": 2},
    "pathogenic_count": 1,
}
COSMIC_SUMMARY = {
    "total_variants": 1,
    "by_source": {"COSMIC": 1},
    "by_consequence": {"stop_gained": 1},  # a term the other frames never saw
    "pathogenic_count": 0,
}


def frame(*summaries: dict) -> pd.DataFrame:
    """A minimal paired frame carrying both clinical summary columns."""
    return pd.DataFrame({
        "gene_name": [f"GENE{i}" for i in range(len(summaries))],
        "canonical_clinical_summary": list(summaries),
        "isoform_clinical_summary": list(summaries),
    })


def test_inference_alone_still_fails_on_empty_structs(tmp_path: Path) -> None:
    """Guard the premise: without paired_schema this is a hard write failure."""
    with pytest.raises(pa.ArrowNotImplementedError, match="no child field"):
        frame(EMPTY_SUMMARY, EMPTY_SUMMARY).to_parquet(tmp_path / "inferred.parquet")


def test_all_empty_clinical_frame_writes(tmp_path: Path) -> None:
    """A run where NO gene has a clinical hit — the all_paired-level hazard."""
    df = frame(EMPTY_SUMMARY, EMPTY_SUMMARY)
    path = tmp_path / "all_paired.parquet"
    _write_parquet_atomic(df, path, paired_schema(df))
    assert len(pd.read_parquet(path)) == 2


def test_per_gene_slice_of_an_empty_gene_writes(tmp_path: Path) -> None:
    """The original bug: one gene with no hits, sliced out of a populated frame."""
    df = frame(GNOMAD_SUMMARY, EMPTY_SUMMARY)
    schema = paired_schema(df)
    for gene, sub in df.groupby("gene_name"):
        path = tmp_path / f"{gene}_paired.parquet"
        pq.write_table(pa.Table.from_pandas(sub, schema=schema, preserve_index=False), path)
    assert len(list(tmp_path.glob("*_paired.parquet"))) == 2


def test_schema_is_identical_across_frames_with_different_keys() -> None:
    """Two shards must agree on the column type, whatever variants they saw."""
    a = paired_schema(frame(GNOMAD_SUMMARY))
    b = paired_schema(frame(COSMIC_SUMMARY))
    c = paired_schema(frame(EMPTY_SUMMARY))
    for column in CLINICAL_SUMMARY_COLUMNS:
        types = {s.field(s.get_field_index(column)).type for s in (a, b, c)}
        assert len(types) == 1, f"{column} drifted across frames: {types}"


def test_differently_keyed_shards_concat_after_a_round_trip(tmp_path: Path) -> None:
    """The merge step: per-shard files must be readable into one frame."""
    for i, summary in enumerate((GNOMAD_SUMMARY, COSMIC_SUMMARY, EMPTY_SUMMARY)):
        df = frame(summary)
        _write_parquet_atomic(df, tmp_path / f"shard{i}.parquet", paired_schema(df))
    merged = pd.concat(
        [pd.read_parquet(p) for p in sorted(tmp_path.glob("shard*.parquet"))],
        ignore_index=True,
    )
    assert len(merged) == 3


def test_counts_survive_the_map_round_trip(tmp_path: Path) -> None:
    """Declaring a map must not lose the data it is holding."""
    df = frame(GNOMAD_SUMMARY)
    path = tmp_path / "one.parquet"
    _write_parquet_atomic(df, path, paired_schema(df))
    summary = pd.read_parquet(path)["isoform_clinical_summary"][0]
    assert summary["total_variants"] == 2
    assert summary["pathogenic_count"] == 1
    assert dict(summary["by_source"]) == {"gnomAD": 2}  # maps read back as pairs


def test_frames_without_clinical_columns_are_untouched() -> None:
    """A run with the clinical module skipped must still get a usable schema."""
    df = pd.DataFrame({"gene_name": ["A"], "isoform_plm_vep_status": ["ok"]})
    schema = paired_schema(df)
    assert schema.names == ["gene_name", "isoform_plm_vep_status"]
