"""Provenance for the TIS population a run was computed over.

Scores and percentiles only mean something against the population behind them,
and that population is set by a flag plus whichever combined catalog happened to
be on disk — neither of which the parquet reveals. These tests pin the record,
and pin the guard that keeps the record honest: a run must not be able to claim
long-read filtering it did not apply.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from swissisoform.config import POPULATION_SIDECAR, ScoringConfig
from swissisoform.runner import RunSpec, _write_population_sidecar
from swissisoform.sourceresolve import collapse_to_source, resolution_columns


def tagged_frame() -> pd.DataFrame:
    """A catalog carrying HeLa verdicts; K562 was never scored (no long-read data)."""
    return pd.DataFrame({
        "Tid": ["T1", "T2", "T3", "T4"],
        "TisType": ["Annotated", "Truncated", "Extended", "Truncated"],
        "Symbol": ["G1", "G1", "G2", "G3"],
        "HeLa_resolved": [True, True, False, None],
        "HeLa_source_transcript": ["T1", "T2", None, None],
        "K562_resolved": [None] * 4,  # never scored: no long-read data
        "K562_source_transcript": [None] * 4,
    })


def untagged_frame() -> pd.DataFrame:
    """The same catalog built before the source-resolution cascade existed."""
    return tagged_frame().drop(
        columns=[c for c in tagged_frame().columns if "resolved" in c or "source_transcript" in c]
    )


def spec(**kw) -> RunSpec:
    base = dict(
        gene_names=None, restricted_df=None, cell_lines=["HeLa", "K562"],
        single_sample=False, min_cell_lines=1, skip=set(), run_name="probe",
        fasta_out=Path("/dev/null"),
    )
    base.update(kw)
    return RunSpec(**base)


def population_of(df: pd.DataFrame, *, drop: bool) -> dict:
    """Mirror runner.prepare's measurement, which the sidecar just serializes."""
    pairs = resolution_columns(df)
    before = len(df)
    after = collapse_to_source(df, keep_unevaluated=not drop)
    extra = {}
    if drop and pairs:
        kept = len(collapse_to_source(df, keep_unevaluated=True))
        extra = {"rows_if_unevaluated_kept": kept, "rows_dropped_by_flag": kept - len(after)}
    return {
        **extra,
        "drop_unsupported_tis": drop,
        "collapse_applied": bool(pairs),
        "rows_before_collapse": before,
        "rows_after_collapse": len(after),
        "rows_dropped": before - len(after),
        "samples_evaluated_by_long_read": {
            r.removesuffix("_resolved"): int(df[r].notna().sum()) for r, _ in pairs
        },
    }


def read_sidecar(tmp_path: Path, df: pd.DataFrame, *, drop: bool, **kw) -> dict:
    _write_population_sidecar(tmp_path, spec(**kw), population_of(df, drop=drop), None)
    return json.loads((tmp_path / POPULATION_SIDECAR).read_text())


def test_records_the_flag_and_what_it_selected(tmp_path: Path) -> None:
    on = read_sidecar(tmp_path, tagged_frame(), drop=True)
    assert on["drop_unsupported_tis"] is True
    assert "long-read-supported TIS only" in on["selects"]
    off = read_sidecar(tmp_path, tagged_frame(), drop=False)
    assert off["drop_unsupported_tis"] is False
    assert "all TIS" in off["selects"]


def test_records_which_samples_had_long_read_data(tmp_path: Path) -> None:
    """The asymmetry that makes the flag consequential must be on the record."""
    got = read_sidecar(tmp_path, tagged_frame(), drop=True)
    evaluated = got["samples_evaluated_by_long_read"]
    assert evaluated["HeLa"] == 3
    assert evaluated["K562"] == 0


def test_row_counts_reconcile(tmp_path: Path) -> None:
    got = read_sidecar(tmp_path, tagged_frame(), drop=True)
    assert got["rows_before_collapse"] - got["rows_dropped"] == got["rows_after_collapse"]


def test_the_flag_actually_drops_more_than_the_default(tmp_path: Path) -> None:
    on = read_sidecar(tmp_path, tagged_frame(), drop=True)
    off = read_sidecar(tmp_path, tagged_frame(), drop=False)
    assert on["rows_after_collapse"] < off["rows_after_collapse"]


def test_records_the_flags_marginal_cost(tmp_path: Path) -> None:
    """How far this population sits from the production one, not just from raw."""
    on = read_sidecar(tmp_path, tagged_frame(), drop=True)
    off = read_sidecar(tmp_path, tagged_frame(), drop=False)
    assert on["rows_if_unevaluated_kept"] == off["rows_after_collapse"]
    assert on["rows_dropped_by_flag"] == off["rows_after_collapse"] - on["rows_after_collapse"]


def test_marginal_cost_is_absent_when_the_flag_is_off(tmp_path: Path) -> None:
    off = read_sidecar(tmp_path, tagged_frame(), drop=False)
    assert "rows_dropped_by_flag" not in off  # nothing to compare against


def test_a_no_op_collapse_is_visible_in_the_artifact(tmp_path: Path) -> None:
    """Without verdict columns the collapse does nothing — say so in the file."""
    got = read_sidecar(tmp_path, untagged_frame(), drop=False)
    assert got["collapse_applied"] is False
    assert got["rows_dropped"] == 0


def test_records_the_combined_catalog_input(tmp_path: Path) -> None:
    got = read_sidecar(tmp_path, tagged_frame(), drop=True, rebuild_combined=True)
    assert got["combined_catalog"]["rebuilt_this_run"] is True
    assert "path" in got["combined_catalog"]


def test_records_the_source_resolution_settings(tmp_path: Path) -> None:
    got = read_sidecar(
        tmp_path, tagged_frame(), drop=True,
        divergence_threshold=0.7, window_upstream=50, window_downstream=150,
    )
    assert got["source_resolution"] == {
        "skipped": False,
        "divergence_threshold": 0.7,
        "window_upstream": 50,
        "window_downstream": 150,
    }


def test_min_cell_lines_comes_from_the_run_config(tmp_path: Path) -> None:
    class Cfg:
        scoring = ScoringConfig()

    _write_population_sidecar(tmp_path, spec(), population_of(tagged_frame(), drop=False), Cfg())
    got = json.loads((tmp_path / POPULATION_SIDECAR).read_text())
    assert got["min_cell_lines"] == Cfg.scoring.min_cell_lines


def test_missing_verdicts_warn_instead_of_passing_silently(caplog) -> None:
    """A population-defining step must never no-op quietly."""
    with caplog.at_level(logging.WARNING):
        out = collapse_to_source(untagged_frame(), keep_unevaluated=False)
    assert len(out) == len(untagged_frame())  # unchanged
    assert any("no source-resolution verdict columns" in r.message for r in caplog.records)


@pytest.mark.parametrize("missing", [None, float("nan"), pd.NA])
def test_every_missing_marker_reads_as_unevaluated(missing: object) -> None:
    """object/None today, but a nullable-boolean column must not crash the collapse."""
    df = tagged_frame()
    df["HeLa_resolved"] = [True, True, False, missing]
    assert len(collapse_to_source(df, keep_unevaluated=False)) == 2


def test_single_sample_mode_says_so(tmp_path: Path) -> None:
    got = read_sidecar(tmp_path, tagged_frame(), drop=False, single_sample=True)
    assert "single-sample" in got["combined_catalog"]["source"]


@pytest.mark.parametrize("drop", [True, False])
def test_sidecar_is_valid_json_with_a_stable_shape(tmp_path: Path, drop: bool) -> None:
    got = read_sidecar(tmp_path, tagged_frame(), drop=drop)
    for key in (
        "run_name", "written_utc", "cell_lines", "min_cell_lines", "source_resolution",
        "combined_catalog", "drop_unsupported_tis", "selects", "collapse_applied",
        "rows_before_collapse", "rows_after_collapse", "rows_dropped",
        "samples_evaluated_by_long_read",
    ):
        assert key in got, f"missing {key}"
