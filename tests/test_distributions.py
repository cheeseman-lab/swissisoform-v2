"""Frozen metric distributions — builder + reader.

Offline: every test builds a version from a synthetic frame written to tmp_path,
so nothing here needs the genome-wide parquet or the feature catalog on disk.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from swissisoform import distributions as dist
from swissisoform.setup import distributions as build_mod

N_PAIRED = 60  # comfortably over MIN_STRATUM_N so the floor guard stays quiet


def _frame() -> pd.DataFrame:
    """Synthetic paired-TIS frame: two paired strata plus every separate type."""
    orf = (
        ["truncated"] * N_PAIRED
        + ["extended"] * N_PAIRED
        # Individually under the floor; together the `separate` roll-up clears it,
        # which is exactly the situation the roll-up exists for.
        + ["uorf"] * 20
        + ["uoorf"] * 8
        + ["internal_oof"] * 7
    )
    n = len(orf)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "orf_type": orf,
            # Deterministic ramp: quantiles are exactly checkable.
            "ramp": np.arange(n, dtype=float),
            # Truncations sit an order of magnitude above extensions, so a pooled
            # percentile is provably wrong — the reason strata exist.
            "split": [10.0] * N_PAIRED + [1.0] * (N_PAIRED + 35),
            "sparse": [1.0] * 10 + [None] * (n - 10),
            "allnull": [None] * n,
            "compartment": ["Nucleus"] * 40 + ["Cytoplasm"] * (n - 40),
            "noise": rng.normal(size=n),
            "isoform_scoring_criteria": [
                {"C1_primate_conservation": True if i % 3 == 0 else (None if i % 3 == 1 else False)}
                for i in range(n)
            ],
            "hits": [[1] * (i % 4) for i in range(n)],
        }
    )


def _catalog() -> pd.DataFrame:
    rows = [
        ("ramp", "float"),
        ("split", "float"),
        ("sparse", "float"),
        ("allnull", "float"),
        ("noise", "float"),
        ("compartment", "str"),
        ("hits", "list"),
        ("absent_from_run", "float"),
    ]
    return pd.DataFrame(
        [
            {
                "feature": f,
                "module": "test",
                "pane": "isoform",
                "category": "C",
                "dtype": d,
                "scored": False,
                "include_in_plot": True,
                "exclude_reason": None,
            }
            for f, d in rows
        ]
    )


@pytest.fixture
def built(tmp_path):
    """Build one version from the synthetic frame; return its Distributions."""
    df = _frame()
    table = pa.Table.from_pandas(df, preserve_index=False)
    parquet = tmp_path / "all_paired.parquet"
    pq.write_table(table, parquet)
    catalog = tmp_path / "feature_catalog.csv"
    _catalog().to_csv(catalog, index=False)

    out = tmp_path / "data" / "reference" / "distributions" / "v1"
    build_mod.build([parquet], catalog, out, source_label="synthetic")

    dist.load.cache_clear()
    return dist.load("v1", root=tmp_path)


# ── Builder ───────────────────────────────────────────────────────────────


def test_strata_partition_the_population(built):
    """Each metric's orf_type strata sum to `all`, and `separate` rolls the rest up."""
    g = built.numeric.pivot_table(index="metric", columns="stratum", values="n", aggfunc="first")
    orf = ["truncated", "extended", "uorf", "uoorf", "internal_oof"]
    assert (g[orf].sum(axis=1) == g["all"]).all()
    assert (g[["uorf", "uoorf", "internal_oof"]].sum(axis=1) == g["separate"]).all()


def test_quantiles_are_correct(built):
    """A 0..n-1 ramp puts p50 at the midpoint and the ends at min/max."""
    s = built.summary("ramp")
    n = 2 * N_PAIRED + 35
    assert s["n"] == n and s["n_missing"] == 0
    assert s["min"] == 0.0 and s["max"] == float(n - 1)
    assert s["p50"] == pytest.approx((n - 1) / 2)
    assert built.value_at("ramp", 25) == pytest.approx(0.25 * (n - 1))


def test_nulls_are_counted_not_imputed(built):
    """A sparse metric reports its fill separately rather than reading as narrow."""
    s = built.summary("sparse")
    assert s["n"] == 10
    assert s["n_missing"] == 2 * N_PAIRED + 35 - 10
    assert s["fill_rate"] < 0.1


def test_all_null_metric_survives_as_an_empty_row(built):
    """An unpopulated metric is recorded with n=0, not silently dropped."""
    s = built.summary("allnull")
    assert s is not None and s["n"] == 0 and s["p50"] is None
    assert built.percentile("allnull", 1.0) is None
    assert built.value_at("allnull", 50) is None


def test_list_column_contributes_its_length(built):
    """Hit lists are profiled by element count, inheriting the parent's metadata."""
    s = built.summary("hits__len")
    assert s is not None and s["n"] == 2 * N_PAIRED + 35
    assert s["min"] == 0.0 and s["max"] == 3.0
    assert s["category"] == "C" and s["module"] == "test"


def test_metric_absent_from_the_run_is_skipped(built):
    """A catalog entry with no column in the run does not fabricate a row."""
    assert built.summary("absent_from_run") is None
    assert built.provenance["row_counts"]["metrics_not_in_run"] == 1


def test_histogram_shape(built):
    """Fixed bin count, edges one longer than counts, counts non-negative."""
    edges, counts = built.histogram("ramp")
    assert len(counts) == dist.HIST_BINS and len(edges) == dist.HIST_BINS + 1
    assert (counts >= 0).all()


def test_categorical_fractions_sum_to_one(built):
    """Value frequencies (including the __other__ tail) partition the stratum.

    ``frac`` is stored rounded to 6dp, so the comparisons carry that tolerance.
    """
    sub = built.categorical[
        (built.categorical["metric"] == "compartment") & (built.categorical["stratum"] == "all")
    ]
    assert sub["frac"].sum() == pytest.approx(1.0, abs=1e-5)
    n = 2 * N_PAIRED + 35
    assert built.rate("compartment", "Nucleus") == pytest.approx(40 / n, abs=1e-6)
    # A value outside the stored set is unknown, not zero.
    assert built.rate("compartment", "Mitochondrion") is None


def test_criteria_tristate_rates(built):
    """None (not-evaluable) is counted apart from False, not folded into it."""
    r = built.criterion_rates("C1_primate_conservation")
    n = 2 * N_PAIRED + 35
    assert r["n_true"] + r["n_false"] + r["n_none"] == n
    assert r["n_none"] > 0
    assert r["frac_true"] == pytest.approx(r["n_true"] / (r["n_true"] + r["n_false"]))
    assert r["frac_evaluable"] == pytest.approx((r["n_true"] + r["n_false"]) / n)


def test_provenance_records_the_population_caveat(built):
    """The sidecar carries source, strata sizes, and the caveats a reader needs."""
    p = built.provenance
    assert p["n_isoforms"] == 2 * N_PAIRED + 35
    assert p["strata_n"]["truncated"] == N_PAIRED
    assert p["feature_catalog_sha256"]
    assert any("drop-unsupported-tis" in c for c in p["caveats"])


# ── Reader ────────────────────────────────────────────────────────────────


def test_percentile_value_at_roundtrip(built):
    """value_at then percentile returns the same point, within grid resolution.

    The grid is 101 wide and `percentile` is a rank (fraction at or below), so a
    value landing exactly on a grid point reports one point high. Anything beyond
    that would mean the two directions disagree.
    """
    for p in (10, 25, 50, 75, 90):
        v = built.value_at("noise", p)
        assert abs(built.percentile("noise", v) - p) <= 1


def test_stratification_changes_the_answer(built):
    """The pooled percentile is provably wrong for a metric with split baselines."""
    assert built.value_at("split", 50, "truncated") == 10.0
    assert built.value_at("split", 50, "extended") == 1.0
    assert built.percentile("split", 10.0, "extended") == 100.0
    assert built.percentile("split", 10.0, "truncated") == 100.0
    # Pooled, the truncation baseline reads as extreme when it is typical.
    assert built.percentile("split", 10.0, "all") > built.percentile("split", 1.0, "all")


def test_value_at_refuses_a_too_small_stratum(built):
    """A cutoff from a 5-isoform stratum would be noise; the reader says so."""
    with pytest.raises(dist.DistributionsError, match="MIN_STRATUM_N"):
        built.value_at("ramp", 75, "uoorf")
    assert built.value_at("ramp", 75, "uoorf", allow_small=True) is not None
    # The roll-up is the supported way to get a baseline for the rare types.
    assert built.value_at("ramp", 75, "separate") is not None


def test_percentile_is_a_display_path_and_does_not_enforce_the_floor(built):
    """Placing a value is allowed anywhere; deriving a cutoff is not."""
    assert built.percentile("ramp", 5.0, "uoorf") is not None
    assert built.summary("ramp", "uoorf")["n"] < dist.MIN_STRATUM_N


def test_unknown_metric_and_stratum_return_none(built):
    assert built.summary("nope") is None
    assert built.percentile("nope", 1.0) is None
    assert built.value_at("nope", 50) is None
    assert built.summary("ramp", "no_such_stratum") is None
    assert built.strata("nope") == []


def test_percentile_rejects_missing_and_non_finite_values(built):
    assert built.percentile("ramp", None) is None
    assert built.percentile("ramp", float("nan")) is None


def test_value_at_rejects_out_of_range_percentile(built):
    with pytest.raises(ValueError):
        built.value_at("ramp", 101)


def test_stratum_for_maps_separate_types_to_the_rollup():
    assert dist.stratum_for("uoorf") == dist.STRATUM_SEPARATE
    assert dist.stratum_for("truncated") == "truncated"


def test_load_names_the_build_command_when_a_version_is_missing(tmp_path):
    dist.load.cache_clear()
    with pytest.raises(dist.DistributionsError, match="build_distributions"):
        dist.load("v99", root=tmp_path)


def test_summary_csv_drops_the_array_columns(tmp_path, built):
    """The human view is scalars only — a CSV with 101-point grids is unreadable."""
    csv = tmp_path / "data" / "reference" / "distributions" / "v1" / dist.SUMMARY_FILE
    cols = pd.read_csv(csv, nrows=1).columns
    assert "p50" in cols and "n" in cols
    assert not {"q_grid", "hist_edges", "hist_counts"} & set(cols)


def test_build_refuses_to_clobber_without_force(tmp_path, built):
    """Versions are frozen: rebuilding in place changes what existing tags mean."""
    out = tmp_path / "data" / "reference" / "distributions" / "v1"
    df, catalog = _frame(), tmp_path / "feature_catalog.csv"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp_path / "p.parquet")
    argv = ["--parquet", str(tmp_path / "p.parquet"), "--catalog", str(catalog), "--out", str(out)]
    with pytest.raises(SystemExit, match="frozen"):
        build_mod.main(argv)
    assert build_mod.main(argv + ["--force"]) == 0


def test_build_requires_orf_type(tmp_path):
    """Without orf_type there is no stratification, so the build stops."""
    catalog = tmp_path / "feature_catalog.csv"
    _catalog().to_csv(catalog, index=False)
    parquet = tmp_path / "no_orf.parquet"
    pq.write_table(pa.table({"ramp": [1.0, 2.0]}), parquet)
    with pytest.raises(SystemExit, match="orf_type"):
        build_mod.build([parquet], catalog, tmp_path / "out")


def test_missing_catalog_names_its_regenerate_command(tmp_path):
    with pytest.raises(SystemExit, match="export_feature_catalog"):
        build_mod.load_catalog(tmp_path / "absent.csv")


def test_sidecar_is_valid_json(tmp_path, built):
    path = tmp_path / "data" / "reference" / "distributions" / "v1" / dist.SIDECAR_FILE
    assert json.loads(path.read_text())["artifact"]
