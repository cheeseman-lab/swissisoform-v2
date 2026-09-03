"""The tag-candidate sweep — proposal, cutoff choice, filters, review table.

Offline: every test builds a distributions version from a synthetic frame in
tmp_path, then sweeps it, so nothing here needs the genome-wide parquet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from swissisoform import distributions as dist_mod
from swissisoform import metrics
from swissisoform.setup import distributions as build_mod
from swissisoform.tags import candidates as C
from swissisoform.tags import seeds

N = 200  # per paired stratum, comfortably over MIN_STRATUM_N


def _frame() -> pd.DataFrame:
    """Synthetic run with one metric of each shape the sweep must tell apart."""
    orf = ["truncated"] * N + ["extended"] * N + ["uorf"] * 40
    n = len(orf)
    rng = np.random.default_rng(0)
    half = n // 2
    return pd.DataFrame(
        {
            "orf_type": orf,
            "gene_name": [f"GENE{i}" for i in range(n)],
            "tis_id": [f"tis{i}" for i in range(n)],
            # Two well-separated modes: a break exists and should be found.
            "cmp_bimodal_score": np.r_[rng.normal(0, 0.4, half), rng.normal(10, 0.4, n - half)],
            # Unimodal: no break, no anchor -> a chip, not a tag.
            "isoform_unimodal_score": rng.normal(0, 1, n),
            # A ratio: anchored at 1.0, with ~30% above it.
            "isoform_thing_ratio": np.r_[
                rng.uniform(0.1, 1.0, int(n * 0.7)), rng.uniform(1.0, 4.0, n - int(n * 0.7))
            ],
            # Already a tag.
            "cmp_flag_changed": [True] * (n // 4) + [False] * (n - n // 4),
            # Two metrics identical to each other once thresholded. Split at a
            # third rather than cmp_bimodal_score's half, so they are redundant
            # with each other (Jaccard 1.0) but not with it (0.66) — otherwise all
            # three collapse and the pair under test is never the redundant one.
            "isoform_twin_a_score": np.r_[
                rng.normal(0, 0.3, n // 3), rng.normal(8, 0.3, n - n // 3)
            ],
            "isoform_twin_b_score": np.r_[
                rng.normal(0, 0.3, n // 3), rng.normal(8, 0.3, n - n // 3)
            ],
            "isoform_sparse_delta": [1.0] * 10 + [None] * (n - 10),
            "isoform_constant_delta": [3.0] * n,
        }
    )


def _catalog() -> pd.DataFrame:
    rows = [
        ("cmp_bimodal_score", "float", "cmp", "S", None),
        ("isoform_unimodal_score", "float", "isoform", "S", None),
        ("isoform_thing_ratio", "float", "isoform", "M", None),
        ("cmp_flag_changed", "bool", "cmp", "L", "binary"),
        ("isoform_twin_a_score", "float", "isoform", "S", None),
        ("isoform_twin_b_score", "float", "isoform", "S", None),
        ("isoform_sparse_delta", "float", "isoform", "C", None),
        ("isoform_constant_delta", "float", "isoform", "C", None),
        ("gene_name", "str", "site", "-", "identifier"),
    ]
    return pd.DataFrame(
        [
            {
                "feature": f, "module": "test", "pane": pane, "category": cat, "dtype": d,
                "scored": False, "include_in_plot": d == "float", "exclude_reason": excl,
                "duplicate_of": None, "null_pattern": None,
            }
            for f, d, pane, cat, excl in rows
        ]
    )


@pytest.fixture
def swept(tmp_path):
    """Build distributions from the synthetic frame, then sweep it."""
    df = _frame()
    parquet = tmp_path / "all_paired.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), parquet)
    catalog_csv = tmp_path / "feature_catalog.csv"
    _catalog().to_csv(catalog_csv, index=False)

    out = tmp_path / "data" / "reference" / "distributions" / "v1"
    build_mod.build([parquet], catalog_csv, out, source_label="synthetic")
    dist_mod.load.cache_clear()
    dist = dist_mod.load("v1", root=tmp_path)

    table, chips, funnel = C.build_table(df, _catalog(), dist)
    return {"table": table, "chips": chips, "funnel": funnel, "dist": dist, "df": df}


def _row(table: pd.DataFrame, metric: str) -> pd.Series | None:
    hit = table[table["source_metric"] == metric]
    return hit.iloc[0] if len(hit) else None


# ── Cutoff choice ─────────────────────────────────────────────────────────


def test_break_is_found_between_two_modes(swept):
    """A bimodal metric cuts at the trough, not at a quantile.

    Named ``_score`` deliberately: a ``_delta`` name would anchor at 0.0, and an
    in-band anchor outranks a break by design.
    """
    row = _row(swept["table"], "cmp_bimodal_score")
    assert row is not None
    assert row["cutoff_source"] == "break"
    assert 1.0 < row["cutoff"] < 9.0
    assert row["break_depth"] > C.MIN_BREAK_PROMINENCE


def test_anchor_beats_a_percentile(swept):
    """A ratio cuts at its null hypothesis of 1.0, recorded as an anchor."""
    row = _row(swept["table"], "isoform_thing_ratio")
    assert row is not None
    assert row["cutoff_source"] == "anchor"
    assert row["cutoff"] == pytest.approx(1.0)


def test_unimodal_metric_becomes_a_chip(swept):
    """No anchor and no break means the boolean is arbitrary — range-filter it."""
    assert _row(swept["table"], "isoform_unimodal_score") is None
    assert "isoform_unimodal_score" in set(swept["chips"]["metric"])


def test_find_break_rejects_a_shallow_dip():
    """A dip that is not prominent is histogram noise, not structure."""
    edges = np.linspace(0, 1, 11)
    assert C.find_break(edges, np.array([10, 10, 10, 10, 9, 10, 10, 10, 10, 10])) is None
    value, prominence = C.find_break(edges, np.array([10, 10, 10, 10, 0, 0, 10, 10, 10, 10]))
    assert prominence == pytest.approx(1.0)
    assert 0.35 < value < 0.65


# ── Filters ───────────────────────────────────────────────────────────────


def test_sparse_and_constant_metrics_are_dropped(swept):
    assert _row(swept["table"], "isoform_sparse_delta") is None
    assert _row(swept["table"], "isoform_constant_delta") is None
    dropped = swept["funnel"].dropped
    assert dropped.get("sparse (fill < 0.8)")
    assert dropped.get("constant")


def test_identical_tags_collapse_to_one(swept):
    """Jaccard keeps the first of two identical tags and drops the second."""
    kept = [m for m in ("isoform_twin_a_score", "isoform_twin_b_score")
            if _row(swept["table"], m) is not None]
    assert len(kept) == 1
    assert swept["funnel"].dropped.get(f"redundant (Jaccard >= {C.JACCARD_MAX})")


def test_uncategorised_metrics_are_not_candidates(swept):
    """A coordinate or identity field cannot be any category's checkbox."""
    assert set(swept["table"]["category"]) <= set("CDLMPS")


def test_complementary_directions_collapse(swept):
    """`>= x` and `< x` are the same filter twice; only one survives."""
    per_metric = swept["table"][swept["table"]["stream"] == "sweep"]
    assert per_metric["source_metric"].is_unique


def test_signed_delta_is_proposed_as_a_magnitude(swept):
    """A *_delta is swept as |delta|, never as the sign of the change.

    `delta >= 0` fires on the DIRECTION of a change with no magnitude gate — on
    the real corpus ~87% of its firings fall below the bar the S2 criterion treats
    as a real shift. The tag worth having asks whether the property moved
    appreciably, so only the magnitude is proposed, in one direction.
    """
    signed = swept["table"][
        swept["table"]["source_metric"].astype(str).str.endswith("_delta")
        & (swept["table"]["stream"] == "sweep")
    ]
    assert len(signed) == 0
    mags = swept["table"][swept["table"]["source_metric"].astype(str).str.startswith("abs:")]
    assert (mags["test"].str.contains(">=")).all()


def test_magnitude_has_no_zero_anchor():
    """0.0 is the null for a signed delta; ``|x| >= 0`` is true of everything."""
    assert seeds.anchor_for("cmp_biophysics_gravy_delta")[0] == 0.0
    assert seeds.anchor_for("abs:cmp_biophysics_gravy_delta") is None
    assert seeds.anchor_for("tx:abs_gravy_delta") is None


def test_magnitude_resolves_to_the_absolute_value():
    df = pd.DataFrame({"cmp_x_delta": [-3.0, 2.0, None]})
    got = metrics.resolve("abs:cmp_x_delta", df)
    assert list(got[:2]) == [3.0, 2.0] and pd.isna(got[2])
    assert metrics.resolve("abs:missing", df) is None


def test_merge_decisions_preserves_review_and_keeps_retired_rows():
    """A re-sweep must never discard review work, or the record of a rejection."""
    previous = pd.DataFrame(
        {"tag_id": ["kept", "gone"], "decision": ["remove", "remove"],
         "reviewer_notes": ["[sign-only] why", "[gene property] why"], "stream": ["sweep"] * 2}
    )
    new = pd.DataFrame(
        {"tag_id": ["kept", "fresh"], "decision": ["", ""], "reviewer_notes": ["", ""],
         "stream": ["sweep", "sweep"]}
    )
    out = C.merge_decisions(new, previous)
    assert out.set_index("tag_id").loc["kept", "decision"] == "remove"
    assert out.set_index("tag_id").loc["kept", "reviewer_notes"] == "[sign-only] why"
    assert out.set_index("tag_id").loc["fresh", "decision"] == ""
    retired = out.set_index("tag_id").loc["gone"]
    assert retired["stream"] == "retired"
    assert "superseded" in retired["reviewer_notes"]
    assert "[gene property] why" in retired["reviewer_notes"]

    # Idempotent: sweeping repeatedly must not stack the marker.
    again = C.merge_decisions(new, out)
    note = again.set_index("tag_id").loc["gone", "reviewer_notes"]
    assert note.count("superseded") == 1


# ── Streams ───────────────────────────────────────────────────────────────


def test_boolean_is_a_tag_with_no_cutoff(swept):
    """A boolean column is already tri-state; there is nothing to threshold."""
    row = _row(swept["table"], "cmp_flag_changed")
    assert row is not None
    assert row["kind"] == "bool"
    assert pd.isna(row["cutoff"])
    assert row["fire_pct"] == pytest.approx(25.0, abs=0.5)


def test_llm_candidates_survive_with_blank_rates(swept):
    """Judgment tags cannot be measured until the loop runs; they still appear."""
    llm = swept["table"][swept["table"]["kind"] == "llm"]
    assert len(llm) == len(seeds.LLM_SEEDS)
    assert llm["cutoff"].isna().all()
    assert llm["fire_pct"].isna().all()


def test_every_row_has_a_blank_decision_column(swept):
    """The sweep proposes; it does not decide."""
    assert (swept["table"]["decision"] == "").all()


# ── Tri-state semantics ───────────────────────────────────────────────────


def test_validity_is_declared_not_inferred(swept):
    """Not-evaluable follows `valid_for`, even where the metric is populated."""
    cand = C.Candidate(
        tag_id="t", category="M", label="l", metric="isoform_thing_ratio", kind="code",
        direction=">=", valid_for=("truncated",), source="sweep", cutoff=1.0,
    )
    state = C.evaluate(swept["df"], [cand])["t"]
    is_trunc = (swept["df"]["orf_type"] == "truncated").to_numpy()
    assert (state[~is_trunc] == -1).all()
    assert (state[is_trunc] >= 0).all()


def test_fire_rate_is_over_evaluable_rows_only(swept):
    """A tag undefined for most of the corpus must not look selective for it."""
    state = np.array([1, 1, 0, -1, -1, -1], dtype="int8")
    fire, not_eval = C.rates(state)
    assert fire == pytest.approx(200 / 3)  # 2 of 3 evaluable
    assert not_eval == pytest.approx(50.0)


def test_jaccard_uses_jointly_evaluable_rows(swept):
    """Two tags undefined on disjoint strata are not thereby independent."""
    a = np.array([1, 1, -1, -1], dtype="int8")
    b = np.array([-1, -1, 1, 1], dtype="int8")
    assert C.jaccard(a, b) == 0.0
    assert C.jaccard(a, np.array([1, 0, 1, 0], dtype="int8")) == pytest.approx(0.5)


def test_anchor_at_the_median_is_rejected(swept):
    """A null landing mid-distribution separates nothing, so it is not a cutoff.

    ``isoform_thing_ratio`` puts 1.0 at ~p70, so its anchor stands; a metric whose
    null sits at the median must fall through to break/percentile instead.
    """
    dist = swept["dist"]
    at = dist.percentile("isoform_thing_ratio", 1.0)
    assert not (C.ANCHOR_DEAD_ZONE[0] <= at <= C.ANCHOR_DEAD_ZONE[1])
    assert _row(swept["table"], "isoform_thing_ratio")["cutoff_source"] == "anchor"

    centred = C.Candidate(
        tag_id="c", category="S", label="l", metric="isoform_unimodal_score", kind="code",
        direction=">=", valid_for=seeds.ALL_ORF_TYPES, source="sweep",
    )
    # The unimodal metric's own median is its 0-anchor analogue: nothing to cut on.
    assert C.choose_cutoff(centred, dist).cutoff_source != "anchor"


def test_labels_are_short_human_readable_strings(swept):
    """Sidebar labels, not column names: no underscores, no operators, and short."""
    labels = swept["table"]["proposed_label"]
    assert not labels.str.contains(r"_|>=|<", regex=True).any()
    assert labels.str.len().max() <= 45
    assert labels.str[0].str.isupper().all()


def test_label_registry_covers_direction():
    """The same metric read high and read low are opposite claims, so both differ."""
    hi = seeds.label_for("isoform_conservation_phastcons_at_tis", ">=")
    lo = seeds.label_for("isoform_conservation_phastcons_at_tis", "<")
    assert lo == "Unconserved start site" and hi != lo
    # An uncurated metric still reads as English rather than a raw column name.
    fb = seeds.label_for("cmp_biophysics_some_new_thing", ">=")
    assert fb == "Biophysics some new thing high"
    assert "_" not in fb


# ── Seeds ─────────────────────────────────────────────────────────────────


def test_every_criterion_seed_names_a_live_config_field():
    """A seed pointing at a renamed ScoringConfig field would silently mis-cut."""
    from swissisoform.config import ScoringConfig

    cfg = ScoringConfig()
    missing = [s.config_field for s in seeds.CRITERION_SEEDS
               if s.config_field and not hasattr(cfg, s.config_field)]
    assert missing == []


def test_criterion_seed_tag_ids_are_unique():
    """Three S2 branches share a criterion_id; their tag ids must still differ."""
    ids = [C._slug(f"{s.criterion_id}__{s.metric}")
           for s in seeds.CRITERION_SEEDS if s.metric]
    assert len(ids) == len(set(ids))


def test_transform_resolve_matches_a_raw_column_read():
    """`resolve` is the single point both the profiler and the evaluator use."""
    df = pd.DataFrame({"a": [1.0, 2.0], "isoform_sae_top_gained_delta_max": [1.0, -5.0],
                       "isoform_sae_top_lost_delta_max": [-3.0, 2.0]})
    assert list(metrics.resolve("a", df)) == [1.0, 2.0]
    assert list(metrics.resolve("tx:sae_top_delta", df)) == [3.0, 5.0]
    assert metrics.resolve("nope", df) is None


def test_anchor_rules_do_not_fire_on_percent_identity():
    """1.0 is a ceiling for identity, not a null hypothesis — no anchor."""
    assert seeds.anchor_for("isoform_conservation_frame_primate_mean_pident") is None
    assert seeds.anchor_for("isoform_thing_ratio")[0] == 1.0
    assert seeds.anchor_for("cmp_biophysics_gravy_delta")[0] == 0.0


def test_paired_only_metrics_declare_paired_validity():
    """A shared-region metric is undefined for ORFs that have no shared region."""
    assert seeds.validity_for("isoform_structure_rmsd_shared", None) == seeds.PAIRED_ORF_TYPES
    assert set(seeds.validity_for("some_other_metric", None)) == set(seeds.ALL_ORF_TYPES)
    assert seeds.validity_for("x", "absent_for_separate_orfs") == seeds.PAIRED_ORF_TYPES
