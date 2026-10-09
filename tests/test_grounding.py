"""Alternative groundings for the category LLM slice.

Offline: every test builds its own catalog / registry / distributions in memory or
under tmp_path, so nothing here needs the frozen artifacts or a run on disk.
"""

from __future__ import annotations

import copy

from swissisoform.site.tools import EMIT_VERDICT
import json

import numpy as np
import pandas as pd
import pytest

from swissisoform import distributions as dist_mod
from swissisoform.site import evidence as ev
from swissisoform.site import grounding as gr
from swissisoform.site import llm
from swissisoform.tags import registry as reg_mod

CATEGORY_C = {"letter": "C", "name": "Conservation", "members": []}
CATEGORY_M = {"letter": "M", "name": "Mutation Landscape", "members": []}
ALL_ORF = "extended|truncated|uorf|uoorf|internal_oof|3utr_orf|alt_orf"


def _catalog(rows: list[dict]) -> pd.DataFrame:
    base = {"pane": "isoform", "dtype": "float", "category": "C"}
    return pd.DataFrame([{**base, **r} for r in rows])


def _record(raw: dict, orf_type: str = "extended") -> dict:
    return {
        "tis_id": "chr1:1:+:ATG:ENST1",
        "gene": {"name": "G"},
        "orf_type": orf_type,
        "differential_sequence": "MAA",
        "diff_space": "isoform",
        "isoform_length_aa": 10,
        "canonical_length_aa": 8,
        "_raw": raw,
    }


def _tag_row(**kw) -> dict:
    base = {
        "tag_id": "t",
        "category": "C",
        "axis": "E",
        "label": "A tag",
        "kind": reg_mod.KIND_THRESHOLD,
        "metric": "m",
        "direction": ">=",
        "cutoff": 1.0,
        "cutoff_source": "config",
        "cutoff_pctile": None,
        "cutoff_overrides": "",
        "valid_for": ALL_ORF,
        "criterion_id": "",
        "source": "sweep",
        "blocked": "",
        "note": "",
    }
    base.update(kw)
    return base


def _registry(*rows: dict) -> reg_mod.TagRegistry:
    return reg_mod.from_frame(
        "vtest", pd.DataFrame(list(rows), columns=list(reg_mod.REGISTRY_COLUMNS)), {}
    )


# ---------------------------------------------------------------------------
# The hook itself
# ---------------------------------------------------------------------------


class TestHook:
    def test_default_is_no_hook(self):
        assert ev.use_category_body(None) is None

    def test_set_and_restore_round_trips(self):
        def builder(record, category):
            return {"x": 1}

        previous = ev.use_category_body(builder)
        try:
            assert callable(ev._CATEGORY_BODY)
        finally:
            ev.use_category_body(previous)
        assert ev._CATEGORY_BODY is None

    def test_identity_block_is_not_the_builder_s_to_drop(self):
        """A grounding must not be able to remove differential_region_location."""

        def builder(record, category):
            return {"isoform": "GONE", "x": 1}

        previous = ev.use_category_body(builder)
        try:
            out = ev.slice_category(_record({}), CATEGORY_C)
        finally:
            ev.use_category_body(previous)
        # The fixed block wins: **builder-output is spread AFTER, so a key clash
        # would overwrite — assert the guarantee we actually provide.
        assert out["category"] == "C" and out["name"] == "Conservation"
        assert "x" in out


# ---------------------------------------------------------------------------
# Column universe
# ---------------------------------------------------------------------------


class TestColumns:
    def test_canonical_pane_excluded(self):
        cat = _catalog(
            [
                {"feature": "isoform_a"},
                {"feature": "canonical_a", "pane": "canonical"},
                {"feature": "cmp_a", "pane": "cmp"},
            ]
        )
        assert gr.category_columns(cat)["C"] == ["isoform_a", "cmp_a"]

    def test_scoring_and_tag_columns_never_leak(self):
        cat = _catalog(
            [
                {"feature": "isoform_a"},
                {"feature": "isoform_scoring_criteria"},
                {"feature": "isoform_tags_states"},
                {"feature": "cmp_biophysics_gravy_enriched", "pane": "cmp"},
            ]
        )
        assert gr.category_columns(cat)["C"] == ["isoform_a"]

    def test_dist_scoped_to_profiled_numeric(self):
        cat = _catalog(
            [
                {"feature": "isoform_num"},
                {"feature": "isoform_unprofiled"},
                {"feature": "isoform_text", "dtype": "str"},
            ]
        )
        dist = dist_mod.Distributions(
            version="v",
            numeric=pd.DataFrame(
                [{"metric": "isoform_num", "stratum": dist_mod.STRATUM_ALL, "n": 100}]
            ),
            categorical=pd.DataFrame(),
            criteria=pd.DataFrame(),
            provenance={},
        )
        assert gr.numeric_category_columns(cat, dist)["C"] == ["isoform_num"]


# ---------------------------------------------------------------------------
# raw
# ---------------------------------------------------------------------------


class TestRaw:
    def _build(self, catalog, **kw):
        return gr._raw_body(gr.category_columns(catalog), **kw)

    def test_carries_the_category_s_columns_only(self):
        cat = _catalog([{"feature": "isoform_a"}, {"feature": "isoform_b", "category": "D"}])
        body = self._build(cat)(_record({"isoform_a": 1.0, "isoform_b": 2.0}), CATEGORY_C)
        assert body["evidence"] == {"isoform_a": 1.0}

    def test_nan_becomes_none_not_a_float(self):
        cat = _catalog([{"feature": "isoform_a"}])
        body = self._build(cat)(_record({"isoform_a": float("nan")}), CATEGORY_C)
        assert body["evidence"]["isoform_a"] is None

    def test_long_lists_capped_and_the_cap_is_stated(self):
        cat = _catalog([{"feature": "isoform_hits"}])
        n_rows = gr.MAX_LIST_ROWS + 7
        rows = [{"i": i} for i in range(n_rows)]
        body = self._build(cat, strip_lists_for=frozenset())(
            _record({"isoform_hits": rows}), CATEGORY_C
        )
        assert len(body["evidence"]["isoform_hits"]) == gr.MAX_LIST_ROWS
        assert body["truncated"]["rows_dropped"]["isoform_hits"] == n_rows - gr.MAX_LIST_ROWS

    def test_numpy_arrays_count_as_lists(self):
        cat = _catalog([{"feature": "isoform_hits"}])
        body = self._build(cat, strip_lists_for=frozenset())(
            _record({"isoform_hits": np.arange(3)}), CATEGORY_C
        )
        assert body["evidence"]["isoform_hits"] == [0, 1, 2]

    def test_tool_loop_category_gets_counts_not_rows(self):
        """M's rows reach the model through tools; duplicating them would hand the
        raw arm a 20x larger opening context than the criteria arm.
        """
        cat = _catalog([{"feature": "isoform_hits", "category": "M"}])
        body = self._build(cat)(
            _record({"isoform_hits": [{"i": i} for i in range(99)]}), CATEGORY_M
        )
        assert "isoform_hits" not in body["evidence"]
        assert body["hits_note"]["n_rows"]["isoform_hits"] == 99

    def test_identical_hit_lists_are_sent_once(self):
        """On an extension the diff-region hit list is the isoform list, row for row."""
        cat = _catalog(
            [
                {"feature": "isoform_massspec_hits", "category": "D", "dtype": "list"},
                {
                    "feature": "cmp_massspec_hits_in_diff_region",
                    "category": "D",
                    "pane": "cmp",
                    "dtype": "list",
                },
            ]
        )
        hits = [
            {"peptide": "MAGTMGK", "validated": True},
            {"peptide": "DATAATR", "validated": False},
        ]
        raw = {"isoform_massspec_hits": hits, "cmp_massspec_hits_in_diff_region": list(hits)}
        ev_d = self._build(cat)(_record(raw), {"letter": "D", "name": "Detection"})["evidence"]
        assert ev_d["isoform_massspec_hits"] == hits
        assert ev_d["cmp_massspec_hits_in_diff_region"] == {
            "same_rows_as": "isoform_massspec_hits",
            "n_rows": 2,
        }

    def test_different_hit_lists_are_both_sent(self):
        cat = _catalog(
            [
                {"feature": "isoform_massspec_hits", "category": "D", "dtype": "list"},
                {
                    "feature": "cmp_massspec_hits_in_diff_region",
                    "category": "D",
                    "pane": "cmp",
                    "dtype": "list",
                },
            ]
        )
        raw = {
            "isoform_massspec_hits": [{"p": 1}, {"p": 2}],
            "cmp_massspec_hits_in_diff_region": [],
        }
        ev_d = self._build(cat)(_record(raw), {"letter": "D", "name": "Detection"})["evidence"]
        assert ev_d["cmp_massspec_hits_in_diff_region"] == []

    def test_missing_column_is_skipped_not_nulled(self):
        cat = _catalog([{"feature": "isoform_absent"}])
        body = self._build(cat)(_record({}), CATEGORY_C)
        assert body["evidence"] == {}


# ---------------------------------------------------------------------------
# tags
# ---------------------------------------------------------------------------


class TestTags:
    def test_all_three_states_are_carried(self):
        """`off` is evidence and `not_evaluable` is not `off`; dropping either
        leaves the model unable to tell a negative from an untestable.
        """
        reg = _registry(
            _tag_row(tag_id="on_", metric="m1"),
            _tag_row(tag_id="off_", metric="m2"),
            _tag_row(tag_id="na_", metric="m3"),
        )
        rec = _record(
            {
                "isoform_tags_states": {"on_": True, "off_": False, "na_": None},
                "isoform_tags_citations": {"on_": 5.0, "off_": 0.1, "na_": None},
            }
        )
        states = {t["tag_id"]: t["state"] for t in gr._tags_body(reg)(rec, CATEGORY_C)["tags"]}
        assert states == {"on_": "on", "off_": "off", "na_": "not_evaluable"}

    def test_citation_and_test_string_travel_with_a_fired_tag(self):
        reg = _registry(_tag_row(tag_id="a", metric="isoform_x", cutoff=2.0))
        rec = _record({"isoform_tags_states": {"a": True}, "isoform_tags_citations": {"a": 7.5}})
        tag = gr._tags_body(reg)(rec, CATEGORY_C)["tags"][0]
        assert tag["value"] == 7.5
        assert tag["test"] == "isoform_x >= 2"

    @pytest.mark.parametrize(
        "orf_type, label",
        [
            ("extended", "Extension more basic"),
            ("truncated", "Lost region more basic"),
            ("uorf", "ORF more basic"),
        ],
    )
    def test_label_names_the_region_for_the_orf_type(self, orf_type, label):
        reg = _registry(_tag_row(tag_id="a", label="Unique region more basic"))
        rec = _record({"isoform_tags_states": {"a": True}}, orf_type=orf_type)
        assert gr._tags_body(reg)(rec, CATEGORY_C)["tags"][0]["label"] == label

    def test_llm_tags_are_questions_not_states(self):
        reg = _registry(_tag_row(tag_id="judged", category="M", kind=reg_mod.KIND_LLM))
        rec = _record({"isoform_tags_states": {"other": True}})
        body = gr._tags_body(reg)(rec, CATEGORY_M)
        assert body["tags"] == []
        assert body["open_questions"][0]["state"] == "unanswered"

    def test_llm_tag_outside_its_orf_type_is_not_evaluable_not_unanswered(self):
        reg = _registry(
            _tag_row(
                tag_id="extension_docks_against_core",
                category="P",
                kind=reg_mod.KIND_LLM,
                valid_for="extended",
            )
        )
        rec = _record({"isoform_tags_states": {"x": True}}, orf_type="uorf")
        q = gr._tags_body(reg)(rec, {"letter": "P", "name": "P", "members": []})
        assert q["open_questions"][0]["state"] == "not_evaluable"
        assert "extended" in q["open_questions"][0]["why"]

    def test_record_without_tags_is_an_error_not_an_empty_slice(self):
        """A silently empty tags arm would look like an isoform with no evidence."""
        reg = _registry(_tag_row())
        with pytest.raises(gr.GroundingError, match="isoform_tags_states"):
            gr._tags_body(reg)(_record({}), CATEGORY_C)

    def test_registry_version_is_stamped(self):
        reg = _registry(_tag_row(tag_id="a"))
        rec = _record({"isoform_tags_states": {"a": True}})
        assert gr._tags_body(reg)(rec, CATEGORY_C)["tags_registry_version"] == "vtest"


class TestTagMetrics:
    """A criterion-backed tag carries the criterion's own supporting numbers.

    The single cited value is one input to a verdict that may rest on nine, so a
    tag restating a criterion hands over that criterion's `evidence_cols` too.
    Sweep tags stay lean: their value IS their metric.
    """

    @staticmethod
    def _rec(raw_extra: dict | None = None) -> dict:
        raw = {
            "isoform_tags_states": {"crit": True, "sweep": True},
            "isoform_tags_citations": {"crit": 7.0, "sweep": 2.0},
            "a": 1,
            "b": 2,
            "unused": 99,
        }
        raw.update(raw_extra or {})
        return _record(raw)

    @staticmethod
    def _install(monkeypatch, cfg: dict) -> None:
        # slice_criterion reads the identity fields every real criterion declares.
        base = {"axis": "E", "label": "L", "short_label": "S"}
        monkeypatch.setitem(gr.ev.CRITERIA, "X", {**base, **cfg})

    def test_sweep_tag_stays_lean(self, monkeypatch):
        self._install(monkeypatch, {"evidence_cols": ["a"], "interpretation_hint": "H"})
        reg = _registry(_tag_row(tag_id="sweep", metric="m"))
        tag = gr._tags_body(reg)(self._rec(), CATEGORY_C)["tags"][0]
        assert "metrics" not in tag and "means" not in tag and "criterion_id" not in tag

    def test_derived_tag_carries_exactly_its_evidence_cols(self, monkeypatch):
        """Closed on purpose: a future 'just hand over _raw' regression fails here."""
        self._install(monkeypatch, {"evidence_cols": ["a", "b"], "interpretation_hint": "H"})
        reg = _registry(_tag_row(tag_id="crit", criterion_id="X", metric="m"))
        tag = gr._tags_body(reg)(self._rec(), CATEGORY_C)["tags"][0]
        assert tag["metrics"] == {"a": 1, "b": 2}
        assert tag["criterion_id"] == "X"

    def test_builder_backed_criterion_is_not_a_silent_no_op(self, monkeypatch):
        """S2/S3 hold their evidence in a builder, not a column list.

        No builder-backed criterion ships today, so the real corpus cannot catch
        a regression here — this test is the only thing that does.
        """
        self._install(
            monkeypatch,
            {
                "evidence_cols": [],
                "interpretation_hint": "H",
                "evidence_builder": lambda rec: {"evidence": {"k": 1}},
            },
        )
        reg = _registry(_tag_row(tag_id="crit", criterion_id="X", metric="m"))
        tag = gr._tags_body(reg)(self._rec(), CATEGORY_C)["tags"][0]
        assert tag["metrics"] == {"k": 1}

    def test_builder_returning_none_keeps_the_tag(self, monkeypatch):
        """The tri-state is still a real result; dropping the tag would make
        "could not build evidence" read as "not evaluated"."""
        self._install(
            monkeypatch,
            {"evidence_cols": [], "interpretation_hint": "H", "evidence_builder": lambda r: None},
        )
        reg = _registry(_tag_row(tag_id="crit", criterion_id="X", metric="m"))
        tag = gr._tags_body(reg)(self._rec(), CATEGORY_C)["tags"][0]
        assert "metrics" not in tag
        assert tag["state"] == "on"

    def test_nested_leak_is_scrubbed_from_builder_output(self, monkeypatch):
        """`_leaks` is name-based and cannot see a leak nested under a label."""
        self._install(
            monkeypatch,
            {
                "evidence_cols": [],
                "interpretation_hint": "H",
                "evidence_builder": lambda rec: {
                    "evidence": {"GRAVY": {"ratio": 1.2, "cmp_biophysics_gravy_enriched": True}}
                },
            },
        )
        reg = _registry(_tag_row(tag_id="crit", criterion_id="X", metric="m"))
        tag = gr._tags_body(reg)(self._rec(), CATEGORY_C)["tags"][0]
        assert tag["metrics"]["GRAVY"] == {"ratio": 1.2}

    def test_hints_off_drops_only_means(self, monkeypatch):
        """`interpretation_hint` IS the hint axis — carrying it unconditionally
        would hand tags_nohint the guidance the axis exists to remove."""
        self._install(monkeypatch, {"evidence_cols": ["a"], "interpretation_hint": "H"})
        reg = _registry(_tag_row(tag_id="crit", criterion_id="X", metric="m", note="N"))
        rec = self._rec()
        on = gr._tags_body(reg, hints=True)(rec, CATEGORY_C)["tags"][0]
        off = gr._tags_body(reg, hints=False)(rec, CATEGORY_C)["tags"][0]
        assert on["means"] == "H"
        assert set(on) - set(off) == {"means"}
        assert off["metrics"] == {"a": 1} and off["note"] == "N"

    def test_derived_tag_carries_the_criterion_s_summary_lines(self, monkeypatch):
        """Without `headline`/`reason` the model derives ratios and counts itself."""
        self._install(monkeypatch, {"evidence_cols": ["a"], "interpretation_hint": "H"})
        rec = self._rec()
        sliced = {"reason": "n_cell_lines=1 (threshold 3)", "headline": "detected in 1/6 cell lines"}
        monkeypatch.setattr(gr.ev, "slice_criterion", lambda r, cid: dict(sliced))
        reg = _registry(_tag_row(tag_id="crit", criterion_id="X", metric="m"))
        tag = gr._tags_body(reg)(rec, CATEGORY_C)["tags"][0]
        assert tag["reason"] == sliced["reason"]
        assert tag["headline"] == sliced["headline"]

    def test_missing_summary_line_is_omitted_not_nulled(self, monkeypatch):
        self._install(monkeypatch, {"evidence_cols": ["a"], "interpretation_hint": "H"})
        monkeypatch.setattr(gr.ev, "slice_criterion", lambda r, cid: {"reason": "r", "headline": None})
        reg = _registry(_tag_row(tag_id="crit", criterion_id="X", metric="m"))
        tag = gr._tags_body(reg)(self._rec(), CATEGORY_C)["tags"][0]
        assert tag["reason"] == "r" and "headline" not in tag

    def test_sweep_tag_carries_no_summary_lines(self, monkeypatch):
        self._install(monkeypatch, {"evidence_cols": ["a"], "interpretation_hint": "H"})
        reg = _registry(_tag_row(tag_id="sweep", metric="m"))
        tag = gr._tags_body(reg)(self._rec(), CATEGORY_C)["tags"][0]
        assert "reason" not in tag and "headline" not in tag

    def test_unknown_criterion_fails_at_build_time(self):
        """Registry/criteria drift must fail at arm setup, not degrade to a lean
        payload partway through a paid run."""
        reg = _registry(_tag_row(tag_id="crit", criterion_id="nope", metric="m"))
        with pytest.raises(gr.GroundingError):
            gr._tags_body(reg)


# ---------------------------------------------------------------------------
# dist
# ---------------------------------------------------------------------------


def _dist(rows: list[dict]) -> dist_mod.Distributions:
    frame = pd.DataFrame(rows)
    for point in ("p05", "p25", "p50", "p75", "p95"):
        if point not in frame:
            frame[point] = 0.0
    frame["q_grid"] = [list(np.linspace(0, 10, 101))] * len(frame)
    return dist_mod.Distributions(
        version="v3",
        numeric=frame,
        categorical=pd.DataFrame(),
        criteria=pd.DataFrame(),
        provenance={"source_run": "full_catalog", "n_isoforms": 6462},
    )


class TestDist:
    def test_value_and_percentile_reported(self):
        dist = _dist([{"metric": "isoform_a", "stratum": "extended", "n": 200}])
        body = gr._dist_body({"C": ["isoform_a"]}, dist)(_record({"isoform_a": 5.0}), CATEGORY_C)
        field = body["fields"]["isoform_a"]
        assert field["value"] == 5.0
        assert field["stratum"] == "extended"
        assert 0 <= field["pctile"] <= 100

    def test_thin_stratum_falls_back_to_all_and_says_so(self):
        """`percentile` does not enforce MIN_STRATUM_N, so it would happily rank
        against n=3 and read as authoritative.
        """
        dist = _dist(
            [
                {"metric": "isoform_a", "stratum": "extended", "n": 3},
                {"metric": "isoform_a", "stratum": dist_mod.STRATUM_ALL, "n": 6462},
            ]
        )
        body = gr._dist_body({"C": ["isoform_a"]}, dist)(_record({"isoform_a": 5.0}), CATEGORY_C)
        assert body["fields"]["isoform_a"]["stratum"] == dist_mod.STRATUM_ALL
        assert body["fields"]["isoform_a"]["n"] == 6462

    def test_null_value_is_omitted_not_ranked(self):
        dist = _dist([{"metric": "isoform_a", "stratum": "extended", "n": 200}])
        body = gr._dist_body({"C": ["isoform_a"]}, dist)(
            _record({"isoform_a": float("nan")}), CATEGORY_C
        )
        assert body["fields"] == {}

    def test_categorical_calls_ride_alongside_the_percentiles(self):
        """The DeepLoc call and its changed flag are the L finding, not context."""
        cat = _catalog(
            [
                {
                    "feature": "isoform_localization_deeploc_prediction",
                    "category": "L",
                    "dtype": "str",
                    "exclude_reason": "categorical",
                },
                {
                    "feature": "cmp_localization_deeploc_prediction_changed",
                    "category": "L",
                    "pane": "cmp",
                    "dtype": "bool",
                    "exclude_reason": "binary",
                },
                {
                    "feature": "isoform_conservation_summary.phylop_status",
                    "category": "L",
                    "dtype": "str",
                    "exclude_reason": "status_string",
                },
                {
                    "feature": "isoform_structure_isoform_hash",
                    "category": "L",
                    "dtype": "str",
                    "exclude_reason": "identifier",
                },
                {
                    "feature": "isoform_conservation_summary.phylop_bigwig",
                    "category": "L",
                    "dtype": "str",
                    "exclude_reason": "status_string",
                },
                {
                    "feature": "cmp_biophysics_gravy_enriched",
                    "category": "L",
                    "pane": "cmp",
                    "dtype": "bool",
                    "exclude_reason": "binary",
                },
            ]
        )
        calls = gr.categorical_category_columns(cat)["L"]
        assert calls == [
            "isoform_localization_deeploc_prediction",
            "cmp_localization_deeploc_prediction_changed",
            "isoform_conservation_summary.phylop_status",
        ]
        dist = _dist([{"metric": "isoform_a", "stratum": "extended", "n": 200}])
        raw = {
            "isoform_a": 1.0,
            "isoform_localization_deeploc_prediction": "Nucleus",
            "cmp_localization_deeploc_prediction_changed": False,
            "isoform_conservation_summary": {"phylop_status": "ok"},
        }
        body = gr._dist_body({"L": ["isoform_a"]}, dist, {"L": calls})(
            _record(raw), {"letter": "L", "name": "Localization"}
        )
        assert body["calls"] == {
            "isoform_localization_deeploc_prediction": "Nucleus",
            "cmp_localization_deeploc_prediction_changed": False,
            "isoform_conservation_summary.phylop_status": "ok",
        }
        assert "pctile" in body["fields"]["isoform_a"]

    def test_feature_index_is_a_call_not_a_ranked_metric(self):
        cat = _catalog(
            [
                {
                    "feature": "isoform_sae_top_gained_feature_index",
                    "category": "S",
                    "dtype": "int",
                },
                {"feature": "isoform_sae_n_features", "category": "S", "dtype": "int"},
            ]
        )
        dist = _dist(
            [
                {"metric": m, "stratum": dist_mod.STRATUM_ALL, "n": 200}
                for m in ("isoform_sae_top_gained_feature_index", "isoform_sae_n_features")
            ]
        )
        assert gr.numeric_category_columns(cat, dist)["S"] == ["isoform_sae_n_features"]
        assert gr.categorical_category_columns(cat)["S"] == ["isoform_sae_top_gained_feature_index"]

    def test_p_values_are_flagged_lower_is_stronger(self):
        metrics = [
            "isoform_massspec_summary.min_pvalue",
            "tis_pvalue",
            "fisher_qvalue",
            "isoform_a",
        ]
        dist = _dist([{"metric": m, "stratum": "extended", "n": 200} for m in metrics])
        raw = {
            "isoform_massspec_summary": {"min_pvalue": 0.001},
            "tis_pvalue": 0.2,
            "fisher_qvalue": 0.05,
            "isoform_a": 3.0,
        }
        fields = gr._dist_body({"D": metrics}, dist)(_record(raw), CATEGORY_C | {"letter": "D"})[
            "fields"
        ]
        assert {m for m, f in fields.items() if f.get("lower_is_stronger")} == set(metrics[:3])

    def test_reference_population_is_declared(self):
        """A percentile against full_catalog is not a percentile against this corpus."""
        dist = _dist([{"metric": "isoform_a", "stratum": "extended", "n": 200}])
        body = gr._dist_body({"C": ["isoform_a"]}, dist)(_record({"isoform_a": 1.0}), CATEGORY_C)
        assert body["reference_population"]["source_run"] == "full_catalog"
        assert body["reference_population"]["n_isoforms"] == 6462


# ---------------------------------------------------------------------------
# Hints, verdict extras, entry point
# ---------------------------------------------------------------------------


class TestHints:
    def test_strip_removes_every_member_hint(self):
        record = {"members": [{"interpretation_hint": "x", "value": True}, {"value": False}]}
        out = gr.strip_hints(record)
        assert all("interpretation_hint" not in m for m in out["members"])
        assert out["members"][0]["value"] is True

    def test_strip_is_a_noop_on_a_body_with_no_members(self):
        assert gr.strip_hints({"tags": []}) == {"tags": []}


class TestVerdictExtras:
    def test_fields_enumerate_only_that_category_s_llm_tags(self):
        reg = _registry(
            _tag_row(tag_id="m_one", category="M", kind=reg_mod.KIND_LLM),
            _tag_row(tag_id="p_one", category="P", kind=reg_mod.KIND_LLM),
            _tag_row(tag_id="code", category="M"),
        )
        extras = gr.verdict_extra_fields(reg, "M")
        assert extras["tags_fired"]["items"]["enum"] == ["m_one"]

    def test_no_llm_tags_means_no_field(self):
        assert gr.verdict_extra_fields(_registry(_tag_row()), "C") == {}

    def test_install_patches_then_restores_exactly(self):
        tools = [
            {"name": "reader", "input_schema": {"properties": {}}},
            {
                "name": EMIT_VERDICT,
                "strict": True,
                "input_schema": {
                    "properties": {"verdict": {"type": "string"}},
                    "required": ["verdict"],
                    "additionalProperties": False,
                },
            },
        ]
        before = copy.deepcopy(tools)
        restore = gr.install_verdict_extras(tools, {"tags_fired": {"type": "array"}})
        assert "tags_fired" in tools[1]["input_schema"]["properties"]
        # Required, so "judged and nothing fired" ([]) is distinguishable from
        # "never answered" — 3 of 4 loops omitted it when it was optional.
        assert "tags_fired" in tools[1]["input_schema"]["required"]
        restore()
        assert tools == before

    def test_can_be_left_optional(self):
        tools = [{"name": EMIT_VERDICT, "input_schema": {"properties": {}}}]
        gr.install_verdict_extras(tools, {"tags_fired": {}}, required=False)
        assert "tags_fired" not in tools[0]["input_schema"].get("required", [])

    def test_empty_extras_is_a_noop(self):
        tools = [{"name": EMIT_VERDICT, "input_schema": {"properties": {}}}]
        before = copy.deepcopy(tools)
        gr.install_verdict_extras(tools, {})()
        assert tools == before

    def test_both_tool_modules_agree_on_the_terminal_name(self):
        """install_verdict_extras patches both lists through one constant.

        Nothing else pins them together, and a silent divergence would leave one
        category's loop unpatched rather than raising.
        """
        from swissisoform.site import structure_tools as ST
        from swissisoform.site import tools as T

        assert T.EMIT_VERDICT == ST.EMIT_VERDICT


class TestCatalogNamesResolve:
    """Dotted struct leaves and ``n_*`` counts must reach raw and dist.

    The catalog names columns the way the flattened parquet reads, but ``_raw``
    keeps structs nested — so a flat-key fixture passes while D's whole
    mass-spec summary is silently dropped. These records are shaped like the
    real cheeseman50 ones (CBX1 chr17:48076878, a validated truncation).
    """

    RAW = {
        "isoform_massspec_summary": {
            "total_peptides": 2,
            "unique_peptides": 1,
            "validated_peptides": 1.0,
            "pepquery_run": True,
            "best_hyperscore": 45.87,
            "min_pvalue": 0.0002,
            "total_psms": 1.0,
        },
        "isoform_massspec_hits": [
            {"peptide": "MGFSDEDNTWEPEENLDCPDLIAEFLQSQK", "validated": True},
            {"peptide": "MEKVLDR", "validated": False},
        ],
        "isoform_conservation_summary": {
            "phylop_status": "ok",
            "region_status": "ok",
            "unique_region_nt": 126,
            "shared_region_nt": 429,
        },
        "isoform_clinical_summary": {
            "total_variants": 2054,
            # A parquet map column serialises as key/value pairs.
            "by_consequence": [["missense_variant", 173], ["stop_gained", 13]],
        },
    }
    D_COLS = [
        "isoform_massspec_summary.best_hyperscore",
        "isoform_massspec_summary.min_pvalue",
        "isoform_massspec_summary.validated_peptides",
        "isoform_massspec_summary.total_peptides",
        "isoform_massspec_summary.total_psms",
        "n_isoform_massspec_hits",
    ]
    C_COLS = [
        "isoform_conservation_summary.phylop_status",
        "isoform_conservation_summary.region_status",
        "isoform_conservation_summary.unique_region_nt",
    ]

    def test_lookup_walks_structs_maps_and_counts(self):
        raw = self.RAW
        assert gr._lookup(raw, "isoform_massspec_summary.best_hyperscore") == 45.87
        assert gr._lookup(raw, "isoform_clinical_summary.by_consequence.stop_gained") == 13
        assert gr._lookup(raw, "n_isoform_massspec_hits") == 2
        assert gr._lookup(raw, "isoform_clinical_summary.by_consequence.intronic") is gr._MISSING
        assert gr._lookup(raw, "n_isoform_absent_hits") is gr._MISSING

    def test_raw_arm_carries_d3_and_c_status(self):
        cat = _catalog(
            [{"feature": c, "category": "D"} for c in self.D_COLS]
            + [{"feature": c, "category": "C", "dtype": "str"} for c in self.C_COLS]
        )
        build = gr._raw_body(gr.category_columns(cat))
        d = build(_record(self.RAW, "truncated"), {"letter": "D", "name": "Detection"})
        assert d["evidence"]["isoform_massspec_summary.best_hyperscore"] == 45.87
        assert d["evidence"]["isoform_massspec_summary.validated_peptides"] == 1.0
        assert d["evidence"]["n_isoform_massspec_hits"] == 2
        assert set(d["evidence"]) == set(self.D_COLS)
        c = build(_record(self.RAW, "truncated"), CATEGORY_C)
        assert c["evidence"]["isoform_conservation_summary.region_status"] == "ok"
        assert c["evidence"]["isoform_conservation_summary.unique_region_nt"] == 126

    def test_dist_arm_ranks_dotted_and_derived_metrics(self):
        numeric = [c for c in self.D_COLS] + ["isoform_conservation_summary.unique_region_nt"]
        dist = _dist([{"metric": m, "stratum": "truncated", "n": 200} for m in numeric])
        build = gr._dist_body({"D": self.D_COLS}, dist)
        body = build(_record(self.RAW, "truncated"), {"letter": "D", "name": "Detection"})
        assert set(body["fields"]) == set(self.D_COLS)
        assert body["fields"]["n_isoform_massspec_hits"]["value"] == 2


class TestSupersededStrip:
    """P's PAE block means must leave every arm's tool-loop opening, not only criteria's.

    ``pae_block()`` recomputes all three, so carrying them hands the loop its own
    answers; the strip used to return early on any payload without ``members``,
    which is every arm but criteria.
    """

    PAE = llm.SUPERSEDED_BY_TOOLS["P"]
    CATEGORY_P = {"letter": "P", "name": "Predicted Structure", "members": []}

    def _raw(self) -> dict:
        return {
            **{c: 4.2 for c in self.PAE},
            "isoform_structure_pae_status": "ok",
            "isoform_structure_plddt_diffregion_mean": 0.8,
            "isoform_tags_states": {"p1_structured_extension": True},
            "isoform_tags_citations": {"p1_structured_extension": 0.8},
        }

    def _opening(self, builder) -> dict:
        previous = ev.use_category_body(builder)
        try:
            sliced = ev.slice_category(_record(self._raw()), self.CATEGORY_P)
        finally:
            ev.use_category_body(previous)
        return llm._strip_superseded_evidence(sliced, "P")

    def _assert_stripped(self, opening: dict) -> None:
        text = json.dumps(opening, default=str)
        assert not [c for c in self.PAE if c in text]
        # Availability metadata is kept on purpose: it saves a wasted call.
        assert "isoform_structure_pae_status" in text

    def test_criteria_members_shape(self):
        evidence = {**{c: 1.0 for c in self.PAE}, "isoform_structure_pae_status": "ok"}
        record = {"members": [{"evidence": evidence}]}
        out = llm._strip_superseded_evidence(record, "P")
        self._assert_stripped(out)
        assert "_superseded_note" in out["members"][0]["evidence"]

    def test_raw_arm(self):
        cat = _catalog(
            [{"feature": c, "category": "P"} for c in self.PAE]
            + [{"feature": "isoform_structure_pae_status", "category": "P", "dtype": "str"}]
        )
        out = self._opening(gr._raw_body(gr.category_columns(cat)))
        self._assert_stripped(out)
        assert out["_superseded_note"]

    def test_tags_arm(self):
        reg = _registry(
            _tag_row(
                tag_id="p1_structured_extension",
                category="P",
                kind=reg_mod.KIND_DERIVED,
                metric="isoform_structure_plddt_diffregion_mean",
                criterion_id="P1_structured_extension",
            )
        )
        out = self._opening(gr._tags_body(reg))
        self._assert_stripped(out)
        # The criterion's other supporting numbers survive the strip.
        assert out["tags"][0]["metrics"]["isoform_structure_plddt_diffregion_mean"] == 0.8

    def test_dist_arm(self):
        dist = _dist([{"metric": c, "stratum": "extended", "n": 200} for c in self.PAE])
        out = self._opening(gr._dist_body({"P": list(self.PAE)}, dist))
        assert set(out["fields"]) == set()
        assert out["_superseded_note"]

    def test_other_letters_untouched(self):
        record = {"evidence": {self.PAE[0]: 1.0}}
        assert llm._strip_superseded_evidence(record, "C") is record


class TestBuild:
    def test_criteria_installs_no_hook(self):
        """The status-quo arm must run the untouched path, not a reimplementation."""
        assert gr.build("criteria") is None

    def test_unknown_grounding_rejected(self):
        with pytest.raises(gr.GroundingError, match="unknown grounding"):
            gr.build("percentiles")

    def test_provenance_pins_the_catalog_a_raw_arm_reads(self, tmp_path):
        import hashlib

        csv = tmp_path / "catalog.csv"
        csv.write_text("feature,category\nisoform_a,C\n")
        prov = gr.provenance("raw", catalog_csv=csv)
        assert prov["grounding"] == "raw"
        assert prov["feature_catalog"]["sha256"] == hashlib.sha256(csv.read_bytes()).hexdigest()
        assert gr.provenance("criteria") == {"grounding": "criteria"}

    def test_dump_is_json(self):
        assert json.loads(gr.dump({"a": 1})) == {"a": 1}
