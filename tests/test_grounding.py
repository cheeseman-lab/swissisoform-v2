"""Alternative groundings for the category LLM slice.

Offline: every test builds its own catalog / registry / distributions in memory or
under tmp_path, so nothing here needs the frozen artifacts or a run on disk.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from swissisoform import distributions as dist_mod
from swissisoform.site import evidence as ev
from swissisoform.site import grounding as gr
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
        monkeypatch.setitem(gr.ev.CRITERIA, "X", cfg)

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
                "name": "emit_verdict",
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
        tools = [{"name": "emit_verdict", "input_schema": {"properties": {}}}]
        gr.install_verdict_extras(tools, {"tags_fired": {}}, required=False)
        assert "tags_fired" not in tools[0]["input_schema"].get("required", [])

    def test_empty_extras_is_a_noop(self):
        tools = [{"name": "emit_verdict", "input_schema": {"properties": {}}}]
        before = copy.deepcopy(tools)
        gr.install_verdict_extras(tools, {})()
        assert tools == before


class TestBuild:
    def test_criteria_installs_no_hook(self):
        """The status-quo arm must run the untouched path, not a reimplementation."""
        assert gr.build("criteria") is None

    def test_unknown_grounding_rejected(self):
        with pytest.raises(gr.GroundingError, match="unknown grounding"):
            gr.build("percentiles")

    def test_dump_is_json(self):
        assert json.loads(gr.dump({"a": 1})) == {"a": 1}
