"""The additive tag layer — registry, evaluator, module.

Offline: every test builds a registry frame in memory or under tmp_path, so
nothing here needs the frozen distributions, the feature catalog, or a run.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from swissisoform.config import PipelineConfig, ScoringConfig
from swissisoform.models import ORFType, TranslationInitiationSite
from swissisoform.modules.tags import (
    CITATIONS_COLUMN,
    STATES_COLUMN,
    VERSION_COLUMN,
    TagModule,
)
from swissisoform.tags import derived as derived_mod
from swissisoform.tags import evaluate as tag_eval
from swissisoform.tags import registry as reg_mod

ALL_ORF = "extended|truncated|uorf|uoorf|internal_oof|3utr_orf|alt_orf"


def _row(**kwargs) -> dict:
    """A registry row with every column present, overridden by *kwargs*."""
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
    base.update(kwargs)
    return base


def _registry(*rows: dict, version: str = "vtest") -> reg_mod.TagRegistry:
    return reg_mod.from_frame(
        version, pd.DataFrame(list(rows), columns=list(reg_mod.REGISTRY_COLUMNS)), {}
    )


def _frame(orf_types: list[str], **columns) -> pd.DataFrame:
    return pd.DataFrame({"orf_type": orf_types, **columns})


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_round_trips_through_parquet(self, tmp_path):
        frame = pd.DataFrame(
            [_row(tag_id="a"), _row(tag_id="b", kind=reg_mod.KIND_LLM)],
            columns=list(reg_mod.REGISTRY_COLUMNS),
        )
        vdir = tmp_path / "data" / "reference" / "tags" / "v9"
        vdir.mkdir(parents=True)
        frame.to_parquet(vdir / reg_mod.REGISTRY_FILE, index=False)
        (vdir / reg_mod.SIDECAR_FILE).write_text(json.dumps({"cutoff_source": "config"}))

        reg = reg_mod.load("v9", tmp_path)
        assert len(reg) == 2
        assert reg.provenance["cutoff_source"] == "config"
        assert reg.get("a").valid_for[0] == "extended"

    def test_missing_version_names_the_build_command(self, tmp_path):
        with pytest.raises(reg_mod.TagRegistryError, match="build_tag_registry"):
            reg_mod.load("nope", tmp_path)

    def test_missing_column_is_an_error_not_a_default(self):
        frame = pd.DataFrame([{"tag_id": "a"}])
        with pytest.raises(reg_mod.TagRegistryError, match="missing columns"):
            reg_mod.from_frame("v", frame, {})

    def test_duplicate_tag_id_rejected(self):
        with pytest.raises(reg_mod.TagRegistryError, match="duplicate tag_id"):
            _registry(_row(tag_id="dup"), _row(tag_id="dup", metric="other"))

    def test_llm_and_blocked_are_not_code_fired(self):
        reg = _registry(
            _row(tag_id="ok"),
            _row(tag_id="judged", kind=reg_mod.KIND_LLM),
            _row(tag_id="stale", blocked="metric renamed"),
        )
        assert [t.tag_id for t in reg.code_fired()] == ["ok"]
        assert reg.state_columns() == ("ok",)

    def test_axis_follows_category(self):
        assert reg_mod.axis_for("C") == "E"
        assert reg_mod.axis_for("D") == "E"
        assert reg_mod.axis_for("S") == "F"

    def test_overrides_parse_from_json(self):
        reg = _registry(
            _row(
                tag_id="c1",
                kind=reg_mod.KIND_DERIVED,
                cutoff_overrides=json.dumps({"c1_pident_min": 0.9}),
            )
        )
        assert reg.get("c1").cutoff_overrides == {"c1_pident_min": 0.9}

    def test_empty_overrides_is_an_empty_dict(self):
        assert _registry(_row()).get("t").cutoff_overrides == {}


# ---------------------------------------------------------------------------
# effective_scoring
# ---------------------------------------------------------------------------


class TestEffectiveScoring:
    def test_no_overrides_returns_base_unchanged(self):
        base = ScoringConfig()
        assert tag_eval.effective_scoring(_registry(_row()), base) is base

    def test_override_applied(self):
        reg = _registry(
            _row(
                kind=reg_mod.KIND_DERIVED,
                cutoff_overrides=json.dumps({"c1_pident_min": 0.42}),
            )
        )
        assert tag_eval.effective_scoring(reg, ScoringConfig()).c1_pident_min == 0.42

    def test_int_fields_stay_int(self):
        """A parquet double must not turn an int threshold into a float."""
        reg = _registry(
            _row(
                kind=reg_mod.KIND_DERIVED,
                cutoff_overrides=json.dumps({"min_cell_lines": 4.0}),
            )
        )
        value = tag_eval.effective_scoring(reg, ScoringConfig()).min_cell_lines
        assert value == 4
        assert isinstance(value, int)

    def test_unknown_field_raises_rather_than_silently_ignoring(self):
        reg = _registry(
            _row(
                kind=reg_mod.KIND_DERIVED,
                cutoff_overrides=json.dumps({"renamed_away": 1.0}),
            )
        )
        with pytest.raises(tag_eval.TagEvaluationError, match="not a ScoringConfig field"):
            tag_eval.effective_scoring(reg, ScoringConfig())


# ---------------------------------------------------------------------------
# Firing
# ---------------------------------------------------------------------------


class TestFire:
    def test_threshold_ge(self):
        reg = _registry(_row(tag_id="hi", metric="m", direction=">=", cutoff=2.0))
        df = _frame(["extended"] * 3, m=[1.0, 2.0, 3.0])
        states, _ = tag_eval.fire(df, [], reg)
        assert list(states["hi"]) == [False, True, True]

    def test_threshold_lt(self):
        reg = _registry(_row(tag_id="lo", metric="m", direction="<", cutoff=2.0))
        df = _frame(["extended"] * 3, m=[1.0, 2.0, 3.0])
        states, _ = tag_eval.fire(df, [], reg)
        assert list(states["lo"]) == [True, False, False]

    def test_null_metric_is_not_evaluable_not_false(self):
        """The distinction the whole layer exists for: NA never becomes False."""
        reg = _registry(_row(tag_id="hi", metric="m", cutoff=2.0))
        df = _frame(["extended"] * 2, m=[float("nan"), 5.0])
        states, citations = tag_eval.fire(df, [], reg)
        assert pd.isna(states["hi"].iloc[0])
        assert states["hi"].iloc[1] is True or states["hi"].iloc[1]
        assert pd.isna(citations["hi"].iloc[0])

    def test_outside_valid_for_is_not_evaluable(self):
        reg = _registry(
            _row(tag_id="paired", metric="m", cutoff=0.0, valid_for="extended|truncated")
        )
        df = _frame(["extended", "uorf"], m=[5.0, 5.0])
        states, citations = tag_eval.fire(df, [], reg)
        assert bool(states["paired"].iloc[0]) is True
        assert pd.isna(states["paired"].iloc[1])
        # A tag that did not apply cites nothing, even though the column has a value.
        assert pd.isna(citations["paired"].iloc[1])

    def test_empty_valid_for_means_everywhere(self):
        reg = _registry(_row(tag_id="any", metric="m", cutoff=0.0, valid_for=""))
        df = _frame(["uorf"], m=[5.0])
        states, _ = tag_eval.fire(df, [], reg)
        assert bool(states["any"].iloc[0]) is True

    def test_bool_tag_is_tristate(self):
        reg = _registry(_row(tag_id="b", kind=reg_mod.KIND_BOOL, metric="flag", cutoff=None))
        df = _frame(["extended"] * 3, flag=[True, False, None])
        states, citations = tag_eval.fire(df, [], reg)
        assert bool(states["b"].iloc[0]) is True
        assert bool(states["b"].iloc[1]) is False
        assert pd.isna(states["b"].iloc[2])
        # A boolean has no number behind it.
        assert citations["b"].isna().all()

    def test_missing_column_yields_an_all_null_column_not_a_missing_one(self):
        """The struct's fields must depend on the registry, not on the run."""
        reg = _registry(_row(tag_id="absent", metric="nope", cutoff=1.0))
        df = _frame(["extended"], m=[1.0])
        states, _ = tag_eval.fire(df, [], reg)
        assert list(states.columns) == ["absent"]
        assert states["absent"].isna().all()

    def test_llm_tags_never_fired_by_code(self):
        reg = _registry(_row(tag_id="judged", kind=reg_mod.KIND_LLM))
        df = _frame(["extended"], m=[1.0])
        states, _ = tag_eval.fire(df, [], reg)
        assert states.empty or list(states.columns) == []

    def test_column_order_follows_the_registry(self):
        reg = _registry(
            _row(tag_id="b", metric="m", cutoff=0.0),
            _row(tag_id="a", metric="m", cutoff=0.0),
        )
        df = _frame(["extended"], m=[1.0])
        states, citations = tag_eval.fire(df, [], reg)
        assert list(states.columns) == ["b", "a"]
        assert list(citations.columns) == ["b", "a"]

    def test_missing_orf_type_is_an_error(self):
        with pytest.raises(tag_eval.TagEvaluationError, match="orf_type"):
            tag_eval.fire(pd.DataFrame({"m": [1.0]}), [], _registry(_row()))

    def test_derived_tag_needs_aligned_sites(self):
        reg = _registry(
            _row(tag_id="d", kind=reg_mod.KIND_DERIVED, criterion_id="L1_localization_change")
        )
        df = _frame(["extended", "extended"], m=[1.0, 2.0])
        with pytest.raises(tag_eval.TagEvaluationError, match="in row order"):
            tag_eval.fire(df, [], reg)


# ---------------------------------------------------------------------------
# Derived tags call the real scorers
# ---------------------------------------------------------------------------


def _site(tis_id: str = "t1") -> TranslationInitiationSite:
    return TranslationInitiationSite(
        tis_id=tis_id,
        gene_name="G",
        transcript_id="ENST1",
        chrom="chr1",
        position=100,
        strand="+",
        start_codon="ATG",
        orf_type=ORFType.TRUNCATED,
    )


class TestDerived:
    def test_scorer_map_covers_every_criterion_and_names_match(self):
        """A criterion rename would otherwise silently score the wrong thing."""
        assert derived_mod.check_names(_site(), ScoringConfig()) == []
        assert len(derived_mod.SCORER_BY_CRITERION) == 16

    def test_irreducible_are_all_registered(self):
        for criterion_id in derived_mod.IRREDUCIBLE:
            assert criterion_id in derived_mod.SCORER_BY_CRITERION

    def test_unannotated_site_scores_not_evaluable(self):
        reg = _registry(
            _row(
                tag_id="l1",
                category="L",
                axis="F",
                kind=reg_mod.KIND_DERIVED,
                criterion_id="L1_localization_change",
                metric="",
                cutoff=None,
            )
        )
        df = _frame(["truncated"], m=[1.0])
        states, _ = tag_eval.fire(df, [_site()], reg)
        assert states["l1"].isna().all()

    def test_derived_tag_sees_the_registry_cutoff(self):
        """The gate stays; only the number moves."""
        site = _site()
        site.isoform_annotations["conservation_frame"] = {
            "summary": {"status": "ok"},
            "primate_mean_pident": 0.6,
        }
        df = _frame(["truncated"], m=[1.0])
        row = dict(
            tag_id="c1",
            kind=reg_mod.KIND_DERIVED,
            criterion_id="C1_primate_conservation",
            metric="",
            cutoff=None,
        )
        strict = _registry(_row(**row, cutoff_overrides=json.dumps({"c1_pident_min": 0.8})))
        loose = _registry(_row(**row, cutoff_overrides=json.dumps({"c1_pident_min": 0.5})))
        assert bool(tag_eval.fire(df, [site], strict)[0]["c1"].iloc[0]) is False
        assert bool(tag_eval.fire(df, [site], loose)[0]["c1"].iloc[0]) is True

    def test_unknown_criterion_is_a_build_error_not_a_null(self):
        with pytest.raises(KeyError):
            derived_mod.score_criterion("X9_invented", _site(), ScoringConfig())


# ---------------------------------------------------------------------------
# Struct conversion + the module
# ---------------------------------------------------------------------------


class TestModule:
    def test_writes_three_columns_and_the_annotation(self):
        reg = _registry(_row(tag_id="hi", metric="m", cutoff=2.0), version="v7")
        df = _frame(["extended", "extended"], m=[1.0, 3.0])
        sites = [_site("a"), _site("b")]

        out = TagModule(reg, PipelineConfig()).annotate_frame(df, sites)

        assert list(out[VERSION_COLUMN]) == ["v7", "v7"]
        assert out[STATES_COLUMN].iloc[0] == {"hi": False}
        assert out[STATES_COLUMN].iloc[1] == {"hi": True}
        assert out[CITATIONS_COLUMN].iloc[1] == {"hi": 3.0}
        # The object graph carries the same values as the frame.
        assert sites[1].isoform_annotations["tags"]["states"] == {"hi": True}
        assert sites[1].isoform_annotations["tags"]["registry_version"] == "v7"

    def test_input_frame_not_mutated(self):
        reg = _registry(_row(tag_id="hi", metric="m", cutoff=2.0))
        df = _frame(["extended"], m=[3.0])
        TagModule(reg).annotate_frame(df, [_site()])
        assert STATES_COLUMN not in df.columns

    def test_structs_use_none_so_parquet_writes_a_real_null(self):
        reg = _registry(_row(tag_id="hi", metric="m", cutoff=2.0))
        df = _frame(["extended"], m=[float("nan")])
        out = TagModule(reg).annotate_frame(df, [_site()])
        assert out[STATES_COLUMN].iloc[0] == {"hi": None}
        assert out[CITATIONS_COLUMN].iloc[0] == {"hi": None}

    def test_summary_counts_the_three_states(self):
        reg = _registry(_row(tag_id="hi", metric="m", cutoff=2.0))
        df = _frame(["extended"] * 3, m=[1.0, 3.0, float("nan")])
        module = TagModule(reg)
        out = module.annotate_frame(df, [_site(str(i)) for i in range(3)])
        assert module.summary(out)["hi"] == {"on": 1, "off": 1, "not_evaluable": 1}


class TestParquetRoundTrip:
    def test_nulls_survive_the_struct(self, tmp_path):
        """A null tag must not come back False after a parquet round-trip."""
        reg = _registry(_row(tag_id="hi", metric="m", cutoff=2.0))
        df = _frame(["extended", "extended"], m=[3.0, float("nan")])
        out = TagModule(reg).annotate_frame(df, [_site("a"), _site("b")])

        path = tmp_path / "paired.parquet"
        out.to_parquet(path, index=False)
        back = pd.read_parquet(path)
        assert back[STATES_COLUMN].iloc[0] == {"hi": True}
        assert back[STATES_COLUMN].iloc[1] == {"hi": None}


# ---------------------------------------------------------------------------
# Runner wiring — the layer is additive, so its absence must never fail a run
# ---------------------------------------------------------------------------


def _spec(**kwargs):
    from swissisoform.runner import RunSpec

    base = dict(
        gene_names=["X"],
        restricted_df=None,
        cell_lines=[],
        single_sample=False,
        min_cell_lines=1,
        skip=set(),
        run_name="t",
        fasta_out=None,
    )
    base.update(kwargs)
    return RunSpec(**base)


class TestRunnerWiring:
    def test_missing_registry_warns_and_leaves_the_frame_alone(self, caplog):
        from swissisoform.runner import _attach_tags

        df = _frame(["extended"], m=[1.0])
        with caplog.at_level("WARNING"):
            out = _attach_tags(df, [], PipelineConfig(), _spec(tag_registry_version="v_absent"))
        assert list(out.columns) == ["orf_type", "m"]
        assert "no tag registry at" in caplog.text

    def test_skip_modules_tags_is_honoured(self):
        from swissisoform.runner import _attach_tags

        df = _frame(["extended"], m=[1.0])
        out = _attach_tags(df, [], PipelineConfig(), _spec(skip={"tags"}))
        assert list(out.columns) == ["orf_type", "m"]

    def test_evaluation_failure_does_not_lose_the_run(self, tmp_path, monkeypatch):
        """A late optional stage must not cost a multi-hour run its output."""
        from swissisoform import runner

        reg = _registry(
            _row(tag_id="d", kind=reg_mod.KIND_DERIVED, criterion_id="L1_localization_change")
        )
        monkeypatch.setattr(runner, "load_tag_registry", lambda _v: reg)
        df = _frame(["extended", "extended"], m=[1.0, 2.0])
        # No sites for a registry that has a derived tag -> TagEvaluationError.
        out = runner._attach_tags(df, [], PipelineConfig(), _spec())
        assert list(out.columns) == ["orf_type", "m"]

    def test_tags_is_a_known_skip_module(self):
        """--skip-modules tags must not be rejected by the CLI's whitelist."""
        from swissisoform.references import ALL_POST_MODULES

        assert "tags" in ALL_POST_MODULES


class TestSchemaOverrides:
    def test_struct_types_are_declared_not_inferred(self):
        """An all-null tag must still write as bool, or shards disagree."""
        from swissisoform.modules.tags import schema_overrides

        df = pd.DataFrame({VERSION_COLUMN: ["v_absent"]})
        assert schema_overrides(df) == {}

    def test_no_tag_columns_means_no_overrides(self):
        from swissisoform.modules.tags import schema_overrides

        assert schema_overrides(pd.DataFrame({"orf_type": ["extended"]})) == {}


class TestEmptyVocabulary:
    def test_no_code_fired_tags_emits_no_columns(self):
        """An empty struct is a type Parquet cannot represent."""
        reg = _registry(_row(tag_id="judged", kind=reg_mod.KIND_LLM))
        df = _frame(["extended"], m=[1.0])
        out = TagModule(reg).annotate_frame(df, [_site()])
        assert list(out.columns) == ["orf_type", "m"]


class TestBuilderGuards:
    def test_renamed_criterion_fails_the_build(self, monkeypatch):
        """A criterion rename must not leave a derived tag scoring the wrong thing."""
        from swissisoform.setup import tags as build_mod

        monkeypatch.setattr(
            build_mod.derived_mod, "check_names", lambda *_: ["C1_primate_conservation"]
        )
        with pytest.raises(build_mod.TagBuildError, match="no longer match"):
            build_mod._check_scorer_names()

    def test_scorer_names_match_today(self):
        from swissisoform.setup import tags as build_mod

        build_mod._check_scorer_names()  # raises if the map has drifted
