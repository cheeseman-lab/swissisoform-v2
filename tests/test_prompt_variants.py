"""Prompt marker-splice and the arm matrix.

Offline. The round-trip tests read the three tracked base prompts, which is the
point: they are what must not drift.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figures" / "prompt_variants"))

import assemble as A  # noqa: E402
import variants as V  # noqa: E402

BASE = ROOT / "scripts" / "site" / "prompts"

SAMPLE = """<!-- @block:input_contract -->
You receive: the old contract.
<!-- @end -->

Middle text that never varies.

<!-- @block:directionality -->
Directionality — get these right:
- a rule
<!-- @end -->

<!-- @block:judgment_tags -->
<!-- @end -->

Trailing text.
"""


def _norm(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------------------------------------------------------------------------
# Marker parsing
# ---------------------------------------------------------------------------


class TestParse:
    def test_finds_every_section(self):
        assert sorted(A.parse_blocks(SAMPLE)) == sorted(A.SECTIONS)

    def test_unclosed_block_rejected(self):
        with pytest.raises(A.AssembleError, match="never closed"):
            A.parse_blocks("<!-- @block:input_contract -->\ntext\n")

    def test_nested_block_rejected(self):
        text = "<!-- @block:a -->\n<!-- @block:b -->\n<!-- @end -->\n<!-- @end -->\n"
        with pytest.raises(A.AssembleError, match="not closed before"):
            A.parse_blocks(text)

    def test_duplicate_section_rejected(self):
        text = (
            "<!-- @block:input_contract -->\nx\n<!-- @end -->\n"
            "<!-- @block:input_contract -->\ny\n<!-- @end -->\n"
        )
        with pytest.raises(A.AssembleError, match="declared twice"):
            A.parse_blocks(text)

    def test_stray_end_rejected(self):
        with pytest.raises(A.AssembleError, match="no open @block"):
            A.parse_blocks("<!-- @end -->\n")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestRender:
    def test_markers_never_reach_the_model(self):
        for grounding in ("criteria", "raw", "tags", "dist"):
            for hints in (True, False):
                out = A.render(SAMPLE, grounding=grounding, hints=hints)
                assert "@block" not in out and "@end" not in out

    def test_criteria_with_hints_is_the_base_file_verbatim(self):
        """The status-quo arm must not be a rewrite of the production prompt."""
        out = A.render(SAMPLE, grounding="criteria", hints=True)
        stripped = "\n".join(
            ln
            for ln in SAMPLE.splitlines()
            if not (ln.strip().startswith("<!-- @block:") or ln.strip() == "<!-- @end -->")
        )
        assert _norm(out) == _norm(stripped)

    def test_nohint_drops_the_directionality_block_only(self):
        out = A.render(SAMPLE, grounding="criteria", hints=False)
        assert "Directionality" not in out
        assert "Middle text that never varies." in out
        assert "Trailing text." in out

    def test_grounding_swaps_the_contract(self):
        out = A.render(SAMPLE, grounding="tags", hints=True)
        assert "the old contract" not in out
        assert "controlled vocabulary of binary findings" in out

    def test_criteria_keeps_the_original_contract(self):
        out = A.render(SAMPLE, grounding="criteria", hints=True)
        assert "the old contract" in out

    def test_missing_section_is_an_error(self):
        with pytest.raises(A.AssembleError, match="missing section"):
            A.render(
                "<!-- @block:input_contract -->\nx\n<!-- @end -->\n", grounding="raw", hints=True
            )


# ---------------------------------------------------------------------------
# The tracked base prompts
# ---------------------------------------------------------------------------


class TestBasePrompts:
    @pytest.mark.parametrize("name", A.BASE_PROMPTS)
    def test_declares_every_section_exactly_once(self, name):
        """One missing marker and an arm silently keeps a block it should swap."""
        spans = A.parse_blocks((BASE / name).read_text(encoding="utf-8"))
        assert sorted(spans) == sorted(A.SECTIONS)

    @pytest.mark.parametrize("name", A.BASE_PROMPTS)
    def test_hint_on_round_trips_to_the_tracked_file(self, name):
        text = (BASE / name).read_text(encoding="utf-8")
        out = A.render(text, grounding="criteria", hints=True)
        stripped = "\n".join(
            ln
            for ln in text.splitlines()
            if not (ln.strip().startswith("<!-- @block:") or ln.strip() == "<!-- @end -->")
        )
        assert _norm(out) == _norm(stripped)

    @pytest.mark.parametrize("name", A.BASE_PROMPTS)
    def test_n_terminal_rule_survives_hint_stripping(self, name):
        """It is a property of the annotation, not guidance. Dropping it
        reintroduces the N/C-terminal confusion the PR #24 audit found fixed.
        """
        text = (BASE / name).read_text(encoding="utf-8")
        out = A.render(text, grounding="criteria", hints=False)
        assert "always N-terminal" in out

    @pytest.mark.parametrize("name", A.BASE_PROMPTS)
    def test_orf_kind_block_survives_hint_stripping(self, name):
        text = (BASE / name).read_text(encoding="utf-8")
        out = A.render(text, grounding="criteria", hints=False)
        assert "SEPARATE ORF" in out

    @pytest.mark.parametrize("name", A.BASE_PROMPTS)
    def test_directionality_is_gone_when_hints_are_off(self, name):
        text = (BASE / name).read_text(encoding="utf-8")
        out = A.render(text, grounding="criteria", hints=False)
        assert "Directionality — get these right" not in out


# ---------------------------------------------------------------------------
# Materialize
# ---------------------------------------------------------------------------


class TestMaterialize:
    def test_writes_all_three_prompts_and_the_schema(self, tmp_path):
        dest = A.materialize(tmp_path / "arm", grounding="raw", hints=True, base_dir=BASE)
        for name in A.BASE_PROMPTS:
            assert (dest / name).exists()
        schema = json.loads((dest / A.SCHEMA_REL).read_text())
        assert list(schema["properties"]) == ["verdict", "reasoning", "evidence_used"]

    def test_tags_fired_never_reaches_the_single_shot_decoder(self, tmp_path):
        """Declared as a property it is offered to C/D/L/S too, and the smoke test
        showed Conservation filling it with an M tag. Relax additionalProperties
        so the tool-loop payload validates without the field being emittable.
        """
        extras = {"M": {"tags_fired": {"type": "array", "items": {"type": "string"}}}}
        dest = A.materialize(
            tmp_path / "arm", grounding="tags", hints=True, base_dir=BASE, verdict_extras=extras
        )
        schema = json.loads((dest / A.SCHEMA_REL).read_text())
        assert "tags_fired" not in schema["properties"]
        # Closed, not relaxed: `additionalProperties: true` was tried and made it
        # worse — an open door instead of a labelled slot, so Detection started
        # emitting the field too.
        assert schema["additionalProperties"] is False
        assert schema["properties"]["verdict"]["enum"] == [
            "interesting",
            "neutral",
            "not_interesting",
        ]

        # The tool loop gets its own schema; it is validated with jsonschema after
        # the fact, never decoded against, so the field is safe to declare there.
        tool_schema = json.loads((dest / A.TOOL_SCHEMA_REL).read_text())
        assert "tags_fired" in tool_schema["properties"]
        assert tool_schema["additionalProperties"] is False

    def test_tool_schema_is_absent_when_there_are_no_judgment_tags(self, tmp_path):
        dest = A.materialize(tmp_path / "arm", grounding="raw", hints=True, base_dir=BASE)
        assert not (dest / A.TOOL_SCHEMA_REL).exists()

    def test_tool_schema_filename_matches_what_llm_looks_for(self):
        """Two constants, two files; a rename in one would silently disable the
        override and send the loop back to the shared schema."""
        from swissisoform.site import llm

        assert str(A.TOOL_SCHEMA_REL) == str(llm.TOOL_VERDICT_SCHEMA)

    def test_non_tags_arms_keep_the_schema_closed(self, tmp_path):
        dest = A.materialize(tmp_path / "arm", grounding="raw", hints=True, base_dir=BASE)
        schema = json.loads((dest / A.SCHEMA_REL).read_text())
        assert schema["additionalProperties"] is False

    def test_missing_base_prompt_is_an_error(self, tmp_path):
        with pytest.raises(A.AssembleError, match="missing base prompt"):
            A.materialize(tmp_path / "arm", grounding="raw", hints=True, base_dir=tmp_path)


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------


class TestVariants:
    def test_eight_arms_covering_the_full_cross(self):
        assert len(V.VARIANTS) == 8
        assert {(v.grounding, v.hints) for v in V.VARIANTS} == {
            (g, h) for g in V.GROUNDINGS for h in (True, False)
        }

    def test_arm_ids_are_unique(self):
        assert len({v.arm_id for v in V.VARIANTS}) == 8

    def test_status_quo_is_criteria_with_hints(self):
        assert V.BY_ID["criteria_hint"].note == "status quo"

    def test_capture_dir_is_namespaced_by_corpus(self):
        """Two corpora sharing a capture dir would truncate each other's index."""
        v = V.BY_ID["tags_hint"]
        assert v.capture_dir("cheeseman50") != v.capture_dir("cheeseman_test")

    def test_select_defaults_to_everything(self):
        assert len(V.select(None)) == 8 and len(V.select([])) == 8

    def test_select_rejects_a_typo_rather_than_running_nothing(self):
        with pytest.raises(KeyError, match="unknown arm"):
            V.select(["tags_hints"])


class TestTerminology:
    """The base prompts say "members" outside the marked block — in the task
    sentence, the all-None verdict rule, and the Bad/Good example. An arm whose
    payload has no members must be told what to read those as.
    """

    @pytest.mark.parametrize("grounding", ["raw", "tags", "dist"])
    def test_alternative_arms_get_a_terminology_mapping(self, grounding):
        out = A.render(SAMPLE, grounding=grounding, hints=True)
        assert "Terminology:" in out
        assert A.NOUNS[grounding] in out

    def test_criteria_arm_gets_none(self):
        """Its payload really does have members; a mapping would be noise."""
        assert "Terminology:" not in A.render(SAMPLE, grounding="criteria", hints=True)

    @pytest.mark.parametrize("name", A.BASE_PROMPTS)
    def test_mapping_survives_hint_stripping(self, name):
        """It is a statement about the payload, not interpretive guidance."""
        text = (BASE / name).read_text(encoding="utf-8")
        assert "Terminology:" in A.render(text, grounding="tags", hints=False)


class TestJudgmentTags:
    """The smoke test found both halves of this missing: nothing asked M/P to
    emit `tags_fired`, and the single-shot categories were offered another
    category's enum and filled it."""

    @pytest.mark.parametrize("name", A.BASE_PROMPTS)
    def test_only_the_tags_arm_mentions_tags_fired(self, name):
        text = (BASE / name).read_text(encoding="utf-8")
        tool = name != "category-pass.txt"
        for grounding in ("criteria", "raw", "dist"):
            out = A.render(text, grounding=grounding, hints=True, is_tool_prompt=tool)
            assert "tags_fired" not in out

    def test_tool_prompts_are_asked_to_emit(self):
        text = (BASE / "category-pass-M.txt").read_text(encoding="utf-8")
        out = A.render(text, grounding="tags", hints=True, is_tool_prompt=True)
        assert "pass the ids that fired as `tags_fired`" in out
        assert "Always include `tags_fired`, even when it is empty" in out

    def test_single_shot_prompt_is_told_it_has_none(self):
        text = (BASE / "category-pass.txt").read_text(encoding="utf-8")
        out = A.render(text, grounding="tags", hints=True, is_tool_prompt=False)
        assert "does not apply to it" in out

    def test_instruction_survives_hint_stripping(self):
        """It states what to emit, not how to weigh evidence."""
        text = (BASE / "category-pass-P.txt").read_text(encoding="utf-8")
        out = A.render(text, grounding="tags", hints=False, is_tool_prompt=True)
        assert "tags_fired" in out

    def test_enums_are_unioned_not_overwritten(self):
        """One category_read.json serves all six categories, so a per-category
        enum is impossible — flattening let P's enum reach Conservation."""
        merged = A._union_extras(
            {
                "M": {"tags_fired": {"type": "array", "items": {"enum": ["m1", "m2"]}}},
                "P": {"tags_fired": {"type": "array", "items": {"enum": ["p1"]}}},
            }
        )
        assert merged["tags_fired"]["items"]["enum"] == ["m1", "m2", "p1"]

    def test_union_does_not_mutate_its_input(self):
        source = {"M": {"tags_fired": {"items": {"enum": ["m1"]}}}}
        A._union_extras({**source, "P": {"tags_fired": {"items": {"enum": ["p1"]}}}})
        assert source["M"]["tags_fired"]["items"]["enum"] == ["m1"]
