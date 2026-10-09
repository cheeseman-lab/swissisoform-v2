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

from swissisoform.site import llm  # noqa: E402

BASE = ROOT / "scripts" / "site" / "prompts"

SAMPLE = """<!-- @block:input_contract -->
You receive: the old contract.
<!-- @end -->

<!-- @block:roster -->
- C Conservation: C1 a member.
<!-- @end -->

Middle text that never varies.

<!-- @block:directionality -->
Directionality — get these right:
- a rule
<!-- @end -->

<!-- @block:judgment_tags -->
<!-- @end -->

Trailing text.

<!-- @block:id_examples -->
Never write C1.
<!-- @end -->

<!-- @block:id_example -->
Bad: "C1 reports 97.7%."
<!-- @end -->

<!-- @block:machinery -->
Never cite the `state` or a threshold.
<!-- @end -->
"""


def _norm(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------------------------------------------------------------------------
# Marker parsing
# ---------------------------------------------------------------------------


class TestParse:
    def test_finds_every_section(self):
        assert set(A.parse_blocks(SAMPLE)) == set(A.SECTIONS + A.ID_SECTIONS[A.BASE_PROMPTS[0]])

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

    def test_nohint_keeps_the_directionality_block(self):
        """The hint axis varies hint content only; dropping the system rules too
        made a hint effect indistinguishable from a gate effect.
        """
        assert A.render(SAMPLE, grounding="criteria", hints=False) == A.render(
            SAMPLE, grounding="criteria", hints=True
        )

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
        assert set(spans) == set(A.SECTIONS + A.ID_SECTIONS[name])

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
    def test_production_loader_strips_every_marker(self, name):
        """Production reads these files too; a marker is not an instruction."""
        assert "<!--" not in llm.load_system_prompt(BASE / name)

    @pytest.mark.parametrize("name", A.BASE_PROMPTS)
    def test_status_quo_arm_is_what_production_sends(self, name):
        text = (BASE / name).read_text(encoding="utf-8")
        status_quo = A.render(text, grounding="criteria", hints=True, name=name)
        assert status_quo.strip() == llm.load_system_prompt(BASE / name)

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
    @pytest.mark.parametrize("grounding", ["criteria", "raw", "tags", "dist"])
    def test_directionality_and_p2_gate_survive_hint_stripping(self, name, grounding):
        text = (BASE / name).read_text(encoding="utf-8")
        out = A.render(text, grounding=grounding, hints=False, name=name)
        assert "Directionality — get these right" in out
        if name != "category-pass-M.txt":
            assert "RMSD" in out and "pLDDT" in out

    @pytest.mark.parametrize("name", A.BASE_PROMPTS)
    @pytest.mark.parametrize("hints", [True, False])
    @pytest.mark.parametrize("grounding", sorted(A.ID_FREE_GROUNDINGS))
    def test_id_free_arms_name_no_member_ids_or_thresholds(self, grounding, hints, name):
        """raw/dist payloads carry neither; the prompt taught them to write both."""
        text = (BASE / name).read_text(encoding="utf-8")
        out = A.render(text, grounding=grounding, hints=hints, name=name)
        assert not re.findall(r"\b[CDLMPS][1-3]\b", out)
        # The dist contract's own "no thresholds" and the tempering rule, whose
        # anchors (pLDDT 0.70, ratio 1.0, phyloP 0) mean something outside the
        # pipeline, are the permitted mentions.
        body = re.sub(r"The one exception is tempering\..*?\n\n", "", out, flags=re.S)
        assert "threshold" not in body.lower().replace("no thresholds", "")
        assert "`reason`" not in body and "`state`" not in body
        assert "Directionality — get these right" in out
        if name == A.BASE_PROMPTS[0]:
            assert "core-fold change (Cα RMSD) is only meaningful" in out

    @pytest.mark.parametrize("grounding", ["raw", "tags", "dist"])
    def test_swapped_contract_keeps_the_role_sentence(self, grounding):
        text = (BASE / A.BASE_PROMPTS[0]).read_text(encoding="utf-8")
        out = A.render(text, grounding=grounding, hints=True)
        assert out.startswith("You are interpreting ONE evidence CATEGORY")

    @pytest.mark.parametrize(
        "name, member",
        [
            ("category-pass.txt", "C1 primate AA-identity"),
            ("category-pass-M.txt", "M1 germline tolerance"),
            ("category-pass-P.txt", "P1 Fold Confidence"),
        ],
    )
    @pytest.mark.parametrize("grounding", ["criteria", "tags"])
    def test_member_backed_arms_keep_the_roster(self, grounding, name, member):
        text = (BASE / name).read_text(encoding="utf-8")
        assert member in A.render(text, grounding=grounding, hints=True, name=name)

    @pytest.mark.parametrize("name", ["category-pass-M.txt", "category-pass-P.txt"])
    def test_id_free_tool_prompts_keep_their_orf_kind_rules(self, name):
        """Only the ids go: the per-ORF-type validity rules must survive the swap."""
        text = (BASE / name).read_text(encoding="utf-8")
        out = A.render(text, grounding="raw", hints=True, name=name)
        assert "EXTENSION (`extended`)" in out and "SEPARATE ORF (`uorf`" in out
        assert "`am_pathogenicity` is absent by construction" in out or name.endswith("P.txt")

    def test_id_free_rendering_needs_the_roster_block(self):
        no_roster = re.sub(r"<!-- @block:roster -->.*?<!-- @end -->\n", "", SAMPLE, flags=re.S)
        with pytest.raises(A.AssembleError, match="roster"):
            A.render(no_roster, grounding="raw", hints=True)


# ---------------------------------------------------------------------------
# Materialize
# ---------------------------------------------------------------------------


class TestMaterialize:
    def test_writes_all_three_prompts_and_the_schema(self, tmp_path):
        dest = A.materialize(tmp_path / "arm", grounding="raw", hints=True, base_dir=BASE)
        for name in A.BASE_PROMPTS:
            assert (dest / name).exists()
        schema = json.loads((dest / A.SCHEMA_REL).read_text())
        assert list(schema["properties"]) == ["reasoning", "evidence_used"]

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
        # The enum used to be pinned here too. With the verdict gone there is
        # nothing to pin, and `additionalProperties is False` above is the half
        # that was load-bearing — it is what keeps tags_fired out of the
        # single-shot decoder.
        assert "verdict" not in schema["properties"]

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


class TestReplicate:
    """The noise-floor arm. Its whole value is being indistinguishable from
    `criteria_hint` except in where it writes — any real difference makes it a
    ninth variant and useless as a measure of sampling variance.
    """

    def test_replicate_is_reachable_by_name(self):
        assert "criteria_hint_rep" in V.BY_ID
        assert V.select(["criteria_hint_rep"])[0].arm_id == "criteria_hint_rep"

    def test_replicate_is_not_in_the_default_selection(self):
        """Appending it to VARIANTS would silently make every existing
        "run all arms" command a 9-arm, ~$20-more run.
        """
        assert len(V.VARIANTS) == 8
        assert len(V.select(None)) == 8
        assert "criteria_hint_rep" not in {v.arm_id for v in V.select(None)}

    def test_replicate_matches_the_status_quo_framing(self):
        rep, base = V.BY_ID["criteria_hint_rep"], V.BY_ID["criteria_hint"]
        assert (rep.grounding, rep.hints) == (base.grounding, base.hints)

    def test_replicate_prompts_are_byte_identical_to_the_status_quo(self, tmp_path):
        """The assertion that makes it a floor: same grounding and hints must
        assemble the same prompt, or the two arms differ in framing as well as
        in sampling.
        """
        rep, base = V.BY_ID["criteria_hint_rep"], V.BY_ID["criteria_hint"]
        a = A.materialize(tmp_path / "rep", grounding=rep.grounding, hints=rep.hints, base_dir=BASE)
        b = A.materialize(
            tmp_path / "base", grounding=base.grounding, hints=base.hints, base_dir=BASE
        )
        for name in A.BASE_PROMPTS:
            assert (a / name).read_bytes() == (b / name).read_bytes(), name
        assert (a / A.SCHEMA_REL).read_bytes() == (b / A.SCHEMA_REL).read_bytes()

    def test_replicate_writes_somewhere_else(self):
        """Same framing, different paths — otherwise it overwrites the arm it is
        supposed to be compared against.
        """
        rep, base = V.BY_ID["criteria_hint_rep"], V.BY_ID["criteria_hint"]
        assert rep.out_run != base.out_run
        assert rep.capture_dir("cheeseman50") != base.capture_dir("cheeseman50")


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
        for grounding in ("criteria", "raw", "dist"):
            out = A.render(text, grounding=grounding, hints=True, name=name)
            assert "tags_fired" not in out

    def test_tool_prompts_are_asked_to_emit(self):
        text = (BASE / "category-pass-M.txt").read_text(encoding="utf-8")
        out = A.render(text, grounding="tags", hints=True, name="category-pass-M.txt")
        assert "pass the ids that fired as `tags_fired`" in out
        assert "Always include `tags_fired`, even when it is empty" in out

    def test_single_shot_prompt_is_told_it_has_none(self):
        text = (BASE / "category-pass.txt").read_text(encoding="utf-8")
        out = A.render(text, grounding="tags", hints=True)
        assert "does not apply to it" in out

    def test_instruction_survives_hint_stripping(self):
        """It states what to emit, not how to weigh evidence."""
        text = (BASE / "category-pass-P.txt").read_text(encoding="utf-8")
        out = A.render(text, grounding="tags", hints=False, name="category-pass-P.txt")
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


class TestVerdictIsGone:
    """The four declarations that used to cross-check each other via the enum.

    category_read.json, the two terminal-tool schemas and the prompt prose are
    validated against each other only at runtime, and the enum was what made a
    mismatch loud. With it removed nothing else pins them, so pin them here.
    """

    def test_shared_schema_has_no_verdict(self):
        schema = json.loads((BASE / "output_schemas" / "category_read.json").read_text())
        assert schema["required"] == ["reasoning"]
        assert "verdict" not in schema["properties"]

    def test_neither_terminal_tool_declares_a_verdict(self):
        from swissisoform.site import structure_tools as ST
        from swissisoform.site import tools as T

        for tools, const in ((T.M_TOOLS, T.EMIT_VERDICT), (ST.P_TOOLS, ST.EMIT_VERDICT)):
            emit = next(x for x in tools if x["name"] == const)
            assert "verdict" not in emit["input_schema"]["properties"]
            assert "verdict" not in emit["input_schema"]["required"]

    def test_no_prompt_asks_for_a_verdict(self):
        """The only guard against the prose drifting back.

        `emit_verdict` stays as the terminal tool's name -- it is persisted in
        every {letter}_trace.json outcome -- so the tool call itself is allowed.
        """
        for name in A.BASE_PROMPTS:
            text = (BASE / name).read_text()
            leftovers = [
                line
                for line in text.splitlines()
                if "verdict" in line and "emit_verdict" not in line
            ]
            assert not leftovers, f"{name} still asks for a verdict: {leftovers}"


def test_raw_and_dist_state_the_threshold_rule_once_like_every_arm():
    """The id-free machinery block must not repeat the always-kept tempering paragraph."""
    from pathlib import Path

    base = (
        Path(__file__).resolve().parents[1] / "scripts" / "site" / "prompts" / "category-pass.txt"
    ).read_text()
    arms = ("criteria", "tags", "raw", "dist")
    rendered = {g: A.render(base, grounding=g, hints=True) for g in arms}
    counts = {g: t.count("Never report that a value cleared") for g, t in rendered.items()}
    assert set(counts.values()) == {1}, counts
    assert all("move the verdict" not in t for t in rendered.values())
