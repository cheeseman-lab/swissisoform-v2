"""Tests for the Prometheus judge harness.

Weighted toward the things that would silently corrupt a result rather than error:
a paraphrased Prometheus template, a ``[RESULT]`` default, a bootstrap that ignores
clustering, a rubric that cannot fail a bad answer.
"""

from __future__ import annotations

import random

import pytest

from swissisoform.judge import ARMS, BASELINE, REPLICATE, UNITS
from swissisoform.judge import checks as K
from swissisoform.judge import prompts as PR
from swissisoform.judge import quantize as Q
from swissisoform.judge import reference as RF
from swissisoform.judge import rubrics as RB
from swissisoform.judge import weigh as W
from swissisoform.judge.weigh import Comparison, Score


class TestPrompts:
    def test_templates_are_the_published_shape(self):
        """A paraphrased template still answers, so nothing errors -- the scores
        just stop meaning what the benchmarks measured.
        """
        for template, headers in (
            (
                PR.ABSOLUTE_TEMPLATE,
                (
                    "###Task Description:",
                    "###The instruction to evaluate:",
                    "###Response to evaluate:",
                    "###Score Rubrics:",
                    "###Feedback: ",
                ),
            ),
            (
                PR.RELATIVE_TEMPLATE,
                (
                    "###Task Description:",
                    "###Instruction:",
                    "###Response A:",
                    "###Response B:",
                    "###Score Rubric:",
                    "###Feedback: ",
                ),
            ),
        ):
            for header in headers:
                assert header in template, header
            assert template.endswith("###Feedback: ")
            assert "1. Write a detailed feedback" in template
            assert "4. Please do not generate any other opening" in template

    def test_absolute_asks_for_an_integer_and_relative_for_a_letter(self):
        assert "an integer number between 1 and 5" in PR.ABSOLUTE_TEMPLATE
        assert "(A or B)" in PR.RELATIVE_TEMPLATE

    def test_no_reference_answer_block(self):
        """We have no gold verdicts; a model-written one would be a ninth framing
        smuggled in as ground truth.
        """
        assert "Reference Answer" not in PR.ABSOLUTE_TEMPLATE

    def test_filled_prompt_has_no_leftover_placeholders(self):
        out = PR.absolute_prompt(instruction="I", response="R", rubric="U")
        assert "{" not in out and "}" not in out
        assert out.startswith(PR.ABS_SYSTEM)

    def test_chat_wrapping(self):
        assert PR.chat("x") == "<s>[INST] x [/INST]"

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Feedback: fine. [RESULT] 4", 4),
            ("Feedback: fine. [RESULT] (2)", 2),
            ("[result] 5", 5),
            # Prose can quote the instruction's own format string, so the LAST
            # match is the score.
            ("mentions [RESULT] 1 then concludes [RESULT] 5", 5),
        ],
    )
    def test_parse_score(self, text, expected):
        assert PR.parse_score(text)[0] == expected

    @pytest.mark.parametrize("bad", ["", "no marker", "[RESULT] 6", "[RESULT] 0", "[RESULT] A"])
    def test_unparseable_score_raises_rather_than_defaulting(self, bad):
        """A silent 3 is indistinguishable from a real middling score and would
        drag every mean toward the centre.
        """
        with pytest.raises(PR.ParseError):
            PR.parse_score(bad)

    @pytest.mark.parametrize(
        "text,expected", [("[RESULT] A", "A"), ("[RESULT] (B)", "B"), ("x [result] b", "B")]
    )
    def test_parse_choice(self, text, expected):
        assert PR.parse_choice(text)[0] == expected

    @pytest.mark.parametrize("bad", ["", "[RESULT] C", "[RESULT] 3", "no marker"])
    def test_unparseable_choice_raises(self, bad):
        with pytest.raises(PR.ParseError):
            PR.parse_choice(bad)


class TestChoiceProbability:
    """The verdict must be read at ``[RESULT]``, not at the first A/B in prose.

    Prometheus opens its feedback "Response A ..." as narrative order, so an
    unanchored scan reported P(A) = 0.999 on byte-identical responses whose parsed
    verdicts ran 8/7. These pin the anchor.
    """

    @staticmethod
    def _steps(verdict_a: float, verdict_b: float, *, prose_a: float = -0.001):
        """Feedback prose that leads with "A", then the real verdict."""
        import math

        return [
            {"Response": -0.01, "Both": -4.0},
            {" A": prose_a, " B": -9.0},
            {" reads": -0.1, " fails": -2.0},
            {"[RESULT]": -0.02, "\n": -5.0},
            {" A": math.log(verdict_a), " B": math.log(verdict_b)},
        ]

    def test_reads_the_verdict_not_the_prose(self):
        # Prose is ~certain "A"; the verdict is 0.3/0.7 against it.
        assert PR.choice_probability(self._steps(0.3, 0.7), "[RESULT] B") == pytest.approx(0.3)

    def test_prose_certainty_does_not_leak_in(self):
        # Same verdict, prose made even more certain: the answer must not move.
        near = PR.choice_probability(self._steps(0.3, 0.7, prose_a=-1e-9), "[RESULT] B")
        assert near == pytest.approx(0.3)

    def test_last_marker_wins(self):
        """Matches parse_choice, which takes the last [RESULT]."""
        import math

        steps = self._steps(0.9, 0.1) + [
            {"[RESULT]": -0.01, ".": -6.0},
            {" B": math.log(0.8), " A": math.log(0.2)},
        ]
        assert PR.choice_probability(steps, "[RESULT] A [RESULT] B") == pytest.approx(0.2)

    def test_missing_loser_scores_at_the_floor(self):
        """A confident verdict drops the loser from top-k; keep the call anyway.

        Dropping these would discard the most decisive verdicts and leave only
        indecisive ones.
        """
        import math

        steps = self._steps(0.5, 0.5)[:-1] + [{" A": math.log(0.99), "x": math.log(1e-4)}]
        prob = PR.choice_probability(steps, "[RESULT] A")
        assert prob is not None and prob > 0.99

    def test_no_marker_gives_none(self):
        assert PR.choice_probability(self._steps(0.3, 0.7)[:3], "no verdict here") is None

    def test_reconstruction_mismatch_gives_none(self):
        """Rebuilt text disagreeing with the completion means the alignment is not
        trustworthy; refuse rather than score an arbitrary position.
        """
        assert PR.choice_probability(self._steps(0.3, 0.7), "nothing to see") is None

    def test_no_logprobs_gives_none(self):
        assert PR.choice_probability([], "[RESULT] A") is None

    def test_verdict_far_past_the_marker_is_not_scored(self):
        """Only a short window after the marker counts -- 283 of 7,261 completions
        continue with "Response A/B" prose, which is not a verdict.
        """
        import math

        steps = self._steps(0.3, 0.7)[:4] + [{" ": -0.01}] * 6
        steps += [{" A": math.log(0.9), " B": math.log(0.1)}]
        assert PR.choice_probability(steps, "[RESULT]") is None

    def test_emitted_text_is_the_argmax_path(self):
        assert PR.emitted_text(self._steps(0.3, 0.7)) == "Response A reads[RESULT] B"

    @pytest.mark.parametrize("spelling", ["A", " A", "(A)", "[A]"])
    def test_letter_spellings(self, spelling):
        import math

        steps = self._steps(0.5, 0.5)[:-1] + [{spelling: math.log(0.7), " B": math.log(0.3)}]
        assert PR.choice_probability(steps, "[RESULT] A") == pytest.approx(0.7)


class TestLogprobFlattening:
    """The rebuilt text must be the model's, with nothing substituted into it."""

    class _Info:
        def __init__(self, text, logprob):
            self.decoded_token, self.logprob = text, logprob

    class _Completion:
        def __init__(self, steps):
            self.logprobs = steps

    def _flat(self, steps):
        from swissisoform.judge.serve import _flatten_logprobs

        return _flatten_logprobs(self._Completion(steps))

    def test_empty_decode_is_preserved_not_replaced_by_an_id(self):
        """``x or str(token_id)`` put digits in the text and broke the anchor.

        Mixtral decodes a lone leading-space token to "", which is falsy; the id
        went in instead, turning [RESULT] into [28705RESULT] so the marker could
        not be found.
        """
        steps = self._flat([{28705: self._Info("", -0.02), 7: self._Info("w", -6.0)}])
        assert "" in steps[0]
        assert "28705" not in steps[0]

    def test_marker_survives_an_empty_token_inside_it(self):
        import math

        steps = self._flat(
            [
                {1: self._Info("[", -0.01)},
                {28705: self._Info("", -0.02)},
                {3: self._Info("RESULT", -0.01)},
                {5: self._Info("]", -0.01)},
                {8: self._Info("A", math.log(0.35)), 9: self._Info("B", math.log(0.65))},
            ]
        )
        assert PR.emitted_text(steps) == "[RESULT]B"
        assert PR.choice_probability(steps, "[RESULT] B") == pytest.approx(0.35)

    def test_absent_attribute_is_marked_as_an_id(self):
        """A genuinely missing decode is still distinguishable from model text."""

        class Bare:
            logprob = -0.5

        steps = self._flat([{99: Bare()}])
        assert steps[0] == {"<id:99>": -0.5}

    def test_colliding_decodes_keep_the_likelier(self):
        steps = self._flat([{1: self._Info(" A", -3.0), 2: self._Info(" A", -0.5)}])
        assert steps[0][" A"] == -0.5

    def test_no_logprobs_gives_empty_list(self):
        assert self._flat(None) == []


class TestForcedVerdict:
    """The second pass for completions that never wrote a verdict.

    Measured on the gate probes: 7 of 40 stopped at 341 tokens of a 512 cap, ended
    a complete sentence, and emitted no marker. Appending "[RESULT]" to the prompt
    puts the verdict at generated position 0, where it cannot be skipped.
    """

    @staticmethod
    def _step(p_a: float):
        import math

        return [{" A": math.log(p_a), " B": math.log(1 - p_a)}]

    def test_reads_position_zero_without_a_marker(self):
        """The forced pass has no [RESULT] to anchor on, by construction."""
        assert PR.verdict_probability(self._step(0.4)) == pytest.approx(0.4)

    def test_no_letters_gives_none(self):
        assert PR.verdict_probability([{"the": -0.1, "score": -2.0}]) is None

    def test_empty_gives_none(self):
        assert PR.verdict_probability([]) is None
        assert PR.verdict_probability(None) is None

    def test_preference_prefers_the_direct_channel(self):
        """A readable completion must not be overridden by a stale forced step."""
        import math

        direct = [
            {"[RESULT]": -0.01},
            {" A": math.log(0.8), " B": math.log(0.2)},
        ]
        assert PR.preference(direct, "[RESULT] A", self._step(0.1)) == pytest.approx(0.8)

    def test_preference_falls_back_when_no_verdict_was_written(self):
        """This is the case the regex loses entirely: prose, no marker, no letter."""
        rambled = [{"Both": -0.01}, {" responses": -0.01}, {" are": -0.01}]
        assert PR.choice_probability(rambled, "Both responses are supported.") is None
        assert PR.preference(rambled, "Both responses are supported.", self._step(0.3)) == (
            pytest.approx(0.3)
        )

    def test_preference_none_when_neither_channel_fired(self):
        assert PR.preference([], "no verdict", []) is None


class TestRewardExtraction:
    """Pulling (instruction, response) back out of the Prometheus pairwise prompts.

    The reward model must read byte-identical input to what the judge read, or the
    two judges are not comparable -- which is the only reason to run both.
    """

    @staticmethod
    def _prompt(instruction: str, response_a: str, response_b: str) -> str:
        return PR.relative_prompt(
            instruction=instruction,
            response_a=response_a,
            response_b=response_b,
            rubric="does it hold up",
        )

    def _extract(self, tmp_path, rows):
        import importlib.util
        import json

        path = tmp_path / "requests.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        spec = importlib.util.spec_from_file_location("rr", "scripts/judge/run_reward.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.extract_pairs(path)

    def test_round_trips_instruction_and_response(self, tmp_path):
        row = {
            "kind": "pairwise",
            "id": "pw|s|C|x|y|0",
            "slug": "s",
            "unit": "C",
            "arm_a": "x",
            "arm_b": "y",
            "prompt": self._prompt("EVIDENCE PAYLOAD", "ARM X SAID THIS", "ARM Y SAID THAT"),
        }
        pairs = self._extract(tmp_path, [row])
        assert len(pairs) == 1
        slug, unit, arm, instruction, response = pairs[0]
        assert (slug, unit, arm) == ("s", "C", "x")
        assert instruction == "EVIDENCE PAYLOAD"
        assert response == "ARM X SAID THIS"

    def test_one_row_per_arm_output(self, tmp_path):
        """An arm appears in 36 pairings per cell; it must be scored once."""
        rows = [
            {
                "kind": "pairwise",
                "id": f"pw|s|C|x|{other}|0",
                "slug": "s",
                "unit": "C",
                "arm_a": "x",
                "arm_b": other,
                "prompt": self._prompt("PAYLOAD", "ARM X", "OTHER"),
            }
            for other in ("y", "z", "w")
        ]
        assert len(self._extract(tmp_path, rows)) == 1

    def test_no_marker_leakage(self, tmp_path):
        """A template marker in the extracted text means the split drifted."""
        row = {
            "kind": "pairwise",
            "id": "pw|s|C|x|y|0",
            "slug": "s",
            "unit": "C",
            "arm_a": "x",
            "arm_b": "y",
            "prompt": self._prompt("PAYLOAD", "ARM X", "ARM Y"),
        }
        _, _, _, instruction, response = self._extract(tmp_path, row and [row])[0]
        assert "###" not in instruction
        assert "###" not in response
        assert "ARM Y" not in response

    def test_absolute_requests_are_ignored(self, tmp_path):
        rows = [{"kind": "absolute", "id": "abs|s|C|x|R1", "slug": "s", "unit": "C", "prompt": "x"}]
        assert self._extract(tmp_path, rows) == []


class TestRubrics:
    """Rubrics load from text files under scripts/judge/prompts/.

    Text rather than Python constants so a wording change is reviewable as a diff
    of the words. These tests pin the properties that the measured failures argue
    for, not the prose itself -- the prose is meant to be edited.
    """

    def test_every_active_rubric_loads(self):
        for rid in RB.all_ids():
            r = RB.by_id(rid)
            assert r.criterion.strip()
            assert r.path.exists()

    def test_two_category_axes_and_three_synthesis(self):
        """Two, not four. A calibration axis scored a deliberately miscalibrated
        verdict 4/5, and a directionality axis scored 4, 4, then 5/5 across three
        wordings; both were removed. An axis that cannot fail its own anchor adds
        variance without discriminating between arms.
        """
        assert RB.CATEGORY_IDS == (
            "R1_evidential_support",
            "R2_not_evaluable_discipline",
        )
        assert len(RB.SYNTHESIS_IDS) == 3

    def test_absolute_rubrics_declare_five_levels(self):
        """A rubric short of a level still returns scores -- on a scale its own
        text does not describe -- so the loader refuses it.
        """
        for rid in (*RB.CATEGORY_IDS, *RB.SYNTHESIS_IDS):
            assert len(RB.by_id(rid).levels) == 5

    def test_pairwise_declares_no_levels(self):
        """Relative grading emits A or B, so Score lines there would be a sign the
        wrong template was edited.
        """
        assert RB.pairwise().levels == ()

    def test_a_malformed_rubric_is_refused(self, tmp_path):
        (tmp_path / "absolute").mkdir()
        (tmp_path / "absolute" / "R1_evidential_support.txt").write_text(
            "criterion\n\nScore 1: a\nScore 2: b\n"
        )
        with pytest.raises(RB.RubricError, match="missing Score line"):
            RB.load("R1_evidential_support", tmp_path)

    def test_a_missing_rubric_is_refused(self, tmp_path):
        with pytest.raises(RB.RubricError, match="no rubric file"):
            RB.load("R1_evidential_support", tmp_path)

    def test_every_rubric_rejects_style_and_length(self):
        """Every measured failure rewarded fluent, confident prose: 100% position
        bias on identical text, and two axes scoring obviously-wrong verdicts 4-5.
        So each file says so explicitly.
        """
        for rid in RB.all_ids():
            assert "NOT style or length" in RB.by_id(rid).criterion, rid

    def test_pairwise_makes_overreach_a_defect(self):
        """The anchors caught the judge rewarding confident overreach. The
        tiebreaker has to fire BEFORE any preference for detail, or a longer,
        more assertive answer wins on length alone.
        """
        text = RB.pairwise().criterion
        assert "TIEBREAKER" in text
        assert "asserts LESS" in text
        assert "DEFECT" in text

    def test_surviving_axes_name_a_locatable_referent(self):
        """The axes that work ask whether something is *in the payload*; the two
        that failed asked the judge to weigh or cross-check.
        """
        assert "present in that evidence" in RB.by_id("R1_evidential_support").criterion
        assert "not_evaluable" in RB.by_id("R2_not_evaluable_discipline").criterion

    def test_removed_axes_have_no_files(self):
        for gone in ("R3_calibration", "R4_directionality"):
            assert gone not in RB.all_ids()
            with pytest.raises(RB.RubricError):
                RB.load(gone)

    def test_render_is_the_file_verbatim(self):
        """What a reviewer reads in the diff is what the model receives."""
        r = RB.by_id("R1_evidential_support")
        assert r.render() == r.path.read_text(encoding="utf-8").strip()

    def test_evidence_used_is_in_no_rubric(self):
        """Non-empty in only ~40% of outputs in every arm, so it cannot separate
        arms; scoring it adds variance with no discriminating power.
        """
        for rid in RB.all_ids():
            assert "evidence_used" not in RB.by_id(rid).criterion


class TestNumberExtraction:
    @pytest.mark.parametrize(
        "text,expected",
        [
            # Range dashes are not minus signs. The first run of the fabrication
            # check reported 54% almost entirely because of these.
            ("range (0.05-0.12)", {"0.05", "0.12"}),
            ("run 91-97% in alignments", {"91", "97"}),
            ("elements 14-18, 21-28", {"14", "18", "21", "28"}),
            ("at the 41st-75th percentiles", {"41"}),  # 75 is a landmark
            # A real negative survives.
            ("delta of -0.42", {"-0.42"}),
            ("around -18 to -16", {"-18", "-16"}),
        ],
    )
    def test_ranges_do_not_become_negatives(self, text, expected):
        assert K.numbers_in(text) == expected

    def test_percentile_landmarks_are_ignored(self):
        assert K.numbers_in("above the 95th percentile") == set()

    def test_thousands_separators(self):
        assert "13690" in K.numbers_in("13,690 variant rows")

    @pytest.mark.parametrize(
        "stored,cited",
        [
            (4.031841, "4.03"),  # rounded
            (0.0063124, "0.0063"),  # truncated
            (0.00025, "2.5e-4"),  # scientific
            (3.4e-61, "3.4e-61"),
            (3.4e-61, "1e-61"),  # order of magnitude
            (0.41, "41"),  # as a percentage
        ],
    )
    def test_correct_citation_is_not_fabrication(self, stored, cited):
        assert cited in K.reference_numbers({"v": stored})

    def test_arithmetic_on_lengths_is_not_fabrication(self):
        """Cites 243 - 45: "removes the N-terminal 198 residues"."""
        allowed = K.reference_numbers(
            {"isoform": {"canonical_length_aa": 243, "isoform_length_aa": 45}}
        )
        assert "198" in allowed

    def test_sequence_length_is_not_fabrication(self):
        """Cites len(differential_sequence): "the added 55-aa extension".

        That number appears nowhere in the payload as a number.
        """
        allowed = K.reference_numbers({"isoform": {"differential_sequence": "M" * 55}})
        assert "55" in allowed

    def test_a_genuinely_invented_number_is_caught(self):
        found = K.check_fabrication(
            arm="a",
            slug="s",
            unit="C",
            reasoning="phyloP over the region is 7.77",
            reference={"criteria": [{"id": "C3", "value": 4.03}]},
        )
        assert found and "7.77" in found[0].detail

    def test_tool_loop_units_are_skipped(self):
        """M and P queried the full tables through readers while the reference
        holds a 30-row sample, so absence there is not evidence of invention.
        """
        for unit in ("M", "P"):
            assert (
                K.check_fabrication(
                    arm="a", slug="s", unit=unit, reasoning="value 9999", reference={}
                )
                == []
            )


class TestNotEvaluable:
    def test_unmeasured_fields_include_nulls(self):
        found = K.unmeasured_fields(
            {"criteria": [{"id": "P1", "value": None}, {"id": "P2", "status": "no_cache"}]}
        )
        assert "P1" in found
        assert any(v == "no_cache" for v in found.values())

    def test_absence_claim_against_an_unmeasured_field_fires(self):
        found = K.check_not_evaluable(
            arm="a",
            slug="s",
            unit="P",
            reasoning="There is no secondary structure in the extension.",
            reference={"criteria": [{"id": "secondary_structure", "value": None}]},
            keywords=["secondary structure"],
        )
        assert found

    def test_generic_field_fragments_do_not_fire(self):
        """Normal prose like "no evidence for X" must not fire.

        Matching it against a field named `..._evidence` fired on every P cell in
        every arm on the first run.
        """
        assert (
            K.check_not_evaluable(
                arm="a",
                slug="s",
                unit="P",
                reasoning="There is no evidence of a fold change.",
                reference={"criteria": [{"id": "p1_evidence", "value": None}]},
            )
            == []
        )


class TestWeighing:
    def _synthetic(self, spread: float, seed: int = 7):
        rng = random.Random(seed)
        truth = {a: spread * i for i, a in enumerate(ARMS)}
        out = []
        for slug in (f"iso{i}" for i in range(50)):
            for i, a in enumerate(ARMS):
                for b in ARMS[i + 1 :]:
                    p = 1 / (1 + pow(2.718281828, -(truth[a] - truth[b])))
                    w, loser = (a, b) if rng.random() < p else (b, a)
                    out.append(Comparison(slug=slug, unit="C", winner=w, loser=loser))
        return out, truth

    def test_bt_recovers_the_ordering(self):
        comps, truth = self._synthetic(0.6)
        fit = W.bradley_terry(comps)
        ranked = [a for a, _ in sorted(fit.items(), key=lambda kv: kv[1])]
        assert ranked == sorted(ARMS, key=lambda a: truth[a])

    def test_bt_pins_the_baseline_at_zero(self):
        comps, _ = self._synthetic(0.6)
        assert W.bradley_terry(comps)[BASELINE] == pytest.approx(0.0, abs=1e-9)

    def test_bt_shrinks_rather_than_diverging_on_a_sweep(self):
        """With 50 isoforms a clean sweep is common; unregularised BT would give
        an infinite strength for it.
        """
        comps = [
            Comparison(slug=f"iso{i}", unit="C", winner=ARMS[1], loser=BASELINE) for i in range(50)
        ]
        fit = W.bradley_terry(comps)
        assert fit[ARMS[1]] > 1.0
        assert fit[ARMS[1]] < 100.0

    def test_bt_on_no_comparisons_is_empty_not_an_error(self):
        """A real outcome for a unit where the judge contradicted itself throughout."""
        assert W.bradley_terry([]) == {}

    def test_null_corpus_intervals_span_zero(self):
        rng = random.Random(11)
        comps = [
            Comparison(
                slug=f"iso{i}",
                unit="C",
                **({"winner": a, "loser": b} if rng.random() < 0.5 else {"winner": b, "loser": a}),
            )
            for i in range(50)
            for j, a in enumerate(ARMS)
            for b in ARMS[j + 1 :]
        ]
        fit = W.cluster_bootstrap_bt(comps, n=200)
        spanning = [a for a, v in fit.items() if a != BASELINE and not v.excludes_zero]
        assert len(spanning) >= len(ARMS) - 2

    def test_centering_removes_isoform_difficulty(self):
        """A conserved truncation outscores a uORF under every arm, so raw means
        conflate arm quality with which isoforms are easy.
        """
        scores = []
        for i in range(30):
            hard = -1.5 if i % 2 else 1.5
            for j, arm in enumerate(ARMS):
                scores.append(
                    Score(
                        slug=f"iso{i}",
                        unit="C",
                        arm=arm,
                        rubric="R1",
                        score=max(1, min(5, round(3 + hard + 0.25 * j))),
                    )
                )
        centered = W.center_scores(scores)
        series = [centered[(a, "C", "R1")] for a in ARMS]
        assert series == sorted(series)

    def test_single_arm_cell_is_dropped(self):
        """With one arm the centered value is 0 by construction and carries no
        comparison.
        """
        assert W.center_scores([Score(slug="i", unit="C", arm="a", rubric="R", score=5)]) == {}

    def test_order_inconsistent_pairs_are_dropped_not_split(self):
        forward = {
            ("s", "C", "a", "b"): "a",
            ("s", "C", "b", "a"): "a",  # consistent
            ("s", "D", "a", "b"): "a",
            ("s", "D", "b", "a"): "b",  # the judge changed its mind
        }
        comps, checks = W.resolve_orders(forward)
        assert len(comps) == 1 and comps[0].unit == "C"
        assert checks["D"].inconsistent == 1
        assert checks["D"].inconsistency_rate == 1.0

    def test_unparseable_pair_is_counted_separately(self):
        comps, checks = W.resolve_orders({("s", "C", "a", "b"): None, ("s", "C", "b", "a"): "a"})
        assert comps == [] and checks["C"].unparseable == 1

    def test_missing_reverse_order_is_not_a_verdict(self):
        comps, checks = W.resolve_orders({("s", "C", "a", "b"): "a"})
        assert comps == [] and checks == {}

    def test_floor_is_per_unit(self):
        base = {("i0", "C"): "x", ("i1", "C"): "x", ("i0", "S"): "x", ("i1", "S"): "x"}
        rep = {("i0", "C"): "x", ("i1", "C"): "x", ("i0", "S"): "y", ("i1", "S"): "x"}
        floor = W.verdict_floor(base, rep, ("C", "S"))
        assert floor.by_unit["C"] == 0.0
        assert floor.by_unit["S"] == 0.5
        assert floor.overall == 0.25

    def test_effect_in_floor_units(self):
        floor = W.Floor(by_unit={"C": 0.04, "S": 0.20}, overall=0.117)
        assert floor.units_of(0.28, "C") == pytest.approx(7.0)
        assert floor.units_of(0.20, "S") == pytest.approx(1.0)

    def test_zero_floor_is_infinite_not_a_crash(self):
        floor = W.Floor(by_unit={"C": 0.0}, overall=0.1)
        assert floor.units_of(0.1, "C") == float("inf")

    def test_factorial_excludes_the_replicate(self):
        """The replicate is not a cell of the 4x2 design; including it would
        invent a fifth grounding and double-weight `criteria`.
        """
        strengths = {a: W.Interval(point=float(i), lo=0, hi=0) for i, a in enumerate(ARMS)}
        strengths[REPLICATE] = W.Interval(point=99.0, lo=0, hi=0)
        eff = W.factorial_effects(strengths)
        assert set(eff["grounding"]) == {"criteria", "raw", "tags", "dist"}
        assert set(eff["hint"]) == {"hint", "nohint"}
        assert all(v < 99 for v in eff["grounding"].values())


class TestConstants:
    def test_seven_units_six_categories_plus_synthesis(self):
        assert len(UNITS) == 7
        assert UNITS[-1] == "synthesis"

    def test_replicate_is_not_one_of_the_eight_design_arms(self):
        assert len(ARMS) == 8
        assert REPLICATE not in ARMS


class TestTemplateFidelity:
    """Pin our vendored templates against the published package.

    Skipped where `prometheus_eval` is absent (the base env, which runs the
    request builder) and enforced in the judge env. This test earned its keep
    immediately: the first version of prompts.py carried a `"Feedback: "` prefix
    in the output-format line that the real template does not have. A paraphrased
    template still answers, so nothing errors -- the scores just stop meaning what
    the benchmarks measured.
    """

    def _norm(self, text: str) -> str:
        """Placeholder names differ; the prose must not."""
        for a, b in (
            ("{orig_instruction}", "{instruction}"),
            ("{orig_response}", "{response}"),
            ("{orig_response_A}", "{response_A}"),
            ("{orig_response_B}", "{response_B}"),
            ("{score_rubric}", "{rubric}"),
            ("{orig_criteria}", "{rubric}"),
        ):
            text = text.replace(a, b)
        return text.strip()

    def test_absolute_matches_package(self):
        pytest.importorskip("prometheus_eval")
        from prometheus_eval.prompts import ABSOLUTE_PROMPT_WO_REF

        assert self._norm(PR.ABSOLUTE_TEMPLATE) == self._norm(ABSOLUTE_PROMPT_WO_REF)

    def test_relative_matches_package(self):
        pytest.importorskip("prometheus_eval")
        from prometheus_eval.prompts import RELATIVE_PROMPT_WO_REF

        assert self._norm(PR.RELATIVE_TEMPLATE) == self._norm(RELATIVE_PROMPT_WO_REF)

    def test_system_prompts_match_package(self):
        pytest.importorskip("prometheus_eval")
        from prometheus_eval.prompts import ABS_SYSTEM_PROMPT, REL_SYSTEM_PROMPT

        assert PR.ABS_SYSTEM.strip() == ABS_SYSTEM_PROMPT.strip()
        assert PR.REL_SYSTEM.strip() == REL_SYSTEM_PROMPT.strip()

    def test_we_use_the_reference_free_variant(self):
        """The package's default templates require a reference answer. We have no
        gold verdicts, and a model-written one would be a ninth framing smuggled
        in as ground truth -- so the WO_REF variant is the right one, and the
        omission has to be consistent: the block *and* its mention in the task
        description.
        """
        assert "Reference Answer" not in PR.ABSOLUTE_TEMPLATE
        assert "reference answer" not in PR.ABSOLUTE_TEMPLATE
        assert "reference answer" not in PR.RELATIVE_TEMPLATE


class TestReferenceFitting:
    """Trimming the reference to the judge's context.

    `--check-context` found 414 of 37,350 prompts over the 32,768 limit, the worst
    at 48,166 real tokens against a 4-chars/token estimate of 26,274 -- the
    estimate ran 1.73x optimistic on these JSON payloads. vLLM truncates silently,
    so an over-long prompt scores normally while the judge never sees the end of
    the evidence.
    """

    def _counter(self, chars_per_token=4):
        return lambda text: len(text) // chars_per_token

    def test_a_reference_that_fits_is_untouched(self):
        ref = {"criteria": [{"id": "C1", "value": 1.0}]}
        fitted = RF.fit_reference(ref, self._counter(), budget=10_000)
        assert fitted.reference == ref
        assert not fitted.trimmed
        assert fitted.steps_applied == []

    def test_spread_goes_before_anything_else(self):
        """p05-p95 is the least informative thing in the payload; value and
        percentile carry the signal.
        """
        ref = {
            "criteria": [{"id": "C1", "value": 1.0}],
            "distribution": {
                "m": {"value": 4.0, "pctile": 88, "p05": 1, "p25": 2, "p50": 3, "p95": 9}
            },
        }
        fitted = RF.fit_reference(ref, self._counter(), budget=40)
        assert "drop_distribution_spread" == fitted.steps_applied[0]
        kept = fitted.reference["distribution"]["m"]
        assert kept["value"] == 4.0 and kept["pctile"] == 88
        assert "p05" not in kept

    def test_hit_counts_survive_so_absence_is_not_implied(self):
        """Emptying a hit list would read as "no variants found", which is the
        not-evaluable-as-absent failure the rubrics exist to catch.
        """
        ref = {
            "criteria": [
                {"id": "M1", "hits": [{"pos": i} for i in range(30)], "n_hits_total": 13690}
            ]
        }
        fitted = RF.fit_reference(ref, self._counter(), budget=30)
        member = fitted.reference["criteria"][0]
        assert member["n_hits_total"] == 13690
        assert "rows omitted" in str(member["hits"])

    def test_truncation_is_announced_in_the_payload(self):
        """If it still will not fit, say so where the judge can read it rather
        than let vLLM cut it off invisibly.
        """
        ref = {"criteria": [{"id": f"C{i}", "reason": "x" * 400} for i in range(40)]}
        fitted = RF.fit_reference(ref, self._counter(), budget=60)
        assert "truncated" in fitted.steps_applied
        assert "_truncation_note" in fitted.reference
        assert "not measured as absent" in fitted.reference["_truncation_note"]

    def test_fitting_stops_as_soon_as_it_fits(self):
        """Each step costs evidence, so no step should run that is not needed."""
        ref = {
            "criteria": [{"id": "C1", "value": 1.0}],
            "distribution": {"m": {"value": 4.0, "pctile": 88, "p05": 1, "p95": 9}},
        }
        fitted = RF.fit_reference(ref, self._counter(), budget=45)
        assert fitted.steps_applied == ["drop_distribution_spread"]
        assert "distribution" in fitted.reference

    def test_truncation_lands_just_under_budget(self):
        """Halving overshot: on the real synthesis cells it cut ~30k tokens to
        ~15k against a 29k budget, throwing away 14k of evidence to save 1k.
        """
        ref = {"criteria": [{"id": f"C{i}", "reason": "x" * 900} for i in range(40)]}
        fitted = RF.fit_reference(ref, self._counter(), budget=2000)
        assert 0.9 * 2000 <= fitted.tokens <= 2000

    def test_reported_token_count_is_the_trimmed_one(self):
        ref = {"criteria": [{"id": f"C{i}", "reason": "y" * 200} for i in range(20)]}
        count = self._counter()
        fitted = RF.fit_reference(ref, count, budget=80)
        assert fitted.tokens == count(RF.render(fitted.reference))
        assert fitted.tokens <= 80


class TestQuantize:
    """The 4-bit recipe and its calibration set.

    Both fail silently when wrong. A quantized MoE router still loads and still
    answers, it just routes tokens to different experts; a calibration set drawn
    from one payload shape protects the wrong weight channels.
    """

    def test_router_is_in_the_ignore_list(self):
        """Mixtral sends each token to 2 of 8 experts through a gate. AutoAWQ
        excluded it by default; llm-compressor does not, so it is named here.
        """
        assert any("gate" in entry for entry in Q.IGNORE)
        assert "lm_head" in Q.IGNORE

    def test_router_assertion_rejects_a_bad_ignore_list(self):
        Q.assert_router_excluded(("lm_head", "re:.*gate$"))  # no raise
        with pytest.raises(Q.QuantizeError, match="does not exclude"):
            Q.assert_router_excluded(("lm_head",))

    def test_scheme_is_what_vllm_reads(self):
        assert Q.SCHEME == "W4A16"

    def test_output_is_not_the_source(self):
        """The bf16 must survive: it is the reference configuration and what the
        4-bit copy is compared against.
        """
        assert Q.DEFAULT_OUT != Q.MODEL_DIR
        assert Q.MODEL_DIR not in Q.DEFAULT_OUT.parents

    def _requests_file(self, tmp_path, per_unit=40):
        import json

        path = tmp_path / "requests.jsonl"
        with path.open("w") as fh:
            for unit in ("C", "D", "L", "M", "P", "S", "synthesis"):
                for i in range(per_unit):
                    fh.write(json.dumps({"unit": unit, "prompt": f"{unit}-prompt-{i}"}) + "\n")
        return path

    def test_calibration_spreads_across_units(self, tmp_path):
        """Requests are written ordered by cell, so the head of the file is nearly
        all one isoform's C and D payloads -- and an M payload with its variant
        tables tokenizes nothing like a C one.
        """
        samples = Q.load_calibration(self._requests_file(tmp_path), n_samples=70)
        counts = {
            u: sum(s.startswith(u + "-") for s in samples)
            for u in ("C", "D", "L", "M", "P", "S", "synthesis")
        }
        assert all(counts.values())
        assert max(counts.values()) - min(counts.values()) <= 1

    def test_calibration_is_deterministic(self, tmp_path):
        path = self._requests_file(tmp_path)
        assert Q.load_calibration(path, n_samples=30) == Q.load_calibration(path, n_samples=30)

    def test_missing_requests_refuses_rather_than_using_web_text(self, tmp_path):
        """Falling back to pile-val is the failure this prevents: generic web text
        is out of distribution, and the job runs offline.
        """
        with pytest.raises(Q.QuantizeError, match="not optional"):
            Q.load_calibration(tmp_path / "absent.jsonl")

    def test_chunks_are_full_blocks(self):
        class FakeTok:
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": list(range(len(text)))}

        blocks = Q.chunk_prompts(["x" * 2048, "y" * 2048], FakeTok(), seq_len=512, n_chunks=8)
        assert len(blocks) == 8
        assert all(len(b) == 512 for b in blocks)

    def test_chunks_interleave_across_prompts(self):
        """One 25k-token M prompt would otherwise supply 50 consecutive blocks and
        undo the unit spread load_calibration creates.
        """

        class FakeTok:
            def __call__(self, text, add_special_tokens=False):
                base = ord(text[0]) * 100_000
                return {"input_ids": list(range(base, base + len(text)))}

        blocks = Q.chunk_prompts(["a" * 4096, "b" * 4096], FakeTok(), seq_len=512, n_chunks=4)
        assert [b[0] // 100_000 for b in blocks] == [ord("a"), ord("b"), ord("a"), ord("b")]

    def test_text_shorter_than_one_block_is_refused(self):
        class FakeTok:
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": list(range(len(text)))}

        with pytest.raises(Q.QuantizeError, match="shorter than one block"):
            Q.chunk_prompts(["tiny"], FakeTok(), seq_len=512)
