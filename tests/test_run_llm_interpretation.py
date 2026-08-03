"""Tests for ``scripts/site/run_llm_interpretation.py``.

NO network: all LLM calls are monkeypatched. The tests cover prompt assembly,
the --dry-run path, idempotency, and early-error behaviour for missing
ANTHROPIC_API_KEY / missing prompts directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module() -> ModuleType:
    """Load the LLM-interpretation logic module."""
    from swissisoform.site import llm

    return llm


@pytest.fixture
def mod() -> ModuleType:
    return _load_module()


@pytest.fixture
def prompts_dir(tmp_path: Path) -> Path:
    """A prompts/ dir with a fake system.txt and output_schema.json."""
    p = tmp_path / "prompts"
    p.mkdir()
    (p / "system.txt").write_text("YOU ARE A TIS INTERPRETER. RESPOND IN JSON.", encoding="utf-8")
    schema = {
        "type": "object",
        "required": ["gene_name", "isoforms"],
        "properties": {
            "gene_name": {"type": "string"},
            "gene_summary": {"type": "string"},
            "isoforms": {"type": "array"},
        },
    }
    (p / "output_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    return p


@pytest.fixture
def records_dir(tmp_path: Path) -> Path:
    """A records dir with one TRNT1 evidence record."""
    p = tmp_path / "records"
    p.mkdir()
    record = {
        "gene": {"name": "TRNT1", "uniprot_id": "Q96Q11", "function": "CCA-adder"},
        "isoforms": [{"tis_id": "chr3:1:+:ATG:ENST1", "orf_type": "truncated"}],
    }
    (p / "TRNT1.json").write_text(json.dumps(record), encoding="utf-8")
    return p


@pytest.fixture
def patch_prompt_paths(monkeypatch, mod: ModuleType, prompts_dir: Path) -> None:
    """Repoint the module's prompt path constants at the fake prompts dir."""
    monkeypatch.setattr(mod, "SYSTEM_PROMPT_PATH", prompts_dir / "system.txt")
    monkeypatch.setattr(mod, "OUTPUT_SCHEMA_PATH", prompts_dir / "output_schema.json")


# ── Prompt assembly ───────────────────────────────────────────────────────


def test_build_prompt_contains_system_record_and_schema(mod):
    record = {"gene": {"name": "TRNT1"}, "isoforms": []}
    schema = {"type": "object", "marker": "SCHEMA_MARKER"}
    prompt = mod.build_prompt(record, "SYSTEM_MARKER", schema)
    assert prompt.system == "SYSTEM_MARKER"
    # User msg embeds the evidence record JSON
    assert '"name": "TRNT1"' in prompt.user
    # And ends with the schema suffix
    assert "Respond with valid JSON matching this schema:" in prompt.user
    assert "SCHEMA_MARKER" in prompt.user


def test_estimated_input_tokens(mod):
    prompt = mod.Prompt(system="x" * 400, user="y" * 800)
    assert prompt.estimated_input_tokens == (400 + 800) // 4


# ── Dry-run CLI path ──────────────────────────────────────────────────────


def test_dry_run_uses_synthetic_fallback_when_records_missing(
    mod, patch_prompt_paths, tmp_path, capsys, monkeypatch
):
    # No records dir at all → synthetic fallback in dry-run.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    exit_code = mod.main(
        [
            "--records",
            str(tmp_path / "nonexistent"),
            "--out",
            str(tmp_path / "out"),
            "--dry-run",
            "--gene",
            "TRNT1",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "=== TRNT1 ===" in captured
    assert "model:" in captured
    assert "system prompt chars:" in captured
    assert "user prompt chars:" in captured
    assert "estimated input tokens:" in captured


def test_dry_run_against_real_records(
    mod, patch_prompt_paths, records_dir, tmp_path, capsys, monkeypatch
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    exit_code = mod.main(
        [
            "--records",
            str(records_dir),
            "--out",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "TRNT1" in captured


# ── Missing-config errors ─────────────────────────────────────────────────


def test_missing_api_key_raises_when_not_dry_run(
    mod, patch_prompt_paths, records_dir, tmp_path, monkeypatch
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        mod.main(
            [
                "--records",
                str(records_dir),
                "--out",
                str(tmp_path / "out"),
            ]
        )


def test_missing_prompts_dir_raises_early(mod, tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "SYSTEM_PROMPT_PATH", tmp_path / "nope/system.txt")
    monkeypatch.setattr(mod, "OUTPUT_SCHEMA_PATH", tmp_path / "nope/output_schema.json")
    with pytest.raises(FileNotFoundError, match="System prompt not found"):
        mod.main(["--dry-run", "--gene", "TRNT1"])


def test_missing_output_schema_raises_early(mod, tmp_path, monkeypatch, prompts_dir):
    monkeypatch.setattr(mod, "SYSTEM_PROMPT_PATH", prompts_dir / "system.txt")
    monkeypatch.setattr(mod, "OUTPUT_SCHEMA_PATH", tmp_path / "missing_schema.json")
    with pytest.raises(FileNotFoundError, match="Output schema not found"):
        mod.main(["--dry-run", "--gene", "TRNT1"])


# ── Idempotency ───────────────────────────────────────────────────────────


def test_existing_output_not_overwritten_without_force(
    mod, patch_prompt_paths, records_dir, tmp_path, monkeypatch
):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    pre_existing = {"gene_name": "TRNT1", "marker": "DO_NOT_OVERWRITE"}
    (out_dir / "TRNT1.json").write_text(json.dumps(pre_existing), encoding="utf-8")

    # Patch call_llm so we'd notice if it ran (it shouldn't).
    def _fail(*args, **kwargs):
        raise AssertionError("call_llm should not be invoked for idempotent skip")

    monkeypatch.setattr(mod, "call_llm", _fail)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")

    exit_code = mod.main(
        [
            "--records",
            str(records_dir),
            "--out",
            str(out_dir),
        ]
    )
    assert exit_code == 0
    after = json.loads((out_dir / "TRNT1.json").read_text(encoding="utf-8"))
    assert after == pre_existing


def test_force_overwrites_existing_output(
    mod, patch_prompt_paths, records_dir, tmp_path, monkeypatch
):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "TRNT1.json").write_text(json.dumps({"marker": "OLD"}), encoding="utf-8")

    new_payload = {"gene_name": "TRNT1", "gene_summary": "fresh", "isoforms": []}
    monkeypatch.setattr(mod, "call_llm", lambda *a, **kw: json.dumps(new_payload))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")

    exit_code = mod.main(
        [
            "--records",
            str(records_dir),
            "--out",
            str(out_dir),
            "--force",
        ]
    )
    assert exit_code == 0
    after = json.loads((out_dir / "TRNT1.json").read_text(encoding="utf-8"))
    assert after == new_payload


# ── Successful end-to-end with mocked LLM ─────────────────────────────────


def test_successful_run_writes_output(
    mod, patch_prompt_paths, records_dir, tmp_path, monkeypatch, capsys
):
    new_payload = {"gene_name": "TRNT1", "gene_summary": "test summary", "isoforms": []}
    monkeypatch.setattr(mod, "call_llm", lambda *a, **kw: json.dumps(new_payload))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")

    out_dir = tmp_path / "out"
    exit_code = mod.main(
        [
            "--records",
            str(records_dir),
            "--out",
            str(out_dir),
        ]
    )
    assert exit_code == 0
    written = json.loads((out_dir / "TRNT1.json").read_text(encoding="utf-8"))
    assert written == new_payload
    captured = capsys.readouterr().out
    assert "OK" in captured
    assert "TRNT1" in captured


def test_json_parse_failure_logs_to_failures_dir_and_returns_nonzero(
    mod, patch_prompt_paths, records_dir, tmp_path, monkeypatch
):
    monkeypatch.setattr(mod, "call_llm", lambda *a, **kw: "not json at all")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")

    out_dir = tmp_path / "out"
    exit_code = mod.main(
        [
            "--records",
            str(records_dir),
            "--out",
            str(out_dir),
        ]
    )
    assert exit_code == 1
    failure_file = out_dir / "_failures" / "TRNT1.txt"
    assert failure_file.exists()
    assert failure_file.read_text(encoding="utf-8") == "not json at all"
    # The main output file should NOT have been written.
    assert not (out_dir / "TRNT1.json").exists()


def test_parse_response_strips_markdown_fence(mod):
    payload = {"gene_name": "X", "isoforms": []}
    fenced = "```json\n" + json.dumps(payload) + "\n```"
    assert mod.parse_response(fenced) == payload


# ── PASS_REGISTRY (shared per-pass machinery) ─────────────────────────────


def test_pass_registry_lists_v1_plus_two_v2_passes(mod):
    """The PASS_REGISTRY exposes v1 (default) plus category + synthesis."""
    assert set(mod.PASS_REGISTRY) == {"default", "category", "synthesis"}


def test_pass_default_uses_v1_system_prompt(mod):
    """The default pass loads system.txt and writes to <gene>.json (back-compat)."""
    spec = mod.PASS_REGISTRY["default"]
    assert spec.system_prompt_filename == "system.txt"
    assert spec.output_filename_template == "{gene}.json"


# ── Category + synthesis pass dispatch ────────────────────────────────────


def _ISO_FIXTURE_RECORD() -> dict:
    """Synthetic per-gene evidence record exercising the category pass."""
    return {
        "gene": {
            "name": "GENE_A",
            "uniprot_id": None,
            "function": None,
            "subcellular_location": None,
        },
        "isoforms": [
            {
                "tis_id": "chr1:100:+:ATG:ENST_A",
                "orf_type": "truncated",
                "differential_sequence": "M",
                "diff_space": "canonical",
                "isoform_length_aa": 50,
                "canonical_length_aa": 60,
                "alt_start_codon": "ATG",
                "kozak_context": "AAAA",
                "scoring": {"criteria": {}},
                "key_metrics": {},
                "pathogenic_variants_in_unique": [],
                # A current-shape _raw: StructureModule's fold-cache hashes plus
                # the differential-region coordinates. The P readers resolve
                # through these, and _tool_setup refuses to run without them.
                "_raw": {
                    "isoform_structure_canonical_hash": "a" * 40,
                    "isoform_structure_isoform_hash": "b" * 40,
                    "isoform_structure_backend": "esmfold2",
                    "diff_space": "canonical",
                    "diff_start": 0,
                    "diff_end": 10,
                },
            }
        ],
    }


def test_category_pass_dry_run_emits_one_call_per_category(monkeypatch, tmp_path, capsys):
    """Category pass iterates over the 6 CDLMPS categories × N isoforms."""
    from swissisoform.site import llm as rli

    records_dir = tmp_path / "records"
    records_dir.mkdir()
    (records_dir / "GENE_A.json").write_text(json.dumps(_ISO_FIXTURE_RECORD()))
    out_dir = tmp_path / "out"
    rc = rli.main(
        [
            "--records",
            str(records_dir),
            "--out",
            str(out_dir),
            "--pass",
            "category",
            "--dry-run",
        ]
    )
    captured = capsys.readouterr().out
    assert rc == 0
    # 6 CDLMPS categories × 1 isoform = 6 dry-run "category:" prints (all 15
    # criteria incl. S2/S3 are covered across the categories).
    assert captured.count("category:") == 6


def test_synthesis_pass_refuses_without_prereqs(monkeypatch, tmp_path):
    from swissisoform.site import llm as rli

    records_dir = tmp_path / "records"
    records_dir.mkdir()
    (records_dir / "GENE_A.json").write_text(
        json.dumps(
            {
                "gene": {"name": "GENE_A"},
                "isoforms": [
                    {
                        "tis_id": "chr1:100:+:ATG:ENST_A",
                        "scoring": {"existence_score": 5, "functional_score": 5},
                    }
                ],
            }
        )
    )
    out_dir = tmp_path / "out"
    rc = rli.main(
        [
            "--records",
            str(records_dir),
            "--out",
            str(out_dir),
            "--pass",
            "synthesis",
            "--dry-run",
        ]
    )
    # Synthesis requires the categories.json prereq; missing → exit code 2.
    assert rc == 2


def test_category_pass_skips_isoforms_with_existing_output(monkeypatch, tmp_path, capsys):
    """Re-running the category pass without --force is a no-op for done isoforms."""
    from swissisoform.site import llm as rli

    records_dir = tmp_path / "records"
    records_dir.mkdir()
    (records_dir / "GENE_A.json").write_text(json.dumps(_ISO_FIXTURE_RECORD()))
    out_dir = tmp_path / "out"
    tis_slug = rli._tis_slug("chr1:100:+:ATG:ENST_A")
    (out_dir / tis_slug).mkdir(parents=True)
    (out_dir / tis_slug / "categories.json").write_text("{}")

    rc = rli.main(
        [
            "--records",
            str(records_dir),
            "--out",
            str(out_dir),
            "--pass",
            "category",
            "--dry-run",
        ]
    )
    captured = capsys.readouterr().out
    assert rc == 0
    # With output present and --force absent, dry-run should NOT emit any
    # "category:" lines for this isoform — it skips entirely.
    assert captured.count("category:") == 0
    # And it should explicitly report the skip:
    assert "[skip]" in captured


def test_synthesis_input_record_pulls_category_reads(monkeypatch, tmp_path):
    """synthesis._build_synthesis_record reads categories.json into category_reads."""
    from swissisoform.site import llm as rli

    tis_slug = "chr1-100-ATG-ENST_A"
    base = tmp_path / tis_slug
    base.mkdir()
    category_payload = {
        "Conservation": {
            "verdict": "interesting",
            "reasoning": "primate frame intact (frac_intact=0.96) with strong phyloP.",
        },
        "Mutation Landscape": {
            "verdict": "neutral",
            "reasoning": "no disease enrichment; germline signal weak.",
        },
    }
    (base / "categories.json").write_text(json.dumps(category_payload))
    iso = {
        "tis_id": "chr1:100:+:ATG:ENST_A",
        "scoring": {"existence_score": 5, "functional_score": 5},
        # _raw feeds slice_criterion; the L1 columns drive the localization triad.
        "_raw": {
            "canonical_localization_deeploc_prediction": "Nucleus",
            "canonical_localization_deeploc_top_prob": 0.91,
            "isoform_localization_deeploc_prediction": "Cytoplasm",
            "isoform_localization_deeploc_top_prob": 0.83,
        },
    }
    gene_block = {
        "name": "GENE_A",
        "function": "GENE_A is a chromatin reader.",
        "keywords": "histone binding; chromatin",
        "subcellular_location": "nucleus",
    }
    rec = rli._build_synthesis_record(iso, "GENE_A", base, gene_block)
    assert rec["isoform"]["tis_id"] == "chr1:100:+:ATG:ENST_A"
    assert "Conservation" in rec["category_reads"]
    assert "Mutation Landscape" in rec["category_reads"]
    assert rec["category_reads"]["Conservation"]["verdict"] == "interesting"
    # The gene's established function is threaded in as the divergence baseline.
    assert rec["gene"]["function"] == "GENE_A is a chromatin reader."
    assert rec["gene"]["keywords"] == "histone binding; chromatin"
    # Localization calibration triad: literature vs DeepLoc-on-canonical (+conf) vs
    # DeepLoc-on-isoform (+conf), assembled from the L1 criterion evidence.
    loc = rec["localization"]
    assert loc["known_from_literature"] == "nucleus"
    assert loc["predicted_canonical"] == {"compartment": "Nucleus", "top_prob": 0.91}
    assert loc["predicted_isoform"] == {"compartment": "Cytoplasm", "top_prob": 0.83}
    # The record states the isoform once at the top level; the 15 criteria
    # entries no longer each repeat it.
    assert rec["isoform"]["tis_id"] == "chr1:100:+:ATG:ENST_A"
    assert all("isoform" not in v for v in rec["criteria_evidence"].values())
    # Backward-compat: no gene arg → name-only block, no crash.
    rec_bare = rli._build_synthesis_record(iso, "GENE_A", base)
    assert rec_bare["gene"] == {
        "name": "GENE_A", "function": None, "keywords": None, "subcellular_location": None,
    }
    # All 15 criteria (incl. S2/S3) are carried in the raw evidence for synthesis —
    # S2/S3 tolerate the _raw-less fixture (empty evidence, never raise).
    assert "P2_shared_structural_change" in rec["criteria_evidence"]
    assert "S2_biophysics" in rec["criteria_evidence"]
    assert "S3_sae" in rec["criteria_evidence"]


# ── Tool loop (M category) ────────────────────────────────────────────────
#
# Still NO network: a fake `anthropic` module replaces the SDK and returns a
# scripted sequence of content blocks, so the loop's turn handling, guards and
# trace output are exercised deterministically.


class _Block:
    """Stand-in for an SDK content block (attribute access, not dict access)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _text(text: str) -> _Block:
    return _Block(type="text", text=text)


def _tool_use(name: str, tool_input: dict, tid: str = "tu") -> _Block:
    return _Block(type="tool_use", name=name, input=tool_input, id=tid)


class _Response:
    def __init__(self, content, stop_reason="tool_use"):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = _Block(
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )


class _FakeMessages:
    def __init__(self, script, calls):
        self._script = list(script)
        self.calls = calls

    def create(self, **kwargs):
        # Snapshot the message list: the loop mutates one list in place, so
        # storing the reference would make every recorded turn look identical
        # to the last one.
        self.calls.append({**kwargs, "messages": list(kwargs.get("messages") or [])})
        # Repeat the final scripted response once exhausted, so a loop that keeps
        # going hits max_turns rather than an IndexError.
        return self._script.pop(0) if len(self._script) > 1 else self._script[0]


def _fake_anthropic(script, calls):
    """A module-like object exposing just the ``Anthropic`` client the loop uses."""
    import types as _types

    class _Client:
        def __init__(self, api_key=None):
            self.messages = _FakeMessages(script, calls)

    return _types.SimpleNamespace(Anthropic=_Client)


def _dispatch_ok(name, tool_input):
    return {"n": 7, "tool": name, "input": tool_input}


VERDICT = {"verdict": "interesting", "reasoning": "clustered pathogenic variants."}


def test_tool_loop_returns_verdict_after_data_calls(mod, monkeypatch):
    calls: list = []
    script = [
        _Response([_tool_use("variant_position_histogram", {"region": "unique"}, "t1")]),
        _Response([_tool_use("query_variants", {"clinsig": "pathogenic"}, "t2")]),
        _Response([_tool_use("emit_verdict", VERDICT, "t3")], stop_reason="tool_use"),
    ]
    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: _fake_anthropic(script, calls))

    verdict, trace = mod.run_tool_loop(
        system="S", user="U", tools=[], dispatch=_dispatch_ok,
        model="claude-sonnet-5", max_tokens=1000, api_key="k",
    )
    assert verdict == VERDICT
    assert trace["outcome"] == "emit_verdict"
    assert trace["n_data_calls"] == 2
    assert len(trace["turns"]) == 3
    assert len(calls) == 3


def test_tool_loop_records_tool_inputs_and_results_in_the_trace(mod, monkeypatch):
    """The transcript is the audit artifact — it must carry what was asked and returned."""
    calls: list = []
    script = [
        _Response([_tool_use("variant_position_histogram", {"region": "unique"}, "t1")]),
        _Response([_tool_use("query_variants", {"limit": 5}, "t2")]),
        _Response([_tool_use("emit_verdict", VERDICT, "t3")]),
    ]
    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: _fake_anthropic(script, calls))

    _, trace = mod.run_tool_loop(
        system="S", user="U", tools=[], dispatch=_dispatch_ok,
        model="claude-sonnet-5", max_tokens=1000, api_key="k",
    )
    first = trace["turns"][0]["tool_results"][0]
    assert first["name"] == "variant_position_histogram"
    assert first["input"] == {"region": "unique"}
    assert first["result"]["n"] == 7
    assert trace["turns"][0]["assistant"][0]["type"] == "tool_use"


def test_tool_loop_rejects_emit_verdict_before_min_data_calls(mod, monkeypatch):
    """A model that skips straight to a verdict is sent back to the data."""
    calls: list = []
    script = [
        _Response([_tool_use("emit_verdict", VERDICT, "t0")]),  # premature
        _Response([_tool_use("query_variants", {}, "t1")]),
        _Response([_tool_use("variant_effect_stats", {}, "t2")]),
        _Response([_tool_use("emit_verdict", VERDICT, "t3")]),
    ]
    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: _fake_anthropic(script, calls))

    verdict, trace = mod.run_tool_loop(
        system="S", user="U", tools=[], dispatch=_dispatch_ok,
        model="claude-sonnet-5", max_tokens=1000, api_key="k",
    )
    assert verdict == VERDICT
    assert trace["turns"][0]["tool_results"][0]["rejected"]
    # The rejection is fed back as an error tool_result, and the loop continues.
    second_request_messages = calls[1]["messages"]
    tool_results = second_request_messages[-1]["content"]
    assert tool_results[0]["is_error"] is True


def test_tool_loop_errored_tool_calls_do_not_count_as_data(mod, monkeypatch):
    """Only a call that returned data satisfies min_data_calls."""
    calls: list = []
    script = [
        _Response([_tool_use("bogus", {}, "t1")]),
        _Response([_tool_use("emit_verdict", VERDICT, "t2")]),
        _Response([_tool_use("query_variants", {}, "t3")]),
        _Response([_tool_use("variant_effect_stats", {}, "t4")]),
        _Response([_tool_use("emit_verdict", VERDICT, "t5")]),
    ]

    def dispatch(name, tool_input):
        return {"error": "nope"} if name == "bogus" else {"n": 1}

    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: _fake_anthropic(script, calls))
    _, trace = mod.run_tool_loop(
        system="S", user="U", tools=[], dispatch=dispatch,
        model="claude-sonnet-5", max_tokens=1000, api_key="k",
    )
    assert trace["n_data_calls"] == 2  # the errored call did not count


def test_tool_loop_raises_with_trace_after_max_turns(mod, monkeypatch):
    calls: list = []
    script = [_Response([_tool_use("query_variants", {}, "t")])]  # never terminates
    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: _fake_anthropic(script, calls))

    with pytest.raises(mod.ToolLoopError) as e:
        mod.run_tool_loop(
            system="S", user="U", tools=[], dispatch=_dispatch_ok,
            model="claude-sonnet-5", max_tokens=1000, api_key="k", max_turns=3,
        )
    assert e.value.trace["outcome"] == "max_turns_exhausted"
    assert len(e.value.trace["turns"]) == 3
    assert len(calls) == 3


def test_tool_loop_accepts_a_plain_json_verdict(mod, monkeypatch):
    """A model that answers in prose JSON instead of calling the terminal tool."""
    calls: list = []
    script = [
        _Response([_tool_use("query_variants", {}, "t1")]),
        _Response([_tool_use("variant_effect_stats", {}, "t2")]),
        _Response([_text(json.dumps(VERDICT))], stop_reason="end_turn"),
    ]
    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: _fake_anthropic(script, calls))

    verdict, trace = mod.run_tool_loop(
        system="S", user="U", tools=[], dispatch=_dispatch_ok,
        model="claude-sonnet-5", max_tokens=1000, api_key="k",
    )
    assert verdict == VERDICT
    assert trace["outcome"] == "text_verdict"


def test_tool_loop_nudges_once_then_gives_up(mod, monkeypatch):
    calls: list = []
    script = [_Response([_text("I would rather not.")], stop_reason="end_turn")]
    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: _fake_anthropic(script, calls))

    with pytest.raises(mod.ToolLoopError):
        mod.run_tool_loop(
            system="S", user="U", tools=[], dispatch=_dispatch_ok,
            model="claude-sonnet-5", max_tokens=1000, api_key="k",
        )
    # One nudge was sent before giving up, and only two turns were spent.
    assert len(calls) == 2
    assert "emit_verdict" in calls[1]["messages"][-1]["content"][0]["text"]


def test_tool_loop_appends_assistant_content_wholesale(mod, monkeypatch):
    """Thinking-block safety: the SDK objects go back verbatim, not rebuilt."""
    calls: list = []
    thinking = _Block(type="thinking", thinking="hmm")
    tu = _tool_use("query_variants", {}, "t1")
    script = [
        _Response([thinking, tu]),
        _Response([_tool_use("variant_effect_stats", {}, "t2")]),
        _Response([_tool_use("emit_verdict", VERDICT, "t3")]),
    ]
    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: _fake_anthropic(script, calls))

    mod.run_tool_loop(
        system="S", user="U", tools=[], dispatch=_dispatch_ok,
        model="claude-sonnet-5", max_tokens=1000, api_key="k",
    )
    assistant_msg = calls[1]["messages"][1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"][0] is thinking  # same object, not a copy
    assert assistant_msg["content"][1] is tu


def test_tool_loop_truncates_an_oversized_tool_result(mod, monkeypatch):
    calls: list = []
    script = [
        _Response([_tool_use("query_variants", {}, "t1")]),
        _Response([_tool_use("variant_effect_stats", {}, "t2")]),
        _Response([_tool_use("emit_verdict", VERDICT, "t3")]),
    ]
    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: _fake_anthropic(script, calls))
    huge = {"rows": ["x" * 1000] * 200}  # ~200k chars serialised

    mod.run_tool_loop(
        system="S", user="U", tools=[], dispatch=lambda n, i: huge,
        model="claude-sonnet-5", max_tokens=1000, api_key="k",
    )
    body = calls[1]["messages"][-1]["content"][0]["content"]
    assert len(body) < mod.MAX_TOOL_RESULT_CHARS + 200
    assert body.endswith("[tool result truncated]")


def test_tool_loop_omits_temperature_for_no_sampling_models(mod, monkeypatch):
    calls: list = []
    script = [
        _Response([_tool_use("query_variants", {}, "t1")]),
        _Response([_tool_use("variant_effect_stats", {}, "t2")]),
        _Response([_tool_use("emit_verdict", VERDICT, "t3")]),
    ]
    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: _fake_anthropic(script, calls))
    mod.run_tool_loop(
        system="S", user="U", tools=[], dispatch=_dispatch_ok,
        model="claude-sonnet-5", max_tokens=1000, api_key="k",
    )
    assert "temperature" not in calls[0]
    assert calls[0]["thinking"] == {"type": "disabled"}

    calls_old: list = []
    script_old = [
        _Response([_tool_use("query_variants", {}, "t1")]),
        _Response([_tool_use("variant_effect_stats", {}, "t2")]),
        _Response([_tool_use("emit_verdict", VERDICT, "t3")]),
    ]
    monkeypatch.setattr(
        mod, "_try_import_anthropic", lambda: _fake_anthropic(script_old, calls_old)
    )
    mod.run_tool_loop(
        system="S", user="U", tools=[], dispatch=_dispatch_ok,
        model="claude-sonnet-4-6", max_tokens=1000, api_key="k", temperature=0.0,
    )
    assert calls_old[0]["temperature"] == 0.0


def test_tool_loop_requires_the_official_sdk(mod, monkeypatch):
    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: None)
    with pytest.raises(RuntimeError, match="anthropic"):
        mod.run_tool_loop(
            system="S", user="U", tools=[], dispatch=_dispatch_ok,
            model="claude-sonnet-5", max_tokens=1000, api_key="k",
        )


# ── Category pass wiring for the tool loop ────────────────────────────────


TIS_ID = "chr1:100:+:ATG:ENST_A"


@pytest.fixture
def variants_long(tmp_path: Path) -> Path:
    """Minimal variants_long.parquet covering the fixture isoform."""
    import pandas as pd

    from swissisoform.site import tools as site_tools

    rows = [
        {
            "tis_id": TIS_ID, "gene_name": "GENE_A", "variant_id": "cv1",
            "source": "ClinVar", "clinical_significance": "Pathogenic",
            "isoform_protein_pos": 12.0, "in_isoform_unique": True,
            "in_isoform_shared": False, "consequence": "missense_variant",
            "am_pathogenicity": 0.9, "plm_delta_llr": -6.0, "effect_damaging": True,
            "cosmic_sample_count": None, "allele_frequency": None, "hgvsp": "p.A12T",
        }
    ]
    p = tmp_path / "variants_long.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    site_tools._load.cache_clear()
    site_tools._index_by_tis.cache_clear()
    return p


def _category_run_args(records_dir: Path, out_dir: Path, extra: list[str]) -> list[str]:
    return ["--records", str(records_dir), "--out", str(out_dir), "--pass", "category", *extra]


@pytest.fixture
def only_m(monkeypatch, mod):
    """Restrict the tool-loop registry to M.

    Each tool category has its own tool names, and a dispatch returns an error
    for names it does not own — which correctly does not count toward
    ``min_data_calls``. So a test scripting M's tools must not also drive P, or
    P's loop spins until ``max_turns``. Isolating the registry keeps each test
    about one category.
    """
    monkeypatch.setattr(mod, "TOOL_CATEGORY_PROMPTS", {"M": "category-pass-M.txt"})


@pytest.fixture
def only_p(monkeypatch, mod):
    """Restrict the tool-loop registry to P. See :func:`only_m`."""
    monkeypatch.setattr(mod, "TOOL_CATEGORY_PROMPTS", {"P": "category-pass-P.txt"})


@pytest.fixture
def category_records(tmp_path: Path) -> Path:
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    (records_dir / "GENE_A.json").write_text(json.dumps(_ISO_FIXTURE_RECORD()))
    return records_dir


def test_category_pass_runs_m_as_a_tool_loop(
    mod, monkeypatch, tmp_path, category_records, variants_long, only_m
):
    """M goes through the loop; the other five stay single-shot."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    single_shot: list = []
    monkeypatch.setattr(
        mod, "call_llm",
        lambda *a, **kw: single_shot.append(1) or json.dumps({"verdict": "neutral",
                                                             "reasoning": "single shot."}),
    )
    calls: list = []
    script = [
        _Response([_tool_use("variant_position_histogram", {"region": "unique"}, "t1")]),
        _Response([_tool_use("query_variants", {"clinsig": "pathogenic"}, "t2")]),
        _Response([_tool_use("emit_verdict", VERDICT, "t3")]),
    ]
    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: _fake_anthropic(script, calls))

    out_dir = tmp_path / "out"
    rc = mod.main(
        _category_run_args(category_records, out_dir, ["--variants-long", str(variants_long)])
    )
    assert rc == 0

    tis_slug = mod._tis_slug(TIS_ID)
    payload = json.loads((out_dir / tis_slug / "categories.json").read_text())
    assert payload["Mutation Landscape"]["verdict"] == "interesting"
    # The other five categories still went through the single-shot path.
    assert len(single_shot) == 5
    assert payload["Conservation"]["reasoning"] == "single shot."


def test_category_pass_writes_the_m_trace(
    mod, monkeypatch, tmp_path, category_records, variants_long, only_m
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(mod, "call_llm", lambda *a, **kw: json.dumps({"verdict": "neutral",
                                                                     "reasoning": "x"}))
    script = [
        _Response([_tool_use("variant_position_histogram", {}, "t1")]),
        _Response([_tool_use("query_variants", {}, "t2")]),
        _Response([_tool_use("emit_verdict", VERDICT, "t3")]),
    ]
    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: _fake_anthropic(script, []))

    out_dir = tmp_path / "out"
    mod.main(
        _category_run_args(category_records, out_dir, ["--variants-long", str(variants_long)])
    )
    trace = json.loads((out_dir / mod._tis_slug(TIS_ID) / "M_trace.json").read_text())
    assert trace["outcome"] == "emit_verdict"
    assert len(trace["turns"]) == 3
    # The readers really ran against the parquet, not a stub.
    assert trace["turns"][0]["tool_results"][0]["result"]["n"] == 1


def test_strip_hits_removes_rows_but_keeps_the_aggregate(mod):
    """The scalars are the intended starting context; the sampled rows are not."""
    record = {
        "category": "M",
        "name": "Mutation Landscape",
        "isoform": {"tis_id": TIS_ID},
        "members": [
            {
                "criterion_id": "M1_pathogenic_variant_enrichment",
                "evidence": {"gnomad_depletion_ratio": 0.4},
                "reason": "depleted",
                "interpretation_hint": "germline constraint",
                "hits": [{"variant_id": f"v{i}"} for i in range(30)],
                "n_hits_total": 1619,
                "n_hits_shown": 30,
            }
        ],
    }
    out = mod._strip_hits_for_tools(record)
    member = out["members"][0]
    assert member["hits"] == []
    assert member["n_hits_shown"] == 0
    # The count survives, so the model knows how much data is queryable.
    assert member["n_hits_total"] == 1619
    assert "1619 variant records exist" in member["hits_note"]
    # Everything that is actually the aggregate is untouched.
    assert member["evidence"] == {"gnomad_depletion_ratio": 0.4}
    assert member["reason"] == "depleted"
    assert member["interpretation_hint"] == "germline constraint"
    assert out["isoform"] == {"tis_id": TIS_ID}
    # And the caller's record is not mutated.
    assert len(record["members"][0]["hits"]) == 30


def test_strip_hits_shrinks_a_real_m_slice(mod):
    """Regression on the number that motivated this: rows are ~93% of the payload."""
    import json as _json

    from swissisoform.site.evidence import CATEGORIES, slice_category

    rec = _json.loads(
        (ROOT / "data/output/cheeseman_test/llm_evidence/TRNT1.json").read_text()
    )
    iso = {**rec["isoforms"][0], "gene": {"name": "TRNT1"}}
    cat = next(c for c in CATEGORIES if c["letter"] == "M")
    full = slice_category(iso, cat)
    stripped = mod._strip_hits_for_tools(full)
    n_full = len(_json.dumps(full, indent=2))
    n_stripped = len(_json.dumps(stripped, indent=2))
    assert n_stripped < n_full / 5
    # The two M members declare the same evidence_hits_col, so the identical rows
    # were serialised twice; both copies are gone.
    assert all(m["hits"] == [] for m in stripped["members"])


def test_tool_loop_opening_context_carries_no_variant_rows(
    mod, monkeypatch, tmp_path, category_records, variants_long, only_m
):
    """End-to-end: what actually reaches the API has the rows removed."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(mod, "call_llm", lambda *a, **kw: json.dumps({"verdict": "neutral",
                                                                     "reasoning": "x"}))
    calls: list = []
    script = [
        _Response([_tool_use("query_variants", {}, "t1")]),
        _Response([_tool_use("variant_effect_stats", {}, "t2")]),
        _Response([_tool_use("emit_verdict", VERDICT, "t3")]),
    ]
    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: _fake_anthropic(script, calls))

    out_dir = tmp_path / "out"
    mod.main(
        _category_run_args(category_records, out_dir, ["--variants-long", str(variants_long)])
    )
    opening = calls[0]["messages"][0]["content"][0]["text"]
    payload = json.loads(opening)
    assert all(m["hits"] == [] for m in payload["members"])
    assert "hits_note" in payload["members"][0]
    # The trace records the size so the multi-turn cost is measurable after a run.
    trace = json.loads((out_dir / mod._tis_slug(TIS_ID) / "M_trace.json").read_text())
    assert trace["opening_context_chars"] == len(opening)


def test_no_tools_path_still_sends_the_hits(
    mod, monkeypatch, tmp_path, category_records
):
    """Stripping is tool-loop-only: the single-shot read still needs its sample."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    prompts: list = []

    def capture(prompt, **kw):
        prompts.append(prompt.user)
        return json.dumps({"verdict": "neutral", "reasoning": "x"})

    monkeypatch.setattr(mod, "call_llm", capture)
    mod.main(_category_run_args(category_records, tmp_path / "out", ["--no-tools"]))
    assert any('"hits"' in p for p in prompts)
    assert not any("hits_note" in p for p in prompts)


def test_category_pass_writes_a_separate_tool_usage_report(
    mod, monkeypatch, tmp_path, category_records, variants_long, only_m
):
    """Tool categories are full-price and multi-turn, so their cost is reported apart."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(mod, "call_llm", lambda *a, **kw: json.dumps({"verdict": "neutral",
                                                                     "reasoning": "x"}))
    script = [
        _Response([_tool_use("query_variants", {}, "t1")]),
        _Response([_tool_use("variant_effect_stats", {}, "t2")]),
        _Response([_tool_use("emit_verdict", VERDICT, "t3")]),
    ]
    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: _fake_anthropic(script, []))

    out_dir = tmp_path / "out"
    mod.main(
        _category_run_args(category_records, out_dir, ["--variants-long", str(variants_long)])
    )
    report = json.loads((out_dir / "_usage_category_tools.json").read_text())
    assert report["batch"] is False
    # Three turns' usage summed, not just the last one...
    assert report["total"]["input"] == 300
    # ...and counted as three API round trips, not one category.
    assert report["total"]["calls"] == 3


def test_category_pass_records_an_error_when_the_loop_fails(
    mod, monkeypatch, tmp_path, category_records, variants_long, only_m
):
    """A failed M loop must not sink the other five categories."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(mod, "call_llm", lambda *a, **kw: json.dumps({"verdict": "neutral",
                                                                     "reasoning": "x"}))
    script = [_Response([_tool_use("query_variants", {}, "t1")])]  # never terminates
    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: _fake_anthropic(script, []))

    out_dir = tmp_path / "out"
    rc = mod.main(
        _category_run_args(
            category_records, out_dir,
            ["--variants-long", str(variants_long), "--max-tool-turns", "2"],
        )
    )
    assert rc == 1  # the run reports the failure
    payload = json.loads((out_dir / mod._tis_slug(TIS_ID) / "categories.json").read_text())
    assert "error" in payload["Mutation Landscape"]
    assert payload["Conservation"]["verdict"] == "neutral"
    # The partial transcript is still written — a failed loop is the one worth reading.
    trace = json.loads((out_dir / mod._tis_slug(TIS_ID) / "M_trace.json").read_text())
    assert trace["outcome"] == "max_turns_exhausted"


def test_category_pass_fails_loudly_when_variants_long_is_missing(
    mod, monkeypatch, tmp_path, category_records, only_m
):
    """No silent fallback: an M verdict with tools and one without aren't comparable."""
    from swissisoform.site import tools as site_tools

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    out_dir = tmp_path / "out"
    with pytest.raises(site_tools.VariantsLongMissing, match="build_evidence_records"):
        mod.main(
            _category_run_args(
                category_records, out_dir, ["--variants-long", str(tmp_path / "gone.parquet")]
            )
        )


def test_no_tools_flag_runs_every_category_single_shot(
    mod, monkeypatch, tmp_path, category_records
):
    """The A/B escape hatch — and it does not need the variants table at all."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    single_shot: list = []
    monkeypatch.setattr(
        mod, "call_llm",
        lambda *a, **kw: single_shot.append(1) or json.dumps({"verdict": "neutral",
                                                             "reasoning": "single shot."}),
    )
    out_dir = tmp_path / "out"
    rc = mod.main(_category_run_args(category_records, out_dir, ["--no-tools"]))
    assert rc == 0
    assert len(single_shot) == 6
    assert not (out_dir / mod._tis_slug(TIS_ID) / "M_trace.json").exists()


def test_dry_run_needs_no_variants_table_and_still_counts_six(
    mod, tmp_path, category_records, capsys
):
    """--dry-run makes no API calls, so there is no loop to set up."""
    out_dir = tmp_path / "out"
    rc = mod.main(_category_run_args(category_records, out_dir, ["--dry-run"]))
    assert rc == 0
    assert capsys.readouterr().out.count("category:") == 6


def test_batch_path_excludes_tool_categories_from_the_batch(
    mod, monkeypatch, tmp_path, category_records, variants_long, only_m
):
    """Multi-turn is structurally unbatchable — M runs interactively after the batch."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    submitted: dict = {}

    def fake_batch(items, **kw):
        submitted["items"] = items
        return {
            cid: {
                "text": json.dumps({"verdict": "neutral", "reasoning": "batched."}),
                "usage": mod._empty_usage(),
                "error": None,
            }
            for cid, _ in items
        }

    monkeypatch.setattr(mod, "call_llm_batch", fake_batch)
    script = [
        _Response([_tool_use("query_variants", {}, "t1")]),
        _Response([_tool_use("variant_effect_stats", {}, "t2")]),
        _Response([_tool_use("emit_verdict", VERDICT, "t3")]),
    ]
    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: _fake_anthropic(script, []))

    out_dir = tmp_path / "out"
    rc = mod.main(
        _category_run_args(
            category_records, out_dir, ["--batch", "--variants-long", str(variants_long)]
        )
    )
    assert rc == 0
    # 5 batched categories, not 6.
    assert len(submitted["items"]) == 5
    payload = json.loads((out_dir / mod._tis_slug(TIS_ID) / "categories.json").read_text())
    assert payload["Conservation"]["reasoning"] == "batched."
    assert payload["Mutation Landscape"]["verdict"] == "interesting"
    # Batch-priced and full-priced work are reported separately.
    assert (out_dir / "_usage_category.json").exists()
    assert (out_dir / "_usage_category_tools.json").exists()


def test_variants_long_defaults_to_a_sibling_of_records(mod, tmp_path):
    """The evidence stage writes llm_evidence/ and variants_long.parquet together."""
    import argparse

    args = argparse.Namespace(records=tmp_path / "run" / "llm_evidence", variants_long=None)
    assert mod._resolve_variants_long(args) == tmp_path / "run" / "variants_long.parquet"

    args.variants_long = tmp_path / "explicit.parquet"
    assert mod._resolve_variants_long(args) == tmp_path / "explicit.parquet"


# ── P category wiring ─────────────────────────────────────────────────────


P_VERDICT = {"verdict": "interesting", "reasoning": "extension folds but is unplaced."}


def test_p_category_runs_as_a_tool_loop(
    mod, monkeypatch, tmp_path, category_records, only_p
):
    """P loops over its own readers; the other five stay single-shot."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    single_shot: list = []
    monkeypatch.setattr(
        mod, "call_llm",
        lambda *a, **kw: single_shot.append(1) or json.dumps({"verdict": "neutral",
                                                             "reasoning": "single shot."}),
    )
    calls: list = []
    script = [
        _Response([_tool_use("plddt_profile", {"side": "canonical"}, "t1")]),
        _Response([_tool_use("pae_block", {"side": "canonical"}, "t2")]),
        _Response([_tool_use("emit_verdict", P_VERDICT, "t3")]),
    ]
    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: _fake_anthropic(script, calls))

    out_dir = tmp_path / "out"
    rc = mod.main(_category_run_args(category_records, out_dir, []))
    assert rc == 0

    tis_slug = mod._tis_slug(TIS_ID)
    payload = json.loads((out_dir / tis_slug / "categories.json").read_text())
    assert payload["Predicted Structure"]["verdict"] == "interesting"
    assert len(single_shot) == 5
    # And the trace lands under P, not M.
    trace = json.loads((out_dir / tis_slug / "P_trace.json").read_text())
    assert trace["outcome"] == "emit_verdict"
    assert trace["n_data_calls"] == 2


def test_p_category_fails_loudly_without_the_hash_columns(
    mod, monkeypatch, tmp_path, only_p
):
    """A run predating StructureModule's hash columns cannot address the cache."""
    from swissisoform.site import structure_tools as p_tools

    records_dir = tmp_path / "records"
    records_dir.mkdir()
    stale = _ISO_FIXTURE_RECORD()
    stale["isoforms"][0]["_raw"] = {}  # pre-hash-column record
    (records_dir / "GENE_A.json").write_text(json.dumps(stale))

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    with pytest.raises(p_tools.StructureHashesMissing, match="StructureModule"):
        mod.main(_category_run_args(records_dir, tmp_path / "out", []))


# ── Output-shape normalisation + schema validation ────────────────────────


# The exact shape observed on a live cheeseman_test run: the model returned an
# array-declared tool parameter as one string of <item> markup.
_OBSERVED_STRING = (
    "\n<item>Canonical region 1-32 mean pLDDT 0.899</item>"
    "\n<item>PAE to the body averages 5.09 A</item>"
)
_ARRAY_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string"},
        "evidence_used": {"type": "array", "items": {"type": "string"}},
    },
}


def test_normalise_coerces_the_observed_item_markup(mod):
    out, coerced = mod._normalise_tool_input(
        {"verdict": "interesting", "evidence_used": _OBSERVED_STRING}, _ARRAY_SCHEMA
    )
    assert out["evidence_used"] == [
        "Canonical region 1-32 mean pLDDT 0.899",
        "PAE to the body averages 5.09 A",
    ]
    assert coerced == ["evidence_used"]
    assert out["verdict"] == "interesting"  # non-array fields untouched


def test_normalise_leaves_a_correct_list_alone(mod):
    payload = {"evidence_used": ["already", "a list"]}
    out, coerced = mod._normalise_tool_input(payload, _ARRAY_SCHEMA)
    assert out["evidence_used"] == ["already", "a list"]
    assert coerced == []


def test_normalise_wraps_an_untagged_string(mod):
    out, _ = mod._normalise_tool_input({"evidence_used": "one plain note"}, _ARRAY_SCHEMA)
    assert out["evidence_used"] == ["one plain note"]


def test_normalise_handles_missing_and_empty(mod):
    assert mod._normalise_tool_input({}, _ARRAY_SCHEMA)[0] == {}
    assert mod._normalise_tool_input({"evidence_used": ""}, _ARRAY_SCHEMA)[0][
        "evidence_used"
    ] == []


def test_normalise_is_schema_driven_not_field_specific(mod):
    """Any array-declared parameter is covered, not just evidence_used."""
    schema = {"type": "object", "properties": {"citations": {"type": "array"}}}
    out, coerced = mod._normalise_tool_input(
        {"citations": "<item>a</item><item>b</item>"}, schema
    )
    assert out["citations"] == ["a", "b"]
    assert coerced == ["citations"]


def test_normalise_is_a_noop_without_a_schema(mod):
    payload = {"evidence_used": _OBSERVED_STRING}
    assert mod._normalise_tool_input(payload, None)[0] == payload


# Verbatim from the CBX1 Predicted-Structure verdict (cheeseman_test, live P run):
# the model closed `reasoning` and opened `evidence_used` inside the string it was
# writing, so the tool call arrived with only `verdict` and `reasoning` keys and
# the raw markup rendered on the website.
_LEAKED_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string"},
        "reasoning": {"type": "string"},
        "evidence_used": {"type": "array", "items": {"type": "string"}},
    },
}
_LEAKED_REASONING = (
    "The truncation removes a short but confidently folded element."
    '</reasoning>\n<parameter name="evidence_used">'
    '["5-residue strand at 35-39, pLDDT 0.92", "PAE 1.39 A to residues 44-60"]'
)


def test_normalise_recovers_a_parameter_leaked_into_the_previous_value(mod):
    out, touched = mod._normalise_tool_input(
        {"verdict": "interesting", "reasoning": _LEAKED_REASONING}, _LEAKED_SCHEMA
    )
    assert out["reasoning"] == "The truncation removes a short but confidently folded element."
    assert "<parameter" not in out["reasoning"] and "</reasoning>" not in out["reasoning"]
    assert out["evidence_used"] == [
        "5-residue strand at 35-39, pLDDT 0.92",
        "PAE 1.39 A to residues 44-60",
    ]
    assert touched == ["reasoning(leak-trimmed)", "evidence_used(recovered)"]


def test_normalise_recovers_item_markup_from_a_leaked_parameter(mod):
    """The two failure modes compose: a leaked param whose value is <item> markup."""
    leaked = (
        "prose</reasoning>\n"
        '<parameter name="evidence_used"><item>first</item><item>second</item>'
    )
    out, _ = mod._normalise_tool_input({"reasoning": leaked}, _LEAKED_SCHEMA)
    assert out["evidence_used"] == ["first", "second"]


def test_normalise_never_clobbers_a_properly_emitted_parameter(mod):
    out, _ = mod._normalise_tool_input(
        {"reasoning": _LEAKED_REASONING, "evidence_used": ["the real one"]}, _LEAKED_SCHEMA
    )
    assert out["evidence_used"] == ["the real one"]
    assert out["reasoning"].endswith("element.")


def test_normalise_ignores_a_leaked_parameter_not_in_the_schema(mod):
    out, touched = mod._normalise_tool_input(
        {"reasoning": 'x</reasoning><parameter name="bogus">y'}, _LEAKED_SCHEMA
    )
    assert "bogus" not in out
    assert out["reasoning"] == "x"
    assert touched == ["reasoning(leak-trimmed)"]


def test_normalise_leaves_a_clean_reasoning_alone(mod):
    payload = {"verdict": "neutral", "reasoning": "No </other> tag matters here."}
    out, touched = mod._normalise_tool_input(payload, _LEAKED_SCHEMA)
    assert out["reasoning"] == payload["reasoning"]
    assert touched == []


def test_tool_loop_returns_a_normalised_verdict(mod, monkeypatch):
    """End-to-end: the loop must not hand a string-typed array to its caller."""
    calls: list = []
    script = [
        _Response([_tool_use("query_variants", {}, "t1")]),
        _Response([_tool_use("variant_effect_stats", {}, "t2")]),
        _Response(
            [
                _tool_use(
                    "emit_verdict",
                    {"verdict": "neutral", "reasoning": "x", "evidence_used": _OBSERVED_STRING},
                    "t3",
                )
            ]
        ),
    ]
    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: _fake_anthropic(script, calls))
    tools = [{"name": "emit_verdict", "input_schema": _ARRAY_SCHEMA}]

    verdict, trace = mod.run_tool_loop(
        system="S", user="U", tools=tools, dispatch=_dispatch_ok,
        model="claude-sonnet-5", max_tokens=1000, api_key="k",
    )
    assert isinstance(verdict["evidence_used"], list)
    assert len(verdict["evidence_used"]) == 2
    # The transcript records that the stored shape differs from what was emitted.
    assert trace["turns"][-1]["normalised"] == ["evidence_used"]


def test_schema_validation_is_actually_running(mod):
    """Guard against the dependency silently vanishing again.

    ``validate_against_schema`` is defensive — it returns [] when jsonschema is
    absent — so without this test the whole mechanism can rot back to a no-op
    unnoticed, which is exactly what happened between a64d8fa and this commit.
    """
    assert mod._try_import_jsonschema() is not None, "jsonschema missing from the env"

    schema = json.loads(
        (ROOT / "scripts/site/prompts/output_schemas/category_read.json").read_text()
    )
    # A genuine violation: evidence_used declared array, supplied as a string.
    bad = {"verdict": "neutral", "reasoning": "x", "evidence_used": _OBSERVED_STRING}
    assert mod.validate_against_schema(bad, schema), "validator did not flag a real violation"
    good = {"verdict": "neutral", "reasoning": "x", "evidence_used": ["a", "b"]}
    assert mod.validate_against_schema(good, schema) == []


def test_schema_warnings_print_without_verbose(mod, capsys):
    """Violations must not hide behind --verbose; that is how this went unnoticed."""
    schema = json.loads(
        (ROOT / "scripts/site/prompts/output_schemas/category_read.json").read_text()
    )
    bad = {"verdict": "not_a_valid_enum_value", "reasoning": "x"}
    mod._emit_schema_warnings(bad, schema, "some/label", verbose=False)
    assert "schema warning [some/label]" in capsys.readouterr().err


def _p_record_with_pae() -> dict:
    return {
        "category": "P",
        "name": "Predicted Structure",
        "isoform": {"tis_id": TIS_ID},
        "members": [
            {
                "criterion_id": "P1_structured_extension",
                "evidence": {
                    "isoform_structure_plddt_diffregion_mean": 0.82,
                    "isoform_structure_pae_diff_vs_diff": 2.3,
                    "isoform_structure_pae_body_vs_body": 2.5,
                    "isoform_structure_pae_diff_vs_body": 20.1,
                    "isoform_structure_pae_status": "ok",
                },
            }
        ],
    }


def test_superseded_pae_means_are_dropped_for_p(mod):
    """pae_block() with defaults IS diff_vs_body — shipping it pre-computed
    hands the model the answer to its own tool call, and 10/18 live reasonings
    cited it rather than querying.
    """
    ev = mod._strip_superseded_evidence(_p_record_with_pae(), "P")["members"][0]["evidence"]
    for gone in (
        "isoform_structure_pae_diff_vs_diff",
        "isoform_structure_pae_body_vs_body",
        "isoform_structure_pae_diff_vs_body",
    ):
        assert gone not in ev
    assert "_superseded_note" in ev


def test_pae_status_is_retained(mod):
    """Availability metadata, not a measurement — keeping it saves a wasted call."""
    ev = mod._strip_superseded_evidence(_p_record_with_pae(), "P")["members"][0]["evidence"]
    assert ev["isoform_structure_pae_status"] == "ok"


def test_non_pae_evidence_survives(mod):
    ev = mod._strip_superseded_evidence(_p_record_with_pae(), "P")["members"][0]["evidence"]
    assert ev["isoform_structure_plddt_diffregion_mean"] == 0.82


def test_other_categories_are_untouched(mod):
    """Only P declares superseded columns; M and the single-shot four keep everything."""
    rec = _p_record_with_pae()
    for letter in ("M", "C", "S"):
        out = mod._strip_superseded_evidence(rec, letter)
        assert out is rec  # returned unchanged, not even copied
    # And the caller is not mutated by the P path.
    mod._strip_superseded_evidence(rec, "P")
    assert "isoform_structure_pae_diff_vs_body" in rec["members"][0]["evidence"]


def test_p_dispatch_is_bound_to_the_isoforms_own_raw_row(mod, tmp_path, category_records):
    """Each isoform's readers must resolve through its own hashes, not a shared one."""
    records = mod.load_records(category_records)
    iso = mod._first_isoform(records)
    assert iso["_raw"]["isoform_structure_isoform_hash"] == "b" * 40

    from swissisoform.site import structure_tools as p_tools

    dispatch = p_tools.make_p_dispatch(iso["_raw"], cache_dir=tmp_path)
    # Empty cache dir → the readers report absence rather than raising.
    out = dispatch("plddt_profile", {})
    assert out["status"] == "no_cache"
    assert out["scale"] == "0-1"
