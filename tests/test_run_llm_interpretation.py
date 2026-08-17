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
    """A model that answers in prose JSON instead of calling the terminal tool.

    Passes the schema, so this pins the accept AFTER validation rather than the
    old truthiness check — the payload is clean and both data calls were made.
    """
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
        verdict_schema=_SCHEMA,
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


def test_hit_stripping_is_scoped_to_m(mod):
    """P keeps its hits; only M's are superseded by a reader.

    M's are a 30-row sample of a table ``query_variants`` reads in full; P3's are
    the complete scan its verdict was computed from. The strip was written for M
    when P had no ``evidence_hits_col``; P3 later gave it one and silently
    inherited the behaviour.
    """
    import types as _types

    def _record(criterion: str) -> dict:
        return {
            "members": [
                {
                    "criterion_id": criterion,
                    "evidence": {},
                    "hits": [{"x": i} for i in range(3)],
                    "n_hits_total": 3,
                    "n_hits_shown": 3,
                }
            ]
        }

    def _opening(letter: str, criterion: str) -> dict:
        return json.loads(
            mod._capture_tool_opening(
                config={"system": "sys", "tools": []},
                iso={"tis_id": TIS_ID},
                category_record=_record(criterion),
                args=_types.SimpleNamespace(model="m", max_tokens=1, temperature=0.0),
                letter=letter,
            )
        )

    m = _opening("M", "M1_pathogenic_variant_enrichment")["members"][0]
    assert m["hits"] == []
    assert m["n_hits_shown"] == 0
    assert "hits_note" in m

    p = _opening("P", "P3_secondary_structure")["members"][0]
    assert p["hits"] == [{"x": 0}, {"x": 1}, {"x": 2}]
    assert p["n_hits_shown"] == 3
    # No note either: its wording ("variant records") would be wrong for SSE
    # elements, and there is nothing to point the model at the readers for.
    assert "hits_note" not in p

    assert mod.STRIP_HITS_FOR_TOOLS == frozenset({"M"})


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


def test_category_pass_holds_back_the_file_when_the_loop_fails(
    mod, monkeypatch, tmp_path, category_records, variants_long, only_m
):
    """A failed M loop must not sink the other five categories, nor be staged.

    categories.json is the isoform's own idempotency key, so writing one that
    carries {"error": ...} would block its retry and let the rerun report success.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(mod, "call_llm", lambda *a, **kw: json.dumps({"verdict": "neutral",
                                                                     "reasoning": "x"}))
    script = [_Response([_tool_use("query_variants", {}, "t1")])]  # never terminates
    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: _fake_anthropic(script, []))

    out_dir = tmp_path / "out"
    iso_dir = out_dir / mod._tis_slug(TIS_ID)
    rc = mod.main(
        _category_run_args(
            category_records, out_dir,
            ["--variants-long", str(variants_long), "--max-tool-turns", "2"],
        )
    )
    assert rc == 1  # the run reports the failure
    assert not (iso_dir / "categories.json").exists()
    # The five good verdicts and the error are kept, under a name nothing reads.
    payload = json.loads((iso_dir / "categories.partial.json").read_text())
    assert "error" in payload["Mutation Landscape"]
    assert payload["Conservation"]["verdict"] == "neutral"
    # The partial transcript is still written — a failed loop is the one worth reading.
    trace = json.loads((iso_dir / "M_trace.json").read_text())
    assert trace["outcome"] == "max_turns_exhausted"


def test_a_held_back_isoform_is_retried_and_clears_its_partial(
    mod, monkeypatch, tmp_path, category_records, variants_long, only_m, capsys
):
    """The point of holding the write back: the rerun regenerates, without --force."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(mod, "call_llm", lambda *a, **kw: json.dumps({"verdict": "neutral",
                                                                     "reasoning": "x"}))
    out_dir = tmp_path / "out"
    iso_dir = out_dir / mod._tis_slug(TIS_ID)
    argv = _category_run_args(category_records, out_dir, ["--variants-long", str(variants_long)])

    monkeypatch.setattr(
        mod, "_try_import_anthropic",
        lambda: _fake_anthropic([_Response([_tool_use("query_variants", {}, "t1")])], []),
    )
    assert mod.main(argv + ["--max-tool-turns", "2"]) == 1

    # Same records and out dir, no --force: the isoform is absent, so it runs again.
    monkeypatch.setattr(
        mod, "_try_import_anthropic",
        lambda: _fake_anthropic(
            [
                _Response([_tool_use("query_variants", {}, "t1")]),
                _Response([_tool_use("variant_effect_stats", {}, "t2")]),
                _Response([_tool_use("emit_verdict", VERDICT, "t3")]),
            ],
            [],
        ),
    )
    assert mod.main(argv) == 0
    payload = json.loads((iso_dir / "categories.json").read_text())
    assert payload["Mutation Landscape"]["verdict"] == VERDICT["verdict"]
    assert not (iso_dir / "categories.partial.json").exists()

    # A third run now genuinely has nothing to do — and says so, rather than
    # reading as a clean "0/0 successful".
    capsys.readouterr()
    assert mod.main(argv) == 0
    assert "1 isoform(s) reused" in capsys.readouterr().out


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


def test_tool_loop_returns_the_tool_input_verbatim(mod, monkeypatch):
    """The loop must not rewrite what the model emitted.

    emit_verdict is declared strict, so the API guarantees the shape; coercing
    client-side would only paper over a broken contract. A wrong payload is
    meant to reach _emit_schema_warnings and be reported, not silently fixed.
    """
    calls: list = []
    emitted = {"verdict": "neutral", "reasoning": "x", "evidence_used": ["a", "b"]}
    script = [
        _Response([_tool_use("query_variants", {}, "t1")]),
        _Response([_tool_use("variant_effect_stats", {}, "t2")]),
        _Response([_tool_use("emit_verdict", emitted, "t3")]),
    ]
    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: _fake_anthropic(script, calls))
    tools = [{"name": "emit_verdict", "input_schema": _ARRAY_SCHEMA}]

    verdict, trace = mod.run_tool_loop(
        system="S", user="U", tools=tools, dispatch=_dispatch_ok,
        model="claude-sonnet-5", max_tokens=1000, api_key="k",
    )
    assert verdict == emitted
    assert "normalised" not in trace["turns"][-1]


def test_tool_loop_does_not_repair_a_malformed_verdict(mod, monkeypatch):
    """A string where an array is declared passes straight through, unrepaired.

    This is the shape a live run produced before emit_verdict became strict. It
    is preserved as a fixture because the schema validator must still flag it —
    see test_schema_validation_is_actually_running.
    """
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
    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: _fake_anthropic(script, []))
    verdict, _ = mod.run_tool_loop(
        system="S", user="U",
        tools=[{"name": "emit_verdict", "input_schema": _ARRAY_SCHEMA}],
        dispatch=_dispatch_ok, model="claude-sonnet-5", max_tokens=1000, api_key="k",
    )
    assert verdict["evidence_used"] == _OBSERVED_STRING


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



# ── Prompt capture (--save-prompts) ───────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_prompt_capture(mod):
    """Capture state is a module global; never let it leak between tests."""
    yield
    mod._PROMPT_DIR = None
    mod._PROMPT_OUT_DIR = None
    mod._PROMPT_INDEX.clear()


@pytest.fixture
def capture_dir(tmp_path: Path) -> Path:
    """Where capture tests write.

    Every test pins --save-prompts-dir here. The default resolves into the real
    data/llm_inputs/, so a test that omitted it would litter the repo;
    _default_prompt_dir is covered separately without running a pass.
    """
    return tmp_path / "captured"


def _capture_flags(capture_dir: Path) -> list[str]:
    return ["--dry-run", "--save-prompts", "--save-prompts-dir", str(capture_dir)]


def _split_capture(text: str) -> tuple[dict[str, str], str, str]:
    """Split a captured .txt into (header fields, system, user)."""
    head, rest = text.split("\n=== SYSTEM ===\n", 1)
    system, user = rest.split("\n\n=== USER ===\n", 1)
    fields = dict(line[2:].split(": ", 1) for line in head.splitlines())
    return fields, system, user.rstrip("\n")


def test_prompt_capture_is_off_by_default(tmp_path, category_records, mod, capture_dir):
    """No flag, no artifacts — the normal path is untouched.

    Also asserts the real data/llm_inputs/ default is not created, since an accidental
    write there would land in the repo rather than in tmp_path.
    """
    out_dir = tmp_path / "out"
    assert mod.main(_category_run_args(category_records, out_dir, ["--dry-run"])) == 0
    assert not capture_dir.exists()
    assert not mod._default_prompt_dir(out_dir).exists()


def test_default_prompt_dir_is_data_llm_keyed_by_run(mod):
    """Default is data/llm_inputs/{run}, with run read off --out, not --out itself."""
    resolved = mod._default_prompt_dir(Path("data/output/cheeseman_test/llm"))
    assert resolved == mod.DEFAULT_PROMPT_DIR / "cheeseman_test"
    assert mod.DEFAULT_PROMPT_DIR == ROOT / "data" / "llm_inputs"
    # Per-run subdir keeps two presets from clobbering each other's index.
    other = mod._default_prompt_dir(Path("data/output/cheeseman_13gene/llm"))
    assert other != resolved
    # A non-standard --out falls back to its own basename rather than guessing.
    assert mod._default_prompt_dir(Path("/scratch/adhoc")) == mod.DEFAULT_PROMPT_DIR / "adhoc"


def test_save_prompts_writes_one_txt_per_category(tmp_path, category_records, mod, capture_dir):
    """6 CDLMPS categories × 1 isoform, plus an index listing all of them.

    M and P land as ``*_tools.txt``: a dry run captures their real tool-loop
    opening rather than the single-shot prompt they never send.
    """
    out_dir = tmp_path / "out"
    run_args = _category_run_args(category_records, out_dir, _capture_flags(capture_dir))
    assert mod.main(run_args) == 0

    prompts = capture_dir
    files = sorted(p.name for p in (prompts / "category" / mod._tis_slug(TIS_ID)).glob("*.txt"))
    assert files == ["C.txt", "D.txt", "L.txt", "M_tools.txt", "P_tools.txt", "S.txt"]

    index = json.loads((prompts / "index_category.json").read_text())
    assert len(index) == 6
    assert {e["gene"] for e in index} == {"GENE_A"}
    assert {e["pass"] for e in index} == {"category"}


def test_save_prompts_dir_overrides_the_default_location(tmp_path, category_records, mod):
    out_dir = tmp_path / "out"
    elsewhere = tmp_path / "elsewhere"
    rc = mod.main(
        _category_run_args(
            category_records,
            out_dir,
            ["--dry-run", "--save-prompts", "--save-prompts-dir", str(elsewhere)],
        )
    )
    assert rc == 0
    assert (elsewhere / "index_category.json").exists()
    assert not mod._default_prompt_dir(out_dir).exists()


def test_dry_run_captures_tool_openings_without_any_api_call(
    tmp_path, category_records, mod, monkeypatch
, capture_dir):
    """The tool-loop opening is computable client-side, so a dry run gets it free.

    Nothing is stubbed here: no API key, no fake SDK, and the M/P data
    preconditions (variants parquet, fold cache) are never checked, because the
    opening comes from the evidence record and the static tool schemas. Any
    attempt to reach the network or run a loop would fail this test.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        mod, "_try_import_anthropic",
        lambda: pytest.fail("dry-run capture must not touch the anthropic SDK"),
    )
    monkeypatch.setattr(
        mod, "call_llm", lambda *a, **kw: pytest.fail("dry-run capture must not call the API")
    )

    out_dir = tmp_path / "out"
    run_args = _category_run_args(category_records, out_dir, _capture_flags(capture_dir))
    assert mod.main(run_args) == 0

    captured = capture_dir / "category" / mod._tis_slug(TIS_ID) / "M_tools.txt"
    fields, system, rest = _split_capture(captured.read_text(encoding="utf-8"))
    assert fields["mode"] == "tool_loop"
    assert system
    user, tools_json = rest.split("\n\n=== TOOLS ===\n", 1)
    assert json.loads(user)["category"] == "M"
    assert {t["name"] for t in json.loads(tools_json)} >= {"query_variants", "emit_verdict"}


def test_two_passes_into_one_dir_keep_separate_indexes(
    tmp_path, category_records, mod, capture_dir
):
    """Each pass runs as its own process; a shared index.json would be truncated.

    Regression: category then synthesis into the same --out left an index
    listing only synthesis's isoforms while the category .txt files sat beside
    it, so the index under-reported the corpus with no error.
    """
    out_dir = tmp_path / "out"
    tis_slug = mod._tis_slug(TIS_ID)
    run_args = _category_run_args(category_records, out_dir, _capture_flags(capture_dir))
    assert mod.main(run_args) == 0

    # Satisfy the synthesis prereq, then run it into the same directory.
    (out_dir / tis_slug).mkdir(parents=True, exist_ok=True)
    (out_dir / tis_slug / "categories.json").write_text(json.dumps({"Conservation": {}}))
    rc = mod.main(
        [
            "--records", str(category_records), "--out", str(out_dir),
            "--pass", "synthesis", *_capture_flags(capture_dir),
        ]
    )
    assert rc == 0

    prompts = capture_dir
    category_index = json.loads((prompts / "index_category.json").read_text())
    synthesis_index = json.loads((prompts / "index_synthesis.json").read_text())
    assert len(category_index) == 6
    assert len(synthesis_index) == 1
    # Together the indexes account for every .txt actually written.
    assert len(category_index) + len(synthesis_index) == len(list(prompts.rglob("*.txt")))


def test_dry_run_without_save_prompts_leaves_tool_categories_alone(
    tmp_path, category_records, mod
):
    """Capture-only tool configs are built for --save-prompts, nothing else.

    Guards the branch added to _tool_categories: a plain dry run must keep
    returning {} so every category stays on the single-shot print path.
    """
    from swissisoform.site import llm as rli

    args = rli.build_parser().parse_args(
        _category_run_args(category_records, tmp_path / "out", ["--dry-run"])
    )
    assert rli._tool_categories(args, rli.SYSTEM_PROMPT_PATH.parent, None) == {}


def test_captured_prompt_is_byte_identical_to_the_assembled_prompt(
    tmp_path, category_records, mod
, capture_dir):
    """The whole point: the file IS the prompt, not a summary of it."""
    from swissisoform.site.evidence import CATEGORIES, slice_category

    out_dir = tmp_path / "out"
    run_args = _category_run_args(category_records, out_dir, _capture_flags(capture_dir))
    assert mod.main(run_args) == 0

    prompts_root = mod.SYSTEM_PROMPT_PATH.parent
    spec = mod.PASS_REGISTRY["category"]
    expected = mod.build_prompt(
        slice_category(
            {**_ISO_FIXTURE_RECORD()["isoforms"][0], "gene": {"name": "GENE_A"}},
            next(c for c in CATEGORIES if c["letter"] == "C"),
        ),
        mod.load_system_prompt(prompts_root / spec.system_prompt_filename),
        mod.load_output_schema(prompts_root / spec.output_schema_filename),
    )

    path = capture_dir / "category" / mod._tis_slug(TIS_ID) / "C.txt"
    fields, system, user = _split_capture(path.read_text(encoding="utf-8"))
    assert system == expected.system
    assert user == expected.user
    # Header self-consistency, so a truncation can't hide behind a stale count.
    assert int(fields["system_chars"]) == len(expected.system)
    assert int(fields["user_chars"]) == len(expected.user)


def test_capture_header_is_machine_parseable(tmp_path, category_records, mod, capture_dir):
    """Every header line splits cleanly, so the call stays reconstructible."""
    out_dir = tmp_path / "out"
    run_args = _category_run_args(category_records, out_dir, _capture_flags(capture_dir))
    assert mod.main(run_args) == 0
    text = (capture_dir / "category" / mod._tis_slug(TIS_ID) / "C.txt").read_text()
    head = text.split("\n=== SYSTEM ===\n", 1)[0]
    for line in head.splitlines():
        assert line.startswith("# ")
        assert len(line[2:].split(": ", 1)) == 2


def test_capture_header_records_the_no_sampling_gate(tmp_path, category_records, mod, capture_dir):
    """claude-sonnet-5 sends thinking=disabled and no temperature."""
    out_dir = tmp_path / "out"
    rc = mod.main(
        _category_run_args(
            category_records,
            out_dir,
            [*_capture_flags(capture_dir), "--model", "claude-sonnet-5"],
        )
    )
    assert rc == 0
    fields, _, _ = _split_capture(
        (capture_dir / "category" / mod._tis_slug(TIS_ID) / "C.txt").read_text()
    )
    assert fields["model"] == "claude-sonnet-5"
    assert fields["thinking"] == "disabled"
    assert "temperature" not in fields


def test_capture_header_records_temperature_for_sampling_models(
    tmp_path, category_records, mod
, capture_dir):
    """An older model keeps the temperature knob — the header follows the gate."""
    out_dir = tmp_path / "out"
    rc = mod.main(
        _category_run_args(
            category_records,
            out_dir,
            [*_capture_flags(capture_dir), "--model", "claude-sonnet-4-6"],
        )
    )
    assert rc == 0
    fields, _, _ = _split_capture(
        (capture_dir / "category" / mod._tis_slug(TIS_ID) / "C.txt").read_text()
    )
    assert fields["temperature"] == "0.0"
    assert "thinking" not in fields


def test_synthesis_capture_writes_one_txt_per_isoform(tmp_path, category_records, mod, capture_dir):
    out_dir = tmp_path / "out"
    tis_slug = mod._tis_slug(TIS_ID)
    # Satisfy the category prereq without running it.
    (out_dir / tis_slug).mkdir(parents=True)
    (out_dir / tis_slug / "categories.json").write_text(json.dumps({"Conservation": {}}))

    rc = mod.main(
        [
            "--records", str(category_records), "--out", str(out_dir),
            "--pass", "synthesis", *_capture_flags(capture_dir),
        ]
    )
    assert rc == 0
    captured = capture_dir / "synthesis" / f"{tis_slug}.txt"
    assert captured.exists()
    fields, _, _ = _split_capture(captured.read_text(encoding="utf-8"))
    assert fields["pass"] == "synthesis"
    assert fields["tis_id"] == TIS_ID


def test_tool_loop_capture_records_the_opening_and_the_tool_schemas(
    mod, monkeypatch, tmp_path, category_records, variants_long, only_m
, capture_dir):
    """M_trace.json holds what the model DID; M_tools.txt holds what it was TOLD.

    The trace stores only ``opening_context_chars`` and never the system prompt
    or the tool definitions, so this is the only record of the loop's input.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(
        mod, "call_llm",
        lambda *a, **kw: json.dumps({"verdict": "neutral", "reasoning": "single shot."}),
    )
    script = [
        _Response([_tool_use("variant_position_histogram", {}, "t1")]),
        _Response([_tool_use("query_variants", {}, "t2")]),
        _Response([_tool_use("emit_verdict", VERDICT, "t3")]),
    ]
    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: _fake_anthropic(script, []))

    out_dir = tmp_path / "out"
    rc = mod.main(
        _category_run_args(
            category_records,
            out_dir,
            [
                "--variants-long", str(variants_long),
                "--save-prompts", "--save-prompts-dir", str(capture_dir),
            ],
        )
    )
    assert rc == 0

    tis_slug = mod._tis_slug(TIS_ID)
    cat_dir = capture_dir / "category" / tis_slug
    text = (cat_dir / "M_tools.txt").read_text(encoding="utf-8")

    # The single-shot prompt for M is assembled but never sent, so it must not
    # appear in the corpus as though it were.
    assert not (cat_dir / "M.txt").exists()
    assert (cat_dir / "C.txt").exists()

    fields, system, rest = _split_capture(text)
    assert fields["mode"] == "tool_loop"
    assert system  # the category-pass-M.txt system prompt, absent from the trace
    user, tools_json = rest.split("\n\n=== TOOLS ===\n", 1)
    assert json.loads(user)["category"] == "M"
    tools = json.loads(tools_json)
    assert {t["name"] for t in tools} >= {"query_variants", "emit_verdict"}
    # The trace only ever recorded the length of this text.
    trace = json.loads((out_dir / tis_slug / "M_trace.json").read_text())
    assert trace["opening_context_chars"] == len(user)


# ── Run identity (run_id on both sides, corpus drift) ─────────────────────
#
# The corpus is keyed by isoform and overwritten in place, and outputs are
# reused across runs, so a .txt and a categories.json that look like a pair
# routinely are not one. A shared run id is what makes that checkable.


def _emit_verdict_script():
    """A three-turn M loop that terminates properly."""
    return [
        _Response([_tool_use("query_variants", {}, "t1")]),
        _Response([_tool_use("variant_effect_stats", {}, "t2")]),
        _Response([_tool_use("emit_verdict", VERDICT, "t3")]),
    ]


def _live_capture_args(category_records, out_dir, variants_long, capture_dir):
    return _category_run_args(
        category_records,
        out_dir,
        [
            "--variants-long", str(variants_long),
            "--save-prompts", "--save-prompts-dir", str(capture_dir),
        ],
    )


def test_one_invocation_stamps_prompt_and_output_with_the_same_run_id(
    mod, monkeypatch, tmp_path, category_records, variants_long, only_m, capture_dir
):
    """The join between the corpus and the verdicts it produced."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(mod, "call_llm", lambda *a, **kw: json.dumps({"verdict": "neutral",
                                                                     "reasoning": "x"}))
    monkeypatch.setattr(
        mod, "_try_import_anthropic", lambda: _fake_anthropic(_emit_verdict_script(), [])
    )
    out_dir = tmp_path / "out"
    assert mod.main(_live_capture_args(category_records, out_dir, variants_long, capture_dir)) == 0

    tis_slug = mod._tis_slug(TIS_ID)
    stamp = json.loads((out_dir / tis_slug / "categories.meta.json").read_text())
    fields, _, _ = _split_capture(
        (capture_dir / "category" / tis_slug / "C.txt").read_text(encoding="utf-8")
    )
    assert stamp["run_id"] and fields["run_id"] == stamp["run_id"]
    assert fields["run_mode"] == "live" and stamp["run_mode"] == "live"
    assert stamp["pass"] == "category" and stamp["prompts_captured"] is True
    # The tool-loop opening is stamped by the same run, from the live path.
    m_fields, _, _ = _split_capture(
        (capture_dir / "category" / tis_slug / "M_tools.txt").read_text(encoding="utf-8")
    )
    assert m_fields["run_id"] == stamp["run_id"]


def test_reused_output_keeps_its_old_run_id_and_the_capture_says_so(
    mod, monkeypatch, tmp_path, category_records, variants_long, only_m, capture_dir, capsys
):
    """The drift that actually happens: the rerun skips, so the corpus goes stale."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(mod, "call_llm", lambda *a, **kw: json.dumps({"verdict": "neutral",
                                                                     "reasoning": "x"}))
    monkeypatch.setattr(
        mod, "_try_import_anthropic", lambda: _fake_anthropic(_emit_verdict_script(), [])
    )
    out_dir = tmp_path / "out"
    argv = _live_capture_args(category_records, out_dir, variants_long, capture_dir)
    assert mod.main(argv) == 0
    first = json.loads((out_dir / mod._tis_slug(TIS_ID) / "categories.meta.json").read_text())

    # Second invocation: the isoform already has output, so nothing is called and
    # nothing is captured — but the corpus still sits there looking current.
    capsys.readouterr()
    monkeypatch.setattr(
        mod, "_try_import_anthropic", lambda: _fake_anthropic(_emit_verdict_script(), [])
    )
    assert mod.main(argv) == 0
    captured = capsys.readouterr()
    unchanged = json.loads((out_dir / mod._tis_slug(TIS_ID) / "categories.meta.json").read_text())
    assert unchanged["run_id"] == first["run_id"]  # reused, not rewritten
    assert "captured 0 prompts" in captured.out
    assert "1 isoform(s)" in captured.err and "from a different run" in captured.err
    # And the leftover .txt files are named as leftovers.
    assert "left over from earlier runs" in captured.err


def test_dry_run_capture_says_it_describes_no_output(
    mod, monkeypatch, tmp_path, category_records, variants_long, only_m, capture_dir, capsys
):
    """--dry-run --save-prompts makes no API call, so it can match nothing."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(mod, "call_llm", lambda *a, **kw: json.dumps({"verdict": "neutral",
                                                                     "reasoning": "x"}))
    monkeypatch.setattr(
        mod, "_try_import_anthropic", lambda: _fake_anthropic(_emit_verdict_script(), [])
    )
    out_dir = tmp_path / "out"
    assert mod.main(_live_capture_args(category_records, out_dir, variants_long, capture_dir)) == 0

    # --force so the dry run re-dumps the corpus rather than skipping the
    # already-done isoform: this is how the corpus is normally refreshed, and it
    # is exactly the invocation whose output looks like it belongs to the run.
    capsys.readouterr()
    dry_dir = tmp_path / "captured_dry"
    assert mod.main(
        _category_run_args(
            category_records, out_dir,
            ["--dry-run", "--force", "--save-prompts", "--save-prompts-dir", str(dry_dir)],
        )
    ) == 0
    err = capsys.readouterr().err
    assert "run_mode=dry_run" in err and "describes none of the 1" in err
    fields, _, _ = _split_capture(
        (dry_dir / "category" / mod._tis_slug(TIS_ID) / "C.txt").read_text(encoding="utf-8")
    )
    assert fields["run_mode"] == "dry_run"


def test_output_without_a_run_stamp_is_reported_as_unknown(
    mod, monkeypatch, tmp_path, category_records, variants_long, only_m, capture_dir, capsys
):
    """Outputs written before stamping existed must be flagged, not silently trusted."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    out_dir = tmp_path / "out"
    iso_dir = out_dir / mod._tis_slug(TIS_ID)
    iso_dir.mkdir(parents=True)
    (iso_dir / "categories.json").write_text("{}")  # legacy artifact, no sidecar

    assert mod.main(_live_capture_args(category_records, out_dir, variants_long, capture_dir)) == 0
    err = capsys.readouterr().err
    assert "no run stamp" in err and "1 isoform(s)" in err


def test_capture_sends_the_same_output_schema_the_real_call_does(mod, tmp_path):
    """The rebuilt params were missing output_config, so the header lied about it."""
    schema = {"type": "object", "properties": {"verdict": {"enum": ["a"]}}}
    prompt = mod.Prompt(system="sys", user="usr")
    mod._begin_run(dry_run=False)
    mod.enable_prompt_capture(tmp_path / "cap", pass_name="category", out_dir=None)

    mod._capture_single_shot(
        "category/x/C", prompt, model=mod.DEFAULT_MODEL, max_tokens=10,
        temperature=0.0, meta={"pass": "category"}, output_schema=schema,
    )
    mod._capture_single_shot(
        "category/x/D", prompt, model=mod.DEFAULT_MODEL, max_tokens=10,
        temperature=0.0, meta={"pass": "category"},
    )
    with_schema, _, _ = _split_capture((tmp_path / "cap/category/x/C.txt").read_text())
    without, _, _ = _split_capture((tmp_path / "cap/category/x/D.txt").read_text())
    assert with_schema["structured"] == "True"
    assert without["structured"] == "False"
    # The header claim is derived from the params, which now carry the schema.
    params = mod._build_request_params(
        model=mod.DEFAULT_MODEL, max_tokens=10, temperature=0.0,
        system="sys", user="usr", warn=False, output_schema=schema,
    )
    assert "output_config" in params


def test_transport_for_matches_the_branch_call_llm_takes(mod, monkeypatch):
    """One decision, two readers — capture must not restate call_llm's condition."""
    plain = "claude-sonnet-4-6"  # not in _NO_SAMPLING_MODELS
    assert mod.DEFAULT_MODEL in mod._NO_SAMPLING_MODELS

    monkeypatch.setattr(mod, "_try_import_mozzarellm", lambda: object())
    assert mod._transport_for(plain, structured=False) == "mozzarellm"
    assert mod._transport_for(plain, structured=True) == "sdk"  # output_config has no home
    assert mod._transport_for(plain, structured=False, tools=True) == "sdk"
    assert mod._transport_for(mod.DEFAULT_MODEL, structured=False) == "sdk"  # rejects temperature

    monkeypatch.setattr(mod, "_try_import_mozzarellm", lambda: None)
    assert mod._transport_for(plain, structured=False) == "sdk"


# ── Structured outputs (output_config.format) ─────────────────────────────
#
# A batched category once lost its verdict to `Invalid control character at
# char 1252`: the model wrote a literal newline inside its `reasoning` string.
# Nothing constrained the format — the schema was prose in the prompt and a
# linter after the fact. output_config makes it a decoding constraint.
#
# Subset behaviour below was probed against claude-sonnet-5, not assumed: the
# API accepts (and ignores) minLength/maxLength but hard-400s on uniqueItems.


def test_output_format_schema_drops_only_the_rejected_keywords(mod):
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["verdict"],
        "properties": {
            "verdict": {"enum": ["a", "b"]},
            "reasoning": {"type": "string", "minLength": 1, "maxLength": 1500},
            "tags": {"type": "array", "uniqueItems": True, "items": {"enum": ["x"]}},
        },
    }
    out = mod._output_format_schema(schema)

    assert "uniqueItems" not in json.dumps(out)  # hard 400 from the API
    # Accepted-but-ignored: kept so the prompt copy and the wire copy agree on
    # everything the API tolerates.
    assert out["properties"]["reasoning"]["maxLength"] == 1500
    assert out["properties"]["verdict"]["enum"] == ["a", "b"]
    assert out["properties"]["tags"]["items"]["enum"] == ["x"]
    assert out["additionalProperties"] is False
    assert out["required"] == ["verdict"]
    assert schema["properties"]["tags"]["uniqueItems"] is True, "input must not be mutated"


# The filter has to classify each value before walking it: a subschema is
# filtered, a name -> subschema map has only its VALUES filtered, and instance
# data is left alone. "Is it a dict?" answers none of those, and got all three
# of the following wrong.


def test_a_property_named_like_a_keyword_survives(mod):
    """Dropping it would leave `required` naming a key additionalProperties forbids."""
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["maximum", "verdict"],
        "properties": {
            "maximum": {"type": "number", "minimum": 0},  # field name, not a keyword
            "verdict": {"enum": ["a"]},
        },
    }
    out = mod._output_format_schema(schema)
    assert sorted(out["properties"]) == ["maximum", "verdict"]
    assert set(out["required"]) <= set(out["properties"]), "unsatisfiable wire schema"
    # ...while a genuine keyword INSIDE that property is still stripped.
    assert "minimum" not in out["properties"]["maximum"]


def test_list_valued_subschemas_are_filtered(mod):
    """anyOf/oneOf/allOf and tuple-form items are lists, and were skipped entirely."""
    schema = {
        "properties": {
            "hits": {"anyOf": [{"type": "array", "uniqueItems": True}, {"type": "null"}]},
            "pair": {"items": [{"type": "integer", "minimum": 0}, {"type": "string"}]},
            "both": {"allOf": [{"maxItems": 3}], "oneOf": [{"multipleOf": 2}]},
        }
    }
    assert "uniqueItems" not in json.dumps(mod._output_format_schema(schema))
    assert "minimum" not in json.dumps(mod._output_format_schema(schema))
    assert "maxItems" not in json.dumps(mod._output_format_schema(schema))
    assert "multipleOf" not in json.dumps(mod._output_format_schema(schema))


def test_instance_data_is_never_filtered(mod):
    """const/enum/default hold values, not schemas — filtering rewrites the assertion.

    Object-valued enum survived only because lists were skipped, so recursing
    into lists without this exemption would trade one bug for another.
    """
    schema = {
        "properties": {
            "cfg": {
                "const": {"minimum": 3, "label": "x"},
                "enum": [{"maximum": 9}],
                "default": {"maxItems": 1},
            }
        }
    }
    cfg = mod._output_format_schema(schema)["properties"]["cfg"]
    assert cfg["const"] == {"minimum": 3, "label": "x"}
    assert cfg["enum"] == [{"maximum": 9}]
    assert cfg["default"] == {"maxItems": 1}


def test_defs_names_and_string_lists_pass_through(mod):
    """$defs is another name -> schema map; required/type lists are not schemas."""
    schema = {
        "$defs": {"minimum": {"type": "integer", "maximum": 5}},
        "type": ["string", "null"],
        "required": ["minimum"],
    }
    out = mod._output_format_schema(schema)
    assert list(out["$defs"]) == ["minimum"]  # the definition's NAME
    assert "maximum" not in out["$defs"]["minimum"]  # its keyword, still stripped
    assert out["type"] == ["string", "null"] and out["required"] == ["minimum"]


def test_real_schemas_are_wire_safe(mod):
    """The two live schemas survive the subset filter."""
    root = ROOT / "scripts/site/prompts/output_schemas"
    for name in ("category_read.json", "synthesis.json"):
        out = mod._output_format_schema(json.loads((root / name).read_text()))
        assert "uniqueItems" not in json.dumps(out), name
        assert out["additionalProperties"] is False, name


def test_request_params_gate_output_config_on_the_schema(mod):
    """No schema, no output_config — so the tool-loop and legacy paths are
    provably unchanged by this feature.
    """
    kw = dict(model="claude-sonnet-5", max_tokens=10, temperature=0.0, system="s", user="u")

    assert "output_config" not in mod._build_request_params(**kw)

    params = mod._build_request_params(**kw, output_schema={"type": "object"})
    assert params["output_config"] == {
        "format": {"type": "json_schema", "schema": {"type": "object"}}
    }


def test_incomplete_response_names_truncation_and_refusal(mod):
    """Structured outputs guarantee parseable JSON only if the model FINISHED.

    Both remaining failure modes otherwise surface as a generic JSONDecodeError,
    indistinguishable from a malformed generation.
    """
    with pytest.raises(mod.IncompleteResponse, match="max_tokens"):
        mod._raise_on_incomplete("max_tokens")
    with pytest.raises(mod.IncompleteResponse, match="refusal"):
        mod._raise_on_incomplete("refusal")
    mod._raise_on_incomplete("end_turn")  # does not raise
    mod._raise_on_incomplete(None)


def test_failed_response_is_saved_for_diagnosis(mod, tmp_path):
    """The bug that motivated this was undiagnosable: the raw text was replaced
    by {"error": ...} and lost.
    """
    bad = '{"verdict": "neutral", "reasoning": "line one\nline two"}'
    with pytest.raises(json.JSONDecodeError) as excinfo:
        mod.parse_response(bad)
    assert "control character" in str(excinfo.value)

    mod._save_failed_response(
        tmp_path, "chr1-1-+-ATG-ENST1-1", "Structural Characteristics", bad, excinfo.value
    )
    (written,) = list((tmp_path / "_failures").glob("*.txt"))
    body = written.read_text()
    assert "line one\nline two" in body
    assert "JSONDecodeError" in body


def test_save_failed_response_never_masks_the_original_error(mod, tmp_path):
    """Diagnostics are best-effort; a write failure must not replace the real one."""
    mod._save_failed_response(tmp_path, "slug", "cat", None, ValueError("x"))  # no text
    assert not (tmp_path / "_failures").exists()

    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    mod._save_failed_response(blocked, "slug", "cat", "text", ValueError("x"))  # no raise


# ── Verdict validation (strict:true does not police string contents) ──────
#
# Two P verdicts were persisted containing tool-call markup written as prose
# inside the `reasoning` string. On one, the model noticed and re-emitted a clean
# call in the same turn — which the loop discarded by taking the first block.

_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "reasoning"],
    "properties": {
        "verdict": {"type": "string", "enum": ["interesting", "neutral", "not_interesting"]},
        "reasoning": {"type": "string", "minLength": 1, "maxLength": 60},
        "evidence_used": {"type": "array", "items": {"type": "string"}},
    },
}
_DIRTY = {"verdict": "neutral",
          "reasoning": 'Fold is modest.</reasoning>\n<parameter name="evidence_used">'}
_CLEAN = {"verdict": "neutral", "reasoning": "Fold is modest and not integrated."}


def test_verdict_violations_flags_markup_and_length(mod):
    assert mod._verdict_violations(_CLEAN, _SCHEMA) == []

    (markup,) = mod._verdict_violations(_DIRTY, _SCHEMA)
    assert "tool-call markup" in markup

    long = {"verdict": "neutral", "reasoning": "x" * 61}
    assert any("too long" in v for v in mod._verdict_violations(long, _SCHEMA))

    assert mod._verdict_violations("not a dict", _SCHEMA) == ["payload is str, expected object"]


def test_strip_verdict_markup_keeps_the_prose(mod):
    out = mod._strip_verdict_markup(_DIRTY)
    assert out["reasoning"] == "Fold is modest."
    assert out["verdict"] == "neutral"
    assert mod._verdict_violations(out, _SCHEMA) == []


def test_tool_loop_prefers_the_valid_terminal_block(mod, monkeypatch):
    """The CBX1 shape: the model self-corrects within one turn.

    Its correction carried the better payload and we returned the first block.
    """
    calls: list = []
    script = [
        _Response([_tool_use("query_variants", {}, "t1")]),
        _Response([_tool_use("query_variants", {}, "t2")]),
        _Response([
            _tool_use("emit_verdict", _DIRTY, "t3a"),
            _tool_use("emit_verdict", _CLEAN, "t3b"),
        ]),
    ]
    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: _fake_anthropic(script, calls))

    verdict, trace = mod.run_tool_loop(
        system="S", user="U", tools=[], dispatch=_dispatch_ok,
        model="claude-sonnet-5", max_tokens=1000, api_key="k",
        verdict_schema=_SCHEMA,
    )
    assert verdict == _CLEAN
    assert trace["outcome"] == "emit_verdict"
    assert len(calls) == 3, "the correction is used in-turn, costing no extra round trip"


def test_tool_loop_rejects_and_retries_a_lone_bad_verdict(mod, monkeypatch):
    """The CDC34 shape: one corrupt call, nothing to fall back to."""
    calls: list = []
    script = [
        _Response([_tool_use("query_variants", {}, "t1")]),
        _Response([_tool_use("query_variants", {}, "t2")]),
        _Response([_tool_use("emit_verdict", _DIRTY, "t3")]),
        _Response([_tool_use("emit_verdict", _CLEAN, "t4")]),
    ]
    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: _fake_anthropic(script, calls))

    verdict, trace = mod.run_tool_loop(
        system="S", user="U", tools=[], dispatch=_dispatch_ok,
        model="claude-sonnet-5", max_tokens=1000, api_key="k",
        verdict_schema=_SCHEMA,
    )
    assert verdict == _CLEAN
    # The rejection is fed back as an error tool_result naming the problem.
    rejected = [r for t in trace["turns"] for r in (t.get("tool_results") or [])
                if "rejected" in r]
    assert len(rejected) == 1
    assert "tool-call markup" in rejected[0]["rejected"]


def test_tool_loop_salvages_when_the_verdict_never_validates(mod, monkeypatch):
    calls: list = []
    script = [
        _Response([_tool_use("query_variants", {}, "t1")]),
        _Response([_tool_use("query_variants", {}, "t2")]),
        _Response([_tool_use("emit_verdict", _DIRTY, "t3")]),
    ]
    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: _fake_anthropic(script, calls))

    verdict, trace = mod.run_tool_loop(
        system="S", user="U", tools=[], dispatch=_dispatch_ok,
        model="claude-sonnet-5", max_tokens=1000, api_key="k",
        max_turns=4,
        verdict_schema=_SCHEMA,
    )
    assert trace["outcome"] == "emit_verdict_salvaged"
    assert verdict["reasoning"] == "Fold is modest."


def test_tool_loop_without_a_schema_is_unchanged(mod, monkeypatch):
    """No schema, no validation — the tool-loop path is untouched by default."""
    calls: list = []
    script = [
        _Response([_tool_use("query_variants", {}, "t1")]),
        _Response([_tool_use("query_variants", {}, "t2")]),
        _Response([_tool_use("emit_verdict", _DIRTY, "t3")]),
    ]
    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: _fake_anthropic(script, calls))

    verdict, trace = mod.run_tool_loop(
        system="S", user="U", tools=[], dispatch=_dispatch_ok,
        model="claude-sonnet-5", max_tokens=1000, api_key="k",
    )
    assert verdict == _DIRTY and trace["outcome"] == "emit_verdict"


# ── The other verdict door: plain JSON, no tool call ──────────────────────
#
# `strict: true` polices tool INPUTS, so a verdict arriving as text is unpoliced
# by construction. It used to be accepted on `payload.get("verdict")` alone —
# the same payload the tool door would reject and retry.


def _data_turns(n: int = 2) -> list:
    """n reader-tool turns, so min_data_calls is satisfied."""
    return [_Response([_tool_use("query_variants", {}, f"t{i}")]) for i in range(n)]


def _nudge_reasons(trace: dict) -> list[str]:
    return [t["nudge_reason"] for t in trace["turns"] if t.get("nudge_reason")]


def test_text_verdict_carrying_markup_is_rejected_then_re_answered(mod, monkeypatch):
    """The leaked-markup payload, delivered by the door strict:true cannot reach."""
    script = [
        *_data_turns(),
        _Response([_text(json.dumps(_DIRTY))], stop_reason="end_turn"),
        _Response([_text(json.dumps(_CLEAN))], stop_reason="end_turn"),
    ]
    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: _fake_anthropic(script, []))

    verdict, trace = mod.run_tool_loop(
        system="S", user="U", tools=[], dispatch=_dispatch_ok,
        model="claude-sonnet-5", max_tokens=1000, api_key="k",
        verdict_schema=_SCHEMA,
    )
    assert verdict == _CLEAN and trace["outcome"] == "text_verdict"
    (reason,) = _nudge_reasons(trace)
    assert "tool-call markup" in reason


def test_off_enum_text_verdict_is_rejected(mod, monkeypatch):
    """The site matches verdicts by string literal, so an off-enum value fires nothing."""
    script = [
        *_data_turns(),
        _Response([_text(json.dumps({"verdict": "Interesting", "reasoning": "ok"}))],
                  stop_reason="end_turn"),
        _Response([_text(json.dumps(_CLEAN))], stop_reason="end_turn"),
    ]
    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: _fake_anthropic(script, []))

    verdict, trace = mod.run_tool_loop(
        system="S", user="U", tools=[], dispatch=_dispatch_ok,
        model="claude-sonnet-5", max_tokens=1000, api_key="k",
        verdict_schema=_SCHEMA,
    )
    assert verdict == _CLEAN
    (reason,) = _nudge_reasons(trace)
    assert "verdict" in reason and "is not one of" in reason


def test_text_verdict_before_the_data_calls_is_rejected(mod, monkeypatch):
    """Answering in prose must not be a way around the read-first rule."""
    script = [
        _Response([_text(json.dumps(_CLEAN))], stop_reason="end_turn"),  # turn 1, no data yet
        *_data_turns(),
        _Response([_text(json.dumps(_CLEAN))], stop_reason="end_turn"),
    ]
    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: _fake_anthropic(script, []))

    verdict, trace = mod.run_tool_loop(
        system="S", user="U", tools=[], dispatch=_dispatch_ok,
        model="claude-sonnet-5", max_tokens=1000, api_key="k",
        verdict_schema=_SCHEMA,
    )
    assert verdict == _CLEAN and trace["n_data_calls"] == 2
    (reason,) = _nudge_reasons(trace)
    assert "call at least 2 reader tools" in reason


def test_two_bad_text_verdicts_are_salvaged(mod, monkeypatch):
    """Same last resort the tool door gets: good prose, markup cut at the tail."""
    script = [
        *_data_turns(),
        _Response([_text(json.dumps(_DIRTY))], stop_reason="end_turn"),
        _Response([_text(json.dumps(_DIRTY))], stop_reason="end_turn"),
    ]
    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: _fake_anthropic(script, []))

    verdict, trace = mod.run_tool_loop(
        system="S", user="U", tools=[], dispatch=_dispatch_ok,
        model="claude-sonnet-5", max_tokens=1000, api_key="k",
        verdict_schema=_SCHEMA,
    )
    assert trace["outcome"] == "emit_verdict_salvaged"
    assert verdict["reasoning"] == "Fold is modest."


def test_two_premature_text_verdicts_raise_rather_than_salvage(mod, monkeypatch):
    """A premature verdict is not corrupt, so it is never collected for salvage."""
    script = [
        _Response([_text(json.dumps(_CLEAN))], stop_reason="end_turn"),
        _Response([_text(json.dumps(_CLEAN))], stop_reason="end_turn"),
    ]
    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: _fake_anthropic(script, []))

    with pytest.raises(mod.ToolLoopError) as excinfo:
        mod.run_tool_loop(
            system="S", user="U", tools=[], dispatch=_dispatch_ok,
            model="claude-sonnet-5", max_tokens=1000, api_key="k",
            verdict_schema=_SCHEMA,
        )
    assert "call at least 2 reader tools" in str(excinfo.value)
    assert excinfo.value.trace["outcome"] == "no_tool_call"


def test_text_verdict_without_a_schema_still_checks_the_data_calls(mod, monkeypatch):
    """No schema means no content validation — the read-first rule is independent."""
    script = [
        _Response([_text(json.dumps(_DIRTY))], stop_reason="end_turn"),  # no data calls yet
        *_data_turns(),
        _Response([_text(json.dumps(_DIRTY))], stop_reason="end_turn"),
    ]
    monkeypatch.setattr(mod, "_try_import_anthropic", lambda: _fake_anthropic(script, []))

    verdict, trace = mod.run_tool_loop(
        system="S", user="U", tools=[], dispatch=_dispatch_ok,
        model="claude-sonnet-5", max_tokens=1000, api_key="k",
    )
    assert verdict == _DIRTY  # unvalidated, exactly as before
    assert trace["outcome"] == "text_verdict"
    (reason,) = _nudge_reasons(trace)
    assert "call at least 2 reader tools" in reason


def test_rejection_message_does_not_echo_the_markup_back(mod):
    """The rejection is fed to the model; jsonschema embeds the offending value,
    so an untidied message would re-inject the syntax we are correcting.
    """
    dirty = {"verdict": "neutral",
             "reasoning": "prose " * 40 + '</reasoning><parameter name="x">'}
    violations = mod._verdict_violations(dirty, _SCHEMA)
    msg = "Rejected: " + "; ".join(violations)

    assert not mod._VERDICT_MARKUP.search(msg), "markup must not survive into the reply"
    assert len(msg) < 400, f"rejection ballooned to {len(msg)} chars"
    assert "too long" in msg and "tool-call markup" in msg
