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
                "_raw": {},
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
    }
    rec = rli._build_synthesis_record(iso, "GENE_A", base)
    assert rec["isoform"]["tis_id"] == "chr1:100:+:ATG:ENST_A"
    assert "Conservation" in rec["category_reads"]
    assert "Mutation Landscape" in rec["category_reads"]
    assert rec["category_reads"]["Conservation"]["verdict"] == "interesting"
    # All 15 criteria (incl. S2/S3) are carried in the raw evidence for synthesis —
    # S2/S3 tolerate the _raw-less fixture (empty evidence, never raise).
    assert "P2_shared_structural_change" in rec["criteria_evidence"]
    assert "S2_biophysics" in rec["criteria_evidence"]
    assert "S3_sae" in rec["criteria_evidence"]
