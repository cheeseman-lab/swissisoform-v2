"""Tests for the conservation annotation module."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from swissisoform.config import ConservationConfig, PipelineConfig
from swissisoform.modules.conservation import (
    CONSERVATION_LABELS,
    ConservationModule,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config_no_conservation() -> PipelineConfig:
    """PipelineConfig with no conservation config."""
    return PipelineConfig(conservation=None)


@pytest.fixture()
def config_with_db(tmp_path: Path) -> PipelineConfig:
    """PipelineConfig with a fake diamond db path."""
    db_path = tmp_path / "test.dmnd"
    db_path.touch()
    return PipelineConfig(
        conservation=ConservationConfig(diamond_db=db_path),
    )


# ---------------------------------------------------------------------------
# TestConservationModule
# ---------------------------------------------------------------------------


class TestConservationModule:
    """Tests for the ConservationModule."""

    def test_annotate_no_config(self, config_no_conservation):
        """No conservation config -> empty hits, score None (couldn't run),
        status=not_run. Distinguishes from "ran and found nothing" (score 0).
        """
        mod = ConservationModule(config_no_conservation)
        result = mod.annotate("MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPS")
        assert result["hits"] == []
        # score is None because we COULDN'T run, not because we found nothing
        assert result["summary"]["conservation_score"] is None
        assert result["summary"]["tool_used"] is None
        assert result["summary"]["status"] == "not_run"

    def test_annotate_precomputed(self, config_no_conservation):
        """Precomputed results are returned directly."""
        precomputed_result = {
            "hits": [{"subject_id": "sp|P04637", "pident": 99.0}],
            "summary": {
                "conservation_score": 9,
                "conservation_label": "Near-perfect conservation",
                "best_pident": 99.0,
                "best_evalue": 1e-100,
                "n_hits": 1,
                "tool_used": "diamond",
            },
        }
        protein = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPS"
        mod = ConservationModule(
            config_no_conservation,
            precomputed={protein: precomputed_result},
        )
        result = mod.annotate(protein)
        assert result is precomputed_result
        assert result["summary"]["conservation_score"] == 9

    def test_score_pident(self):
        """Test the scoring scale at each boundary."""
        score = ConservationModule._score_pident
        assert score(100.0) == 9
        assert score(95.0) == 9
        assert score(94.9) == 8
        assert score(85.0) == 8
        assert score(84.9) == 7
        assert score(75.0) == 7
        assert score(74.9) == 6
        assert score(65.0) == 6
        assert score(64.9) == 5
        assert score(55.0) == 5
        assert score(54.9) == 3
        assert score(45.0) == 3
        assert score(44.9) == 2
        assert score(40.0) == 2
        assert score(39.9) == 1
        assert score(30.0) == 1
        assert score(29.9) == 0
        assert score(0.0) == 0

    def test_parse_blast_tabular(self, tmp_path):
        """Create a mock tabular output file, verify parsing."""
        output_file = str(tmp_path / "results.tsv")
        content = textwrap.dedent("""\
            query\tsp|P04637|P53_HUMAN\t95.5\t200\t9\t0\t1\t200\t1\t200\t1e-50\t350.0
            query\tsp|Q9NZC2|TREM2_HUMAN\t72.3\t150\t40\t2\t10\t159\t5\t154\t1e-20\t180.5
        """)
        with open(output_file, "w") as fh:
            fh.write(content)

        hits = ConservationModule._parse_blast_tabular(output_file)
        assert len(hits) == 2

        # First hit
        assert hits[0]["subject_id"] == "sp|P04637|P53_HUMAN"
        assert hits[0]["pident"] == 95.5
        assert hits[0]["length"] == 200
        assert hits[0]["evalue"] == 1e-50
        assert hits[0]["bitscore"] == 350.0
        # qstart/qend are 0-indexed (original 1-based minus 1)
        assert hits[0]["qstart"] == 0
        assert hits[0]["qend"] == 199
        assert hits[0]["sstart"] == 1
        assert hits[0]["send"] == 200

        # Second hit
        assert hits[1]["subject_id"] == "sp|Q9NZC2|TREM2_HUMAN"
        assert hits[1]["pident"] == 72.3

    def test_parse_blast_tabular_missing_file(self, tmp_path):
        """Nonexistent file returns empty list."""
        hits = ConservationModule._parse_blast_tabular(
            str(tmp_path / "nonexistent.tsv"),
        )
        assert hits == []

    def test_parse_blast_tabular_empty_file(self, tmp_path):
        """Empty file returns empty list."""
        output_file = str(tmp_path / "empty.tsv")
        with open(output_file, "w") as fh:
            fh.write("")
        hits = ConservationModule._parse_blast_tabular(output_file)
        assert hits == []

    def test_empty_protein(self, config_no_conservation):
        """Empty string -> empty result."""
        mod = ConservationModule(config_no_conservation)
        result = mod.annotate("")
        assert result["hits"] == []
        assert result["summary"]["conservation_score"] == 0

    def test_star_only_protein(self, config_no_conservation):
        """Protein that is just '*' -> empty result."""
        mod = ConservationModule(config_no_conservation)
        result = mod.annotate("*")
        assert result["hits"] == []
        assert result["summary"]["conservation_score"] == 0

    def test_run_no_sites_lost(self, synthetic_tis, config):
        """Wrapper preserves all sites."""
        mod = ConservationModule(config)
        result = mod.run(synthetic_tis)
        assert len(result) == len(synthetic_tis)
        for site in result:
            assert "conservation" in site.isoform_annotations

    def test_module_scope(self):
        """SCOPE should be 'C' (per-candidate)."""
        assert ConservationModule.SCOPE == "C"

    def test_module_name(self):
        """MODULE_NAME should be 'conservation'."""
        assert ConservationModule.MODULE_NAME == "conservation"

    @patch(
        "swissisoform.modules.conservation.ConservationModule._detect_tool",
        return_value=None,
    )
    def test_detect_tool_none(self, mock_detect):
        """When no tool is available, score is None (couldn't run)."""
        cfg = PipelineConfig(
            conservation=ConservationConfig(diamond_db=Path("/fake/db.dmnd")),
        )
        mod = ConservationModule(cfg)
        assert mod._tool is None
        result = mod.annotate("MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPS")
        assert result["summary"]["tool_used"] is None
        # score is None because we COULDN'T run — not "ran, found nothing" (0)
        assert result["summary"]["conservation_score"] is None
        assert result["summary"]["status"] == "not_run"

    @patch(
        "swissisoform.modules.conservation.ConservationModule._detect_tool",
        return_value="diamond",
    )
    @patch(
        "swissisoform.modules.conservation.ConservationModule._run_search",
    )
    def test_annotate_with_mock_search(self, mock_search, mock_detect, tmp_path):
        """Mock the search to simulate hits and verify annotation."""
        db_path = tmp_path / "test.dmnd"
        db_path.touch()
        cfg = PipelineConfig(
            conservation=ConservationConfig(diamond_db=db_path),
        )

        mock_search.return_value = [
            {
                "subject_id": "sp|P04637|P53_HUMAN",
                "pident": 88.0,
                "length": 200,
                "evalue": 1e-40,
                "bitscore": 300.0,
                "qstart": 0,
                "qend": 199,
                "sstart": 1,
                "send": 200,
            },
            {
                "subject_id": "sp|Q9NZC2|TREM2_HUMAN",
                "pident": 55.0,
                "length": 100,
                "evalue": 1e-10,
                "bitscore": 120.0,
                "qstart": 0,
                "qend": 99,
                "sstart": 5,
                "send": 104,
            },
        ]

        mod = ConservationModule(cfg)
        result = mod.annotate("MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPS")

        assert len(result["hits"]) == 2
        assert result["summary"]["conservation_score"] == 8  # 88% -> score 8
        assert result["summary"]["conservation_label"] == "Very strong conservation"
        assert result["summary"]["best_pident"] == 88.0
        assert result["summary"]["best_evalue"] == 1e-40
        assert result["summary"]["n_hits"] == 2
        assert result["summary"]["tool_used"] == "diamond"

    def test_conservation_label(self):
        """Verify label text for each score level."""
        assert CONSERVATION_LABELS[0] == "No homology"
        assert CONSERVATION_LABELS[1] == "Weak homology"
        assert CONSERVATION_LABELS[2] == "Partial homology"
        assert CONSERVATION_LABELS[5] == "Moderate conservation"
        assert CONSERVATION_LABELS[7] == "Strong conservation"
        assert CONSERVATION_LABELS[8] == "Very strong conservation"
        assert CONSERVATION_LABELS[9] == "Near-perfect conservation"
        # All 0-9 scores have labels
        for i in range(10):
            assert i in CONSERVATION_LABELS

    def test_annotate_result_structure(self, config_no_conservation):
        """Verify the annotation dict has the expected top-level keys."""
        mod = ConservationModule(config_no_conservation)
        result = mod.annotate("MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPS")
        assert "hits" in result
        assert "summary" in result
        assert isinstance(result["hits"], list)
        assert isinstance(result["summary"], dict)
        summary = result["summary"]
        for key in [
            "conservation_score",
            "conservation_label",
            "best_pident",
            "best_evalue",
            "n_hits",
            "tool_used",
        ]:
            assert key in summary, f"Missing summary key: {key}"

    def test_run_writes_to_isoform_annotations(self, synthetic_tis, config):
        """run() writes to site.isoform_annotations['conservation']."""
        mod = ConservationModule(config)
        mod.run(synthetic_tis)
        for site in synthetic_tis:
            ann = site.isoform_annotations["conservation"]
            assert "hits" in ann
            assert "summary" in ann

    @patch(
        "swissisoform.modules.conservation.ConservationModule._detect_tool",
        return_value="diamond",
    )
    @patch(
        "swissisoform.modules.conservation.ConservationModule._run_search",
    )
    def test_run_with_precomputed(self, mock_search, mock_detect, synthetic_tis):
        """Precomputed results bypass search entirely."""
        protein = synthetic_tis[0].isoform_protein
        precomputed_result = {
            "hits": [{"subject_id": "sp|TEST", "pident": 50.0}],
            "summary": {
                "conservation_score": 3,
                "conservation_label": "Stop codon interrupting homology",
                "best_pident": 50.0,
                "best_evalue": 1e-5,
                "n_hits": 1,
                "tool_used": "precomputed",
            },
        }
        cfg = PipelineConfig(conservation=None)
        mod = ConservationModule(cfg, precomputed={protein: precomputed_result})
        result = mod.annotate(protein)
        assert result["summary"]["conservation_score"] == 3
        # _run_search should not have been called
        mock_search.assert_not_called()
