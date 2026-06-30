"""Tests for the unified source-transcript resolution cascade.

Covers the divergent-case decision helper and the most-abundant tie-break in
isolation, plus an end-to-end ``resolve_sources`` pass over a synthetic
per-sample table: the long-read filter, window-purity branching, abundance
labeling, the threshold knobs, and the tag-only invariant.
"""

from __future__ import annotations

import pandas as pd

from swissisoform.models import TranscriptCoordinates
from swissisoform.sourceresolve.diagnostics import divergent_site_distribution
from swissisoform.sourceresolve.resolve import (
    _divergent_decision,
    _most_abundant,
    resolve_sources,
)
from tests.test_sourceresolve_mrna import FakeGenome


class TestMostAbundant:
    def test_picks_highest(self):
        assert _most_abundant(["A", "B"], {"A": 3.0, "B": 9.0}) == "B"

    def test_tie_breaks_on_smallest_tid(self):
        assert _most_abundant(["B", "A"], {"A": 5.0, "B": 5.0}) == "A"

    def test_empty(self):
        assert _most_abundant([], {}) is None


class TestDivergentDecision:
    def test_no_threshold_is_most_abundant(self):
        src = _divergent_decision(["A", "B"], {"A": 8.0, "B": 2.0}, dominance_frac=None)
        assert src == "A"

    def test_dominance_pass(self):
        src = _divergent_decision(["A", "B"], {"A": 8.0, "B": 2.0}, dominance_frac=0.75)
        assert src == "A"  # 8 / 10 = 0.8 >= 0.75

    def test_dominance_fail(self):
        src = _divergent_decision(["A", "B"], {"A": 8.0, "B": 2.0}, dominance_frac=0.9)
        assert src is None  # 0.8 < 0.9

    def test_dominance_exactly_at_threshold_passes(self):
        # 5 / 10 = 0.5 >= 0.5 → resolved (default threshold behavior).
        src = _divergent_decision(["A", "B"], {"A": 5.0, "B": 5.0}, dominance_frac=0.5)
        assert src == "A"  # tie-break to smallest tid


def _coords(tid, strand, exons, chrom="chr1"):
    return TranscriptCoordinates(transcript_id=tid, chrom=chrom, strand=strand, exons=exons)


def _genome():
    # region A "A"*100, region G "G"*100, filler "T"*100, then 5 CDS blocks
    # each opening with ATG at 300/360/420/480/540.
    seq = "A" * 100 + "G" * 100 + "T" * 100 + ("ATG" + "C" * 57) * 5
    return FakeGenome({"chr1": seq})


def _skeletons():
    return {
        # pure site (gstart 300): identical 5'UTR (region A) within the window.
        "T_p1": _coords("T_p1", "+", [(0, 100), (300, 360)]),
        "T_p2": _coords("T_p2", "+", [(50, 100), (300, 360)]),
        # divergent site (gstart 360): 5'UTR region A vs region G.
        "T_d1": _coords("T_d1", "+", [(0, 100), (360, 420)]),
        "T_d2": _coords("T_d2", "+", [(100, 200), (360, 420)]),
        # single-candidate site (gstart 420).
        "T_one": _coords("T_one", "+", [(0, 100), (420, 480)]),
        # long-read filter collapses a divergent pair to one survivor (gstart 480).
        "T_m1": _coords("T_m1", "+", [(0, 100), (480, 540)]),
        "T_m2": _coords("T_m2", "+", [(100, 200), (480, 540)]),
        # all-filtered-out site (gstart 540): neither candidate is expressed.
        "T_x1": _coords("T_x1", "+", [(0, 100), (540, 600)]),
        "T_x2": _coords("T_x2", "+", [(100, 200), (540, 600)]),
    }


def _filtered_df():
    rows = [
        ("T_p1", 300), ("T_p2", 300),
        ("T_d1", 360), ("T_d2", 360),
        ("T_one", 420),
        ("T_m1", 480), ("T_m2", 480),
        ("T_x1", 540), ("T_x2", 540),
    ]
    return pd.DataFrame(
        {
            "Tid": [t for t, _ in rows],
            "GenomePos": [f"chr1:{g}-{g + 60}:+" for _, g in rows],
            "StartCodon": ["ATG"] * len(rows),
            "NormTISCounts": [100.0] * len(rows),
        }
    )


def _write_iso(path, counts: dict[str, float]):
    pd.DataFrame(
        {"feature_id": list(counts), "count": [counts[t] for t in counts]}
    ).to_csv(path, sep="\t", index=False)
    return path


# Long-read counts: T_x* absent (0) so they fail the min_count=3 filter; T_m2
# absent so the divergent (480) pair collapses to the single survivor T_m1.
_ISO = {
    "T_p1": 30.0, "T_p2": 10.0,
    "T_d1": 32.0, "T_d2": 8.0,  # both present (>=3); top fraction 0.8
    "T_one": 5.0,
    "T_m1": 12.0, "T_m2": 0.0,
    "T_x1": 0.0, "T_x2": 0.0,
}


class TestResolveSources:
    def _run(self, tmp_path, **kwargs):
        iso = _write_iso(tmp_path / "iso.tsv", _ISO)
        out = resolve_sources(
            _filtered_df(),
            exon_skeletons=_skeletons(),
            genome=_genome(),
            isoquant_table=iso,
            window_upstream=10,
            window_downstream=10,
            isoquant_min_count=3.0,
            **kwargs,
        )
        return out.set_index("Tid")

    def test_tag_only_keeps_every_row(self, tmp_path):
        out = self._run(tmp_path)
        assert len(out) == 9  # nothing dropped

    def test_single_candidate_site(self, tmp_path):
        out = self._run(tmp_path)
        assert out.loc["T_one", "window_status"] == "single"
        assert bool(out.loc["T_one", "resolved"]) is True
        assert out.loc["T_one", "source_transcript"] == "T_one"
        assert out.loc["T_one", "source_evidence"] == "window_pure"
        assert out.loc["T_one", "tie_initiation_efficiency"] == 100.0 / 5.0

    def test_pure_site_picks_most_abundant(self, tmp_path):
        out = self._run(tmp_path)
        assert out.loc["T_p1", "window_status"] == "pure"
        assert bool(out.loc["T_p1", "resolved"]) is True
        assert out.loc["T_p1", "source_transcript"] == "T_p1"  # 30 > 10
        assert out.loc["T_p1", "source_evidence"] == "window_pure"
        # both rows of the site share the broadcast verdict
        assert out.loc["T_p2", "source_transcript"] == "T_p1"

    def test_divergent_default_threshold_resolves(self, tmp_path):
        # Default divergence_dominance_frac=0.5; top fraction 0.8 >= 0.5.
        out = self._run(tmp_path)
        assert out.loc["T_d1", "window_status"] == "divergent"
        assert bool(out.loc["T_d1", "resolved"]) is True
        assert out.loc["T_d1", "source_transcript"] == "T_d1"  # 32 > 8
        assert out.loc["T_d1", "source_evidence"] == "divergent_pass"

    def test_divergent_no_threshold_most_abundant(self, tmp_path):
        out = self._run(tmp_path, divergence_dominance_frac=None)
        assert bool(out.loc["T_d1", "resolved"]) is True
        assert out.loc["T_d1", "source_transcript"] == "T_d1"
        assert out.loc["T_d1", "source_evidence"] == "divergent_pass"

    def test_divergent_dominance_pass(self, tmp_path):
        out = self._run(tmp_path, divergence_dominance_frac=0.75)
        assert bool(out.loc["T_d1", "resolved"]) is True
        assert out.loc["T_d1", "source_transcript"] == "T_d1"  # 0.8 >= 0.75

    def test_divergent_dominance_fail_is_unresolved(self, tmp_path):
        out = self._run(tmp_path, divergence_dominance_frac=0.9)
        assert out.loc["T_d1", "window_status"] == "divergent"
        assert bool(out.loc["T_d1", "resolved"]) is False
        assert pd.isna(out.loc["T_d1", "source_transcript"])
        assert out.loc["T_d1", "source_evidence"] == "unresolved"

    def test_long_read_filter_collapses_divergent_to_single(self, tmp_path):
        out = self._run(tmp_path)
        # T_m2 unexpressed → only T_m1 survives → single, not divergent.
        assert out.loc["T_m1", "window_status"] == "single"
        assert out.loc["T_m1", "source_transcript"] == "T_m1"

    def test_all_filtered_out_is_no_support(self, tmp_path):
        out = self._run(tmp_path)
        # No long-read-supported candidate → no_support, distinct from unresolved.
        assert out.loc["T_x1", "window_status"] == "no_support"
        assert bool(out.loc["T_x1", "resolved"]) is False
        assert pd.isna(out.loc["T_x1", "source_transcript"])
        assert out.loc["T_x1", "source_evidence"] == "no_support"


class TestDivergentSiteDistribution:
    def test_records_only_divergent_sites(self, tmp_path):
        iso = _write_iso(tmp_path / "iso.tsv", _ISO)
        dist = divergent_site_distribution(
            _filtered_df(),
            exon_skeletons=_skeletons(),
            genome=_genome(),
            isoquant_table=iso,
            window_upstream=10,
            window_downstream=10,
            isoquant_min_count=3.0,
        )
        # only the gstart-360 pair (T_d1/T_d2) diverges with both present
        assert len(dist) == 1
        row = dist.iloc[0]
        assert row["transcripts"] == ["T_d1", "T_d2"]  # ranked by abundance
        assert row["abundances"] == [32.0, 8.0]
        assert row["fractions"] == [0.8, 0.2]
        assert row["top_fraction"] == 0.8
        assert row["top_to_runnerup_ratio"] == 4.0
