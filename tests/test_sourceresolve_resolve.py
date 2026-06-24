"""Tests for the per-sample source-transcript resolution orchestrator.

Covers the three-way union decision (``_union``) directly — tier assignment,
source precedence (long_read > short_read > sequence), and the TIE quotient —
plus an end-to-end ``resolve_sources`` pass over a synthetic per-sample table
asserting the verdict columns and the tag-only invariant.
"""

from __future__ import annotations

import pandas as pd

from swissisoform.models import TranscriptCoordinates
from swissisoform.sourceresolve.resolve import ArmResult, _union, resolve_sources
from tests.test_sourceresolve_mrna import FakeGenome


class TestUnion:
    """The three-way union decision logic, isolated from sequence/IO."""

    def test_sequence_only_tier1_picks_max_tpm_candidate(self):
        res = _union(
            seq_pure=True,
            windowed_tids=["A", "B"],
            salmon_arm=None,
            iso_arm=None,
            salmon_abundance={"A": 3.0, "B": 9.0},
            norm_tis=18.0,
        )
        assert res.resolved is True
        assert res.agreement_tier == 1
        assert res.source_transcript == "B"  # higher salmon TPM
        assert res.source_evidence == "sequence"
        assert res.tie_initiation_efficiency == 2.0  # 18 / 9 (src TPM)

    def test_long_read_tier2(self):
        res = _union(
            seq_pure=False,
            windowed_tids=["A", "B"],
            salmon_arm=None,
            iso_arm=ArmResult(keep=True, source_tid="C", abundance_sum=20.0),
            salmon_abundance={},
            norm_tis=10.0,
        )
        assert res.agreement_tier == 2
        assert res.source_transcript == "C"
        assert res.source_evidence == "long_read"

    def test_short_read_tier3(self):
        res = _union(
            seq_pure=False,
            windowed_tids=["A", "B"],
            salmon_arm=ArmResult(keep=True, source_tid="D", abundance_sum=15.0),
            iso_arm=None,
            salmon_abundance={"D": 5.0},
            norm_tis=30.0,
        )
        assert res.agreement_tier == 3
        assert res.source_transcript == "D"
        assert res.source_evidence == "short_read"
        assert res.tie_initiation_efficiency == 2.0  # 30 / 15 (abundance_sum)

    def test_unresolved_is_tagged_not_dropped(self):
        res = _union(
            seq_pure=False,
            windowed_tids=["A", "B"],
            salmon_arm=ArmResult(keep=False, source_tid=None, abundance_sum=None),
            iso_arm=None,
            salmon_abundance={},
            norm_tis=10.0,
        )
        assert res.resolved is False
        assert res.agreement_tier == 0
        assert res.source_transcript is None
        assert res.source_evidence == "unresolved"
        assert res.tie_initiation_efficiency is None

    def test_long_read_precedence_over_short_read(self):
        res = _union(
            seq_pure=False,
            windowed_tids=["A", "B"],
            salmon_arm=ArmResult(keep=True, source_tid="D", abundance_sum=15.0),
            iso_arm=ArmResult(keep=True, source_tid="C", abundance_sum=20.0),
            salmon_abundance={"D": 5.0},
            norm_tis=10.0,
        )
        assert res.agreement_tier == 2  # long-read present
        assert res.source_transcript == "C"  # long-read wins the source
        assert res.source_evidence == "long_read"

    def test_seq_pure_keeps_tier1_but_long_read_picks_source(self):
        # S sets tier=1; B still wins the source-transcript precedence.
        res = _union(
            seq_pure=True,
            windowed_tids=["A", "B"],
            salmon_arm=None,
            iso_arm=ArmResult(keep=True, source_tid="C", abundance_sum=20.0),
            salmon_abundance={"A": 1.0, "B": 2.0},
            norm_tis=10.0,
        )
        assert res.agreement_tier == 1
        assert res.source_transcript == "C"
        assert res.source_evidence == "long_read"


def _coords(tid, strand, exons, chrom="chr1"):
    return TranscriptCoordinates(transcript_id=tid, chrom=chrom, strand=strand, exons=exons)


def _genome():
    # 5'UTR region A = "A"*100, region G = "G"*100, region T = "T"*100,
    # then three CDS blocks each opening with ATG.
    seq = (
        "A" * 100  # [0,100)
        + "G" * 100  # [100,200)
        + "T" * 100  # [200,300)
        + "ATG" + "C" * 57  # [300,360)  CDS-pure
        + "ATG" + "C" * 57  # [360,420)  CDS-expr
        + "ATG" + "C" * 57  # [420,480)  CDS-unres
    )
    return FakeGenome({"chr1": seq})


def _skeletons():
    return {
        # pure site (gstart 300): two transcripts, identical local sequence
        # (both 5'UTR region A) — differ only further 5', outside the window.
        "T_p1": _coords("T_p1", "+", [(0, 100), (300, 360)]),
        "T_p2": _coords("T_p2", "+", [(50, 100), (300, 360)]),
        # expression-resolved site (gstart 360): divergent 5'UTRs (A vs G).
        "T_e1": _coords("T_e1", "+", [(0, 100), (360, 420)]),
        "T_e2": _coords("T_e2", "+", [(100, 200), (360, 420)]),
        # unresolved site (gstart 420): divergent 5'UTRs, neither expressed.
        "T_u1": _coords("T_u1", "+", [(0, 100), (420, 480)]),
        "T_u2": _coords("T_u2", "+", [(100, 200), (420, 480)]),
    }


def _filtered_df():
    rows = [
        ("T_p1", 300), ("T_p2", 300),
        ("T_e1", 360), ("T_e2", 360),
        ("T_u1", 420), ("T_u2", 420),
    ]
    return pd.DataFrame(
        {
            "Tid": [t for t, _ in rows],
            "GenomePos": [f"chr1:{g}-{g + 60}:+" for _, g in rows],
            "StartCodon": ["ATG"] * len(rows),
            "NormTISCounts": [100.0] * len(rows),
        }
    )


def _write_quant(path, rows):
    pd.DataFrame(
        {
            "Name": [r[0] for r in rows],
            "Length": 1000,
            "EffectiveLength": 900.0,
            "TPM": [r[1] for r in rows],
            "NumReads": [r[1] * 10 for r in rows],
        }
    ).to_csv(path, sep="\t", index=False)


class TestResolveSources:
    def _run(self, tmp_path):
        a, b = tmp_path / "rep1.sf", tmp_path / "rep2.sf"
        # Only T_e1 is expressed (>=0.1 TPM in BOTH reps); T_e2/T_u* absent.
        _write_quant(a, [("T_e1", 50.0), ("T_e2", 0.0)])
        _write_quant(b, [("T_e1", 40.0), ("T_e2", 0.0)])
        out = resolve_sources(
            _filtered_df(),
            exon_skeletons=_skeletons(),
            genome=_genome(),
            salmon_quant=[a, b],
            isoquant_table=None,
            window=10,
            salmon_min_tpm=0.1,
        )
        return out.set_index("Tid")

    def test_tag_only_keeps_every_row(self, tmp_path):
        out = resolve_sources(
            _filtered_df(),
            exon_skeletons=_skeletons(),
            genome=_genome(),
            salmon_quant=None,
            isoquant_table=None,
            window=10,
        )
        assert len(out) == 6  # nothing dropped

    def test_pure_site_is_tier1_sequence(self, tmp_path):
        out = self._run(tmp_path)
        assert out.loc["T_p1", "agreement_tier"] == 1
        assert bool(out.loc["T_p1", "resolved"]) is True
        assert out.loc["T_p1", "source_evidence"] == "sequence"
        # both rows of the site carry the same verdict (broadcast by init_site)
        assert out.loc["T_p2", "agreement_tier"] == 1

    def test_expression_resolves_ambiguous_site_to_tier3(self, tmp_path):
        out = self._run(tmp_path)
        assert out.loc["T_e1", "agreement_tier"] == 3
        assert out.loc["T_e1", "source_transcript"] == "T_e1"
        assert out.loc["T_e1", "source_evidence"] == "short_read"
        assert out.loc["T_e1", "tie_initiation_efficiency"] == 100.0 / 45.0  # mean TPM 45

    def test_unexpressed_ambiguous_site_is_unresolved(self, tmp_path):
        out = self._run(tmp_path)
        assert bool(out.loc["T_u1", "resolved"]) is False
        assert out.loc["T_u1", "agreement_tier"] == 0
        assert pd.isna(out.loc["T_u1", "source_transcript"])  # None → NaN in object column
        assert out.loc["T_u1", "source_evidence"] == "unresolved"
