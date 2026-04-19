"""Tests for TIS filtering pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from swissisoform.filtering import (
    filter_tis,
    identify_reference_transcripts,
    normalize_tis_counts,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tis_df(rows: list[dict]) -> pd.DataFrame:
    """Build a TIS DataFrame with sensible defaults for missing columns."""
    defaults = {
        "Gid": "ENSG00000000001.1",
        "Tid": "ENST00000000001.1",
        "Symbol": "GENE1",
        "GeneType": "protein_coding",
        "GenomePos": "chr1:100-200:+",
        "StartCodon": "ATG",
        "Start": 10,
        "Stop": 100,
        "TisType": "Annotated",
        "TISGroup": 1,
        "TISCounts": 50,
        "TISPvalue": 0.001,
        "RiboPvalue": 0.001,
        "RiboPStatus": "T",
        "FisherPvalue": 0.0005,
        "TISQvalue": 0.01,
        "FrameQvalue": 0.01,
        "FisherQvalue": 0.005,
        "AALen": 30,
        "Seq": "ATGCCC",
        "AASeq": "MP",
        "Chromosome": "chr1",
        "Strand": "+",
        "Locus": 100,
        "MANE_Select": True,
        "transcript_support_level": "1",
    }
    full_rows = []
    for row in rows:
        full = {**defaults, **row}
        full_rows.append(full)
    return pd.DataFrame(full_rows)


# ---------------------------------------------------------------------------
# normalize_tis_counts
# ---------------------------------------------------------------------------


class TestNormalizeTisCounts:
    """Tests for normalize_tis_counts."""

    def test_rpm_with_explicit_total(self):
        df = _make_tis_df([{"TISCounts": 100}, {"TISCounts": 200}])
        result = normalize_tis_counts(df, total_reads=1000)
        assert result["NormTISCounts"].iloc[0] == pytest.approx(100_000.0)
        assert result["NormTISCounts"].iloc[1] == pytest.approx(200_000.0)

    def test_rpm_auto_total_deduplicates(self):
        """When total_reads is None, use deduplicated sum of TISCounts."""
        df = _make_tis_df([
            {"TISCounts": 100, "Chromosome": "chr1", "Locus": 100, "Strand": "+",
             "Tid": "ENST00000000001.1"},
            # Same genomic position, different transcript -- should be counted once
            {"TISCounts": 100, "Chromosome": "chr1", "Locus": 100, "Strand": "+",
             "Tid": "ENST00000000002.1"},
            # Different genomic position
            {"TISCounts": 200, "Chromosome": "chr1", "Locus": 200, "Strand": "+",
             "Tid": "ENST00000000003.1"},
        ])
        result = normalize_tis_counts(df)
        # Deduplicated total = 100 + 200 = 300
        expected_first = (100 / 300) * 1e6
        assert result["NormTISCounts"].iloc[0] == pytest.approx(expected_first)

    def test_does_not_mutate_input(self):
        df = _make_tis_df([{"TISCounts": 100}])
        original_cols = set(df.columns)
        normalize_tis_counts(df, total_reads=1000)
        assert set(df.columns) == original_cols


# ---------------------------------------------------------------------------
# identify_reference_transcripts
# ---------------------------------------------------------------------------


class TestIdentifyReferenceTranscripts:
    """Tests for identify_reference_transcripts."""

    def test_returns_tids(self):
        df = _make_tis_df([
            {"Tid": "ENST1", "MANE_Select": True, "transcript_support_level": "1"},
        ])
        result = identify_reference_transcripts(df)
        assert isinstance(result, list)
        assert "ENST1" in result

    def test_filters_by_tsl(self):
        df = _make_tis_df([
            {"Tid": "ENST1", "MANE_Select": False, "transcript_support_level": "1"},
            {"Tid": "ENST2", "MANE_Select": False, "transcript_support_level": "5"},
        ])
        result = identify_reference_transcripts(df, transcript_support_levels=["1", "2", "3"])
        assert "ENST1" in result
        assert "ENST2" not in result

    def test_respects_mane_select(self):
        """MANE_Select transcripts should be kept regardless of TSL."""
        df = _make_tis_df([
            {"Tid": "ENST1", "MANE_Select": True, "transcript_support_level": "5"},
        ])
        result = identify_reference_transcripts(df, transcript_support_levels=["1"])
        assert "ENST1" in result

    def test_filters_by_significance(self):
        df = _make_tis_df([
            {"Tid": "ENST1", "MANE_Select": True, "TISPvalue": 0.5},
        ])
        result = identify_reference_transcripts(
            df, tis_enrichment_max_p=0.01
        )
        assert "ENST1" not in result

    def test_filters_by_min_counts(self):
        df = _make_tis_df([
            {"Tid": "ENST1", "MANE_Select": True, "TISCounts": 5},
        ])
        result = identify_reference_transcripts(df, min_tis_counts=10)
        assert "ENST1" not in result


# ---------------------------------------------------------------------------
# filter_tis
# ---------------------------------------------------------------------------


class TestFilterTis:
    """Tests for filter_tis."""

    def _make_standard_df(self) -> pd.DataFrame:
        """Build a DataFrame with a mix of passing and failing TIS."""
        return _make_tis_df([
            # Passes all filters
            {
                "Tid": "ENST1", "MANE_Select": True,
                "TISCounts": 50, "NormTISCounts": 1.0,
                "TISPvalue": 0.001, "RiboPvalue": 0.001, "FisherQvalue": 0.01,
                "TisType": "Annotated", "Start": 10, "GenomePos": "chr1:100-200:+",
            },
            # Low normalized counts
            {
                "Tid": "ENST1", "MANE_Select": True,
                "TISCounts": 1, "NormTISCounts": 0.01,
                "TISPvalue": 0.001, "RiboPvalue": 0.001, "FisherQvalue": 0.01,
                "TisType": "Extended", "Start": 50, "GenomePos": "chr1:150-250:+",
            },
            # Not significant
            {
                "Tid": "ENST2", "MANE_Select": True,
                "TISCounts": 50, "NormTISCounts": 1.0,
                "TISPvalue": 0.5, "RiboPvalue": 0.001, "FisherQvalue": 0.01,
                "TisType": "Extended", "Start": 10, "GenomePos": "chr2:100-200:+",
            },
            # Not reference transcript
            {
                "Tid": "ENST3", "MANE_Select": False,
                "transcript_support_level": "5",
                "TISCounts": 50, "NormTISCounts": 1.0,
                "TISPvalue": 0.001, "RiboPvalue": 0.001, "FisherQvalue": 0.01,
                "TisType": "Extended", "Start": 10, "GenomePos": "chr3:100-200:+",
            },
        ])

    def test_returns_filtered_and_dropped(self):
        df = self._make_standard_df()
        filtered, dropped = filter_tis(df, return_dropped=True)
        assert len(filtered) + len(dropped) == len(df)

    def test_drops_low_counts(self):
        df = self._make_standard_df()
        _, dropped = filter_tis(df, return_dropped=True)
        low_count_dropped = dropped[dropped["DropReason"].str.contains("LowReadcounts", na=False)]
        assert len(low_count_dropped) >= 1

    def test_drops_non_reference(self):
        df = self._make_standard_df()
        _, dropped = filter_tis(df, return_dropped=True)
        non_ref = dropped[
            dropped["DropReason"].str.contains("NotReferenceTranscript", na=False)
        ]
        assert len(non_ref) >= 1

    def test_keeps_annotated_sites(self):
        """Annotated TIS should survive filtering when they pass basic thresholds."""
        df = _make_tis_df([
            {
                "Tid": "ENST1", "MANE_Select": True,
                "TISCounts": 50, "NormTISCounts": 1.0,
                "TISPvalue": 0.001, "RiboPvalue": 0.001, "FisherQvalue": 0.01,
                "TisType": "Annotated", "Start": 10, "GenomePos": "chr1:100-200:+",
            },
        ])
        filtered = filter_tis(df)
        assert len(filtered) == 1
        assert "Annotated" in filtered.iloc[0]["TisType"]

    def test_distance_deduplication(self):
        """Two TIS within 30nt on same transcript: only the higher-count one survives."""
        df = _make_tis_df([
            {
                "Tid": "ENST1", "MANE_Select": True,
                "TISCounts": 100, "NormTISCounts": 5.0,
                "TISPvalue": 0.001, "RiboPvalue": 0.001, "FisherQvalue": 0.01,
                "TisType": "Extended", "Start": 10, "GenomePos": "chr1:100-200:+",
            },
            {
                "Tid": "ENST1", "MANE_Select": True,
                "TISCounts": 50, "NormTISCounts": 2.0,
                "TISPvalue": 0.001, "RiboPvalue": 0.001, "FisherQvalue": 0.01,
                "TisType": "Truncated", "Start": 20,  # within 30nt of Start=10
                "GenomePos": "chr1:110-200:+",
            },
        ])
        filtered, dropped = filter_tis(df, return_dropped=True)
        assert len(filtered) == 1
        assert filtered.iloc[0]["NormTISCounts"] == pytest.approx(5.0)
        assert "UpstreamTIS" in dropped.iloc[0]["DropReason"]

    def test_distance_dedup_prefers_annotated(self):
        """Annotated TIS preferred over higher-count non-annotated TIS.

        When both are within the distance buffer, annotated wins.
        """
        df = _make_tis_df([
            {
                "Tid": "ENST1", "MANE_Select": True,
                "TISCounts": 50, "NormTISCounts": 2.0,
                "TISPvalue": 0.001, "RiboPvalue": 0.001, "FisherQvalue": 0.01,
                "TisType": "Annotated", "Start": 10, "GenomePos": "chr1:100-200:+",
            },
            {
                "Tid": "ENST1", "MANE_Select": True,
                "TISCounts": 200, "NormTISCounts": 10.0,
                "TISPvalue": 0.001, "RiboPvalue": 0.001, "FisherQvalue": 0.01,
                "TisType": "Extended", "Start": 15,  # within 30nt
                "GenomePos": "chr1:105-200:+",
            },
        ])
        filtered = filter_tis(df)
        # Annotated should be kept, higher-count Extended excluded
        assert len(filtered) == 1
        assert "Annotated" in filtered.iloc[0]["TisType"]

    def test_distance_dedup_keeps_distant_tis(self):
        """Two TIS more than 30nt apart should both survive."""
        df = _make_tis_df([
            {
                "Tid": "ENST1", "MANE_Select": True,
                "TISCounts": 100, "NormTISCounts": 5.0,
                "TISPvalue": 0.001, "RiboPvalue": 0.001, "FisherQvalue": 0.01,
                "TisType": "Extended", "Start": 10, "GenomePos": "chr1:100-200:+",
            },
            {
                "Tid": "ENST1", "MANE_Select": True,
                "TISCounts": 50, "NormTISCounts": 2.0,
                "TISPvalue": 0.001, "RiboPvalue": 0.001, "FisherQvalue": 0.01,
                "TisType": "Truncated", "Start": 100,  # > 30nt away
                "GenomePos": "chr1:190-290:+",
            },
        ])
        filtered = filter_tis(df)
        assert len(filtered) == 2

    def test_filter_without_return_dropped(self):
        df = self._make_standard_df()
        result = filter_tis(df, return_dropped=False)
        assert isinstance(result, pd.DataFrame)
        assert "DropReason" in result.columns


class TestExemptAnnotated:
    """``exempt_annotated`` flag: Annotated rows from reference transcripts
    bypass count and significance drops but must still be on a reference
    transcript.
    """

    def _base_row(self, **overrides) -> dict:
        row = {
            "Tid": "ENST_REF", "MANE_Select": True,
            "TISCounts": 100, "NormTISCounts": 5.0,
            "TISPvalue": 0.001, "RiboPvalue": 0.001, "FisherQvalue": 0.01,
            "TisType": "Annotated", "Start": 10,
            "GenomePos": "chr1:100-200:+",
        }
        row.update(overrides)
        return row

    def test_exempt_annotated_keeps_nonsig_annotated(self):
        """An Annotated row that fails significance survives when
        exempt_annotated=True."""
        df = _make_tis_df([
            self._base_row(
                # Fails all three significance filters
                TISPvalue=0.5, RiboPvalue=0.5, FisherQvalue=0.5,
                NormTISCounts=0.001,  # Also fails count filter
            ),
        ])
        filtered, dropped = filter_tis(df, exempt_annotated=True, return_dropped=True)
        assert len(filtered) == 1, (
            f"exempt_annotated=True should keep nonsig Annotated; got "
            f"{len(filtered)} kept, {len(dropped)} dropped"
        )
        assert filtered.iloc[0]["TisType"].startswith("Annotated")

    def test_non_exempt_drops_nonsig_annotated(self):
        """Same row drops when exempt_annotated=False (upstream-reference-faithful)."""
        df = _make_tis_df([
            self._base_row(
                TISPvalue=0.5, RiboPvalue=0.5, FisherQvalue=0.5,
                NormTISCounts=0.001,
            ),
        ])
        filtered, dropped = filter_tis(df, exempt_annotated=False, return_dropped=True)
        assert len(filtered) == 0
        assert len(dropped) == 1
        reason = str(dropped.iloc[0]["DropReason"])
        assert "NotSignificant" in reason or "LowReadcounts" in reason

    def test_exempt_still_requires_reference_transcript(self):
        """Annotated row from a NON-reference transcript still drops with
        NotReferenceTranscript even when exempt_annotated=True.  Exemption
        does not override the reference-transcript gate."""
        df = _make_tis_df([
            self._base_row(
                Tid="ENST_BAD",
                MANE_Select=False,
                transcript_support_level="5",
            ),
        ])
        filtered, dropped = filter_tis(df, exempt_annotated=True, return_dropped=True)
        assert len(filtered) == 0
        assert len(dropped) == 1
        assert "NotReferenceTranscript" in str(dropped.iloc[0]["DropReason"])

    def test_exempt_does_not_exempt_alt_tis(self):
        """Non-Annotated rows still get filtered normally when
        exempt_annotated=True."""
        df = _make_tis_df([
            self._base_row(
                TisType="Extended",
                TISPvalue=0.5, RiboPvalue=0.5, FisherQvalue=0.5,
            ),
        ])
        filtered, dropped = filter_tis(df, exempt_annotated=True, return_dropped=True)
        assert len(filtered) == 0
        assert len(dropped) == 1
