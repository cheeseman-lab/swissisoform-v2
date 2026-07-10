"""Tests for combine_filtered_samples (cross-sample wide pivot).

Regression guard for the pivot path: an earlier implementation used
``pivot_table(..., dropna=False)``, which builds ``MultiIndex.from_product`` over
the four DEDUP_KEY levels and exploded to trillions of cells on the full catalog.
The replacement (drop_duplicates + set_index + unstack) must keep the same wide
layout: one row per init-site, ``{sample}_{metric}`` columns, NaN where a TIS was
not called in a sample.
"""

from __future__ import annotations

import pandas as pd

from swissisoform.combine import combine_filtered_samples


def _sample_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal filtered-sample frame with DEDUP_KEY + SHARED_FIELDS."""
    base = []
    for r in rows:
        d = {
            "Symbol": r["Symbol"],
            "Gid": r.get("Gid", "G1"),
            "Tid": r["Tid"],
            "GenomePos": r["GenomePos"],
            "StartCodon": r["StartCodon"],
            "TisType": r["TisType"],
            "RecatTISType": r["TisType"],
            "AASeq": r["AASeq"],
            "AALen": len(r["AASeq"]),
            "Start": r.get("Start", 0),
            "TISCounts": r.get("TISCounts", 10),
            "NormTISCounts": r.get("NormTISCounts", 1.0),
        }
        d.update({k: v for k, v in r.items() if k not in d})
        base.append(d)
    return pd.DataFrame(base)


def test_wide_pivot_absent_sample_is_nan():
    # HeLa ran source resolution (has resolved/source_transcript); K562 did not.
    # T3 is K562-only; T2 is HeLa-only; T1 is shared.
    hela = _sample_df(
        [
            {"Symbol": "G", "Tid": "T1", "GenomePos": "chr1:100-200:+", "StartCodon": "ATG",
             "TisType": "Annotated", "AASeq": "MAAA", "resolved": True, "source_transcript": "T1"},
            {"Symbol": "G", "Tid": "T2", "GenomePos": "chr1:100-260:+", "StartCodon": "CTG",
             "TisType": "Extended", "AASeq": "MBBB", "resolved": True, "source_transcript": "T2"},
        ]
    )
    k562 = _sample_df(
        [
            {"Symbol": "G", "Tid": "T1", "GenomePos": "chr1:100-200:+", "StartCodon": "ATG",
             "TisType": "Annotated", "AASeq": "MAAA"},
            {"Symbol": "G", "Tid": "T3", "GenomePos": "chr1:100-300:+", "StartCodon": "GTG",
             "TisType": "Truncated", "AASeq": "MCCC"},
        ]
    )

    out = combine_filtered_samples({"HeLa": hela, "K562": k562})

    # One row per unique init-site (T1, T2, T3).
    assert len(out) == 3
    assert sorted(out["Tid"]) == ["T1", "T2", "T3"]

    # Wide {sample}_{metric} columns exist.
    assert "HeLa_TISCounts" in out.columns
    assert "K562_TISCounts" in out.columns
    assert "HeLa_resolved" in out.columns

    # Presence flags: T2 only in HeLa, T3 only in K562, T1 in both.
    row = out.set_index("Tid")
    assert bool(row.loc["T1", "present_HeLa"]) and bool(row.loc["T1", "present_K562"])
    assert bool(row.loc["T2", "present_HeLa"]) and not bool(row.loc["T2", "present_K562"])
    assert not bool(row.loc["T3", "present_HeLa"]) and bool(row.loc["T3", "present_K562"])

    # A K562-only TIS was never scored by HeLa → HeLa_resolved is NaN there.
    assert pd.isna(row.loc["T3", "HeLa_resolved"])
    # HeLa's own resolved verdict survives.
    assert bool(row.loc["T1", "HeLa_resolved"])

    # n_samples counts.
    assert int(row.loc["T1", "n_samples"]) == 2
    assert int(row.loc["T2", "n_samples"]) == 1
