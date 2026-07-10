"""Tests for collapse_to_source — the assembly-boundary mRNA collapse.

Keeps all Annotated rows + each resolved site's source-transcript row; drops the
non-source candidate rows and the rows of non-resolved (no_support / unresolved)
sites. No-op when the verdict columns are absent.
"""

from __future__ import annotations

import pandas as pd

from swissisoform.sourceresolve.collapse import collapse_to_source


def _bare_df() -> pd.DataFrame:
    # Verdict is broadcast per site: every row of a site carries that site's
    # source_transcript. The source row's own Tid equals source_transcript.
    return pd.DataFrame(
        [
            # Annotated rows — always kept.
            {"Tid": "T1", "TisType": "Annotated", "resolved": True, "source_transcript": "T1"},
            {"Tid": "T2", "TisType": "Annotated", "resolved": True, "source_transcript": "T2"},
            # Resolved alt site: T2 is the source, T3 is the dropped candidate.
            {"Tid": "T2", "TisType": "Extension", "resolved": True, "source_transcript": "T2"},
            {"Tid": "T3", "TisType": "Extension", "resolved": True, "source_transcript": "T2"},
            # no_support site: dropped.
            {"Tid": "T4", "TisType": "Truncation", "resolved": False, "source_transcript": None},
            # unresolved (divergent-fail) site: both candidates dropped.
            {"Tid": "T5", "TisType": "uORF", "resolved": False, "source_transcript": None},
            {"Tid": "T6", "TisType": "uORF", "resolved": False, "source_transcript": None},
        ]
    )


class TestCollapseBare:
    def test_keeps_annotated_and_source_only(self):
        out = collapse_to_source(_bare_df())
        kept = sorted(zip(out["Tid"], out["TisType"]))
        assert kept == [
            ("T1", "Annotated"),
            ("T2", "Annotated"),
            ("T2", "Extension"),
        ]

    def test_drops_non_source_candidate(self):
        out = collapse_to_source(_bare_df())
        ext = out[out["TisType"] == "Extension"]
        assert list(ext["Tid"]) == ["T2"]  # T3 dropped

    def test_drops_no_support_and_unresolved(self):
        out = collapse_to_source(_bare_df())
        assert not {"T4", "T5", "T6"} & set(out["Tid"])


class TestCollapsePerSample:
    def test_per_sample_columns(self):
        df = pd.DataFrame(
            [
                {"Tid": "T1", "TisType": "Annotated",
                 "HeLa_resolved": True, "HeLa_source_transcript": "T1"},
                {"Tid": "T2", "TisType": "Extension",
                 "HeLa_resolved": True, "HeLa_source_transcript": "T2"},
                {"Tid": "T3", "TisType": "Extension",
                 "HeLa_resolved": True, "HeLa_source_transcript": "T2"},
                {"Tid": "T4", "TisType": "uORF",
                 "HeLa_resolved": False, "HeLa_source_transcript": None},
            ]
        )
        out = collapse_to_source(df)
        assert sorted(out["Tid"]) == ["T1", "T2"]

    def test_unevaluated_rows_pass_through(self):
        # A TIS called only in a sample without long-read data has NaN verdicts —
        # collapse must keep it, not drop it as a non-source candidate.
        df = pd.DataFrame(
            [
                {"Tid": "T1", "TisType": "Annotated",
                 "HeLa_resolved": True, "HeLa_source_transcript": "T1"},
                {"Tid": "T2", "TisType": "Extension",
                 "HeLa_resolved": True, "HeLa_source_transcript": "T2"},
                {"Tid": "T3", "TisType": "Extension",
                 "HeLa_resolved": True, "HeLa_source_transcript": "T2"},
                {"Tid": "T7", "TisType": "Truncation",  # K562-only: no HeLa verdict
                 "HeLa_resolved": None, "HeLa_source_transcript": None},
            ]
        )
        out = collapse_to_source(df)
        assert sorted(out["Tid"]) == ["T1", "T2", "T7"]

    def test_keep_unevaluated_false_drops_unsupported(self):
        # keep_unevaluated=False (--drop-unsupported-tis): the un-evaluated,
        # non-HeLa-supported T7 is dropped; only Annotated + resolved source survive.
        df = pd.DataFrame(
            [
                {"Tid": "T1", "TisType": "Annotated",
                 "HeLa_resolved": True, "HeLa_source_transcript": "T1"},
                {"Tid": "T2", "TisType": "Extension",
                 "HeLa_resolved": True, "HeLa_source_transcript": "T2"},
                {"Tid": "T3", "TisType": "Extension",
                 "HeLa_resolved": True, "HeLa_source_transcript": "T2"},
                {"Tid": "T7", "TisType": "Truncation",  # K562-only: no HeLa verdict
                 "HeLa_resolved": None, "HeLa_source_transcript": None},
            ]
        )
        out = collapse_to_source(df, keep_unevaluated=False)
        assert sorted(out["Tid"]) == ["T1", "T2"]

    def test_string_bool_round_trip(self):
        # CSV round-trips bools to "True"/"False" strings — must still collapse.
        df = pd.DataFrame(
            [
                {"Tid": "T1", "TisType": "Annotated",
                 "resolved": "True", "source_transcript": "T1"},
                {"Tid": "T2", "TisType": "Extension",
                 "resolved": "True", "source_transcript": "T2"},
                {"Tid": "T3", "TisType": "Extension",
                 "resolved": "False", "source_transcript": None},
            ]
        )
        out = collapse_to_source(df)
        assert sorted(out["Tid"]) == ["T1", "T2"]


class TestCollapseNoOp:
    def test_no_verdict_columns_unchanged(self):
        df = pd.DataFrame({"Tid": ["T1", "T2"], "TisType": ["Annotated", "Extension"]})
        out = collapse_to_source(df)
        pd.testing.assert_frame_equal(out, df)

    def test_missing_tistype_is_noop(self):
        df = pd.DataFrame({"Tid": ["T1"], "resolved": [True], "source_transcript": ["T1"]})
        out = collapse_to_source(df)
        pd.testing.assert_frame_equal(out, df)
