"""Flag I: canonical-start expression is built per transcript and exported.

The Annotated (canonical) start's per-cell-line expression is attached to each
alt-TIS so the viewer can compare canonical-start vs alt-start initiation
efficiency (D1/D2).
"""

from __future__ import annotations

import pandas as pd

from swissisoform.assembly import _build_canonical_expression_by_tid


def _row(tid, tistype, **per_sample):
    base = {"Tid": tid, "TisType": tistype}
    base.update(per_sample)
    return base


def _gene_df():
    # Two cell lines; one Annotated (canonical) row + one alt row on the same Tid.
    # Annotated detected in HeLa only; alt detected in HeLa + K562.
    rows = [
        _row(
            "ENST1.1",
            "Annotated",
            present_HeLa=True,
            HeLa_TISCounts=200,
            HeLa_NormTISCounts=10.0,
            HeLa_FisherQvalue=1e-6,
            HeLa_GeneRNASeqCounts=1000,
            present_K562=False,
        ),
        _row(
            "ENST1.1",
            "5'UTR:Extension",
            present_HeLa=True,
            HeLa_TISCounts=50,
            HeLa_NormTISCounts=2.5,
            HeLa_FisherQvalue=1e-3,
            HeLa_GeneRNASeqCounts=1000,
            present_K562=True,
            K562_TISCounts=30,
            K562_NormTISCounts=1.5,
            K562_FisherQvalue=1e-2,
            K562_GeneRNASeqCounts=900,
        ),
    ]
    return pd.DataFrame(rows)


def test_canonical_expression_built_from_annotated_row():
    by_tid = _build_canonical_expression_by_tid(_gene_df(), ["HeLa", "K562"])
    assert "ENST1.1" in by_tid
    canon = by_tid["ENST1.1"]
    # Annotated detected in HeLa only.
    assert set(canon) == {"HeLa"}
    # initiation_efficiency = TISCounts / GeneRNASeqCounts = 200 / 1000.
    assert canon["HeLa"].initiation_efficiency == 0.2
    assert canon["HeLa"].raw_count == 200


def test_canonical_expression_empty_without_samples():
    # Single-sample mode has no combined sample columns.
    assert _build_canonical_expression_by_tid(_gene_df(), None) == {}


def test_canonical_expression_empty_when_no_annotated_row():
    df = _gene_df()
    df = df[df["TisType"] != "Annotated"]
    assert _build_canonical_expression_by_tid(df, ["HeLa", "K562"]) == {}
