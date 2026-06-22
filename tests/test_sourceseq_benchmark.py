"""Tests for the Arm A vs Arm B disambiguation benchmark."""

from __future__ import annotations

import pandas as pd

from swissisoform.sourceseq.benchmark import (
    arm_disambiguation,
    compare_arms,
    per_tis_candidates,
    present_set_agreement,
)


def test_per_tis_candidates_groups_by_init_site():
    df = pd.DataFrame(
        {
            "Tid": ["T1", "T2", "T3", "T4"],
            # T1,T2 share the same + start codon (lo); T3 is minus (hi); T4 not present
            "GenomePos": ["chr1:100-200:+", "chr1:100-250:+", "chr2:400-500:-", "chr1:100-200:+"],
            "StartCodon": ["ATG", "ATG", "CTG", "ATG"],
            "present_HeLa": [True, True, True, False],
        }
    )
    cand = per_tis_candidates(df, "HeLa")
    assert cand == {"chr1:100:+:ATG": {"T1", "T2"}, "chr2:500:-:CTG": {"T3"}}


def test_arm_disambiguation_buckets():
    cand = {
        "a": {"t1", "t2"},  # both present -> ambiguous
        "b": {"t3", "t4"},  # one present  -> resolved
        "c": {"t5", "t6", "t7"},  # none present -> lost
        "d": {"t8"},  # single candidate -> not scored (min 2)
    }
    present = {"t1", "t2", "t3"}
    r = arm_disambiguation(cand, present)
    assert r["multi_candidate_tis"] == 3
    assert (r["lost_0"], r["resolved_1"], r["ambiguous_2plus"]) == (1, 1, 1)
    assert r["mean_survivors"] == (2 + 1 + 0) / 3


def test_present_set_agreement():
    r = present_set_agreement({1, 2, 3}, {2, 3, 4})
    assert r["overlap"] == 2
    assert r["A_only"] == 1 and r["B_only"] == 1
    assert r["precision_A_vs_B"] == 2 / 3
    assert r["recall_A_vs_B"] == 2 / 3
    assert r["jaccard"] == 0.5


def test_compare_arms_shape():
    cand = {"a": {"t1", "t2"}, "b": {"t3", "t4"}}
    out = compare_arms(cand, {"t1"}, {"t3", "t4"})
    assert set(out) == {"agreement", "salmon", "isoquant"}
    assert out["salmon"]["resolved_1"] == 1  # a -> {t1}
    assert out["isoquant"]["ambiguous_2plus"] == 1  # b -> {t3,t4}
