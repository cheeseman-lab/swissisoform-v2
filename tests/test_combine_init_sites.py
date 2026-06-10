"""Tests for ``dedupe_unique_init_sites`` — the genome-LM init-site skeleton."""
from __future__ import annotations

import numpy as np
import pandas as pd

from swissisoform.combine import dedupe_unique_init_sites


def _sample_cols(s: str, *, tis: float, norm: float, present: bool) -> dict:
    return {
        f"{s}_TISCounts": tis,
        f"{s}_NormTISCounts": norm,
        f"{s}_TISPvalue": 0.001 if present else np.nan,
        f"{s}_RiboPvalue": 0.002 if present else np.nan,
        f"{s}_FisherQvalue": 0.01 if present else np.nan,
        f"{s}_GeneRNASeqCounts": 1000.0 if present else np.nan,
        f"{s}_Imputed": False,
        f"present_{s}": present,
    }


def _combined() -> pd.DataFrame:
    """Two transcripts share a plus-strand start (distal splice); one minus-strand canonical."""
    rows = [
        {  # T1 at site A — HeLa + K562
            "Symbol": "GENEA", "Gid": "g1", "Tid": "T1",
            "GenomePos": "chr1:100-400:+", "StartCodon": "ATG",
            "TisType": "Extended", "RecatTISType": "Extended",
            "AASeq": "MAAAA*", "AALen": 5.0, "Start": 100.0, "n_samples": 2,
            **_sample_cols("HeLa", tis=10.0, norm=5.0, present=True),
            **_sample_cols("K562", tis=4.0, norm=2.0, present=True),
        },
        {  # T2 at site A — same start, different protein, HeLa only
            "Symbol": "GENEA", "Gid": "g1", "Tid": "T2",
            "GenomePos": "chr1:100-400:+", "StartCodon": "ATG",
            "TisType": "Extended", "RecatTISType": "Extended",
            "AASeq": "MBBBB*", "AALen": 5.0, "Start": 100.0, "n_samples": 1,
            **_sample_cols("HeLa", tis=6.0, norm=3.0, present=True),
            **_sample_cols("K562", tis=np.nan, norm=np.nan, present=False),
        },
        {  # Site B — minus-strand canonical
            "Symbol": "GENEB", "Gid": "g2", "Tid": "T3",
            "GenomePos": "chr2:200-500:-", "StartCodon": "CTG",
            "TisType": "Annotated", "RecatTISType": "Annotated",
            "AASeq": "MCCCC*", "AALen": 5.0, "Start": 500.0, "n_samples": 1,
            **_sample_cols("HeLa", tis=20.0, norm=9.0, present=True),
            **_sample_cols("K562", tis=np.nan, norm=np.nan, present=False),
        },
    ]
    return pd.DataFrame(rows)


def test_init_site_grouping_collapses_distal_splice():
    out = dedupe_unique_init_sites(_combined())
    assert len(out) == 2  # two transcripts at one start → one site

    a = out[out["init_site"] == "chr1:100:+:ATG"].iloc[0]
    assert a["n_transcripts"] == 2
    assert a["n_proteins"] == 2  # distinct AASeq → distinct protein_hash
    assert len(a["all_protein_hashes"].split(",")) == 2
    assert not a["is_canonical"]
    assert a["gstart"] == 100
    assert a["strand"] == "+"


def test_per_condition_labels_are_peak_usage():
    out = dedupe_unique_init_sites(_combined())
    a = out[out["init_site"] == "chr1:100:+:ATG"].iloc[0]
    assert a["max_norm_HeLa"] == 5.0  # max(5.0, 3.0)
    assert a["max_norm_K562"] == 2.0  # max(2.0, NaN)
    assert bool(a["present_HeLa"]) and bool(a["present_K562"])
    assert a["min_fisher_qvalue"] == 0.01


def test_minus_strand_canonical_anchor():
    out = dedupe_unique_init_sites(_combined())
    b = out[out["init_site"] == "chr2:500:-:CTG"].iloc[0]
    assert b["is_canonical"]
    assert b["gstart"] == 500  # 5' end on minus strand = hi coordinate
    assert b["strand"] == "-"


def test_empty_input():
    assert dedupe_unique_init_sites(pd.DataFrame()).empty
