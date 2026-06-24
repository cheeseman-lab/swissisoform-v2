"""Tests for the transcript-existence (expression) loaders + thresholding."""

from __future__ import annotations

import pandas as pd

from swissisoform.sourceresolve.expression import (
    _bare_tid,
    expressed_in_replicates,
    expressed_transcripts,
    load_isoquant_abundance,
    load_salmon_replicates,
    load_salmon_tpm,
)


def _write_quant(path, rows: list[tuple[str, float]]) -> None:
    """Write a minimal salmon quant.sf (Name, Length, EffectiveLength, TPM, NumReads)."""
    df = pd.DataFrame(
        {
            "Name": [r[0] for r in rows],
            "Length": 1000,
            "EffectiveLength": 900.0,
            "TPM": [r[1] for r in rows],
            "NumReads": [r[1] * 10 for r in rows],
        }
    )
    df.to_csv(path, sep="\t", index=False)


def test_bare_tid():
    assert _bare_tid("ENST00000123.4|ENSG1|OTT|-|GENE-201|GENE|1000|") == "ENST00000123.4"
    assert _bare_tid("ENST00000123.4") == "ENST00000123.4"


def test_load_salmon_tpm(tmp_path):
    p = tmp_path / "quant.sf"
    _write_quant(p, [("ENST1.1|ENSG1|x", 5.0), ("ENST2.1|ENSG2|y", 0.0)])
    assert load_salmon_tpm(p) == {"ENST1.1": 5.0, "ENST2.1": 0.0}


def test_replicate_averaging(tmp_path):
    a, b = tmp_path / "a.sf", tmp_path / "b.sf"
    _write_quant(a, [("ENST1.1|x", 10.0), ("ENST2.1|y", 0.0)])
    _write_quant(b, [("ENST1.1|x", 0.0), ("ENST3.1|z", 4.0)])
    d = load_salmon_replicates([a, b])
    assert d["ENST1.1"] == 5.0  # (10 + 0) / 2
    assert d["ENST2.1"] == 0.0  # present only in rep a, value 0
    assert d["ENST3.1"] == 2.0  # (0 + 4) / 2, absent from rep a


def test_expressed_threshold_short_read():
    abundance = {"A": 1.0, "B": 0.5, "C": 3.0}
    assert expressed_transcripts(abundance, min_value=1.0) == {"A", "C"}


def test_load_isoquant_and_threshold(tmp_path):
    p = tmp_path / "iso.tsv"
    pd.DataFrame({"feature_id": ["ENST1.1", "ENST2.1", "ENST3.1"], "count": [5, 0, 3]}).to_csv(
        p, sep="\t", index=False
    )
    abundance = load_isoquant_abundance(p)
    assert abundance == {"ENST1.1": 5.0, "ENST2.1": 0.0, "ENST3.1": 3.0}
    # long-read presence floor of 3 counts
    assert expressed_transcripts(abundance, min_value=3.0) == {"ENST1.1", "ENST3.1"}


def test_expressed_in_replicates(tmp_path):
    a, b = tmp_path / "a.sf", tmp_path / "b.sf"
    _write_quant(a, [("ENST1.1|x", 5.0), ("ENST2.1|y", 0.5), ("ENST3.1|z", 2.0)])
    _write_quant(b, [("ENST1.1|x", 3.0), ("ENST2.1|y", 4.0), ("ENST3.1|z", 0.2)])
    # stringent rule: present iff TPM >= 1 in BOTH reps
    #   ENST1 (5, 3) ok ; ENST2 (0.5, 4) fails rep a ; ENST3 (2, 0.2) fails rep b
    assert expressed_in_replicates([a, b], min_tpm=1.0, require="all") == {"ENST1.1"}
    # lenient rule: present in at least one rep
    assert expressed_in_replicates([a, b], min_tpm=1.0, require="any") == {
        "ENST1.1",
        "ENST2.1",
        "ENST3.1",
    }
