"""Tests for the long-read (IsoQuant) expression loader + thresholding."""

from __future__ import annotations

import pandas as pd
import pytest

from swissisoform.sourceresolve.expression import (
    _bare_tid,
    expressed_transcripts,
    load_isoquant_abundance,
)


def test_bare_tid():
    assert _bare_tid("ENST00000123.4|ENSG1|OTT|-|GENE-201|GENE|1000|") == "ENST00000123.4"
    assert _bare_tid("ENST00000123.4") == "ENST00000123.4"


def test_expressed_threshold():
    abundance = {"A": 1.0, "B": 0.5, "C": 3.0}
    assert expressed_transcripts(abundance, min_value=1.0) == {"A", "C"}


def test_load_isoquant_single_value_column(tmp_path):
    p = tmp_path / "iso.tsv"
    pd.DataFrame({"feature_id": ["ENST1.1", "ENST2.1", "ENST3.1"], "count": [5, 0, 3]}).to_csv(
        p, sep="\t", index=False
    )
    abundance = load_isoquant_abundance(p)
    assert abundance == {"ENST1.1": 5.0, "ENST2.1": 0.0, "ENST3.1": 3.0}
    # long-read presence floor of 3 counts
    assert expressed_transcripts(abundance, min_value=3.0) == {"ENST1.1", "ENST3.1"}


def test_load_isoquant_bares_feature_ids(tmp_path):
    p = tmp_path / "iso.tsv"
    pd.DataFrame(
        {"feature_id": ["ENST1.1|ENSG1|x", "ENST2.1|ENSG2|y"], "count": [4, 7]}
    ).to_csv(p, sep="\t", index=False)
    assert load_isoquant_abundance(p) == {"ENST1.1": 4.0, "ENST2.1": 7.0}


def test_load_isoquant_sums_replicate_columns(tmp_path):
    p = tmp_path / "iso.tsv"
    # No value_col given → sum across all non-id (replicate) columns, not just the first.
    pd.DataFrame(
        {"feature_id": ["ENST1.1", "ENST2.1"], "rep1": [2, 0], "rep2": [3, 5]}
    ).to_csv(p, sep="\t", index=False)
    assert load_isoquant_abundance(p) == {"ENST1.1": 5.0, "ENST2.1": 5.0}


def test_load_isoquant_explicit_value_column(tmp_path):
    p = tmp_path / "iso.tsv"
    pd.DataFrame(
        {"feature_id": ["ENST1.1", "ENST2.1"], "rep1": [2, 0], "rep2": [3, 5]}
    ).to_csv(p, sep="\t", index=False)
    assert load_isoquant_abundance(p, value_col="rep2") == {"ENST1.1": 3.0, "ENST2.1": 5.0}


def test_load_isoquant_no_value_columns_raises(tmp_path):
    p = tmp_path / "iso.tsv"
    pd.DataFrame({"feature_id": ["ENST1.1"]}).to_csv(p, sep="\t", index=False)
    with pytest.raises(ValueError, match="no abundance columns"):
        load_isoquant_abundance(p)
