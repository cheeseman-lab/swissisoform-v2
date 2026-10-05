"""The cheeseman50 coreset builder, on synthetic embeddings.

Offline: every MFA result here is built in memory, so nothing reads a parquet and
no selection is re-run on real data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figures" / "clustering_dims" / "feature_space"))
sys.path.insert(0, str(ROOT / "figures" / "clustering_dims" / "principled_sampling"))

import build_coreset as bc  # noqa: E402
import featurespace as fs  # noqa: E402

RARE_PER_TYPE = 5
PAIRED_PER_TYPE = 30


def _result(meta: pd.DataFrame, name: str, seed: int) -> fs.MFAResult:
    rng = np.random.default_rng(seed)
    n = len(meta)
    scores = rng.normal(size=(n, 10))
    matrix = fs.FeatureMatrix(
        X=scores.copy(),
        features=[f"f{i}" for i in range(10)],
        blocks=np.array(list("CDLMPSCDLM")),
        meta=meta.reset_index(drop=True),
        observed=np.ones_like(scores, dtype=bool),
        name=name,
    )
    return fs.MFAResult(
        scores=scores,
        loadings=np.eye(10),
        eigenvalues=np.ones(10),
        block_weights={},
        blocks=matrix.blocks,
        matrix=matrix,
    )


def _pool(anchors: pd.DataFrame, extra_anchor_gene_isoform: bool = True) -> pd.DataFrame:
    rows = anchors[["gene_name", "tis_id", "orf_type"]].to_dict("records")
    if extra_anchor_gene_isoform:
        # An anchor gene's *other* isoform: by-gene anchoring would have swept it in.
        rows.append({"gene_name": "CBX1", "tis_id": "chr17:1:-:CTG:ENSTX", "orf_type": "extended"})
    for orf_type in bc.RARE_TYPES:
        for i in range(RARE_PER_TYPE):
            rows.append(
                {"gene_name": f"{orf_type}_g{i}", "tis_id": f"{orf_type}:{i}", "orf_type": orf_type}
            )
    for orf_type in bc.PAIRED_STRATA:
        for i in range(PAIRED_PER_TYPE):
            rows.append(
                {"gene_name": f"{orf_type}_g{i}", "tis_id": f"{orf_type}:{i}", "orf_type": orf_type}
            )
    return pd.DataFrame(rows)


def _results(pool: pd.DataFrame) -> dict[str, fs.MFAResult]:
    out = {"all-ORF": _result(pool, "all-ORF", 0)}
    for k, orf_type in enumerate(bc.PAIRED_STRATA, start=1):
        sub = pool[pool["orf_type"] == orf_type]
        out[f"paired-{orf_type}"] = _result(sub, f"paired-{orf_type}", k)
    return out


@pytest.fixture
def anchors() -> pd.DataFrame:
    return bc.load_anchors()


def test_anchor_file_is_the_22_isoforms_of_12_genes(anchors):
    assert len(anchors) == 22 and anchors["tis_id"].is_unique
    assert anchors["gene_name"].nunique() == 12
    assert fs.ANCHOR_GENES == frozenset(anchors["gene_name"])


def test_build_lands_on_50_with_per_type_strata(anchors):
    results = _results(_pool(anchors))
    picks, final = bc.build_coreset(results, anchors)
    assert len(final) == bc.TARGET
    assert bc.verify(picks, final, anchors, results) == []
    sampled = picks[picks["stage"] == "sampler"]
    assert sampled["orf_type"].value_counts().to_dict() == {"extended": 8, "truncated": 8}
    assert set(sampled["space"]) == {"paired-extended", "paired-truncated"}
    assert (sampled["space"] == "paired-" + sampled["orf_type"]).all()
    assert set(picks.loc[picks["stage"] == "rare_fill", "space"]) == {"all-ORF"}


def test_anchors_are_pinned_by_tis_id_not_gene(anchors):
    """An anchor gene's other isoform is neither an anchor nor pickable."""
    results = _results(_pool(anchors))
    picks, final = bc.build_coreset(results, anchors)
    assert set(final.loc[final["source"] == "anchor", "tis_id"]) == set(anchors["tis_id"])
    assert "chr17:1:-:CTG:ENSTX" not in set(final["tis_id"])
    assert not set(picks["gene_name"]) & set(anchors["gene_name"])


def test_a_missing_anchor_stops_the_build(anchors):
    pool = _pool(anchors)
    pool = pool[pool["tis_id"] != anchors["tis_id"].iloc[0]]
    with pytest.raises(SystemExit, match="anchor"):
        bc.build_coreset(_results(pool), anchors)


def test_stratum_sizes_split_evenly():
    assert bc.stratum_sizes(16) == {"extended": 8, "truncated": 8}
    assert bc.stratum_sizes(17) == {"extended": 9, "truncated": 8}


def test_failed_verify_writes_nothing(anchors, monkeypatch, tmp_path):
    pool = _pool(anchors)
    monkeypatch.setattr(bc, "HERE", tmp_path)
    monkeypatch.setattr(bc, "resolve_parquet", lambda p: [])
    monkeypatch.setattr(
        bc.fs, "build_matrices", lambda p: {k: r.matrix for k, r in _results(pool).items()}
    )
    fits = _results(pool)
    monkeypatch.setattr(bc.fs, "fit_mfa", lambda m: fits[m.name])
    monkeypatch.setattr(bc.fs, "summarize", lambda r: "")
    monkeypatch.setattr(bc, "verify", lambda *a: ["rule broken"])
    with pytest.raises(SystemExit, match="nothing written"):
        bc.main([])
    assert list(tmp_path.iterdir()) == []
