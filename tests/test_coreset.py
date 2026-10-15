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


def test_a_rare_isoform_of_an_anchor_gene_is_never_picked(anchors):
    """By tis_id the anchor gene's *other* rare isoform stays in the pool.

    Placed at the far end of PC1 it is exactly what rare_type_fill picks first,
    and since that gene is already claimed the build used to abort.
    """
    pool = _pool(anchors)
    extra = {"gene_name": "CBX1", "tis_id": "chr17:9:-:CTG:ENSTUORF", "orf_type": "uorf"}
    pool = pd.concat([pool, pd.DataFrame([extra])], ignore_index=True)
    results = _results(pool)
    row = int(np.flatnonzero(pool["tis_id"] == extra["tis_id"])[0])
    results["all-ORF"].scores[row, 0] = 1e6  # the uORF stratum's PC1 maximum

    picks, final = bc.build_coreset(results, anchors)

    assert extra["tis_id"] not in set(picks["tis_id"])
    # No pick lands on an anchor gene, and no two picks share a gene.
    assert not set(picks["gene_name"]) & set(anchors["gene_name"])
    assert picks["gene_name"].is_unique


def test_two_extreme_isoforms_of_one_gene_take_one_rare_slot(anchors):
    """Two uoorfs of one gene, one at each end of PC1 (RPS6KA5 on full_catalog_20261009).

    Y takes the first; Z's first choice is the same gene, so it must fall through
    to the next-best gene (``rank_used`` > 0) instead of aborting the build.
    """
    pool = _pool(anchors)
    twins = [
        {"gene_name": "RPS6KA5", "tis_id": f"chr14:{i}:-:CTG:ENSTRPS", "orf_type": "uoorf"}
        for i in (1, 2)
    ]
    pool = pd.concat([pool, pd.DataFrame(twins)], ignore_index=True)
    results = _results(pool)
    for i, twin in enumerate(twins):
        row = int(np.flatnonzero(pool["tis_id"] == twin["tis_id"])[0])
        results["all-ORF"].scores[row, 0] = 1e6 if i == 0 else -1e6  # Y's and Z's first choice

    picks, final = bc.build_coreset(results, anchors)

    rare = picks[(picks["stage"] == "rare_fill") & (picks["orf_type"] == "uoorf")]
    assert list(rare["gene_name"]).count("RPS6KA5") == 1
    assert int(rare.loc[rare["set"] == "Z", "rank_used"].iloc[0]) == 1
    assert picks["gene_name"].is_unique
    assert len(final) == bc.TARGET
    assert bc.verify(picks, final, anchors, results) == []


@pytest.mark.parametrize("orf_type", ["uorf", "truncated"])
def test_a_chry_isoform_is_never_picked(anchors, orf_type):
    """A PAR gene on chrY, at the far end of PC1 in its own space, stays out."""
    pool = _pool(anchors)
    extra = {"gene_name": "GTPBP6", "tis_id": "chrY:316965:-:ACG:ENSTY", "orf_type": orf_type}
    pool = pd.concat([pool, pd.DataFrame([extra])], ignore_index=True)
    results = _results(pool)
    space = "all-ORF" if orf_type in bc.RARE_TYPES else f"paired-{orf_type}"
    meta = results[space].matrix.meta
    results[space].scores[int(np.flatnonzero(meta["tis_id"] == extra["tis_id"])[0]), 0] = 1e6

    picks, final = bc.build_coreset(results, anchors)

    assert extra["tis_id"] not in set(final["tis_id"])
    assert len(final) == bc.TARGET
    assert bc.verify(picks, final, anchors, results) == []


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


@pytest.mark.parametrize(
    "name", ["isoform_tags_citations.s3_sae", "isoform_scoring_criteria.C1_primate_conservation"]
)
def test_tag_and_scoring_outputs_are_not_feature_dimensions(name):
    """They are computed from the features; as dimensions they double-count."""
    import export_feature_catalog as efc

    include, reason, _ = efc.classify(name, "float", 1.0, 500)
    assert (include, reason) == (False, "derived_from_features")


def test_a_map_column_is_its_keys_not_a_feature_of_its_own(tmp_path):
    """read_flat types each map key; the map itself must not linger as a phantom feature."""
    import export_feature_catalog as efc
    import pyarrow as pa
    import pyarrow.parquet as pq

    from swissisoform.io.parquet import CLINICAL_SUMMARY_TYPE

    summary = {"total_variants": 1, "by_source": [("ClinVar", 1)],
               "by_consequence": [], "pathogenic_count": 0}
    path = tmp_path / "all_paired.parquet"
    pq.write_table(pa.table({
        "tis_id": ["a"],
        "canonical_clinical_summary": pa.array([summary], type=CLINICAL_SUMMARY_TYPE),
    }), path)

    df, types = efc.read_flat([path])

    assert "canonical_clinical_summary.by_source" not in types
    assert "canonical_clinical_summary.by_consequence" not in types
    assert pa.types.is_integer(types["canonical_clinical_summary.by_source.ClinVar"])
    assert set(types) - {"tis_id"} <= set(df.columns) | {"canonical_clinical_summary"}


def test_catalog_reader_does_not_trust_the_first_files_schema(tmp_path):
    """A first shard without the clinical summary must not crash the type map."""
    import export_feature_catalog as efc
    import pyarrow as pa
    import pyarrow.parquet as pq

    from swissisoform.io.parquet import CLINICAL_SUMMARY_TYPE

    first, second = tmp_path / "a.parquet", tmp_path / "b.parquet"
    pq.write_table(pa.table({"tis_id": ["a"]}), first)
    summary = {"total_variants": 1, "by_source": [("ClinVar", 1)],
               "by_consequence": [], "pathogenic_count": 0}
    pq.write_table(pa.table({
        "tis_id": ["b"],
        "canonical_clinical_summary": pa.array([summary], type=CLINICAL_SUMMARY_TYPE),
    }), second)

    df, types = efc.read_flat([first, second])

    assert pa.types.is_integer(types["canonical_clinical_summary.by_source.ClinVar"])
    assert list(df["canonical_clinical_summary.by_source.ClinVar"].fillna(-1)) == [-1, 1]
