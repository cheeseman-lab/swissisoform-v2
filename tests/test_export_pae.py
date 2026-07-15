"""Tests for the per-protein PAE export (website folding panel).

No GPU — seed a temp fold cache with synthetic pae.npy arrays and exercise the
reshape + naming + missing-entry accounting in ``export.pae.export_pae``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from swissisoform.export import pae as pae_mod
from swissisoform.export.pae import export_pae
from swissisoform.structure.fold import cache_path

np = pytest.importorskip("numpy")


def _seed_pae(cache_dir, seq, n, backend="esmfold2"):
    from swissisoform.plm.embed import protein_hash

    base = cache_path(cache_dir, backend, protein_hash(seq.rstrip("*").upper()))
    base.mkdir(parents=True, exist_ok=True)
    arr = np.arange(n * n, dtype=np.float32).reshape(n, n) / 10.0
    np.save(base / "pae.npy", arr.astype(np.float16))
    return arr


def _gene(name, canonical, sites):
    return SimpleNamespace(gene_name=name, canonical_protein=canonical, tis_sites=sites)


def _site(iso, tis_id):
    return SimpleNamespace(isoform_protein=iso, tis_id=tis_id)


def test_export_pae_writes_named_json(tmp_path, monkeypatch):
    monkeypatch.setattr(pae_mod, "DEFAULT_CACHE_DIR", tmp_path / "cache")
    cache = tmp_path / "cache"
    can = "MSKGEELFTGVV"
    iso = "M" + can[3:]  # a truncation-ish isoform, distinct sequence
    _seed_pae(cache, can, len(can))
    iso_arr = _seed_pae(cache, iso, len(iso))

    gene = _gene("GFP", can, [_site(iso, "chr1:100:+:CTG:ENST00000000001.1")])
    outdir = tmp_path / "out"
    n_written, n_missing, pae_dir = export_pae([gene], outdir)

    assert n_written == 2 and n_missing == 0
    seg = "chr1-100---CTG-ENST00000000001.1"
    iso_json = pae_dir / f"GFP__isoform__{seg}.pae.json"
    can_json = pae_dir / f"GFP__canonical__{seg}.pae.json"
    assert iso_json.exists() and can_json.exists()

    payload = json.loads(iso_json.read_text())
    assert payload["L"] == len(iso)
    assert len(payload["pae"]) == len(iso) ** 2
    # Round-trips the row-major array (within fp16 + 1-dp rounding).
    got = np.array(payload["pae"], dtype=np.float32).reshape(payload["L"], payload["L"])
    assert np.allclose(got, iso_arr.astype(np.float32), atol=0.1)


def test_export_pae_counts_missing_when_no_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(pae_mod, "DEFAULT_CACHE_DIR", tmp_path / "cache")
    # No pae.npy seeded → both sides missing.
    gene = _gene("XYZ", "MAAAA", [_site("MBBBB", "chr2:200:-:ATG:ENST00000000002.2")])
    n_written, n_missing, _ = export_pae([gene], tmp_path / "out")
    assert n_written == 0 and n_missing == 2


def test_export_pae_dedupes_shared_canonical(tmp_path, monkeypatch):
    monkeypatch.setattr(pae_mod, "DEFAULT_CACHE_DIR", tmp_path / "cache")
    cache = tmp_path / "cache"
    can = "MSKGEELFTGVV"
    iso1, iso2 = "M" + can[2:], "M" + can[4:]
    for s in (can, iso1, iso2):
        _seed_pae(cache, s, len(s))
    # Two isoforms of one gene → two distinct segments → 2 iso + 2 canonical maps
    # (canonical duplicated per segment by design, but each filename unique).
    gene = _gene(
        "G",
        can,
        [
            _site(iso1, "chr1:10:+:CTG:ENST1.1"),
            _site(iso2, "chr1:20:+:CTG:ENST1.1"),
        ],
    )
    n_written, n_missing, _ = export_pae([gene], tmp_path / "out")
    assert n_written == 4 and n_missing == 0
