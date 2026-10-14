"""The landing table's payload (``/api/landing.json``) and its no-data-loss contract.

The gene cards it replaced read everything from ``load_all()``; the table reads a
column-projected index instead. These tests hold the two to the same answers for
every gene and isoform in the staged run, so the move from cards to table cannot
silently drop a field, and pin how the fired pipeline tags reach the page.
"""

from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("swissisoform_site.app")

REPO_ROOT = Path(__file__).resolve().parents[1]
WEBSITE_SRC = REPO_ROOT / "website" / "src"
WEBSITE_DATA = REPO_ROOT / "website" / "data"


@pytest.fixture(scope="module")
def staged():
    if not (WEBSITE_DATA / "all_paired.parquet").exists():
        pytest.skip("website/data/all_paired.parquet not present — populate per README")
    if str(WEBSITE_SRC) not in sys.path:
        sys.path.insert(0, str(WEBSITE_SRC))
    os.environ["SWISSISOFORM_DATA_DIR"] = str(WEBSITE_DATA)
    from swissisoform_site import data as data_mod
    from swissisoform_site import landing

    data_mod.load_all.cache_clear()
    landing.landing_index.cache_clear()
    return landing


@pytest.fixture(scope="module")
def payload(staged):
    return json.loads(staged.landing_index().body)


@pytest.fixture(scope="module")
def client(staged):
    from swissisoform_site.app import app

    app.testing = True
    return app.test_client()


def _decode_flags(payload, code: str) -> dict[str, tuple[str, int, int]]:
    out = {}
    for k, g in enumerate(payload["groups"]):
        c, n_met, n_ev = code[3 * k], code[3 * k + 1], code[3 * k + 2]
        out[g["letter"]] = (payload["flag_codes"][c], int(n_met), int(n_ev))
    return out


def test_payload_carries_everything_the_gene_cards_showed(staged, payload):
    """Field by field against ``load_all()`` — what the cards rendered from."""
    from swissisoform_site.data import load_all, tis_slug

    genes = load_all()
    by_name = {g["n"]: g for g in payload["genes"]}
    assert set(by_name) == set(genes)
    assert payload["n_genes"] == len(genes)
    assert payload["n_isoforms"] == sum(len(g.isoforms) for g in genes.values())

    for name, rec in genes.items():
        g = by_name[name]
        assert g["u"] == rec.uniprot_id
        assert [payload["keywords"][k] for k in g["k"]] == rec.keywords  # every keyword, in order
        assert len(g["i"]) == len(rec.isoforms)

        # The dropdown's display order: extensions, truncations, then the rest.
        order = (
            [i for i in rec.isoforms if i.orf_type == "extended"]
            + [i for i in rec.isoforms if i.orf_type == "truncated"]
            + [i for i in rec.isoforms if i.orf_type not in ("extended", "truncated")]
        )
        assert [i["t"] for i in g["i"]] == [i.tis_id for i in order]

        best = rec.best_isoform
        assert g["i"][g["b"]]["t"] == best.tis_id
        assert (g["ns"], g["ni"]) == (best.net_score, best.n_interesting)

        for iso, card in zip(g["i"], order, strict=True):
            assert tis_slug(iso["t"]) == tis_slug(card.tis_id)
            assert iso["o"] == card.orf_type
            assert iso["a"] == card.aa_len
            assert [payload["llm_tags"][k] for k in iso["y"]] == card.tags
            flags = _decode_flags(payload, iso["f"])
            for letter, f in card.category_flags.items():
                assert flags[letter] == (f["state"], f["n_met"], f["n_evaluable"])


def test_fired_tags_are_exactly_the_true_states(staged, payload):
    """A chip for every tag whose state is True — never a False or a null one."""
    import pandas as pd

    df = pd.read_parquet(
        WEBSITE_DATA / "all_paired.parquet",
        columns=["tis_id", "orf_type", "isoform_tags_states", "isoform_tags_labels"],
    )
    rows = {r.tis_id: r for r in df.itertuples()}
    ids = [t["id"] for t in payload["tags"]]
    n_checked = 0
    for g in payload["genes"]:
        for iso in g["i"]:
            r = rows[iso["t"]]
            states = r.isoform_tags_states or {}
            assert {ids[k] for k in iso["g"]} == {k for k, v in states.items() if v is True}
            # The label a chip shows is the row's own stratum-worded label.
            for k in iso["g"]:
                t = payload["tags"][k]
                shown = (iso.get("l") or {}).get(str(k)) or t.get("labels", {}).get(iso["o"])
                assert shown == r.isoform_tags_labels[t["id"]]
            n_checked += len(iso["g"])
    assert n_checked, "staged run fired no tags — the test exercised nothing"


def test_tags_carry_registry_categories(payload):
    """With the registry staged, every tag is categorised C–S, in C→S order."""
    if not payload["registry"]["staged"]:
        pytest.skip("no tag_registry.json staged (run prepare_deploy.sh)")
    letters = [g["letter"] for g in payload["groups"]]
    cats = [t["category"] for t in payload["tags"]]
    assert all(c in letters for c in cats)
    assert cats == sorted(cats, key=letters.index)
    assert {t["kind"] for t in payload["tags"]} <= {"derived", "threshold", "bool"}


def test_missing_registry_degrades_to_uncategorised(staged, tmp_path):
    """No staged registry: the chips still render, just without a category."""
    (tmp_path / "all_paired.parquet").symlink_to(WEBSITE_DATA / "all_paired.parquet")
    idx = staged.build_landing_index(tmp_path)
    payload = json.loads(idx.body)
    assert payload["registry"]["staged"] is False
    assert payload["tags"], "fired tags must not disappear without the registry"
    assert all(t["category"] is None for t in payload["tags"])


def test_landing_route_serves_gzip_with_etag(client, staged):
    idx = staged.landing_index()
    r = client.get("/api/landing.json", headers={"Accept-Encoding": "gzip"})
    assert r.status_code == 200
    assert r.headers["Content-Encoding"] == "gzip"
    assert json.loads(gzip.decompress(r.data)) == json.loads(idx.body)
    etag = r.headers["ETag"]
    again = client.get("/api/landing.json", headers={"If-None-Match": etag})
    assert again.status_code == 304
    plain = client.get("/api/landing.json")
    assert "Content-Encoding" not in plain.headers
    assert json.loads(plain.data)["n_genes"] == idx.n_genes


def test_index_is_a_table_shell_fed_by_the_landing_route(client, staged):
    """The page carries the table and the data URL, not one card per gene."""
    body = client.get("/").data.decode()
    assert 'id="gene-table"' in body
    assert "/api/landing.json" in body
    assert "js/landing_table.js" in body
    assert 'id="orf-toggles"' in body  # ORF type filter, filled from the payload's types
    assert "gene-card" not in body
    idx = staged.landing_index()
    assert f'<span class="stat-num">{idx.n_isoforms}</span>' in body
