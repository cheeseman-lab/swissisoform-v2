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
            assert iso["s"] == tis_slug(card.tis_id)  # the page URL, as the cards linked
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
    gz = {"Accept-Encoding": "gzip"}
    r = client.get("/api/landing.json", headers=gz)
    assert r.status_code == 200
    assert r.headers["Content-Encoding"] == "gzip"
    assert json.loads(gzip.decompress(r.data)) == json.loads(idx.body)
    etag = r.headers["ETag"]
    again = client.get("/api/landing.json", headers={**gz, "If-None-Match": etag})
    assert again.status_code == 304
    plain = client.get("/api/landing.json")
    assert "Content-Encoding" not in plain.headers
    assert json.loads(plain.data)["n_genes"] == idx.n_genes
    # Each representation has its own ETag, so the gzip one never validates plain.
    assert plain.headers["ETag"] != etag
    stale = client.get("/api/landing.json", headers={"If-None-Match": etag})
    assert stale.status_code == 200 and "Content-Encoding" not in stale.headers


def test_landing_route_honours_a_refused_gzip(client):
    """``gzip;q=0`` means "not gzip" — a substring test would have sent it anyway."""
    r = client.get("/api/landing.json", headers={"Accept-Encoding": "gzip;q=0, identity"})
    assert r.status_code == 200
    assert "Content-Encoding" not in r.headers
    assert json.loads(r.data)["n_genes"]


def test_null_gene_names_are_dropped_like_load_all(staged, tmp_path):
    """A row with no gene name is skipped, not listed as a gene called "None"."""
    import pandas as pd

    df = pd.read_parquet(WEBSITE_DATA / "all_paired.parquet", columns=list(staged.COLUMNS))
    orphan = df.iloc[[0]].copy()
    orphan["gene_name"] = None
    orphan["tis_id"] = "orphan"
    pd.concat([df, orphan]).to_parquet(tmp_path / "all_paired.parquet")
    payload = json.loads(staged.build_landing_index(tmp_path).body)
    names = {g["n"] for g in payload["genes"]}
    assert not names & {"None", "nan", ""}
    assert payload["n_isoforms"] == len(df)


def test_a_fired_registry_tag_keeps_its_category_even_if_not_code_fired(staged):
    registry = {"tags": [{"tag_id": "llm_x", "category": "M", "kind": "bool", "code_fired": False}]}
    vocab = staged._tag_vocab(registry, {"llm_x"})
    assert [(t["id"], t["category"]) for t in vocab] == [("llm_x", "M")]
    assert staged._tag_vocab(registry, set()) == []  # never fired: not listed


def test_index_logs_llm_coverage_without_load_all(staged):
    idx = staged.landing_index()
    assert 0 <= idx.n_categories <= idx.n_isoforms
    assert 0 <= idx.n_synthesis <= idx.n_isoforms


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


def test_registry_export_refuses_a_different_build(tmp_path):
    """Same version name, other definitions: no export rather than wrong categories."""
    import subprocess

    import pandas as pd

    if not (WEBSITE_DATA / "all_paired.parquet").exists():
        pytest.skip("website/data/all_paired.parquet not present — populate per README")
    cols = ["isoform_tags_registry_version", "isoform_tags_registry_sha256"]
    df = pd.read_parquet(WEBSITE_DATA / "all_paired.parquet", columns=cols)
    script = REPO_ROOT / "scripts" / "export" / "export_site_tag_registry.py"

    def export(frame):
        frame.to_parquet(tmp_path / "run.parquet")
        out = tmp_path / "tag_registry.json"
        out.unlink(missing_ok=True)
        r = subprocess.run(
            [
                sys.executable,
                str(script),
                "--parquet",
                str(tmp_path / "run.parquet"),
                "--out",
                str(out),
            ],
            capture_output=True,
            text=True,
        )
        return r, out

    ok, out = export(df)
    if ok.returncode != 0:
        pytest.skip(f"registry not provisioned here: {ok.stderr.strip()[-200:]}")
    assert json.loads(out.read_text())["sha256"] == df[cols[1]].iloc[0]

    bad, out = export(df.assign(isoform_tags_registry_sha256="0" * 64))
    assert bad.returncode != 0 and "not the build" in bad.stderr
    assert not out.exists()


def test_a_weakened_etag_still_revalidates(client):
    """If-None-Match compares weakly, so a proxy-weakened tag still gets a 304."""
    etag = client.get("/api/landing.json", headers={"Accept-Encoding": "gzip"}).headers["ETag"]
    r = client.get(
        "/api/landing.json", headers={"Accept-Encoding": "gzip", "If-None-Match": f"W/{etag}"}
    )
    assert r.status_code == 304


def test_gene_names_match_the_full_records(staged):
    from swissisoform_site.data import load_all

    assert staged.landing_index().gene_names == set(load_all())


def test_registry_export_warns_when_the_registry_is_not_provisioned(tmp_path):
    """No registry on this machine: no export, exit 0, so a deploy is not blocked."""
    import subprocess

    import pandas as pd

    pd.DataFrame({"isoform_tags_registry_version": ["v_not_provisioned"]}).to_parquet(
        tmp_path / "run.parquet"
    )
    out = tmp_path / "tag_registry.json"
    out.write_text("{}")  # a stale export from another run must not survive
    script = REPO_ROOT / "scripts" / "export" / "export_site_tag_registry.py"
    r = subprocess.run(
        [
            sys.executable,
            str(script),
            "--parquet",
            str(tmp_path / "run.parquet"),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "not provisioned" in r.stderr
    assert not out.exists()
