"""Smoke tests for the SwissIsoform v2 Flask viewer.

Requires the cheeseman_12gene parquet to be visible at
``website/data/all_paired.parquet`` (either real file or a symlink — the
website README documents the layout). Tests skip cleanly if it isn't there.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WEBSITE_SRC = REPO_ROOT / "website" / "src"
WEBSITE_DATA = REPO_ROOT / "website" / "data"


@pytest.fixture(scope="module")
def client():
    """Spin up the Flask test client pointed at website/data."""
    if not (WEBSITE_DATA / "all_paired.parquet").exists():
        pytest.skip("website/data/all_paired.parquet not present — populate per README")

    # Make the website package importable without installing it.
    if str(WEBSITE_SRC) not in sys.path:
        sys.path.insert(0, str(WEBSITE_SRC))

    os.environ["SWISSISOFORM_DATA_DIR"] = str(WEBSITE_DATA)

    # Reset any cached load so the env var actually takes effect this run.
    from swissisoform_site import data as data_mod

    data_mod.load_all.cache_clear()
    data_mod._structure_index.cache_clear()

    from swissisoform_site.app import app

    app.testing = True
    return app.test_client()


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.get_json() == {"ok": True}


def test_index_lists_a_known_gene(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.data.decode()
    assert "TRNT1" in body


def test_gene_page_redirects_to_first_isoform(client):
    """V1 /genes/<gene> is deprecated and 302s to a V2 isoform page."""
    r = client.get("/genes/TRNT1")
    assert r.status_code == 302
    assert "/isoforms/" in r.headers["Location"]
    assert "/genes/TRNT1/isoforms/" in r.headers["Location"]


def test_gene_page_404(client):
    r = client.get("/genes/nonexistent_gene_zzz")
    assert r.status_code == 404


def test_index_page_shows_isoform_dropdown(client):
    """Landing page must render the per-gene isoform dropdown."""
    r = client.get("/")
    assert r.status_code == 200
    body = r.data
    assert b"TRNT1" in body
    # Dropdown markup and a link into the V2 isoform route.
    assert b"iso-dropdown" in body
    assert b"/isoforms/chr" in body


def test_api_data_json(client):
    r = client.get("/api/data.json")
    assert r.status_code == 200
    payload = json.loads(r.data)
    # The cheeseman_12gene parquet ships with 12 genes.
    assert len(payload) >= 12
    for gene in [
        "CBX1",
        "CDC34",
        "CDKN1A",
        "CSNK2A2",
        "EIF2B1",
        "FZR1",
        "MAD2L1",
        "SRSF2",
        "TRIP13",
        "TRNT1",
        "UBE2D2",
        "UBE2M",
    ]:
        assert gene in payload, f"missing gene in /api/data.json: {gene}"
        rec = payload[gene]
        assert "isoforms" in rec
        assert isinstance(rec["isoforms"], list)


def test_slugify_filter():
    """The slugify Jinja filter must scrub ``:`` and ``.`` from tis_ids."""
    if str(WEBSITE_SRC) not in sys.path:
        sys.path.insert(0, str(WEBSITE_SRC))
    from swissisoform_site.app import app

    with app.app_context():
        slug = app.jinja_env.filters["slugify"]("chr3:3129127:+:ATG:ENST00000434583.5")
    assert ":" not in slug
    assert "." not in slug
    assert slug  # non-empty


def test_transcript_skeleton_loaded_for_known_transcript():
    """load_transcript_skeletons reads the V2 parquet and returns dict-by-transcript_id."""
    from pathlib import Path

    if str(WEBSITE_SRC) not in sys.path:
        sys.path.insert(0, str(WEBSITE_SRC))
    from swissisoform_site.data import load_transcript_skeletons

    skel_path = Path("data/output/cheeseman_12gene/transcript_skeletons.parquet")
    if not skel_path.exists():
        pytest.skip("skeleton parquet not present")
    sk = load_transcript_skeletons(skel_path)
    assert len(sk) > 0
    sample = next(iter(sk.values()))
    assert sample.chrom.startswith("chr")
    assert sample.strand in ("+", "-")
    assert len(sample.exons) >= 1


def test_llm_criterion_for_isoform_returns_none_when_missing(tmp_path):
    """llm_criterion_for_isoform tolerates a missing JSON file."""
    if str(WEBSITE_SRC) not in sys.path:
        sys.path.insert(0, str(WEBSITE_SRC))
    from swissisoform_site.data import llm_criterion_for_isoform

    out = llm_criterion_for_isoform(
        llm_dir=tmp_path,
        tis_slug="chr1-100-ATG-ENST_A",
        criterion_id="E1_primate_conservation",
    )
    assert out is None


def test_synthesis_narrative_html_converts_markdown():
    """_markdown_to_html converts the tiny markdown subset, escape-first."""
    if str(WEBSITE_SRC) not in sys.path:
        sys.path.insert(0, str(WEBSITE_SRC))
    from swissisoform_site.data import _markdown_to_html

    out = _markdown_to_html("**Bold** and *em* and `code`.\n\npara two")
    assert "<strong>Bold</strong>" in out
    assert "<em>em</em>" in out
    assert "<code>code</code>" in out
    assert "<p>" in out
    assert "**" not in out


# --------------------------------------------------------------------------- #
# V2 isoform route (/genes/<gene>/isoforms/<tis_slug>)
# --------------------------------------------------------------------------- #


def test_isoform_route_returns_200_for_known_tis(client):
    """The V2 route renders for every (gene, tis_id) in the parquet."""
    import pandas as pd
    from swissisoform_site.data import tis_slug as make_slug

    df = pd.read_parquet(WEBSITE_DATA / "all_paired.parquet", columns=["gene_name", "tis_id"])
    row = df.iloc[0]
    r = client.get(f"/genes/{row['gene_name']}/isoforms/{make_slug(row['tis_id'])}")
    assert r.status_code == 200


def test_isoform_route_returns_404_for_unknown_tis(client):
    """An unknown tis_slug under a known gene 404s."""
    r = client.get("/genes/TRNT1/isoforms/not-a-real-slug")
    assert r.status_code == 404


def test_isoform_page_contains_graphs_and_synthesis_block(client):
    """The rendered V2 page exposes the Plotly graph divs and the Synthesis block."""
    import pandas as pd
    from swissisoform_site.data import tis_slug as make_slug

    df = pd.read_parquet(WEBSITE_DATA / "all_paired.parquet", columns=["gene_name", "tis_id"])
    row = df.iloc[0]
    r = client.get(f"/genes/{row['gene_name']}/isoforms/{make_slug(row['tis_id'])}")
    assert r.status_code == 200
    body = r.data
    assert b"graph-transcript" in body
    assert b"graph-protein" in body
    assert b"Synthesis" in body
    # 12-tile evidence grid replaced the 7-tab strip.
    assert b"tab-strip" not in body
    assert b"evidence-tile" in body


def test_isoform_page_has_12_evidence_tiles(client):
    """The V2 isoform page renders 12 evidence tiles (E1-E6, F1-F6) with axis classes."""
    import pandas as pd
    from swissisoform_site.data import tis_slug as make_slug

    df = pd.read_parquet(WEBSITE_DATA / "all_paired.parquet", columns=["gene_name", "tis_id"])
    row = df.iloc[0]
    r = client.get(f"/genes/{row['gene_name']}/isoforms/{make_slug(row['tis_id'])}")
    assert r.status_code == 200
    body = r.data
    # 12 evidence-tile class instances (one per criterion).
    assert body.count(b"evidence-tile") >= 12
    # Both axis classes present (E + F).
    assert b"axis-E" in body
    assert b"axis-F" in body
    # Folding panel placeholder until Task D lands the dual Mol* viewer.
    assert b"folding-panel" in body


def test_isoform_page_renders_side_by_side_viewers(client):
    """The folding panel renders two side-by-side Mol* viewers + a pLDDT legend."""
    import pandas as pd
    from swissisoform_site.data import tis_slug as make_slug

    df = pd.read_parquet(WEBSITE_DATA / "all_paired.parquet", columns=["gene_name", "tis_id"])
    row = df.iloc[0]
    r = client.get(f"/genes/{row['gene_name']}/isoforms/{make_slug(row['tis_id'])}")
    assert r.status_code == 200
    body = r.data
    # Two side-by-side viewers + legend; the old single/superposed divs are gone.
    assert b"folding-grid" in body
    assert b"folding-legend" in body
    assert b"folding-viewer-canonical" in body
    assert b"folding-viewer-isoform" in body
    assert b"molstar-canonical" not in body
    # Legend names the colouring scheme.
    assert b"pLDDT" in body


def test_isoform_page_has_evidence_modal_element(client):
    """Modal <dialog> element is in the DOM (hidden until JS shows it).

    Tile bodies live inside <template class="tile-body-template"> per tile;
    the browser never renders the inline copy. JS clones the fragment into
    the shared modal on click — tile grid stays unchanged.
    """
    import pandas as pd
    from swissisoform_site.data import tis_slug as make_slug

    df = pd.read_parquet(WEBSITE_DATA / "all_paired.parquet", columns=["gene_name", "tis_id"])
    row = df.iloc[0]
    r = client.get(f"/genes/{row['gene_name']}/isoforms/{make_slug(row['tis_id'])}")
    assert r.status_code == 200
    body = r.data
    # <dialog> element present
    assert b"evidence-modal" in body
    assert b"<dialog" in body
    # Tile body content kept as <template> fragments (not rendered inline).
    assert b"tile-body-template" in body
    # One <template> per tile (12).
    assert body.count(b"tile-body-template") >= 12


def test_isoform_page_truncation_marks_differential_region(client):
    """For a truncation, the folding legend names the differential region."""
    from swissisoform_site.data import tis_slug as make_slug

    # TRNT1 first isoform is a truncation in the cheeseman_12gene parquet.
    r = client.get("/genes/TRNT1/isoforms/" + make_slug("chr3:3129127:+:ATG:ENST00000434583.5"))
    assert r.status_code == 200
    body = r.data
    # The single viewer's legend flags the lost (differential) region by residue range.
    assert b"folding-legend" in body
    assert b"Differential region" in body


def test_diff_evidence_panel_surfaces_modalities(client):
    """The differential-region panel renders modality tabs with real evidence."""
    from swissisoform_site.data import diff_evidence_for, load_all, tis_slug as make_slug

    iso = load_all()["MSRA"].isoforms[0]
    de = diff_evidence_for(iso)
    ids = {s["id"] for s in de["sections"]}
    # Conservation, structure, localization, biophysics must all be surfaced.
    assert {"biophysics", "conservation", "structure", "localization", "initiation"}.issubset(ids)
    bio = next(s for s in de["sections"] if s["id"] == "biophysics")
    assert bio.get("compare_rows"), "biophysics must be comparative (unique vs shared)"
    pi = next(r for r in bio["compare_rows"] if "pI" in r["label"])
    assert pi["unique"] != pi["shared"]  # differential, not whole-protein
    # MSRA's mito->peroxisome retargeting must be flagged on the localization section.
    loc = next(s for s in de["sections"] if s["id"] == "localization")
    assert loc.get("highlight") is True

    r = client.get("/genes/MSRA/isoforms/" + make_slug("chr8:10054582:+:ATG:ENST00000317173.9"))
    assert r.status_code == 200
    body = r.data
    assert b"Differential region \xe2\x80\x94 evidence" in body  # em-dash heading
    assert b"diff-tab" in body
    assert b"Peroxisomal targeting signal" in body  # raw evidence now visible
