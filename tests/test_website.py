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
    assert b"graph-protein" in body
    # AI summary is pinned to the top as a collapsible dropdown.
    assert b"AI summary" in body
    assert b"synthesis-dd" in body
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
    """Folding panel: two side-by-side interactive 3Dmol viewers + a pLDDT legend.

    Each viewer loads its CIF and applies a precomputed per-residue colour map
    (the diff-region recolouring is decided offline, served at /structure-colors/).
    """
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
    # Interactive 3Dmol viewer wired with the offline colour map + CIF download.
    assert b"3Dmol-min.js" in body
    assert b"/structure-colors/" in body
    assert b"Download CIF" in body
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


def test_criterion_evidence_folds_into_score_popups(client):
    """Differential evidence is keyed by criterion id and embedded per modal."""
    from swissisoform_site.data import criterion_evidence_for, load_all, tis_slug as make_slug

    iso = load_all()["MSRA"].isoforms[0]
    ce = criterion_evidence_for(iso)
    # Every criterion has an entry with a plain-English "about" descriptor.
    assert set(ce) >= {
        "E1_primate_conservation",
        "E3_phylop_coding_selection",
        "F1_structured_extension",
        "F2_localization_change",
    }
    assert ce["F2_localization_change"]["about"]
    # F1 hosts comparative biophysics (differential vs shared, not whole-protein).
    bio = next(
        s for s in ce["F1_structured_extension"]["sections"]
        if s["title"].startswith("Biophysics")
    )
    pi = next(r for r in bio["compare_rows"] if "pI" in r["label"])
    assert pi["cols"][0] != pi["cols"][1]  # differential vs shared core
    # F2 renders a canonical-vs-isoform table (not flat rows), and MSRA's
    # mito->peroxisome retargeting flags it.
    loc = ce["F2_localization_change"]["sections"][0]
    assert loc["highlight"] is True
    assert loc["cmp_headers"] == ["Property", "Canonical", "Isoform"]
    assert loc["compare_rows"] and len(loc["compare_rows"][0]["cols"]) == 2

    r = client.get("/genes/MSRA/isoforms/" + make_slug("chr8:10054582:+:ATG:ENST00000317173.9"))
    assert r.status_code == 200
    body = r.data
    assert b"crit-about" in body  # the per-criterion descriptor block
    assert b"click any tile" in body  # standalone panel dissolved into the tiles
    assert b"Peroxisomal targeting signal" in body  # evidence still present (in modal template)


def test_domains_massspec_are_canonical_vs_isoform(client):
    """F3 domains and E6 mass-spec compare the whole canonical vs isoform protein."""
    from swissisoform_site.data import criterion_evidence_for, load_all

    iso = load_all()["CBX1"].isoforms[0]
    ce = criterion_evidence_for(iso)

    f3 = ce["F3_domain_change"]["sections"][0]
    assert f3["cmp_headers"] == ["Feature", "Canonical", "Isoform"]
    dom = next(r for r in f3["compare_rows"] if r["label"] == "InterPro domains")
    assert len(dom["cols"]) == 2  # canonical | isoform counts
    # gained/lost features surface in the Details box
    assert any(h["kind"] in ("gained", "lost") for h in f3.get("hits", []))

    e6 = ce["E6_mass_spec"]["sections"][0]
    assert e6["cmp_headers"] == ["Feature", "Canonical", "Isoform"]
    uniq = next(r for r in e6["compare_rows"] if r["label"] == "Isoform-unique peptides")
    assert uniq["cols"][0] == "—"  # uniqueness is an isoform-only property


def test_comparison_tables_use_two_standard_flavors(client):
    """Every comparison table is one of two flavors — Canonical|Isoform or
    Differential|Shared — so the modals read consistently. E1/E2 frame is the
    one documented exception (its differential side flips with diff_space)."""
    from swissisoform_site.data import criterion_evidence_for, load_all

    genes = load_all()
    offenders = []
    for g in genes.values():
        for iso in g.isoforms:
            for cid, ce in criterion_evidence_for(iso).items():
                for s in ce["sections"]:
                    hdr = s.get("cmp_headers")
                    if not hdr:
                        continue
                    pair = hdr[1:3]
                    ok = pair in (["Canonical", "Isoform"], ["Differential", "Shared"])
                    frame_exception = cid in (
                        "E1_primate_conservation",
                        "E2_mammalian_conservation",
                    )
                    if not ok and not frame_exception:
                        offenders.append((cid, tuple(hdr)))
    assert not offenders, f"non-standard comparison headers: {set(offenders)}"


def test_details_boxes_are_qualitative_lists_only(client):
    """The Details box should host only qualitative info (hit lists — peptides,
    gained/lost domains), never single-scalar key/value rows. Scalars belong in a
    comparison table (two-sided) or a section caption (single)."""
    from swissisoform_site.data import criterion_evidence_for, load_all

    genes = load_all()
    offenders = []
    for g in genes.values():
        for iso in g.isoforms:
            for cid, ce in criterion_evidence_for(iso).items():
                for s in ce["sections"]:
                    if s.get("rows"):
                        offenders.append((cid, s.get("title")))
    assert not offenders, f"Details 'rows' should be tables or captions: {set(offenders)}"


def test_f6_clinical_burden_is_length_normalized(client):
    """F6 reports a per-residue enrichment ratio, not just raw counts."""
    from swissisoform_site.data import criterion_evidence_for, load_all

    iso = load_all()["CBX1"].isoforms[0]
    sec = criterion_evidence_for(iso)["F6_clinical_variant_overlap"]["sections"][0]
    # Standardized Flavor-2 columns: Differential | Shared | Enrichment (the
    # enrichment is the length-normalized per-residue ratio).
    assert sec["cmp_headers"][1:] == ["Differential", "Shared", "Enrichment"]
    allv = next(r for r in sec["compare_rows"] if r["label"] == "Disease variants")
    assert len(allv["cols"]) == 3 and allv["cols"][2].endswith("×")


def test_about_page_renders_glossary(client):
    """The /about route explains the axes, criteria, and the diff_space frame rule."""
    r = client.get("/about")
    assert r.status_code == 200
    body = r.data
    assert b"About SwissIsoform" in body
    assert b"diff_space" in body  # the frame rule is spelled out
    assert b"AlphaMissense" in body and b"canonical frame only" in body
    assert b"Existence (E)" in body and b"Functional (F)" in body
    # nav link is wired on every page
    assert b'href="/about"' in client.get("/").data


def test_every_isoform_page_renders_200(client):
    """Regression guard: the diff-evidence panel must not 500 on any isoform.

    (A hits column of plain strings rather than dicts crashed 11/23 pages.)
    """
    import swissisoform_site.data as data_mod
    from swissisoform_site.data import tis_slug as make_slug

    bad = []
    for gname, g in data_mod.load_all().items():
        for iso in g.isoforms:
            r = client.get(f"/genes/{gname}/isoforms/{make_slug(iso.tis_id)}")
            if r.status_code != 200:
                bad.append((gname, iso.tis_id, r.status_code))
    assert not bad, f"isoform pages returned non-200: {bad}"
