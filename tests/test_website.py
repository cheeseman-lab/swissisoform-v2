"""Smoke tests for the SwissIsoform v2 Flask viewer.

Requires the cheeseman_13gene parquet to be visible at
``website/data/all_paired.parquet`` (either real file or a symlink — the
website README documents the layout). Tests skip cleanly if it isn't there.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

# The Flask viewer is a separate deployable with its own env (Railway/Docker);
# skip these in the bio env, which doesn't carry a working flask/werkzeug stack.
pytest.importorskip("swissisoform_site.app")

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


def test_gene_page_renders_combined_igv_and_isoform_cards(client):
    """/genes/<gene> is the gene overview: combined genomic IGV + isoform cards."""
    r = client.get("/genes/TRNT1")
    assert r.status_code == 200
    body = r.data.decode()
    # The combined per-isoform genomic IGV.
    assert "graph-gene" in body
    assert "GENE_IGV_FIG" in body
    # Clicking a transcript bar navigates to the isoform page.
    assert "plotly_click" in body
    # Isoform cards link through to the per-isoform deep-dive page.
    assert "/genes/TRNT1/isoforms/" in body


def test_gene_page_linkifies_pmids_in_narrative(client):
    """The gene mechanistic narrative renders [PMID:N] citations as PubMed links."""
    r = client.get("/genes/CBX1")
    assert r.status_code == 200
    body = r.data.decode()
    assert "gene-head-fn" in body
    assert "pubmed.ncbi.nlm.nih.gov" in body


def test_pmid_links_filter_escapes_and_links():
    """The pmid_links Jinja filter escapes text and wraps PMIDs in PubMed anchors."""
    from swissisoform_site.app import app

    with app.test_request_context():
        f = app.jinja_env.filters["pmid_links"]
        out = str(f("reads H3K9me3 [PMID:21047797] <script>"))
        multi = str(f("established [PMID:36310139, PMID:30031230, PMID:38769286]"))
    assert '<a href="https://pubmed.ncbi.nlm.nih.gov/21047797/"' in out
    assert "&lt;script&gt;" in out  # surrounding text is escaped
    # Every PMID in a multi-PMID bracket is linked (regression: only the first used to be).
    for pmid in ("36310139", "30031230", "38769286"):
        assert f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/' in multi
    assert multi.count("<a ") == 3
    assert multi.startswith("established [") and multi.rstrip().endswith("]")


def test_gene_view_variant_unique_in_any_isoform():
    """A variant in the unique region of ANY isoform stays flagged unique.

    Regression: the pathogenic-upgrade path in the cross-isoform variant dedup
    used to clobber the OR-merged ``in_unique`` with the current isoform's flag.
    Here V:1 is unique+benign in isoform A but shared+pathogenic in B; the deduped
    lollipop must keep ``in_unique=True`` (unique in A) AND the pathogenic call.
    """
    if str(WEBSITE_SRC) not in sys.path:
        sys.path.insert(0, str(WEBSITE_SRC))
    from types import SimpleNamespace

    from swissisoform_site.app import _make_gene_protein_view

    def _iso(tis_id, diff_end, iso_len, start_codon, variants):
        return SimpleNamespace(
            tis_id=tis_id, transcript_id="ENST1", orf_type="extended",
            diff_space="isoform", diff_end=diff_end, isoform_len=iso_len,
            canonical_len=200, start_codon=start_codon, diff_start=0,
            differential_sequence="", raw={}, variants_all=variants,
            variants_in_unique=[],
        )

    var_in_a = {"variant_id": "V:1", "isoform_protein_pos": 5, "in_isoform_unique": True,
                "clinical_significance": "Benign", "consequence": "missense_variant"}
    var_in_b = {"variant_id": "V:1", "isoform_protein_pos": 60, "in_isoform_unique": False,
                "clinical_significance": "Pathogenic", "consequence": "missense_variant"}
    gene = SimpleNamespace(
        canonical_len=200,
        isoforms=[_iso("A", 10, 210, "CTG", [var_in_a]),
                  _iso("B", 5, 205, "GTG", [var_in_b])],
    )

    view = _make_gene_protein_view(gene)
    v1 = [v for v in view.variants if v["variant_id"] == "V:1"]
    assert len(v1) == 1                     # deduped across the two isoforms
    assert v1[0]["in_unique"] is True       # unique in isoform A → stays unique
    assert "pathogenic" in (v1[0]["significance"] or "").lower()  # most-severe kept


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
    # Dataset-agnostic: whichever run is staged in website/data, every gene in the
    # parquet must surface in the API with a well-formed isoform list. (Pinning an
    # explicit gene list broke whenever the staged run changed.)
    import pandas as pd

    df = pd.read_parquet(WEBSITE_DATA / "all_paired.parquet", columns=["gene_name"])
    expected = {g for g in df["gene_name"].dropna().unique()}
    assert expected, "staged parquet has no genes"
    assert set(payload) >= expected, f"missing from /api/data.json: {expected - set(payload)}"
    for gene in sorted(expected):
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

    skel_path = Path("data/output/cheeseman_13gene/transcript_skeletons.parquet")
    if not skel_path.exists():
        pytest.skip("skeleton parquet not present")
    sk = load_transcript_skeletons(skel_path)
    assert len(sk) > 0
    sample = next(iter(sk.values()))
    assert sample.chrom.startswith("chr")
    assert sample.strand in ("+", "-")
    assert len(sample.exons) >= 1


def test_category_verdicts_for_isoform_returns_empty_when_missing(tmp_path):
    """category_verdicts_for_isoform tolerates a missing JSON file."""
    if str(WEBSITE_SRC) not in sys.path:
        sys.path.insert(0, str(WEBSITE_SRC))
    from swissisoform_site.data import category_verdicts_for_isoform

    out = category_verdicts_for_isoform(
        llm_dir=tmp_path,
        tis_slug="chr1-100-ATG-ENST_A",
    )
    assert out == {}


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


def test_synthesis_tags_whitelisted_against_vocab(tmp_path):
    """synthesis_tags_for_isoform keeps only controlled-vocab tags, in vocab order."""
    if str(WEBSITE_SRC) not in sys.path:
        sys.path.insert(0, str(WEBSITE_SRC))
    import json as _json

    from swissisoform_site.data import ISOFORM_TAG_VOCAB, synthesis_tags_for_isoform

    slug = "chr1-1-ATG-ENST"
    (tmp_path / slug).mkdir()
    (tmp_path / slug / "synthesis.json").write_text(
        _json.dumps({"tags": ["Domain gain", "made up tag", "Localization conflict"]})
    )
    tags = synthesis_tags_for_isoform(llm_dir=tmp_path, tis_slug=slug)
    # off-vocab dropped; survivors in vocab order (Localization conflict precedes Domain gain)
    assert tags == ["Localization conflict", "Domain gain"]
    assert all(t in ISOFORM_TAG_VOCAB for t in tags)
    # missing dir / no tags → empty
    assert synthesis_tags_for_isoform(llm_dir=tmp_path, tis_slug="absent") == []


def test_synthesis_keyed_dict_renders_hypothesis_and_confidence():
    """llm_synthesis_for_isoform html-renders the keyed prose fields; legacy still works."""
    if str(WEBSITE_SRC) not in sys.path:
        sys.path.insert(0, str(WEBSITE_SRC))
    import json as _json
    import tempfile
    from pathlib import Path

    from swissisoform_site.data import llm_synthesis_for_isoform

    d = Path(tempfile.mkdtemp())
    slug = "chr1-1-ATG-ENST"
    (d / slug).mkdir()
    (d / slug / "synthesis.json").write_text(
        _json.dumps({
            "tis_id": "x", "headline": "h",
            "divergence_hypothesis": "adds a **targeting** arm",
            "function_relevance": "matters", "tags": ["Localization conflict", "bogus"],
            "confidence": "medium",
        })
    )
    syn = llm_synthesis_for_isoform(llm_dir=d, tis_slug=slug)
    assert "<strong>targeting</strong>" in syn["divergence_hypothesis_html"]
    assert "function_relevance_html" in syn
    assert syn["tags"] == ["Localization conflict"]  # bogus dropped
    assert syn["confidence"] == "medium"


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
    """The rendered V2 page exposes the folding panel and the Synthesis block.

    The combined genomic IGV moved to the gene page; the per-isoform page keeps
    the folding + evidence + variants deep dive (no ``graph-protein`` panel).
    """
    import pandas as pd
    from swissisoform_site.data import tis_slug as make_slug

    df = pd.read_parquet(WEBSITE_DATA / "all_paired.parquet", columns=["gene_name", "tis_id"])
    row = df.iloc[0]
    r = client.get(f"/genes/{row['gene_name']}/isoforms/{make_slug(row['tis_id'])}")
    assert r.status_code == 200
    body = r.data
    # The residue-axis protein IGV is gone from the isoform page.
    assert b"graph-protein" not in body
    # The folding panel stays on the isoform deep-dive page.
    assert b"iso-panel-folding" in body
    # AI summary is pinned to the top as a collapsible dropdown.
    assert b"AI summary" in body
    assert b"synthesis-dd" in body
    # 12-tile evidence grid replaced the 7-tab strip.
    assert b"tab-strip" not in body
    assert b"evidence-tile" in body


def test_isoform_page_has_evidence_tiles(client):
    """The V2 isoform page renders the scored evidence tiles plus the bespoke
    Biophysics (S2) and SAE (S3) cards, in one flat grid (no group headers).
    """
    import pandas as pd
    from swissisoform_site.data import tis_slug as make_slug

    df = pd.read_parquet(WEBSITE_DATA / "all_paired.parquet", columns=["gene_name", "tis_id"])
    row = df.iloc[0]
    r = client.get(f"/genes/{row['gene_name']}/isoforms/{make_slug(row['tis_id'])}")
    assert r.status_code == 200
    body = r.data
    # 13 scored criterion tiles + the Biophysics and SAE cards, each an evidence-tile.
    assert body.count(b"evidence-tile") >= 15
    # Grouping headers are gone; the flat grid remains.
    assert b"tile-row-h" not in body
    # Biophysics is its own card, keyed by the CDLMPS criterion id (was the
    # pre-rename lowercase "biophysics" when it lived inside the F1 modal).
    assert b'data-criterion="S2_biophysics"' in body
    # Both axis classes still present (E + F border colors).
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
    from swissisoform_site.data import load_all
    from swissisoform_site.data import tis_slug as make_slug

    # Pick any truncation from the staged run rather than pinning one tis_id.
    target = next(
        ((gn, iso.tis_id) for gn, g in load_all().items() for iso in g.isoforms
         if str(getattr(iso, "orf_type", "")).lower() == "truncated"),
        None,
    )
    assert target, "staged parquet has no truncation isoform"
    gene, tis_id = target
    r = client.get(f"/genes/{gene}/isoforms/" + make_slug(tis_id))
    assert r.status_code == 200
    body = r.data
    # The single viewer's legend flags the lost (differential) region by residue range.
    assert b"folding-legend" in body
    assert b"Differential region" in body


def test_criterion_evidence_folds_into_score_popups(client):
    """Differential evidence is keyed by criterion id and embedded per modal."""
    from swissisoform_site.data import (
        biophysics_card_for_isoform,
        criterion_evidence_for,
        load_all,
    )
    from swissisoform_site.data import tis_slug as make_slug

    # Pick an isoform from the staged run that actually exercises this path: a
    # flagged localization change plus a biophysics card. (Pinning MSRA broke
    # whenever the staged run changed.)
    picked = next(
        (
            (gn, iso)
            for gn, g in load_all().items()
            for iso in g.isoforms
            if (criterion_evidence_for(iso).get("L1_localization_change", {}).get("sections"))
            and criterion_evidence_for(iso)["L1_localization_change"]["sections"][0].get(
                "highlight"
            )
            and biophysics_card_for_isoform(iso)
        ),
        None,
    )
    assert picked, "no isoform with a flagged localization change + biophysics card"
    gene_name, iso = picked
    ce = criterion_evidence_for(iso)
    # Every criterion has an entry with a plain-English "about" descriptor.
    assert set(ce) >= {
        "C1_primate_conservation",
        "C3_phylop_coding_selection",
        "P1_structured_extension",
        "L1_localization_change",
    }
    assert ce["L1_localization_change"]["about"]
    # Biophysics is no longer a sub-section of the P1 (folding) modal — it moved to
    # its own descriptive card, so P1 must be folding-only.
    assert not any(
        s["title"].startswith("Biophysics")
        for s in ce["P1_structured_extension"]["sections"]
    ), "P1 should be folding-only; biophysics belongs to the standalone S2 card"
    bio = biophysics_card_for_isoform(iso)
    assert bio is not None, "standalone biophysics card missing"
    sec = bio["evidence"]["sections"][0]
    assert sec["cmp_headers"] == ["Property", "Differential", "Shared", "Enrichment"]
    pi = next(r for r in sec["compare_rows"] if "pI" in r["label"])
    assert pi["cols"][0] != pi["cols"][1]  # differential vs shared core
    # Localization renders a canonical-vs-isoform table (not flat rows), flagged
    # because this isoform's predicted compartment/signals changed.
    loc = ce["L1_localization_change"]["sections"][0]
    assert loc["highlight"] is True
    assert loc["cmp_headers"] == ["Property", "Canonical", "Isoform"]
    assert loc["compare_rows"] and len(loc["compare_rows"][0]["cols"]) == 2

    r = client.get(f"/genes/{gene_name}/isoforms/" + make_slug(iso.tis_id))
    assert r.status_code == 200
    body = r.data
    assert b"crit-about" in body  # the per-criterion descriptor block
    assert b"click any tile" in body  # standalone panel dissolved into the tiles


def test_domains_massspec_are_canonical_vs_isoform(client):
    """F3 domains and E6 mass-spec compare the whole canonical vs isoform protein."""
    from swissisoform_site.data import criterion_evidence_for, load_all

    iso = load_all()["CBX1"].isoforms[0]
    ce = criterion_evidence_for(iso)

    f3 = ce["S1_domain_change"]["sections"][0]
    assert f3["cmp_headers"] == ["Feature", "Canonical", "Isoform"]
    dom = next(r for r in f3["compare_rows"] if r["label"] == "InterPro domains")
    assert len(dom["cols"]) == 2  # canonical | isoform counts
    # gained/lost features surface in the Details box
    assert any(h["kind"] in ("gained", "lost") for h in f3.get("hits", []))

    e6 = ce["D3_mass_spec"]["sections"][0]
    assert e6["cmp_headers"] == ["Feature", "Canonical", "Isoform"]
    uniq = next(r for r in e6["compare_rows"] if r["label"] == "Isoform-unique peptides")
    assert uniq["cols"][0] == "—"  # uniqueness is an isoform-only property


def test_comparison_tables_use_two_standard_flavors(client):
    """Every comparison table is one of two flavors — Canonical|Isoform or
    Differential|Shared — so the modals read consistently. E1/E2 frame is the
    one documented exception (its differential side flips with diff_space).
    """
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
                        "C1_primate_conservation",
                        "C2_mammalian_conservation",
                    )
                    if not ok and not frame_exception:
                        offenders.append((cid, tuple(hdr)))
    assert not offenders, f"non-standard comparison headers: {set(offenders)}"


def test_details_boxes_are_qualitative_lists_only(client):
    """The Details box should host only qualitative info (hit lists — peptides,
    gained/lost domains), never single-scalar key/value rows. Scalars belong in a
    comparison table (two-sided) or a section caption (single).
    """
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
    sec = criterion_evidence_for(iso)["M2_clinical_variant_overlap"]["sections"][0]
    # Standardized Flavor-2 columns: Differential | Shared | Enrichment (the
    # enrichment is the length-normalized per-residue ratio).
    assert sec["cmp_headers"][1:] == ["Differential", "Shared", "Enrichment"]
    allv = next(r for r in sec["compare_rows"] if r["label"] == "Disease variants")
    assert len(allv["cols"]) == 3 and allv["cols"][2].endswith("×")


def test_about_page_renders_glossary(client):
    """The /about route explains the categories, criteria, and diff_space frame rule."""
    r = client.get("/about")
    assert r.status_code == 200
    body = r.data
    assert b"About SwissIsoform" in body
    assert b"diff_space" in body  # the frame rule is spelled out
    assert b"AlphaMissense" in body and b"canonical frame only" in body
    assert b"CDLMPS" in body  # the six-category scheme (replaced the old E/F axes)
    # nav link is wired on every page
    assert b'href="/about"' in client.get("/").data


def test_about_page_is_navigable(client):
    """Onboarding, sticky TOC, and a CDLMPS-ordered collapsible glossary.

    The page opens with a plain-language lede plus a "how to read a card" section,
    carries an in-page TOC, and presents the ~400-line metrics reference as
    collapsible groups ordered C-D-L-M-P-S (scoring framework last).
    """
    body = client.get("/about").data.decode()

    # Onboarding + sticky TOC
    assert 'id="how-to-read"' in body
    assert 'class="about-toc"' in body

    # Frame rule sits after the CDLMPS intro, not before it
    assert body.index('id="cdlmps"') < body.index('id="diff-region"')

    # Glossary groups are collapsible and in CDLMPS order, scoring framework last
    order = re.findall(r'<details class="about-gloss" id="gloss-([a-z-]+)"', body)
    assert order == ["c", "d-init", "d-ms", "l", "m", "p", "s", "scoring"], order

    # No dangling in-page anchors
    ids = set(re.findall(r'id="([^"]+)"', body))
    assert not {h for h in re.findall(r'href="#([^"]+)"', body)} - ids


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
