"""Tests for CRITERIA_FOR_PAGE + category_verdicts_for_isoform in data.py."""

import json
from pathlib import Path

import pytest

pytest.importorskip("swissisoform_site")  # optional website package; skip if not installed

from swissisoform_site.data import CARD_GROUPS, CRITERIA_FOR_PAGE, category_verdicts_for_isoform


def test_criteria_for_page_has_16_entries() -> None:
    # 13 flat criteria + S2 biophysics + S3 SAE + P3 secondary structure.
    assert len(CRITERIA_FOR_PAGE) == 16
    e_count = sum(1 for c in CRITERIA_FOR_PAGE if c["axis"] == "E")
    f_count = sum(1 for c in CRITERIA_FOR_PAGE if c["axis"] == "F")
    assert e_count == 6
    assert f_count == 10


def test_criteria_for_page_ids_match_ber() -> None:
    """Page config must align with CRITERIA in build_evidence_records."""
    from swissisoform.site.evidence import CRITERIA

    assert {c["id"] for c in CRITERIA_FOR_PAGE} == set(CRITERIA)


def test_card_groups_derived_from_backend_categories() -> None:
    """CARD_GROUPS mirrors the backend CATEGORIES (single source of truth)."""
    from swissisoform.site.evidence import CATEGORIES

    assert [g["letter"] for g in CARD_GROUPS] == [c["letter"] for c in CATEGORIES]
    assert [g["members"] for g in CARD_GROUPS] == [c["members"] for c in CATEGORIES]


def test_category_verdicts_returns_empty_when_missing(tmp_path: Path) -> None:
    out = category_verdicts_for_isoform(llm_dir=tmp_path, tis_slug="x")
    assert out == {}


def test_category_verdicts_reads_categories_json(tmp_path: Path) -> None:
    slug = "chr3-3129127-+-ATG-test"
    (tmp_path / slug).mkdir()
    (tmp_path / slug / "categories.json").write_text(
        json.dumps(
            {
                "Conservation": {
                    "verdict": "interesting",
                    "reasoning": "primate frame intact (frac_intact 0.96).",
                }
            }
        )
    )
    out = category_verdicts_for_isoform(llm_dir=tmp_path, tis_slug=slug)
    assert out["Conservation"]["verdict"] == "interesting"
    assert "frac_intact" in out["Conservation"]["reasoning"]


# ── _clean_nan must not collapse one-element arrays ───────────────────────


def test_clean_nan_keeps_a_one_element_array_a_list() -> None:
    """A size-1 numpy object array must stay a list, not become its element.

    ``.item()`` on a size-1 array returns the single element, so a one-hit
    positional column silently arrived as a bare dict. Consumers iterate it and
    get key strings instead of hits — which is how MAD2L1's real 19-residue
    helix (pLDDT 0.97) rendered as "No helix or strand in region". It hit 6
    columns (SSE elements, motifs, massspec, InterProScan), always and only when
    a protein had exactly one hit, which is why it survived: the multi-hit
    proteins everyone checked looked fine.
    """
    np = pytest.importorskip("numpy")
    from swissisoform_site.data import _clean_nan

    one = np.empty(1, dtype=object)
    one[0] = {"type": "helix", "start": 13, "end": 31, "length": 19}
    out = _clean_nan(one)
    assert isinstance(out, list) and len(out) == 1
    assert out[0]["type"] == "helix"

    two = np.empty(2, dtype=object)
    two[0], two[1] = {"a": 1}, {"a": 2}
    assert [d["a"] for d in _clean_nan(two)] == [1, 2]

    empty = np.empty(0, dtype=object)
    assert _clean_nan(empty) == []


def test_clean_nan_still_unwraps_numpy_scalars() -> None:
    """The ndim guard must not break the branch it sits in front of."""
    np = pytest.importorskip("numpy")
    from swissisoform_site.data import _clean_nan

    assert _clean_nan(np.float64(0.966)) == pytest.approx(0.966)
    assert _clean_nan(np.int64(19)) == 19
    assert isinstance(_clean_nan(np.int64(19)), int)
    assert _clean_nan(np.float64("nan")) is None
    assert _clean_nan("abc") == "abc"


# ── P3 evidence modal: two-column SSE listing ─────────────────────────────


class _Iso:
    def __init__(self, raw, orf_type="extended", diff_space="isoform"):
        self.raw, self.orf_type, self.diff_space = raw, orf_type, diff_space


def _sse_raw(*elements):
    return {
        "isoform_structure_sse_status": "ok",
        "isoform_structure_sse_all_elements": list(elements),
    }


def _el(kind, start, end, region, plddt=0.9):
    return {
        "type": kind, "start": start, "end": end,
        "length": end - start + 1, "plddt_mean": plddt, "region": region,
    }


def _p3(iso):
    from swissisoform_site.data import criterion_evidence_for

    return criterion_evidence_for(iso)["P3_secondary_structure"]


def test_p3_modal_splits_unique_and_shared() -> None:
    # All three qualify (>=6 aa, pLDDT >=0.70) so this test is about the
    # unique/shared split, not the threshold — that is covered separately.
    ce = _p3(_Iso(_sse_raw(
        _el("helix", 16, 32, "unique"),
        _el("strand", 40, 46, "shared"),
        _el("helix", 60, 75, "shared"),
    )))
    summary, listing = ce["sections"][0], ce["sections"][1]
    # House grammar: reference (shared) left, the stratum's region right, ratio last.
    assert summary["cmp_headers"] == ["Metric", "Shared", "Extension", "Ratio"]
    # helices: 1 shared / 1 unique;  strands: 1 shared / 0 unique
    assert [r["cols"] for r in summary["compare_rows"]][:2] == [["1", "1", "1×"], ["1", "0", "0×"]]
    assert [r["label"] for r in summary["compare_rows"]][:2] == ["Alpha helices", "Beta strands"]
    # One row per element, padded to the longer side.
    assert len(listing["pairs"]) == 2
    assert listing["pairs"][0]["left"]["span"] == "16–32"
    assert listing["pairs"][1]["left"] is None  # padding


def test_p3_modal_relabels_the_unique_column_for_truncations() -> None:
    """On a truncation the region exists only in the canonical protein, so
    calling that column "isoform-unique" would name the one protein lacking it.
    Both tables name it the way the stratum does.
    """
    ce = _p3(_Iso(_sse_raw(_el("helix", 13, 31, "unique")),
                  orf_type="truncated", diff_space="canonical"))
    assert ce["sections"][0]["cmp_headers"][2] == "Lost region"
    assert ce["sections"][1]["pair_headers"][0] == "Lost region"
    uorf = _p3(_Iso(_sse_raw(_el("helix", 13, 31, "unique")), orf_type="uorf"))
    assert uorf["sections"][1]["pair_headers"][0] == "ORF"


def test_p3_modal_counts_a_boundary_spanning_element_for_the_isoform() -> None:
    """A spanning element has secondary structure in the unique region, so it
    counts for the isoform and lists on the unique side at its full span — no
    separate annotation about where it ends.
    """
    ce = _p3(_Iso(_sse_raw(_el("helix", 13, 35, "spans"))))
    summary, listing = ce["sections"][0], ce["sections"][1]
    assert [r["cols"] for r in summary["compare_rows"]][0] == ["0", "1", "—"]  # alpha helices
    cell = listing["pairs"][0]["left"]
    assert cell["span"] == "13–35"
    assert "note" not in cell
    assert listing["pairs"][0]["right"] is None


def test_p3_modal_absent_when_sse_did_not_run() -> None:
    from swissisoform_site.data import criterion_evidence_for

    ce = criterion_evidence_for(_Iso({"isoform_structure_sse_status": "no_cache"}))
    assert ce["P3_secondary_structure"]["sections"] == []


def test_p3_modal_survives_a_numpy_element_array() -> None:
    """The parquet hands back a numpy object array, not a list."""
    np = pytest.importorskip("numpy")
    arr = np.empty(1, dtype=object)
    arr[0] = _el("helix", 16, 32, "unique")
    ce = _p3(_Iso({
        "isoform_structure_sse_status": "ok",
        "isoform_structure_sse_all_elements": arr,
    }))
    assert ce["sections"][1]["pairs"][0]["left"]["span"] == "16–32"


def test_p3_modal_badges_name_the_structure_type() -> None:
    """The badge reads "alpha helix" / "beta strand", but `kind` stays the bare
    P-SEA name because it keys the CSS class (a space would break it).
    """
    ce = _p3(_Iso(_sse_raw(_el("helix", 16, 32, "unique"), _el("strand", 40, 47, "shared"))))
    left = ce["sections"][1]["pairs"][0]["left"]
    right = ce["sections"][1]["pairs"][0]["right"]
    assert (left["kind"], left["label"]) == ("helix", "alpha helix")
    assert (right["kind"], right["label"]) == ("strand", "beta strand")


def test_p3_modal_moves_sub_threshold_elements_to_their_own_section() -> None:
    """Every count on the card must agree, so the summary and the first listing
    hold only what P3 counts; the rest gets its own heading rather than being
    dropped. TRIP13 is the live case: a 14 aa helix at pLDDT 0.63 made the table
    read 2 elements while the headline read 1.
    """
    ce = _p3(_Iso(_sse_raw(
        _el("strand", 7, 12, "unique", plddt=0.78),       # qualifies
        _el("helix", 21, 34, "spans", plddt=0.63),        # long, but not confident
        _el("strand", 40, 42, "shared", plddt=0.95),      # confident, but short
    )))
    summary, good, weak = ce["sections"]
    # Summary counts the qualifying set only: 0 helices / 1 strand on the unique side.
    assert [r["cols"] for r in summary["compare_rows"]][:2] == [["0", "0", "—"], ["0", "1", "—"]]
    assert good["title"] == "Elements and coordinates"
    assert good["pairs"][0]["left"]["span"] == "7–12"
    assert weak["title"] == "Below threshold"
    assert weak["pairs"][0]["left"]["span"] == "21–34"       # low pLDDT
    assert weak["pairs"][0]["right"]["span"] == "40–42"      # too short


def test_p3_modal_omits_the_below_threshold_section_when_empty() -> None:
    ce = _p3(_Iso(_sse_raw(_el("helix", 16, 32, "unique"))))
    assert [s["title"] for s in ce["sections"]] == [
        "Secondary structure · extension vs shared",
        "Elements and coordinates",
    ]


# ── M1 highlights follow what the scorer read (PR #56 review) ────────────


def _m1_rows(raw):
    from swissisoform_site.data import criterion_evidence_for

    iso = _Iso(raw)
    iso.criteria = {"M1_pathogenic_variant_enrichment": True}
    secs = criterion_evidence_for(iso)["M1_pathogenic_variant_enrichment"]["sections"]
    return {r["label"]: r for s in secs for r in s.get("compare_rows") or []}


def test_m1_gnomad_row_not_hot_on_a_site_computed_depletion() -> None:
    # No module depletion ratio: M1 fired on constraint alone, so the density the
    # page works out from the counts (0.4×, under the cutoff) is context only.
    rows = _m1_rows({
        "isoform_variant_intersection_n_gnomad_in_unique_region": 2,
        "isoform_variant_intersection_n_gnomad_in_shared_region": 10,
        "isoform_variant_intersection_unique_region_nt": 100,
        "isoform_variant_intersection_shared_region_nt": 200,
        "isoform_plm_vep_status": "ok",
        "isoform_plm_vep_constraint_delta": 0.2,
    })
    assert rows["gnomAD variants"]["cols"][2] == "0.4×"
    assert rows["gnomAD variants"]["hot"] is False
    assert rows["Constraint Δ"]["hot"] is True


def test_m1_constraint_row_not_hot_unless_plm_vep_ok() -> None:
    rows = _m1_rows({
        "isoform_variant_intersection_n_gnomad_in_unique_region": 2,
        "isoform_variant_intersection_n_gnomad_in_shared_region": 10,
        "isoform_variant_intersection_gnomad_depletion_ratio": 0.4,
        "isoform_plm_vep_status": "no_cache",
        "isoform_plm_vep_constraint_delta": 0.1,
    })
    assert rows["gnomAD variants"]["hot"] is True
    assert rows["Constraint Δ"]["hot"] is False


def test_fmt_ratio_never_reads_as_scientific_notation() -> None:
    from swissisoform_site.data import _fmt_ratio

    assert _fmt_ratio(0.42) == "0.42×"
    assert _fmt_ratio(3.14) == "3.1×"
    assert _fmt_ratio(162.5) == "162×"
    assert _fmt_ratio(12_000.0) == "12,000×"
    assert _fmt_ratio(0.0) == "0×"
    assert _fmt_ratio(1.2e-05) == "<0.01×"
    assert _fmt_ratio(None) == "—"
    assert _fmt_ratio(float("inf")) == "—"


# ── PR #56 second review ──────────────────────────────────────────────────


def test_s2_headline_reads_the_deltas_s2_scores() -> None:
    """Headline and highlighted lever agree: both use |isoform − canonical| at S2's cutoff."""
    from swissisoform_site.data import biophysics_card_for_isoform

    iso = _Iso({
        "canonical_biophysics_gravy": -0.40, "isoform_biophysics_gravy": -0.05,
        "cmp_biophysics_gravy_delta": 0.35,  # ≥ s2_gravy_delta_min (0.3): S2 fires
        # The region and the shared core barely differ — the old headline's basis.
        "cmp_biophysics_gravy_unique": -0.10, "cmp_biophysics_gravy_shared": -0.05,
    })
    iso.criteria = {"S2_biophysics": True}
    card = biophysics_card_for_isoform(iso)
    assert card["headline"] == "more hydrophobic (+0.35)"
    lever = card["evidence"]["sections"][0]["compare_rows"][0]
    assert lever["label"] == "Hydropathy (GRAVY)" and lever["hot"] is True


def test_c1_table_never_calls_a_lost_region_the_isoform() -> None:
    from swissisoform_site.data import criterion_evidence_for

    iso = _Iso({
        "isoform_conservation_frame_primate_canonical_mean_pident": 0.80,
        "isoform_conservation_frame_primate_mean_pident": 0.95,
    }, orf_type="truncated", diff_space="canonical")
    sec = criterion_evidence_for(iso)["C1_primate_conservation"]["sections"][0]
    assert sec["cmp_headers"][1:3] == ["Canonical ORF", "Lost region"]
    assert "lost region" in sec["subtitle"]


# ── PR #56 third review: highlights read what each scorer reads ───────────


def _rows_for(raw, cid, fired=True, **iso_kw):
    from swissisoform_site.data import criterion_evidence_for

    iso = _Iso(raw, **iso_kw)
    iso.criteria = {cid: fired}
    return {r["label"]: r for s in criterion_evidence_for(iso)[cid]["sections"]
            for r in s.get("compare_rows") or []}


def test_d3_counts_only_validated_is_true() -> None:
    hits = [
        {"peptide": "AAK", "unique_to_isoform": True, "validated": "false"},  # truthy, not True
        {"peptide": "CCK", "unique_to_isoform": True, "validated": True},
    ]
    rows = _rows_for({"isoform_massspec_hits": hits}, "D3_mass_spec")
    assert rows["Validated isoform-unique peptides"]["cols"][1] == "1"
    assert rows["Validated isoform-unique peptides"]["hot"] is True


def test_l1_nan_change_flag_is_not_a_change() -> None:
    rows = _rows_for({
        "cmp_localization_deeploc_signals_canonical": "SP", "cmp_localization_deeploc_signals_isoform": "-",
        "cmp_localization_deeploc_signals_changed": True,
        "cmp_localization_deeploc_membrane_canonical": "M", "cmp_localization_deeploc_membrane_isoform": "M",
        "cmp_localization_deeploc_membrane_changed": float("nan"),  # not evaluated
    }, "L1_localization_change")
    assert rows["Sorting signals"]["hot"] is True
    assert rows["Membrane"]["hot"] is False


def test_c3_ratio_is_the_modules_enrichment() -> None:
    rows = _rows_for({
        "isoform_conservation_phylop_unique_region_mean": 1.5,
        "isoform_conservation_phylop_shared_region_mean": 2.0,
        "isoform_conservation_phylop_enrichment": 0.7,  # the module's, not 1.5 / 2.0
    }, "C3_phylop_coding_selection", fired=False)
    assert rows["phyloP mean"]["cols"][2] == "0.7×"


def test_c3_ratio_is_blank_where_a_negative_mean_flips_it() -> None:
    # -0.5 / -1.0 = 0.5× would read as "half as conserved" for an accelerated region.
    rows = _rows_for({
        "isoform_conservation_phylop_unique_region_mean": -0.5,
        "isoform_conservation_phylop_shared_region_mean": -1.0,
        "isoform_conservation_phylop_enrichment": 0.5,
    }, "C3_phylop_coding_selection", fired=False)
    assert rows["phyloP mean"]["cols"][2] == "—"


def test_d1_counts_a_line_with_an_expression_record() -> None:
    rows = _rows_for({
        "expr_HeLa_raw_count": 12, "expr_HeLa_cpm": 3.0,  # called, no p-value stored
        "canonical_expr_K562_cpm": 5.0,  # K562: canonical only, not called
    }, "D1_multi_cell_line")
    assert rows["HeLa"]["hot"] is True
    assert rows["K562"]["hot"] is False


def test_separate_orf_with_a_shared_region_is_not_called_orf() -> None:
    """A uoORF that kept a shared region (in-frame match) gets the generic noun."""
    from types import SimpleNamespace

    from swissisoform_site.data import _region_noun, has_shared_region

    whole = SimpleNamespace(orf_type="uoorf", diff_space="isoform", diff_end=60, isoform_len=60)
    partial = SimpleNamespace(orf_type="uoorf", diff_space="isoform", diff_end=20, isoform_len=60)
    assert not has_shared_region(whole) and _region_noun(whole) == "ORF"
    assert has_shared_region(partial) and _region_noun(partial) == "differential region"
