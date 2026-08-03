"""Tests for the M-category LLM reader tools over variants_long.parquet.

No network: these exercise the pandas filters and the JSON shapes the tool loop
feeds back to the model. The loop itself is tested in
``test_run_llm_interpretation.py`` with a fake SDK client.
"""

from pathlib import Path

import pandas as pd
import pytest

from swissisoform.site import tools as t
from swissisoform.site.evidence import clinsig_family

TIS_A = "chr1:100:+:ATG:ENST_A"
TIS_B = "chr2:200:+:CTG:ENST_B"


def _row(**kw):
    """One variants_long row with every column present, overridable by kwarg."""
    base = {
        "tis_id": TIS_A,
        "gene_name": "GENE_A",
        "variant_id": "v0",
        "source": "gnomAD",
        "chrom": "chr1",
        "genomic_pos": 1000,
        "ref": "G",
        "alt": "A",
        "hgvsp": "p.Ala1Thr",
        "consequence": "missense_variant",
        "clinical_significance": None,
        "protein_pos": 10.0,
        "isoform_protein_pos": 10.0,
        "in_isoform_unique": True,
        "in_isoform_shared": False,
        "allele_frequency": 1e-5,
        "aa_ref": "A",
        "aa_alt": "T",
        "isoform_aa_ref": "A",
        "isoform_aa_alt": "T",
        "isoform_consequence": "missense_variant",
        "hgvsc": "c.1G>A",
        "rs_id": None,
        "cosmic_sample_count": None,
        "clinvar_title": None,
        "am_class": None,
        "am_pathogenicity": None,
        "plm_delta_llr": None,
        "plm_llr_wt": None,
        "plm_llr_alt": None,
        "effect_damaging": None,
    }
    base.update(kw)
    return base


@pytest.fixture
def variants(tmp_path: Path) -> Path:
    """A small table covering every ClinVar spelling, both regions, all 3 sources."""
    rows = [
        # ClinVar, unique region — the three pathogenic spellings plus the rest.
        _row(variant_id="cv1", source="ClinVar", clinical_significance="Pathogenic",
             isoform_protein_pos=12.0, am_pathogenicity=0.95, plm_delta_llr=-8.0,
             effect_damaging=True, am_class="pathogenic"),
        _row(variant_id="cv2", source="ClinVar",
             clinical_significance="Pathogenic/Likely pathogenic",
             isoform_protein_pos=12.0, am_pathogenicity=0.88, plm_delta_llr=-6.0,
             effect_damaging=True, am_class="pathogenic"),
        _row(variant_id="cv3", source="ClinVar", clinical_significance="Likely pathogenic",
             isoform_protein_pos=12.0, am_pathogenicity=0.80, plm_delta_llr=-4.0,
             effect_damaging=True, am_class="pathogenic"),
        _row(variant_id="cv4", source="ClinVar",
             clinical_significance="Conflicting classifications of pathogenicity",
             isoform_protein_pos=30.0),
        _row(variant_id="cv5", source="ClinVar", clinical_significance="Uncertain significance",
             isoform_protein_pos=31.0),
        _row(variant_id="cv6", source="ClinVar", clinical_significance="Likely benign",
             isoform_protein_pos=32.0),
        _row(variant_id="cv7", source="ClinVar", clinical_significance="Benign/Likely benign",
             isoform_protein_pos=33.0),
        # gnomAD, unique region — no clinical significance, one with no position.
        _row(variant_id="g1", isoform_protein_pos=12.0, allele_frequency=2e-5),
        _row(variant_id="g2", isoform_protein_pos=None, allele_frequency=3e-5),
        # COSMIC, shared region.
        _row(variant_id="cs1", source="COSMIC", in_isoform_unique=False,
             in_isoform_shared=True, isoform_protein_pos=200.0, cosmic_sample_count=42.0,
             consequence="stop_gained"),
        _row(variant_id="cs2", source="COSMIC", in_isoform_unique=False,
             in_isoform_shared=True, isoform_protein_pos=201.0, cosmic_sample_count=7.0),
        # A second isoform, to prove tis_id isolation.
        _row(tis_id=TIS_B, gene_name="GENE_B", variant_id="other1",
             isoform_protein_pos=5.0),
    ]
    p = tmp_path / "variants_long.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    # lru_cache is keyed on the resolved path; tmp_path is unique per test, but
    # clear anyway so a reused path can never serve a stale frame.
    t._load.cache_clear()
    t._index_by_tis.cache_clear()
    return p


# ── clinsig_family ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Pathogenic", "pathogenic"),
        ("Pathogenic/Likely pathogenic", "pathogenic"),
        ("Likely pathogenic", "pathogenic"),
        ("Conflicting classifications of pathogenicity", "conflicting"),
        ("Uncertain significance", "uncertain"),
        ("Benign", "benign"),
        ("Benign/Likely benign", "benign"),
        ("Likely benign", "benign"),
        (None, "none"),
        ("", "none"),
        (float("nan"), "none"),
    ],
)
def test_clinsig_family_buckets_every_observed_spelling(value, expected):
    assert clinsig_family(value) == expected


def test_conflicting_is_not_counted_as_pathogenic():
    """'Conflicting classifications of pathogenicity' contains 'pathogenic'."""
    assert clinsig_family("Conflicting classifications of pathogenicity") != "pathogenic"


# ── _iso filtering ────────────────────────────────────────────────────────


def test_iso_isolates_one_isoform(variants):
    assert len(t._iso(variants, TIS_A)) == 11
    assert len(t._iso(variants, TIS_B)) == 1


def test_iso_unknown_tis_returns_empty_frame_with_columns(variants):
    out = t._iso(variants, "chrZ:1:+:ATG:NOPE")
    assert len(out) == 0
    assert "clinical_significance" in out.columns


def test_iso_region_filter(variants):
    assert len(t._iso(variants, TIS_A, region="unique")) == 9
    assert len(t._iso(variants, TIS_A, region="shared")) == 2
    # "any" and None both mean the whole per-isoform table.
    assert len(t._iso(variants, TIS_A, region="any")) == 11
    assert len(t._iso(variants, TIS_A, region=None)) == 11


def test_iso_pathogenic_family_catches_all_three_spellings(variants):
    """Regression: equality against "Pathogenic" would return 1 of 3 rows here.

    On real data that is a ~21% undercount on the exact question the M category
    exists to answer.
    """
    got = {v for v in t._iso(variants, TIS_A, clinsig="pathogenic")["variant_id"]}
    assert got == {"cv1", "cv2", "cv3"}


def test_iso_benign_family_catches_all_spellings(variants):
    got = {v for v in t._iso(variants, TIS_A, clinsig="benign")["variant_id"]}
    assert got == {"cv6", "cv7"}


def test_iso_clinsig_any_means_any_asserted_call(variants):
    got = {v for v in t._iso(variants, TIS_A, clinsig="any")["variant_id"]}
    assert got == {"cv1", "cv2", "cv3", "cv4", "cv5", "cv6", "cv7"}


def test_iso_clinsig_filter_excludes_gnomad_and_cosmic(variants):
    """clinical_significance is ClinVar-only, so any clinsig filter drops the rest."""
    sources = set(t._iso(variants, TIS_A, clinsig="any")["source"])
    assert sources == {"ClinVar"}
    assert "gnomAD" not in sources and "COSMIC" not in sources


def test_iso_source_filter_reaches_cosmic_recurrence(variants):
    out = t._iso(variants, TIS_A, source="COSMIC")
    assert set(out["variant_id"]) == {"cs1", "cs2"}
    assert out["cosmic_sample_count"].max() == 42.0


def test_iso_consequence_substring_match(variants):
    assert set(t._iso(variants, TIS_A, consequence="stop")["variant_id"]) == {"cs1"}
    assert len(t._iso(variants, TIS_A, consequence="missense")) == 10


def test_iso_rejects_unknown_filter_values(variants):
    for kwargs in ({"region": "middle"}, {"clinsig": "nasty"}, {"source": "dbSNP"}):
        with pytest.raises(ValueError):
            t._iso(variants, TIS_A, **kwargs)


# ── variant_position_histogram ────────────────────────────────────────────


def test_histogram_counts_and_span(variants):
    out = t.variant_position_histogram(variants, TIS_A, region="unique")
    assert out["n"] == 9
    assert out["n_with_position"] == 8  # g2 has no isoform_protein_pos
    assert out["n_missing_pos"] == 1
    assert out["residue_span"] == [12, 33]
    assert out["per_residue_counts"][12] == 4  # cv1, cv2, cv3, g1


def test_histogram_surfaces_the_hotspot(variants):
    """The clustering signal an enrichment ratio cannot express."""
    out = t.variant_position_histogram(variants, TIS_A, region="unique",
                                       clinsig="pathogenic")
    assert out["top_positions"][0] == {"pos": 12, "count": 3}
    assert out["frac_in_top10_positions"] == 1.0
    assert out["n_distinct_positions"] == 1


def test_histogram_empty_result_is_well_formed(variants):
    out = t.variant_position_histogram(variants, TIS_A, region="shared",
                                       clinsig="pathogenic")
    assert out["n"] == 0
    assert out["residue_span"] is None
    assert out["per_residue_counts"] == {}
    assert out["frac_in_top10_positions"] is None


def test_histogram_all_positions_null_is_well_formed(tmp_path):
    p = tmp_path / "variants_long.parquet"
    pd.DataFrame([_row(variant_id="x", isoform_protein_pos=None)]).to_parquet(p, index=False)
    t._load.cache_clear()
    t._index_by_tis.cache_clear()
    out = t.variant_position_histogram(p, TIS_A)
    assert out["n"] == 1
    assert out["n_with_position"] == 0
    assert out["n_missing_pos"] == 1
    assert out["residue_span"] is None


def test_histogram_truncates_but_keeps_stats_over_full_distribution(tmp_path):
    """Truncation caps the emitted histogram, never the concentration stats."""
    n_positions = t.MAX_HISTOGRAM_POSITIONS + 50
    rows = [_row(variant_id=f"v{i}", isoform_protein_pos=float(i)) for i in range(n_positions)]
    # Make position 0 a genuine hotspot so top_positions has something to find.
    rows += [_row(variant_id=f"hot{i}", isoform_protein_pos=0.0) for i in range(5)]
    p = tmp_path / "variants_long.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    t._load.cache_clear()
    t._index_by_tis.cache_clear()

    out = t.variant_position_histogram(p, TIS_A)
    assert out["per_residue_counts_truncated"] is True
    assert len(out["per_residue_counts"]) == t.MAX_HISTOGRAM_POSITIONS
    # Stats reflect all 150 distinct positions, not the 100 emitted.
    assert out["n_distinct_positions"] == n_positions
    assert out["residue_span"] == [0, n_positions - 1]
    assert out["top_positions"][0] == {"pos": 0, "count": 6}


# ── Coordinate space (regression: the flagship reader silently returned
#    nothing for every truncation's unique region) ───────────────────────────


@pytest.fixture
def truncation_variants(tmp_path: Path) -> Path:
    """A truncation, whose unique-region rows carry canonical positions only.

    Mirrors the real table: over the differential region a truncation populates
    ``protein_pos`` (the removed segment exists only in the canonical protein)
    and leaves ``isoform_protein_pos`` null; an extension does the reverse.
    """
    rows = [
        _row(variant_id="u1", source="ClinVar", clinical_significance="Pathogenic",
             protein_pos=5.0, isoform_protein_pos=None),
        _row(variant_id="u2", source="ClinVar", clinical_significance="Pathogenic",
             protein_pos=5.0, isoform_protein_pos=None),
        _row(variant_id="u3", protein_pos=9.0, isoform_protein_pos=None),
        _row(variant_id="s1", in_isoform_unique=False, in_isoform_shared=True,
             protein_pos=120.0, isoform_protein_pos=100.0),
    ]
    p = tmp_path / "variants_long.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    t._load.cache_clear()
    t._index_by_tis.cache_clear()
    return p


def test_position_column_follows_the_isoform_kind():
    assert t._position_column("truncated") == ("protein_pos", "canonical")
    assert t._position_column("extended") == ("isoform_protein_pos", "isoform")
    assert t._position_column("uorf") == ("isoform_protein_pos", "isoform")
    assert t._position_column(None) == ("isoform_protein_pos", "isoform")


def test_histogram_finds_the_truncation_hotspot(truncation_variants):
    """Keying on isoform_protein_pos alone would report an empty distribution."""
    out = t.variant_position_histogram(truncation_variants, TIS_A, region="unique",
                                       orf_type="truncated")
    assert out["n"] == 3
    assert out["n_with_position"] == 3
    assert out["n_missing_pos"] == 0
    assert out["top_positions"][0] == {"pos": 5, "count": 2}
    assert out["residue_span"] == [5, 9]


def test_histogram_labels_the_coordinate_space(truncation_variants):
    """Residue numbers are meaningless without knowing which protein they index."""
    trunc = t.variant_position_histogram(truncation_variants, TIS_A, region="unique",
                                         orf_type="truncated")
    assert trunc["position_space"] == "canonical"
    ext = t.variant_position_histogram(truncation_variants, TIS_A, region="shared",
                                       orf_type="extended")
    assert ext["position_space"] == "isoform"


def test_histogram_on_a_truncation_without_orf_type_reports_the_gap_honestly(
    truncation_variants,
):
    """Wrong space -> empty, but n_missing_pos says so rather than implying zero variants."""
    out = t.variant_position_histogram(truncation_variants, TIS_A, region="unique")
    assert out["n"] == 3
    assert out["n_with_position"] == 0
    assert out["n_missing_pos"] == 3


def test_query_variants_sorts_by_the_populated_position_column(truncation_variants):
    out = t.query_variants(truncation_variants, TIS_A, region="unique",
                           sort_by="position", orf_type="truncated")
    assert out["position_space"] == "canonical"
    assert [r["variant_id"] for r in out["rows"]] == ["u1", "u2", "u3"]


def test_query_variants_returns_both_position_columns(truncation_variants):
    """The model can see which space each row is resolvable in."""
    row = t.query_variants(truncation_variants, TIS_A, region="shared",
                           orf_type="truncated")["rows"][0]
    assert row["protein_pos"] == 120
    assert row["isoform_protein_pos"] == 100


def test_dispatch_binds_orf_type_for_positional_readers(truncation_variants):
    d = t.make_m_dispatch(truncation_variants, TIS_A, "truncated")
    hist = d("variant_position_histogram", {"region": "unique"})
    assert hist["n_with_position"] == 3
    assert hist["position_space"] == "canonical"
    assert d("query_variants", {"region": "unique"})["position_space"] == "canonical"


def test_dispatch_ignores_a_model_supplied_orf_type(truncation_variants):
    """orf_type is a property of the annotation, not a knob the model may turn."""
    d = t.make_m_dispatch(truncation_variants, TIS_A, "truncated")
    out = d("variant_position_histogram", {"region": "unique", "orf_type": "extended"})
    assert out["position_space"] == "canonical"


# ── variant_effect_stats ──────────────────────────────────────────────────


def test_effect_stats_reports_scored_count_separately_from_rows(variants):
    """AlphaMissense scores missense SNVs only, so a mean over 3 of 9 rows is
    not a statement about the 9.
    """
    out = t.variant_effect_stats(variants, TIS_A, region="unique")
    assert out["n_rows"] == 9
    assert out["alphamissense"]["n_scored"] == 3
    assert out["esm_llr"]["n_scored"] == 3


def test_effect_stats_all_null_column_reports_zero_scored(variants):
    out = t.variant_effect_stats(variants, TIS_A, source="COSMIC")
    assert out["n_rows"] == 2
    assert out["alphamissense"]["n_scored"] == 0
    assert out["alphamissense"]["mean"] is None


def test_effect_stats_damaging_counts_split_true_false_null(variants):
    out = t.variant_effect_stats(variants, TIS_A, region="unique")
    assert out["effect_damaging"]["true"] == 3
    assert out["effect_damaging"]["null"] == 6


def test_effect_stats_class_counts(variants):
    out = t.variant_effect_stats(variants, TIS_A, clinsig="pathogenic")
    assert out["alphamissense"]["class_counts"] == {"pathogenic": 3}


def test_effect_caveat_fires_on_extension(variants):
    out = t.variant_effect_stats(variants, TIS_A, region="unique", orf_type="extended")
    assert "caveat" in out
    assert "never canonical coding" in out["caveat"]


def test_effect_caveat_absent_on_truncation(variants):
    """The removed region IS canonical coding, so the canonical-frame scores hold."""
    out = t.variant_effect_stats(variants, TIS_A, region="unique", orf_type="truncated")
    assert "caveat" not in out


def test_effect_caveat_fires_on_separate_orf(variants):
    out = t.variant_effect_stats(variants, TIS_A, orf_type="uorf")
    assert "caveat" in out
    assert "reading frame" in out["caveat"]


# ── query_variants ────────────────────────────────────────────────────────


def test_query_variants_reports_true_total_alongside_capped_rows(variants):
    out = t.query_variants(variants, TIS_A, limit=2)
    assert out["n_matched"] == 11
    assert out["n_returned"] == 2
    assert out["truncated"] is True


def test_query_variants_not_truncated_when_all_returned(variants):
    out = t.query_variants(variants, TIS_A, region="shared")
    assert out["n_matched"] == 2
    assert out["truncated"] is False


def test_query_variants_limit_is_clamped_to_max(variants):
    out = t.query_variants(variants, TIS_A, limit=10_000)
    assert out["n_returned"] <= t.MAX_QUERY_LIMIT


def test_query_variants_sorts_by_position_ascending(variants):
    rows = t.query_variants(variants, TIS_A, region="shared", sort_by="position")["rows"]
    assert [r["isoform_protein_pos"] for r in rows] == [200, 201]


def test_query_variants_sorts_most_damaging_llr_first(variants):
    """plm_delta_llr is negative for damaging substitutions, so ascending is worst-first."""
    rows = t.query_variants(variants, TIS_A, clinsig="pathogenic",
                            sort_by="plm_delta_llr")["rows"]
    assert [r["variant_id"] for r in rows] == ["cv1", "cv2", "cv3"]


def test_query_variants_sorts_cosmic_recurrence_descending(variants):
    rows = t.query_variants(variants, TIS_A, source="COSMIC",
                            sort_by="cosmic_sample_count")["rows"]
    assert [r["variant_id"] for r in rows] == ["cs1", "cs2"]


def test_query_variants_is_deterministic_across_calls(variants):
    """Ties break on variant_id, so repeated calls cannot reorder."""
    first = t.query_variants(variants, TIS_A, sort_by="position", limit=11)["rows"]
    second = t.query_variants(variants, TIS_A, sort_by="position", limit=11)["rows"]
    assert [r["variant_id"] for r in first] == [r["variant_id"] for r in second]


def test_query_variants_rows_are_json_native(variants):
    """NaN/numpy types must not reach json.dumps in the tool result."""
    import json

    out = t.query_variants(variants, TIS_A)
    json.dumps(out)  # raises TypeError if a numpy scalar leaked through
    assert out["rows"][0]["cosmic_sample_count"] is None


def test_query_variants_rejects_unknown_sort(variants):
    with pytest.raises(ValueError):
        t.query_variants(variants, TIS_A, sort_by="vibes")


# ── require_variants_long ─────────────────────────────────────────────────


def test_require_variants_long_names_the_producing_stage(tmp_path):
    with pytest.raises(t.VariantsLongMissing) as e:
        t.require_variants_long(tmp_path / "nope.parquet")
    msg = str(e.value)
    assert "build_evidence_records.py" in msg
    assert "--no-tools" in msg


def test_require_variants_long_returns_path_when_present(variants):
    assert t.require_variants_long(variants) == variants


# ── dispatch ──────────────────────────────────────────────────────────────


def test_dispatch_routes_each_reader(variants):
    d = t.make_m_dispatch(variants, TIS_A, "truncated")
    assert d("variant_position_histogram", {"region": "unique"})["n"] == 9
    assert d("query_variants", {"limit": 1})["n_matched"] == 11
    assert d("variant_effect_stats", {})["n_rows"] == 11


def test_dispatch_translates_region_any_to_no_filter(variants):
    d = t.make_m_dispatch(variants, TIS_A, "truncated")
    assert d("query_variants", {"region": "any"})["n_matched"] == 11


def test_dispatch_injects_orf_type_for_the_caveat(variants):
    d = t.make_m_dispatch(variants, TIS_A, "extended")
    assert "caveat" in d("variant_effect_stats", {})


def test_dispatch_unknown_tool_returns_error_not_exception(variants):
    d = t.make_m_dispatch(variants, TIS_A, "truncated")
    out = d("drop_table", {})
    assert "error" in out
    assert "Unknown tool" in out["error"]


def test_dispatch_bad_argument_returns_error_so_model_can_retry(variants):
    d = t.make_m_dispatch(variants, TIS_A, "truncated")
    assert "error" in d("query_variants", {"region": "middle"})
    assert "error" in d("query_variants", {"nonexistent_kwarg": 1})


# ── tool schemas ──────────────────────────────────────────────────────────


def test_every_tool_schema_is_well_formed():
    for tool in t.M_TOOLS:
        assert tool["name"] and tool["description"]
        assert tool["input_schema"]["type"] == "object"
        assert "properties" in tool["input_schema"]


def test_emit_verdict_is_terminal_and_excluded_from_data_tools():
    names = {tool["name"] for tool in t.M_TOOLS}
    assert t.EMIT_VERDICT in names
    assert t.EMIT_VERDICT not in t.DATA_TOOL_NAMES
    assert t.DATA_TOOL_NAMES == names - {t.EMIT_VERDICT}


def test_emit_verdict_requires_verdict_reasoning_and_evidence():
    emit = next(x for x in t.M_TOOLS if x["name"] == t.EMIT_VERDICT)
    assert set(emit["input_schema"]["required"]) == {"verdict", "reasoning", "evidence_used"}
    enum = emit["input_schema"]["properties"]["verdict"]["enum"]
    assert enum == ["interesting", "neutral", "not_interesting"]


def test_emit_verdict_is_strict_and_schema_stays_strict_compatible():
    """The API rejects a strict tool whose schema it cannot constrain sampling to.

    Guarded here rather than discovered as a 400 mid-run: strict needs
    additionalProperties:false on every object, and rejects minItems above 1.
    """
    emit = next(x for x in t.M_TOOLS if x["name"] == t.EMIT_VERDICT)
    assert emit["strict"] is True
    assert_strict_compatible(emit["input_schema"])


def assert_strict_compatible(schema, path="input_schema"):
    if isinstance(schema, dict):
        if schema.get("type") == "object" and "properties" in schema:
            assert schema.get("additionalProperties") is False, f"{path} allows extra keys"
        if schema.get("type") == "array":
            assert schema.get("minItems", 0) <= 1, f"{path} uses minItems>1"
        for key, value in schema.items():
            assert_strict_compatible(value, f"{path}.{key}")
    elif isinstance(schema, list):
        for i, value in enumerate(schema):
            assert_strict_compatible(value, f"{path}[{i}]")
