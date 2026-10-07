"""Tests for the canonical-vs-isoform comparator."""

from __future__ import annotations

from swissisoform.compare.comparator import (
    _PREDICTORS,
    Comparator,
    _categorical_changes,
    _hits_overlapping,
    compare_genes,
)
from swissisoform.models import (
    DifferentialRegion,
    Gene,
    ORFType,
    TranslationInitiationSite,
)
from swissisoform.modules.biophysics import BiophysicsModule


def _make_site(
    *,
    orf_type: ORFType,
    isoform_protein: str,
    diff_region: DifferentialRegion,
    isoform_annotations: dict | None = None,
) -> TranslationInitiationSite:
    return TranslationInitiationSite(
        tis_id=f"chr1:0:+:ATG:T1:{orf_type.value}",
        gene_name="TEST",
        transcript_id="T1",
        chrom="chr1",
        position=0,
        strand="+",
        start_codon="ATG",
        orf_type=orf_type,
        isoform_protein=isoform_protein,
        diff_region=diff_region,
        isoform_annotations=isoform_annotations or {},
    )


def _make_gene(
    canonical_protein: str, tis_sites: list, canonical_annotations: dict | None = None
) -> Gene:
    return Gene(
        gene_name="TEST",
        gene_id="G1",
        canonical_transcript_id="T1",
        canonical_protein=canonical_protein,
        tis_sites=tis_sites,
        canonical_annotations=canonical_annotations or {},
    )


# ---------------------------------------------------------------------------
# _hits_overlapping
# ---------------------------------------------------------------------------


class TestHitsOverlapping:
    def test_keeps_hits_inside_region(self):
        hits = [
            {"pos": 5, "end": 8, "name": "A"},
            {"pos": 20, "end": 23, "name": "B"},
        ]
        assert _hits_overlapping(hits, 0, 10) == [hits[0]]

    def test_keeps_hits_partially_overlapping(self):
        hits = [{"pos": 8, "end": 12, "name": "X"}]
        assert len(_hits_overlapping(hits, 0, 10)) == 1

    def test_drops_hits_outside_region(self):
        hits = [{"pos": 15, "end": 18}]
        assert _hits_overlapping(hits, 0, 10) == []

    def test_handles_none_coords(self):
        hits = [{"pos": 5, "end": 8}]
        assert _hits_overlapping(hits, None, 10) == []
        assert _hits_overlapping(hits, 0, None) == []

    def test_empty_region(self):
        hits = [{"pos": 5, "end": 8}]
        assert _hits_overlapping(hits, 5, 5) == []

    def test_skips_non_positional_hits(self):
        hits = [{"name": "summary_only"}, {"pos": 5, "end": 7, "name": "A"}]
        subset = _hits_overlapping(hits, 0, 10)
        assert len(subset) == 1
        assert subset[0]["name"] == "A"


# ---------------------------------------------------------------------------
# Scalar deltas
# ---------------------------------------------------------------------------


class TestScalarDeltas:
    def test_isoform_minus_canonical_for_numerics(self):
        gene = _make_gene(
            "MAAA",
            tis_sites=[
                _make_site(
                    orf_type=ORFType.EXTENDED,
                    isoform_protein="MKKMAAA",
                    diff_region=DifferentialRegion(
                        isoform_start=0,
                        isoform_end=3,
                        sequence="MKK",
                        confidence="tail_verified",
                    ),
                    isoform_annotations={"biophysics": {"pI": 9.0, "length": 7, "name": "iso"}},
                )
            ],
            canonical_annotations={"biophysics": {"pI": 6.0, "length": 4, "name": "iso"}},
        )
        Comparator().compare([gene])
        cmp = gene.tis_sites[0].comparison["biophysics"]
        assert cmp["pI_delta"] == 3.0
        assert cmp["length_delta"] == 3
        # No delta on non-numeric (name)
        assert "name_delta" not in cmp

    def test_ignores_bool_fields(self):
        gene = _make_gene(
            "MAA",
            tis_sites=[
                _make_site(
                    orf_type=ORFType.EXTENDED,
                    isoform_protein="MKMAA",
                    diff_region=DifferentialRegion(
                        isoform_start=0,
                        isoform_end=2,
                        sequence="MK",
                        confidence="tail_verified",
                    ),
                    isoform_annotations={"loc": {"flag": True, "score": 0.8}},
                )
            ],
            canonical_annotations={"loc": {"flag": False, "score": 0.5}},
        )
        Comparator().compare([gene])
        cmp = gene.tis_sites[0].comparison["loc"]
        assert "score_delta" in cmp
        # Bools compared via categorical change path, not numeric delta
        assert "flag_delta" not in cmp
        assert cmp["flag_changed"] is True
        assert cmp["flag_canonical"] is False
        assert cmp["flag_isoform"] is True


SIGNALP = _PREDICTORS["signalp"]


class TestOneSidedCategorical:
    """A missing value is "ran, found nothing" or "did not run", and they differ."""

    @staticmethod
    def _signalp(prediction: str | None, site: str | None) -> dict:
        return {"signalp_prediction": prediction, "signalp_cleavage_site": site}

    def test_value_on_one_side_with_both_runs_is_a_change(self):
        """No signal peptide on the canonical, one on the isoform: a gain."""
        out = _categorical_changes(
            self._signalp("OTHER", None), self._signalp("SP", "CS pos: 23-24"), SIGNALP
        )
        assert out["signalp_cleavage_site_changed"] is True

    def test_value_lost_with_both_runs_is_a_change(self):
        out = _categorical_changes(
            self._signalp("SP", "CS pos: 20-21"), self._signalp("OTHER", None), SIGNALP
        )
        assert out["signalp_cleavage_site_changed"] is True

    def test_no_value_on_either_side_with_both_runs_is_no_change(self):
        out = _categorical_changes(
            self._signalp("OTHER", None), self._signalp("OTHER", None), SIGNALP
        )
        assert out["signalp_cleavage_site_changed"] is False

    def test_a_side_that_did_not_run_is_unknown(self):
        """The predictor missed the canonical (no cache): nothing is known there."""
        out = _categorical_changes(
            self._signalp(None, None), self._signalp("SP", "CS pos: 4"), SIGNALP
        )
        assert out["signalp_cleavage_site_changed"] is None
        assert out["signalp_prediction_changed"] is None

    def test_both_values_present_compare_by_value(self):
        same = _categorical_changes(
            self._signalp("SP", "CS pos: 9"), self._signalp("SP", "CS pos: 9"), SIGNALP
        )
        moved = _categorical_changes(
            self._signalp("SP", "CS pos: 9"), self._signalp("SP", "CS pos: 4"), SIGNALP
        )
        assert same["signalp_cleavage_site_changed"] is False
        assert moved["signalp_cleavage_site_changed"] is True

    def test_nan_counts_as_absent(self):
        """DeepLoc's signals column reads back as NaN, not None, when empty."""
        out = _categorical_changes(
            {"deeploc_prediction": "Cytoplasm", "deeploc_signals": float("nan")},
            {"deeploc_prediction": "Nucleus", "deeploc_signals": "Nuclear localization signal"},
            _PREDICTORS["localization"],
        )
        assert out["deeploc_signals_changed"] is True

    def test_a_field_without_a_run_indicator_stays_unknown_when_one_sided(self):
        out = _categorical_changes({"label": None}, {"label": "x"})
        assert out["label_changed"] is None
        both = _categorical_changes({"label": None}, {"label": None})
        assert both["label_changed"] is False

    def test_a_declared_module_flags_only_its_categorical_fields(self):
        """Human TargetP never reports cTP: a probability None on both sides is no category."""
        gene = _make_gene(
            "MAA",
            tis_sites=[
                _make_site(
                    orf_type=ORFType.EXTENDED,
                    isoform_protein="MKMAA",
                    diff_region=DifferentialRegion(
                        isoform_start=0, isoform_end=2, sequence="MK",
                        confidence="tail_verified",
                    ),
                    isoform_annotations={
                        "targetp": {
                            "targetp_prediction": "noTP", "targetp_ctp_prob": None,
                            "targetp_cleavage_site": None,
                        }
                    },
                )
            ],
            canonical_annotations={
                "targetp": {
                    "targetp_prediction": "noTP", "targetp_ctp_prob": None,
                    "targetp_cleavage_site": None,
                }
            },
        )
        Comparator().compare([gene])
        cmp = gene.tis_sites[0].comparison["targetp"]
        assert "targetp_ctp_prob_changed" not in cmp
        assert cmp["targetp_prediction_changed"] is False
        assert cmp["targetp_cleavage_site_changed"] is False

    def test_an_undeclared_module_has_no_ran_check(self):
        """Only declared predictors read a missing value as absent."""
        out = _categorical_changes(
            {"tool_prediction": "x", "tool_site": None}, {"tool_prediction": "x", "tool_site": "y"}
        )
        assert out["tool_site_changed"] is None

    def test_an_undeclared_field_numeric_on_one_side_reads_as_unknown(self):
        """A pane that failed (isoform None) marks the field unknown, not absent."""
        out = _categorical_changes({"gravy": 0.42}, {"gravy": None})
        assert out["gravy_changed"] is None
        assert (out["gravy_canonical"], out["gravy_isoform"]) == (0.42, None)
        # Numeric on both sides is a scalar: a delta, not a flag.
        assert "gravy_changed" not in _categorical_changes({"gravy": 0.4}, {"gravy": 0.5})

    def test_every_predictor_declares_a_ran_field_among_its_calls(self):
        for name, pred in _PREDICTORS.items():
            assert pred.ran in pred.fields, name


# ---------------------------------------------------------------------------
# Positional hit subsetting
# ---------------------------------------------------------------------------


class TestPositionalSubset:
    def test_extension_filters_isoform_hits_to_diff_region(self):
        # Extension: 10 aa added at the N-terminus
        gene = _make_gene(
            "MAAAAAA",
            tis_sites=[
                _make_site(
                    orf_type=ORFType.EXTENDED,
                    isoform_protein="MXXYYZZZKMAAAAAA",
                    diff_region=DifferentialRegion(
                        isoform_start=0,
                        isoform_end=9,
                        sequence="MXXYYZZZK",
                        confidence="tail_verified",
                    ),
                    isoform_annotations={
                        "motifs": {
                            "hits": [
                                {"pos": 2, "end": 4, "name": "in_diff"},
                                {"pos": 11, "end": 14, "name": "outside_diff"},
                            ],
                            "summary": {"total": 2},
                        }
                    },
                )
            ],
            canonical_annotations={"motifs": {"hits": [], "summary": {"total": 0}}},
        )
        Comparator().compare([gene])
        cmp = gene.tis_sites[0].comparison["motifs"]
        assert cmp["n_hits_in_diff_region"] == 1
        assert cmp["hits_in_diff_region"][0]["name"] == "in_diff"
        assert cmp["hits_source_pane"] == "isoform"

    def test_truncation_filters_canonical_hits(self):
        # Truncation: 5 aa lost from N-terminus of canonical
        gene = _make_gene(
            "MMPQRSTUVWX",
            tis_sites=[
                _make_site(
                    orf_type=ORFType.TRUNCATED,
                    isoform_protein="STUVWX",
                    diff_region=DifferentialRegion(
                        canonical_start=0,
                        canonical_end=5,
                        sequence="MMPQR",
                        confidence="tail_verified",
                    ),
                    isoform_annotations={"clinical": {"hits": [], "summary": {}}},
                )
            ],
            canonical_annotations={
                "clinical": {
                    "hits": [
                        {"pos": 2, "end": 3, "variant_id": "v_in_lost"},
                        {"pos": 8, "end": 9, "variant_id": "v_retained"},
                    ],
                    "summary": {"total": 2},
                }
            },
        )
        Comparator().compare([gene])
        cmp = gene.tis_sites[0].comparison["clinical"]
        assert cmp["n_hits_in_diff_region"] == 1
        assert cmp["hits_in_diff_region"][0]["variant_id"] == "v_in_lost"
        assert cmp["hits_source_pane"] == "canonical"

    def test_truncation_massspec_does_not_credit_canonical_peptides(self):
        """D3 fix: canonical lost-region peptides are NOT isoform existence evidence.

        On a truncation, massspec (an isoform-existence module) must return an
        empty diff-region subset with source_pane='isoform' — a canonical
        tryptic peptide in the lost N-terminus evidences the canonical form,
        not the isoform.
        """
        gene = _make_gene(
            "MMPQRSTUVWX",
            tis_sites=[
                _make_site(
                    orf_type=ORFType.TRUNCATED,
                    isoform_protein="STUVWX",
                    diff_region=DifferentialRegion(
                        canonical_start=0,
                        canonical_end=5,
                        sequence="MMPQR",
                        confidence="tail_verified",
                    ),
                    isoform_annotations={"massspec": {"hits": [], "summary": {}}},
                )
            ],
            canonical_annotations={
                "massspec": {
                    "hits": [
                        {"pos": 2, "end": 3, "peptide": "PQR"},
                    ],
                    "summary": {"total": 1},
                }
            },
        )
        Comparator().compare([gene])
        cmp = gene.tis_sites[0].comparison["massspec"]
        assert cmp["n_hits_in_diff_region"] == 0
        assert cmp["hits_in_diff_region"] == []
        assert cmp["hits_source_pane"] == "isoform"


# ---------------------------------------------------------------------------
# S1 — real InterPro domain gain/loss in the diff region
# ---------------------------------------------------------------------------


def _ips_hit(pos: int, *, name: str, interpro_id: str | None, db: str = "Pfam") -> dict:
    return {"pos": pos, "end": pos + 20, "name": name, "interpro_id": interpro_id, "db": db}


class TestRealDomainsChanged:
    def test_gained_real_domain_on_extension(self):
        """A real InterPro domain starting in the isoform diff region, absent
        from canonical, counts as one gained domain.
        """
        gene = _make_gene(
            "MAAAAAA",
            tis_sites=[
                _make_site(
                    orf_type=ORFType.EXTENDED,
                    isoform_protein="MXXYYZZZKMAAAAAA",
                    diff_region=DifferentialRegion(
                        isoform_start=0,
                        isoform_end=9,
                        sequence="MXXYYZZZK",
                        confidence="tail_verified",
                    ),
                    isoform_annotations={
                        "interproscan": {
                            "hits": [_ips_hit(2, name="PF_new", interpro_id="IPR_NEW")],
                            "summary": {"status": "ok"},
                        }
                    },
                )
            ],
            canonical_annotations={
                "interproscan": {"hits": [], "summary": {"status": "ok"}}
            },
        )
        Comparator().compare([gene])
        cmp = gene.tis_sites[0].comparison["interproscan"]
        assert cmp["n_real_domains_changed_in_diff_region"] == 1
        assert cmp["hits_canonical_status"] == "ok"

    def test_repositioned_domain_not_counted(self):
        """A domain present on both panes (by InterPro id) is repositioned, not
        gained/lost → not counted.
        """
        gene = _make_gene(
            "MAAAAAA",
            tis_sites=[
                _make_site(
                    orf_type=ORFType.EXTENDED,
                    isoform_protein="MXXYYZZZKMAAAAAA",
                    diff_region=DifferentialRegion(
                        isoform_start=0,
                        isoform_end=9,
                        sequence="MXXYYZZZK",
                        confidence="tail_verified",
                    ),
                    isoform_annotations={
                        "interproscan": {
                            "hits": [_ips_hit(2, name="PF_same", interpro_id="IPR_SAME")],
                            "summary": {"status": "ok"},
                        }
                    },
                )
            ],
            canonical_annotations={
                "interproscan": {
                    "hits": [_ips_hit(0, name="PF_same", interpro_id="IPR_SAME")],
                    "summary": {"status": "ok"},
                }
            },
        )
        Comparator().compare([gene])
        cmp = gene.tis_sites[0].comparison["interproscan"]
        assert cmp["n_real_domains_changed_in_diff_region"] == 0

    def test_disorder_only_hit_not_a_real_domain(self):
        """A MobiDB-lite (disorder) hit in the diff region is not a real domain."""
        gene = _make_gene(
            "MAAAAAA",
            tis_sites=[
                _make_site(
                    orf_type=ORFType.EXTENDED,
                    isoform_protein="MXXYYZZZKMAAAAAA",
                    diff_region=DifferentialRegion(
                        isoform_start=0,
                        isoform_end=9,
                        sequence="MXXYYZZZK",
                        confidence="tail_verified",
                    ),
                    isoform_annotations={
                        "interproscan": {
                            "hits": [
                                _ips_hit(2, name="disorder", interpro_id=None, db="MobiDB-lite")
                            ],
                            "summary": {"status": "ok"},
                        }
                    },
                )
            ],
            canonical_annotations={
                "interproscan": {"hits": [], "summary": {"status": "ok"}}
            },
        )
        Comparator().compare([gene])
        cmp = gene.tis_sites[0].comparison["interproscan"]
        assert cmp["n_real_domains_changed_in_diff_region"] == 0

    def test_status_not_ok_gives_none(self):
        """When a pane's IPS scan didn't complete, the count is None."""
        gene = _make_gene(
            "MAAAAAA",
            tis_sites=[
                _make_site(
                    orf_type=ORFType.EXTENDED,
                    isoform_protein="MXXYYZZZKMAAAAAA",
                    diff_region=DifferentialRegion(
                        isoform_start=0,
                        isoform_end=9,
                        sequence="MXXYYZZZK",
                        confidence="tail_verified",
                    ),
                    isoform_annotations={
                        "interproscan": {
                            "hits": [_ips_hit(2, name="PF_new", interpro_id="IPR_NEW")],
                            "summary": {"status": "no_cache"},
                        }
                    },
                )
            ],
            canonical_annotations={
                "interproscan": {"hits": [], "summary": {"status": "ok"}}
            },
        )
        Comparator().compare([gene])
        cmp = gene.tis_sites[0].comparison["interproscan"]
        assert cmp["n_real_domains_changed_in_diff_region"] is None


# ---------------------------------------------------------------------------
# Scope-A enrichment
# ---------------------------------------------------------------------------


class TestScopeAEnrichment:
    def test_extension_biophysics_ratio(self, config):
        # MYC1 analog: isoform = extension + canonical body
        extension = "MDAAAPAPAP"  # acidic extension
        canonical = "MASTAV" * 20  # ~120 aa body
        isoform = extension + canonical
        diff_region = DifferentialRegion(
            isoform_start=0,
            isoform_end=len(extension),
            sequence=extension,
            confidence="tail_verified",
        )
        bio = BiophysicsModule(config)

        gene = _make_gene(
            canonical_protein=canonical,
            tis_sites=[
                _make_site(
                    orf_type=ORFType.EXTENDED,
                    isoform_protein=isoform,
                    diff_region=diff_region,
                    isoform_annotations={"biophysics": bio.annotate(isoform)},
                )
            ],
            canonical_annotations={"biophysics": bio.annotate(canonical)},
        )
        Comparator(scope_a_modules=[bio]).compare([gene])

        site = gene.tis_sites[0]
        # diff pane was populated
        assert "biophysics" in site.diff_annotations
        # Scope-A ratio landed in comparison under pI_ratio etc.
        cmp = site.comparison["biophysics"]
        assert "pI_ratio" in cmp
        assert "pI_unique" in cmp
        assert "pI_shared" in cmp
        # Extension is acidic (D), shared body is neutral → pI_unique < pI_shared
        assert cmp["pI_unique"] < cmp["pI_shared"]

    def test_uorf_has_no_enrichment(self, config):
        # uORF: no shared region, Scope-A enrichment should not populate
        bio = BiophysicsModule(config)
        uorf_seq = "MRGSHHHHHH"
        gene = _make_gene(
            canonical_protein="MASTAV" * 20,
            tis_sites=[
                _make_site(
                    orf_type=ORFType.UORF,
                    isoform_protein=uorf_seq,
                    diff_region=DifferentialRegion(
                        isoform_start=0,
                        isoform_end=len(uorf_seq),
                        sequence=uorf_seq,
                        confidence="exact",
                    ),
                    isoform_annotations={"biophysics": bio.annotate(uorf_seq)},
                )
            ],
            canonical_annotations={"biophysics": bio.annotate("MASTAV" * 20)},
        )
        Comparator(scope_a_modules=[bio]).compare([gene])

        cmp = gene.tis_sites[0].comparison["biophysics"]
        # diff pane populated (re-annotated) but no ratio (no shared region)
        assert "biophysics" in gene.tis_sites[0].diff_annotations
        assert "pI_ratio" not in cmp

    def test_length_fallback_skips_scope_a(self, config):
        # length_fallback confidence → Scope-A enrichment skipped
        bio = BiophysicsModule(config)
        gene = _make_gene(
            canonical_protein="MAAAAAAAAA" * 5,
            tis_sites=[
                _make_site(
                    orf_type=ORFType.TRUNCATED,
                    isoform_protein="XXXXX",
                    diff_region=DifferentialRegion(
                        canonical_start=0,
                        canonical_end=45,
                        sequence="MAAAAAAAAA" * 4 + "MAAAA",
                        confidence="length_fallback",
                    ),
                    isoform_annotations={"biophysics": bio.annotate("XXXXX")},
                )
            ],
            canonical_annotations={"biophysics": bio.annotate("MAAAAAAAAA" * 5)},
        )
        Comparator(scope_a_modules=[bio]).compare([gene])

        cmp = gene.tis_sites[0].comparison["biophysics"]
        assert "pI_delta" in cmp  # delta still computed
        assert "pI_ratio" not in cmp  # but Scope-A not trusted


# ---------------------------------------------------------------------------
# Functional wrapper + serialization
# ---------------------------------------------------------------------------


class TestCompareGenes:
    def test_functional_wrapper_works(self):
        gene = _make_gene(
            "MAAA",
            tis_sites=[
                _make_site(
                    orf_type=ORFType.ANNOTATED,
                    isoform_protein="MAAA",
                    diff_region=DifferentialRegion(sequence="", confidence="exact"),
                    isoform_annotations={"biophysics": {"pI": 6.0}},
                )
            ],
            canonical_annotations={"biophysics": {"pI": 6.0}},
        )
        result = compare_genes([gene])
        assert result == [gene]
        cmp = gene.tis_sites[0].comparison["biophysics"]
        assert cmp["pI_delta"] == 0.0
