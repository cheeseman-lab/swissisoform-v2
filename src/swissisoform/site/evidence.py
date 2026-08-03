"""Build per-gene LLM evidence-record JSON from the paired parquet.

Consumes a paired ``all_paired.parquet`` and emits one ``{gene}.json`` per gene
matching the "Per-gene evidence record" schema in
``docs/site_and_llm_plan.md``. Also exposes the V2 per-criterion config
(``CRITERIA``, ``CRITERIA_METRIC_LABELS``) and the ``slice_criterion`` slicer
that the website and LLM passes consume.

This is pure DataFrame → dict conversion: no LLM calls, no network.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PATHOGENIC_CLINSIG_TOKENS = ("pathogenic", "likely_pathogenic", "likely pathogenic")

# Sentinel headline_col for criteria whose headline is computed, not a column.
_START_SITE_USAGE = "__start_site_usage__"
_N_CELL_LINES_DETECTED = "__n_cell_lines_detected__"
_MASSPEC_VALIDATED = "__massspec_validated__"
_CODING_SELECTION = "__coding_selection__"
_PRIMATE_SIMILARITY = "__primate_similarity__"
_MAMMALIAN_SIMILARITY = "__mammalian_similarity__"
# Composed "Unique region {x}% similar across {clade}" headlines (C1/C2):
# sentinel -> (mean_pident column over the differential region, clade word).
_SIMILARITY_HEADLINES = {
    _PRIMATE_SIMILARITY: ("isoform_conservation_frame_primate_mean_pident", "primates"),
    _MAMMALIAN_SIMILARITY: ("isoform_conservation_frame_mammalian_mean_pident", "mammals"),
}
_GNOMAD_FOLD = "__gnomad_fold__"
_DISEASE_FOLD = "__disease_fold__"
_DIVERGING_DOMAINS = "__diverging_domains__"
_SSE_HEADLINE = "__sse_headline__"
# Mirror of ScoringConfig.p3_min_sse_length / p3_min_sse_plddt. The tile is
# rendered without a config object, so the thresholds are restated here; the
# test suite asserts the headline never disagrees with the P3 verdict.
_P3_MIN_LEN = 6
_P3_MIN_PLDDT = 0.70
# Fold-change variant-density headlines (M1/M2): "{noun} {fold}x more/less in
# unique region — {call}". low/high = call word for ratio <1 / >1.
_VARIANT_FOLD_HEADLINES = {
    _GNOMAD_FOLD: {
        "col": "isoform_variant_intersection_gnomad_depletion_ratio",
        "noun": "gnomAD variants",
        "low": "constrained",  # <1: fewer population variants than baseline
        "high": "tolerant",  # >1: more population variants
    },
    _DISEASE_FOLD: {
        "col": "isoform_variant_intersection_disease_enrichment_ratio",
        "noun": "Disease variants",
        "low": "depleted",  # <1: fewer disease variants
        "high": "enriched",  # >1: more disease variants
    },
}
# Composed "iso: {a} | canon: {b}" headlines: sentinel -> (isoform_col, canonical_col).
_ISO_CANON_HEADLINES = {
    "__localization_iso_canon__": (
        "isoform_localization_deeploc_prediction",
        "canonical_localization_deeploc_prediction",
    ),
    "__sorting_iso_canon__": (
        "isoform_targetp_targetp_prediction",
        "canonical_targetp_targetp_prediction",
    ),
}

_INITIATION_EFFICIENCY_SAMPLES = ("HeLa", "K562", "U2OS", "RPE1_Async", "RPE1_Que", "RPE1_Sen")


def _is_missing(value: Any) -> bool:
    """Return True for None, NaN, pd.NA, and empty numpy arrays."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if value is pd.NA:
        return True
    try:
        if isinstance(value, np.ndarray) and value.size == 0:
            return True
    except Exception:
        pass
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return False


def _to_native(value: Any) -> Any:
    """Recursively cast numpy/pandas types to JSON-native Python types."""
    if _is_missing(value):
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return None if math.isnan(v) else v
    if isinstance(value, np.ndarray):
        return [_to_native(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_native(v) for v in value]
    return value


def _get(row: pd.Series, key: str) -> Any:
    """Get a column value, normalised to a JSON-native type or None."""
    if key not in row.index:
        return None
    return _to_native(row[key])


def _scalar_or_none(row: pd.Series, key: str) -> Any:
    value = _get(row, key)
    return value


# ── Clinical-significance normalisation ───────────────────────────────────
#
# ``clinical_significance`` is a ClinVar-only free-text field: gnomAD and COSMIC
# rows carry no value at all, and ClinVar spells the same call several ways
# ("Pathogenic", "Pathogenic/Likely pathogenic", "Likely pathogenic"). Anything
# selecting on it must match by family rather than by equality, or it silently
# undercounts. Both helpers below live here so the LLM tool readers
# (``swissisoform.site.tools``) and the hit-truncation sort agree on what
# "pathogenic" means.

CLINSIG_FAMILIES = ("pathogenic", "benign", "uncertain", "conflicting", "none")


def clinsig_family(value: Any) -> str:
    """Bucket a ClinVar ``clinical_significance`` string into a coarse family.

    Families are the ones a caller actually filters on:
    ``pathogenic`` (Pathogenic + Pathogenic/Likely pathogenic + Likely
    pathogenic), ``benign`` (Benign + Benign/Likely benign + Likely benign),
    ``uncertain``, ``conflicting``, and ``none`` for an absent value (every
    gnomAD/COSMIC row, plus ClinVar rows with no assertion).

    ``conflicting`` is tested first because "Conflicting classifications of
    pathogenicity" contains the substring "pathogenic" and would otherwise be
    counted as a pathogenic call.
    """
    sig = str(value or "").strip().lower()
    if not sig or sig == "nan" or sig == "none":
        return "none"
    if "conflicting" in sig:
        return "conflicting"
    if "pathogenic" in sig:
        return "pathogenic"
    if "uncertain" in sig:
        return "uncertain"
    if "benign" in sig:
        return "benign"
    return "none"


def clinsig_rank(hit: dict[str, Any]) -> int:
    """Truncation-sort priority for one variant hit — lower surfaces first.

    Finer-grained than :func:`clinsig_family` because the sort separates a firm
    Pathogenic call from a Likely pathogenic one. Used by
    :func:`slice_criterion` to decide which hits survive the ``MAX_HITS`` cap.

    Note: a "Conflicting classifications of pathogenicity" value ranks 0 here
    (it contains "pathogenic" and not "likely"), which is more generous than
    :func:`clinsig_family`, where it is its own family. Kept as-is so the
    existing truncation order is unchanged; it only affects which hits appear in
    a capped view, never a count.
    """
    sig = str(hit.get("clinical_significance") or "").lower()
    if "pathogenic" in sig and "likely" not in sig:
        return 0  # Pathogenic
    if "likely_pathogenic" in sig or "likely pathogenic" in sig:
        return 1
    if hit.get("effect_damaging") is True:
        return 2
    if "uncertain" in sig:
        return 4
    if "benign" in sig:
        return 5
    return 3  # other / unknown


def _diff_space_from_orf_type(orf_type: Any) -> str | None:
    if orf_type is None:
        return None
    s = str(orf_type).strip().lower()
    if not s:
        return None
    return "canonical" if s == "truncated" else "isoform"


def _build_scoring(row: pd.Series) -> dict[str, Any]:
    """Pack the CDLMPS criteria (C+D existence, L+M+P+S functional) + reasons."""
    criteria_raw = _get(row, "isoform_scoring_criteria") or {}
    reasons_raw = _get(row, "isoform_scoring_reasons") or {}

    criteria: dict[str, dict[str, Any]] = {}
    keys = list(criteria_raw.keys()) if isinstance(criteria_raw, dict) else []
    for key in keys:
        value = criteria_raw.get(key) if isinstance(criteria_raw, dict) else None
        reason = reasons_raw.get(key) if isinstance(reasons_raw, dict) else None
        criteria[key] = {
            "value": _to_native(value),
            "reason": _to_native(reason),
        }

    return {
        "existence_score": _scalar_or_none(row, "isoform_scoring_existence_score"),
        "existence_evaluable": _scalar_or_none(row, "isoform_scoring_existence_evaluable"),
        "existence_high_confidence": _scalar_or_none(
            row, "isoform_scoring_existence_high_confidence"
        ),
        "functional_score": _scalar_or_none(row, "isoform_scoring_functional_score"),
        "functional_evaluable": _scalar_or_none(row, "isoform_scoring_functional_evaluable"),
        "functional_high_confidence": _scalar_or_none(
            row, "isoform_scoring_functional_high_confidence"
        ),
        "criteria": criteria,
    }


def _validated_unique_peptide_count(row: pd.Series) -> int | None:
    """Count massspec hits flagged unique_to_isoform AND validated."""
    hits = _get(row, "isoform_massspec_hits")
    if hits is None:
        return None
    if not isinstance(hits, list):
        return None
    return sum(
        1 for h in hits if isinstance(h, dict) and h.get("unique_to_isoform") and h.get("validated")
    )


def _build_key_metrics(row: pd.Series) -> dict[str, Any]:
    return {
        "primate_frac_intact": _scalar_or_none(
            row, "isoform_conservation_frame_primate_frac_intact"
        ),
        "mammalian_frac_intact": _scalar_or_none(
            row, "isoform_conservation_frame_mammalian_frac_intact"
        ),
        "phylop_unique": _scalar_or_none(row, "isoform_conservation_phylop_unique_region_mean"),
        "phylop_shared": _scalar_or_none(row, "isoform_conservation_phylop_shared_region_mean"),
        "phylop_at_tis": _scalar_or_none(row, "isoform_conservation_phylop_at_tis"),
        "plddt_diffregion_mean": _scalar_or_none(row, "isoform_structure_plddt_diffregion_mean"),
        "tm_score": _scalar_or_none(row, "isoform_structure_tm_score"),
        "rmsd_global": _scalar_or_none(row, "isoform_structure_rmsd_global"),
        "localization_canonical": _scalar_or_none(row, "canonical_localization_deeploc_prediction"),
        "localization_isoform": _scalar_or_none(row, "isoform_localization_deeploc_prediction"),
        "n_variants_in_unique": _scalar_or_none(
            row, "isoform_variant_intersection_n_in_unique_region"
        ),
        "n_pathogenic_in_unique": _scalar_or_none(
            row, "isoform_variant_intersection_n_pathogenic_in_unique_region"
        ),
        "n_damaging_in_unique": _scalar_or_none(row, "isoform_varianteffect_n_damaging_in_unique"),
        "mean_delta_llr_unique": _scalar_or_none(
            row, "isoform_varianteffect_mean_delta_llr_unique"
        ),
        "mean_am_pathogenicity_unique": _scalar_or_none(
            row, "isoform_varianteffect_mean_am_pathogenicity_unique"
        ),
        "validated_unique_peptides": _validated_unique_peptide_count(row),
    }


def _build_pathogenic_in_unique(row: pd.Series) -> list[dict[str, Any]]:
    """Filter variant_intersection_hits to pathogenic AND in_isoform_unique."""
    hits = _get(row, "isoform_variant_intersection_hits")
    if not isinstance(hits, list):
        return []
    out: list[dict[str, Any]] = []
    for h in hits:
        if not isinstance(h, dict):
            continue
        if not h.get("in_isoform_unique"):
            continue
        sig = h.get("clinical_significance")
        if sig is None:
            continue
        sig_l = str(sig).lower()
        if not any(tok in sig_l for tok in PATHOGENIC_CLINSIG_TOKENS):
            continue
        out.append(
            {
                "id": h.get("variant_id"),
                "source": h.get("source"),
                "consequence": h.get("consequence"),
                "hgvsp": h.get("hgvsp"),
                "clinical_significance": h.get("clinical_significance"),
                "protein_pos": h.get("protein_pos"),
                "isoform_protein_pos": h.get("isoform_protein_pos"),
            }
        )
    return out


def _build_isoform(row: pd.Series) -> dict[str, Any]:
    orf_type = _scalar_or_none(row, "orf_type")
    diff_space = _scalar_or_none(row, "diff_space")
    if diff_space is None:
        diff_space = _diff_space_from_orf_type(orf_type)

    return {
        "tis_id": _scalar_or_none(row, "tis_id"),
        "orf_type": orf_type,
        "alt_start_codon": _scalar_or_none(row, "start_codon"),
        "isoform_length_aa": _scalar_or_none(row, "isoform_len"),
        "canonical_length_aa": _scalar_or_none(row, "canonical_len"),
        "differential_sequence": _scalar_or_none(row, "differential_sequence"),
        "diff_space": diff_space,
        "kozak_context": _scalar_or_none(row, "kozak_context"),
        "scoring": _build_scoring(row),
        "key_metrics": _build_key_metrics(row),
        "pathogenic_variants_in_unique": _build_pathogenic_in_unique(row),
        # V2: raw columns the per-criterion slicer reaches into; kept under _raw
        # so the existing per-gene JSON files surface identical user-facing keys.
        "_raw": {col: _to_native(row[col]) for col in row.index},
    }


def build_gene_record(gene_name: str, sub: pd.DataFrame) -> dict[str, Any]:
    """Build a single gene's evidence record from its TIS rows."""
    head = sub.iloc[0]
    isoforms = [_build_isoform(row) for _, row in sub.iterrows()]
    isoforms.sort(key=lambda r: (r.get("tis_id") or ""))
    return {
        "gene": {
            "name": gene_name,
            "uniprot_id": _scalar_or_none(head, "generef_uniprot_id"),
            "function": _scalar_or_none(head, "generef_function"),
            "subcellular_location": _scalar_or_none(head, "generef_subcellular_location"),
            "keywords": _scalar_or_none(head, "generef_keywords"),
        },
        "isoforms": isoforms,
    }


def _safe_gene_filename(gene_name: str) -> str:
    assert "/" not in gene_name and " " not in gene_name, (
        f"Unsafe gene name for filename: {gene_name!r}"
    )
    return f"{gene_name}.json"


def write_evidence_records(parquet_path: Path, out_dir: Path) -> dict[str, int]:
    """Write one ``{gene}.json`` evidence record per gene. Returns counts."""
    df = pd.read_parquet(parquet_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {"genes": 0, "isoforms": 0}
    for gene_name, sub in df.groupby("gene_name", sort=True):
        record = build_gene_record(str(gene_name), sub)
        out_path = out_dir / _safe_gene_filename(str(gene_name))
        out_path.write_text(
            json.dumps(record, indent=2, sort_keys=False, ensure_ascii=False),
            encoding="utf-8",
        )
        counts["genes"] += 1
        counts["isoforms"] += len(record["isoforms"])
    return counts


_VARIANTS_LONG_BASE_COLS = (
    "variant_id",
    "source",
    "chrom",
    "genomic_pos",
    "ref",
    "alt",
    "hgvsp",
    "consequence",
    "clinical_significance",
    "protein_pos",
    "isoform_protein_pos",
    "in_isoform_unique",
    "in_isoform_shared",
    "allele_frequency",
    "aa_ref",
    "aa_alt",
    "isoform_aa_ref",
    "isoform_aa_alt",
    "isoform_consequence",
)

# Pulled from the per-hit ``metadata`` sub-dict. ``clinvar_title`` reads
# ``metadata.title`` — renamed so the column is self-describing.
_VARIANTS_LONG_METADATA_COLS = (
    ("hgvsc", "hgvsc"),
    ("rs_id", "rs_id"),
    ("cosmic_sample_count", "cosmic_sample_count"),
    ("clinvar_title", "title"),
)

_VARIANTS_LONG_EFFECT_COLS = (
    "am_class",
    "am_pathogenicity",
    "plm_delta_llr",
    "plm_llr_wt",
    "plm_llr_alt",
    "effect_damaging",
)


def write_variants_long(parquet_path: Path, out_path: Path) -> int:
    """Flatten ``isoform_variant_intersection_hits`` + ``isoform_varianteffect_hits``.

    Emits one row per ``(tis_id, variant_id)`` pair. Effect columns are joined
    on ``variant_id`` within each isoform; variants without an effect entry
    have null effect columns. Returns the number of rows written.

    Effect rows with no matching intersection hit are dropped — the table is keyed
    by intersection hits, not by effects.
    """
    df = pd.read_parquet(
        parquet_path,
        columns=[
            "tis_id",
            "gene_name",
            "isoform_variant_intersection_hits",
            "isoform_varianteffect_hits",
        ],
    )

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        hits = _get(row, "isoform_variant_intersection_hits") or []
        effects = _get(row, "isoform_varianteffect_hits") or []
        if not isinstance(hits, list) or not hits:
            continue
        eff_by_id = {
            e.get("variant_id"): e
            for e in effects
            if isinstance(e, dict) and e.get("variant_id") is not None
        }
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            record: dict[str, Any] = {
                "tis_id": _to_native(row["tis_id"]),
                "gene_name": _to_native(row["gene_name"]),
            }
            for c in _VARIANTS_LONG_BASE_COLS:
                record[c] = _to_native(hit.get(c))
            meta = hit.get("metadata")
            if not isinstance(meta, dict):
                meta = {}
            for out_col, meta_key in _VARIANTS_LONG_METADATA_COLS:
                record[out_col] = _to_native(meta.get(meta_key))
            eff = eff_by_id.get(hit.get("variant_id")) or {}
            for c in _VARIANTS_LONG_EFFECT_COLS:
                record[c] = _to_native(eff.get(c))
            rows.append(record)

    out_df = pd.DataFrame(rows)
    if "effect_damaging" in out_df.columns:
        out_df["effect_damaging"] = out_df["effect_damaging"].astype("boolean")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    return len(rows)


def summarise(out_dir: Path) -> None:
    """Print a 5-line spot-check summary."""
    files = sorted(out_dir.glob("*.json"))
    total_genes = len(files)
    total_isoforms = 0
    field_present = 0
    field_total = 0
    best = (-1, None, None)
    worst = (10**9, None, None)
    spec_fields = (
        "primate_frac_intact",
        "phylop_unique",
        "phylop_shared",
        "plddt_diffregion_mean",
        "tm_score",
        "rmsd_global",
        "localization_canonical",
        "localization_isoform",
        "n_variants_in_unique",
        "n_pathogenic_in_unique",
        "n_damaging_in_unique",
        "validated_unique_peptides",
    )
    for fp in files:
        rec = json.loads(fp.read_text(encoding="utf-8"))
        for iso in rec.get("isoforms", []):
            total_isoforms += 1
            km = iso.get("key_metrics") or {}
            for f in spec_fields:
                field_total += 1
                if km.get(f) is not None:
                    field_present += 1
            scoring = iso.get("scoring") or {}
            e_score = scoring.get("existence_score")
            f_score = scoring.get("functional_score")
            total = (e_score or 0) + (f_score or 0)
            if total > best[0]:
                best = (total, rec["gene"]["name"], iso.get("tis_id"))
            if total < worst[0]:
                worst = (total, rec["gene"]["name"], iso.get("tis_id"))
    pct = (100.0 * field_present / field_total) if field_total else 0.0
    print(f"genes: {total_genes}")
    print(f"isoforms (TIS): {total_isoforms}")
    print(f"key_metrics field-fill: {field_present}/{field_total} ({pct:.1f}%)")
    print(f"biggest score (E+F): {best[0]} → {best[1]} / {best[2]}")
    print(f"smallest score (E+F): {worst[0]} → {worst[1]} / {worst[2]}")


# ──────────────────────────────────────────────────────────────────────────
# V2 per-criterion config. Each entry drives
# (a) the slicer (what raw cols to pull) and (b) the UI tile (label, headline).
# Verified against the real cheeseman_13gene parquet column schema.
# ──────────────────────────────────────────────────────────────────────────

CRITERIA: dict[str, dict[str, Any]] = {
    "C1_primate_conservation": {
        "axis": "E",
        "label": "Primate conservation",
        "short_label": "Primates",
        "evidence_cols": [
            "isoform_conservation_frame_primate_mean_pident",
            "isoform_conservation_frame_primate_frac_intact",
            "isoform_conservation_frame_primate_start_codon_conserved",
            "isoform_conservation_frame_primate_n_species_aligned",
            "isoform_conservation_frame_primate_n_species_intact_frame",
            "isoform_conservation_frame_primate_deepest_species",
            "isoform_conservation_frame_primate_max_depth",
            "isoform_conservation_frame_primate_canonical_mean_pident",
            "isoform_conservation_frame_primate_canonical_n_species_aligned",
        ],
        "headline_col": _PRIMATE_SIMILARITY,
        "interpretation_hint": (
            "Is the alternative reading frame conserved across primates? Score on "
            "mean_pident (mean amino-acid % identity to primate orthologs); "
            "frac_intact and start_codon_conserved are context. For a within-gene "
            "baseline compare mean_pident to canonical_mean_pident ONLY — like for "
            "like. frac_intact is the fraction of species whose reading frame "
            "survives intact across the WHOLE queried span, so it is confounded by "
            "span length (a short unique region reads high, a long canonical ORF "
            "reads low for that reason alone). Never compare frac_intact between "
            "regions of different length, and never read such a gap as a "
            "conservation difference."
        ),
    },
    "C2_mammalian_conservation": {
        "axis": "E",
        "label": "Mammalian conservation",
        "short_label": "Mammals",
        "evidence_cols": [
            "isoform_conservation_frame_mammalian_mean_pident",
            "isoform_conservation_frame_mammalian_frac_intact",
            "isoform_conservation_frame_mammalian_start_codon_conserved",
            "isoform_conservation_frame_mammalian_n_species_aligned",
            "isoform_conservation_frame_mammalian_n_species_intact_frame",
            "isoform_conservation_frame_mammalian_deepest_species",
            "isoform_conservation_frame_mammalian_max_depth",
            "isoform_conservation_frame_mammalian_canonical_mean_pident",
            "isoform_conservation_frame_mammalian_canonical_n_species_aligned",
        ],
        "headline_col": _MAMMALIAN_SIMILARITY,
        "interpretation_hint": (
            "Is the alternative reading frame conserved deeper in mammals? Score on "
            "mean_pident (mean amino-acid % identity to mammalian orthologs); "
            "frac_intact is context. For a within-gene baseline compare mean_pident "
            "to canonical_mean_pident ONLY — like for like. frac_intact is the "
            "fraction of species whose reading frame survives intact across the "
            "WHOLE queried span, so it is confounded by span length (a short unique "
            "region reads high, a long canonical ORF reads low for that reason "
            "alone). Never compare frac_intact between regions of different length, "
            "and never read such a gap as a conservation difference."
        ),
    },
    "C3_phylop_coding_selection": {
        "axis": "E",
        "label": "Coding Selection",
        "short_label": "PhyloP",
        "evidence_cols": [
            "isoform_conservation_phylop_unique_region_mean",
            "isoform_conservation_phylop_shared_region_mean",
            "isoform_conservation_phylop_enrichment",
            "isoform_conservation_phylop_at_tis",
            "isoform_conservation_phylop_kozak_mean",
            "isoform_conservation_phastcons_unique_region_mean",
            "isoform_conservation_phastcons_shared_region_mean",
            "isoform_conservation_phastcons_at_tis",
            "isoform_conservation_phastcons_kozak_mean",
        ],
        "headline_col": _CODING_SELECTION,
        "interpretation_hint": (
            "Does the unique coding region show purifying selection by phyloP "
            "(absolute mean ≥ ~2 indicates strong constraint)? The shared region "
            "and enrichment ratio are context only, not the basis for the call."
        ),
    },
    "D1_multi_cell_line": {
        "axis": "E",
        "label": "Expression Breadth",
        "short_label": "Cell lines",
        "evidence_cols": [
            "expr_HeLa_initiation_efficiency",
            "expr_HeLa_p_value",
            "expr_HeLa_cpm",
            "expr_K562_initiation_efficiency",
            "expr_K562_p_value",
            "expr_K562_cpm",
            "expr_U2OS_initiation_efficiency",
            "expr_U2OS_p_value",
            "expr_U2OS_cpm",
            "expr_RPE1_Async_initiation_efficiency",
            "expr_RPE1_Async_p_value",
            "expr_RPE1_Async_cpm",
            "expr_RPE1_Que_initiation_efficiency",
            "expr_RPE1_Que_p_value",
            "expr_RPE1_Que_cpm",
            "expr_RPE1_Sen_initiation_efficiency",
            "expr_RPE1_Sen_p_value",
            "expr_RPE1_Sen_cpm",
        ],
        "headline_col": _N_CELL_LINES_DETECTED,  # computed: # cell lines detected
        "interpretation_hint": (
            "Is the TIS reproducibly detected across multiple cell lines? Count "
            "samples with significant p-values."
        ),
    },
    "D2_initiation_efficiency": {
        "axis": "E",
        "label": "Start-Site Usage",
        "short_label": "Init. eff.",
        "evidence_cols": [
            "expr_HeLa_initiation_efficiency",
            "expr_K562_initiation_efficiency",
            "expr_U2OS_initiation_efficiency",
            "expr_RPE1_Async_initiation_efficiency",
            "expr_RPE1_Que_initiation_efficiency",
            "expr_RPE1_Sen_initiation_efficiency",
            "canonical_expr_HeLa_initiation_efficiency",
            "canonical_expr_K562_initiation_efficiency",
            "canonical_expr_U2OS_initiation_efficiency",
            "canonical_expr_RPE1_Async_initiation_efficiency",
            "canonical_expr_RPE1_Que_initiation_efficiency",
            "canonical_expr_RPE1_Sen_initiation_efficiency",
            "ribo_pvalue",
            "tis_pvalue",
            "fisher_qvalue",
        ],
        # Computed headline: the maximum per-cell-line initiation efficiency
        # (TIS counts / gene RNA-seq counts ratio) across the six samples — see
        # ``_MAX_INITIATION_EFFICIENCY`` handling in ``slice_criterion``.
        "headline_col": _START_SITE_USAGE,
        "interpretation_hint": (
            "How efficiently is this TIS initiated? Each per-cell-line value is the "
            "TIS read counts / gene RNA-seq counts ratio; the headline is the max "
            "across cell lines. Compare to the canonical_ twins for a within-gene "
            "baseline."
        ),
    },
    "D3_mass_spec": {
        "axis": "E",
        "label": "Peptide Evidence",
        "short_label": "MS",
        "evidence_cols": [
            "isoform_massspec_summary",
            "cmp_massspec_n_hits_in_diff_region",
        ],
        "evidence_hits_col": "cmp_massspec_hits_in_diff_region",
        "headline_col": _MASSPEC_VALIDATED,
        "interpretation_hint": (
            "Are there PepQuery2 validated peptides in the isoform's unique region?"
        ),
    },
    "P1_structured_extension": {
        "axis": "F",
        "label": "Fold Confidence",
        "short_label": "Folding",
        "evidence_cols": [
            "isoform_structure_status",
            "isoform_structure_plddt_canonical_mean",
            "isoform_structure_plddt_isoform_mean",
            "isoform_structure_plddt_diffregion_mean",
            "isoform_structure_plddt_diffregion_std",
            "isoform_structure_plddt_delta_shared",
            "isoform_structure_ptm_isoform",
            "isoform_structure_ptm_canonical",
            "isoform_structure_pae_diff_vs_diff",
            "isoform_structure_pae_body_vs_body",
            "isoform_structure_pae_diff_vs_body",
            "isoform_structure_pae_status",
            "isoform_structure_extension_contacts",
        ],
        "headline_col": "isoform_structure_plddt_diffregion_mean",
        "interpretation_hint": (
            "Does the unique region fold confidently (ESMFold2 pLDDT)? Higher "
            "diffregion_mean means more structured. Folding only — the biophysical "
            "distinctness signal (GRAVY / fraction_charged / disorder) is scored "
            "separately under S2, so do not weigh it here. "
            "This member also carries the model-confidence metrics for the whole "
            "prediction — global pTM (ptm_isoform / ptm_canonical) and the PAE blocks "
            "(pae_diff_vs_diff, pae_body_vs_body, pae_diff_vs_body, pae_status). Low "
            "pTM / high PAE means the predicted fold and the relative placement of "
            "regions are unreliable, which is the qualifier the Core Fold Perturbation "
            "(shared-region RMSD) member in this same category must be read against."
        ),
    },
    "L1_localization_change": {
        "axis": "F",
        "label": "Compartment",
        "short_label": "Localization",
        "evidence_cols": [
            "cmp_localization_deeploc_prediction_changed",
            "cmp_localization_deeploc_prediction_canonical",
            "cmp_localization_deeploc_prediction_isoform",
            "canonical_localization_deeploc_prediction",
            "isoform_localization_deeploc_prediction",
            "cmp_localization_deeploc_signals_changed",
            "cmp_localization_deeploc_signals_canonical",
            "cmp_localization_deeploc_signals_isoform",
            "cmp_localization_deeploc_membrane_changed",
            "cmp_localization_deeploc_membrane_canonical",
            "cmp_localization_deeploc_membrane_isoform",
            "canonical_localization_deeploc_signals",
            "isoform_localization_deeploc_signals",
            "canonical_localization_deeploc_membrane",
            "isoform_localization_deeploc_membrane",
            # Top-class confidence + per-compartment probability vector — lets a
            # confident call be told from a borderline one, and confidence
            # SHIFTS register even when the argmax label doesn't flip.
            "isoform_localization_deeploc_top_prob",
            "canonical_localization_deeploc_top_prob",
            "isoform_localization_deeploc_prob_cytoplasm",
            "isoform_localization_deeploc_prob_nucleus",
            "isoform_localization_deeploc_prob_extracellular",
            "isoform_localization_deeploc_prob_cell_membrane",
            "isoform_localization_deeploc_prob_mitochondrion",
            "isoform_localization_deeploc_prob_endoplasmic_reticulum",
            "isoform_localization_deeploc_prob_golgi_apparatus",
            "isoform_localization_deeploc_prob_lysosome_vacuole",
            "isoform_localization_deeploc_prob_peroxisome",
            "canonical_localization_deeploc_prob_cytoplasm",
            "canonical_localization_deeploc_prob_nucleus",
            "canonical_localization_deeploc_prob_extracellular",
            "canonical_localization_deeploc_prob_cell_membrane",
            "canonical_localization_deeploc_prob_mitochondrion",
            "canonical_localization_deeploc_prob_endoplasmic_reticulum",
            "canonical_localization_deeploc_prob_golgi_apparatus",
            "canonical_localization_deeploc_prob_lysosome_vacuole",
            "canonical_localization_deeploc_prob_peroxisome",
        ],
        "headline_col": "__localization_iso_canon__",
        "interpretation_hint": (
            "Do the isoform's localization features change vs canonical "
            "(DeepLoc prediction / sorting signals / membrane association)?"
        ),
    },
    "S1_domain_change": {
        "axis": "F",
        "label": "Domain change",
        "short_label": "Domains",
        "evidence_cols": [
            "isoform_interproscan_summary",
            "cmp_interproscan_n_hits_in_diff_region",
        ],
        "evidence_hits_col": "cmp_interproscan_hits_in_diff_region",
        "headline_col": _DIVERGING_DOMAINS,
        "interpretation_hint": ("Does the differential region overlap with InterProScan domains?"),
    },
    "L2_targeting_change": {
        "axis": "F",
        "label": "Sorting Signals",
        "short_label": "Targeting",
        "evidence_cols": [
            "cmp_signalp_signalp_prediction_changed",
            "cmp_signalp_signalp_prediction_canonical",
            "cmp_signalp_signalp_prediction_isoform",
            "cmp_signalp_signalp_probability_delta",
            "cmp_signalp_signalp_cleavage_site_changed",
            "cmp_signalp_signalp_cleavage_site_canonical",
            "cmp_signalp_signalp_cleavage_site_isoform",
            "cmp_targetp_targetp_prediction_changed",
            "cmp_targetp_targetp_prediction_canonical",
            "cmp_targetp_targetp_prediction_isoform",
            "cmp_targetp_targetp_probability_delta",
            "cmp_targetp_targetp_sp_prob_delta",
            "cmp_targetp_targetp_mtp_prob_delta",
            "cmp_targetp_targetp_cleavage_site_changed",
            "canonical_signalp_signalp_prediction",
            "isoform_signalp_signalp_prediction",
            "canonical_targetp_targetp_prediction",
            "isoform_targetp_targetp_prediction",
        ],
        "headline_col": "__sorting_iso_canon__",
        "interpretation_hint": (
            "Do N-terminal sorting signals differ between canonical and isoform — a "
            "secretory signal peptide (SignalP) or a mitochondrial/chloroplast "
            "transit peptide (TargetP)?"
        ),
    },
    "M1_pathogenic_variant_enrichment": {
        "axis": "F",
        "label": "Germline Variants",
        "short_label": "Germline",
        "evidence_cols": [
            "isoform_variant_intersection_gnomad_depletion_ratio",
            "isoform_variant_intersection_n_gnomad_in_unique_region",
            "isoform_variant_intersection_n_gnomad_in_shared_region",
            "isoform_variant_intersection_unique_region_nt",
            "isoform_variant_intersection_shared_region_nt",
            "isoform_plm_vep_constraint_enrichment",
            "isoform_plm_vep_mean_llr_unique_region",
            "isoform_plm_vep_mean_llr_shared_region",
            "isoform_plm_vep_n_constrained_positions_unique",
            "isoform_plm_vep_n_constrained_positions_shared",
            "isoform_plm_vep_status",
        ],
        "evidence_hits_col": "isoform_variant_intersection_hits",
        "headline_col": _GNOMAD_FOLD,
        "interpretation_hint": (
            "Is the unique region under germline constraint? Two independent "
            "signals: (1) gnomad_depletion_ratio < 1 means germline variation "
            "AVOIDS the unique region (density-normalized vs shared core); "
            "(2) ESM-C constraint_enrichment high means residues there are "
            "predicted intolerant to substitution. This measures tolerance/"
            "constraint, not damaging-variant burden. "
            "The two are either-or evidence with OPPOSITE directionality (gnomAD "
            "low = constrained, ESM-C high = constrained); either alone suffices, "
            "so they need not agree. "
            "VALID ONLY ON TRUNCATIONS, where the region is canonical coding "
            "sequence. On an EXTENSION the unique region was 5'UTR/intron and was "
            "never coding: the gnomAD ratio then measures never-coding variation "
            "(confounded by UTR/splicing selection and coverage) and the ESM-C "
            "score is out-of-distribution, reflecting composition rather than "
            "intolerance — n_constrained_positions_unique is routinely 0 there. "
            "Do not interpret either value, or their disagreement, as evidence "
            "about protein constraint on an extension. On separate-ORF isoforms "
            "there is no shared region, so the ratio is undefined by construction."
        ),
    },
    "M2_clinical_variant_overlap": {
        "axis": "F",
        "label": "Clinical Variants",
        "short_label": "Disease",
        "evidence_cols": [
            "isoform_variant_intersection_disease_enrichment_ratio",
            "isoform_variant_intersection_n_disease_in_unique_region",
            "isoform_variant_intersection_n_disease_in_shared_region",
            "isoform_variant_intersection_n_pathogenic_in_unique_region",
            "isoform_variant_intersection_n_pathogenic_in_shared_region",
            "isoform_variant_intersection_unique_region_nt",
            "isoform_variant_intersection_shared_region_nt",
            "isoform_variant_intersection_n_total",
            "isoform_variant_intersection_n_dropped_outside_coding",
        ],
        "evidence_hits_col": "isoform_variant_intersection_hits",
        "headline_col": _DISEASE_FOLD,
        "interpretation_hint": (
            "Do disease variants (ClinVar + COSMIC) CONCENTRATE in the isoform's "
            "unique coding region? disease_enrichment_ratio > 1 means the unique "
            "region carries a higher disease-variant density than the shared core; "
            "the raw unique/shared disease and pathogenic counts are context."
        ),
    },
    "P3_secondary_structure": {
        "axis": "F",
        "label": "Secondary Structure",
        "short_label": "SSE",
        "evidence_cols": [
            "isoform_structure_sse_status",
            "isoform_structure_sse_longest_helix_diff",
            "isoform_structure_sse_longest_strand_diff",
            "isoform_structure_sse_max_confident_element_diff",
            "isoform_structure_plddt_diffregion_mean",
            "isoform_structure_ptm_isoform",
            "isoform_structure_ptm_canonical",
        ],
        "evidence_hits_col": "isoform_structure_sse_diff_elements",
        "headline_col": _SSE_HEADLINE,
        "interpretation_hint": (
            "Does the differential region contain actual secondary structure — a "
            "helix or strand — as opposed to coil? Assigned from the predicted "
            "coordinates (P-SEA), so each element carries its own mean pLDDT and "
            "BOTH length and confidence are required to score: a geometrically "
            "clean helix running through a disordered stretch is geometry fitted "
            "to a guess, not a structural finding. "
            "Symmetric across ORF types, opposite narrative — on an EXTENSION the "
            "isoform GAINS the element (a candidate functional addition); on a "
            "TRUNCATION it LOSES one, and the element is read off the CANONICAL "
            "structure because the removed segment exists only there, so a hit "
            "means the truncation deletes real structure rather than a "
            "disordered tail. "
            "This says nothing about whether the element is INTEGRATED with the "
            "rest of the fold — whether a gained helix packs against the core, or "
            "a lost one was load-bearing. Read the contact and PAE evidence in "
            "this same category for that; do not infer it from the presence of an "
            "element alone. pLDDT is not a proxy for secondary structure and the "
            "two often disagree: the highest-confidence stretch of a region is "
            "frequently coil."
        ),
    },
    "P2_shared_structural_change": {
        "axis": "F",
        "label": "Core Fold Perturbation",
        "short_label": "Shared RMSD",
        "evidence_cols": [
            "isoform_structure_rmsd_shared",
            "isoform_structure_tm_score_shared",
            "isoform_structure_shared_region_len",
            "isoform_structure_rmsd_shared_status",
            "isoform_structure_plddt_shared_mean_isoform",
            "isoform_structure_plddt_shared_mean_canonical",
            "isoform_structure_rmsd_global",
            "isoform_structure_tm_score",
            "isoform_structure_ptm_isoform",
            "isoform_structure_ptm_canonical",
            "isoform_structure_pae_body_vs_body",
            "isoform_structure_pae_diff_vs_body",
            "isoform_structure_pae_status",
        ],
        "headline_col": "isoform_structure_rmsd_shared",
        "interpretation_hint": (
            "Does the retained (shared) region fold differently in the isoform vs "
            "the canonical protein? The shared region is identical in sequence, so a "
            "high Cα RMSD (Kabsch-superposed on the shared residues only) is "
            "CONSISTENT WITH the extension/truncation reorganizing how that region "
            "folds — most isoforms read ≈ 0. TM-score is a length-normalized "
            "companion. "
            "CONFIDENCE GATE — a high RMSD is NOT on its own evidence of refolding. "
            "The common cause is ESMFold placing a poorly-determined region "
            "differently between two low-confidence models, which is placement/"
            "orientation uncertainty, not a conformational change. Before calling it "
            "a real refold, check the fold-confidence metrics carried here and in the "
            "Fold Confidence member of this same category: global pTM (ptm_isoform / "
            "ptm_canonical), shared-region pLDDT (plddt_shared_mean_isoform / "
            "plddt_shared_mean_canonical), and the PAE blocks (pae_body_vs_body, "
            "pae_diff_vs_body, pae_status). If pTM ≲ 0.50, OR either shared pLDDT < "
            "0.70, OR PAE is high, treat the RMSD as an artifact of low confidence: "
            "state it as an unresolved hypothesis at most, and never make it the "
            "headline or say the isoform 'remodels'/'reorganizes'/'destabilizes' the "
            "core. Always cite the pTM alongside any RMSD claim, and read rmsd_shared "
            "against rmsd_global rather than in isolation. "
            "Only scored when both structures are confidently folded (min "
            "shared-region pLDDT ≥ 0.70); uORF/altORF isoforms have no shared region "
            "and are not evaluable."
        ),
    },
    # S2/S3 are first-class scored criteria whose evidence is nested (a biophysics
    # property table / SAE feature records) rather than flat columns, so they carry
    # no ``evidence_cols``/``headline_col`` and instead delegate to an
    # ``evidence_builder`` hook (attached after the builders are defined, below).
    # ``omit_if_empty`` reproduces the old descriptive behaviour: when the builder
    # has no data the member is dropped from the category display.
    "S2_biophysics": {
        "axis": "F",
        "label": "Biophysics",
        "short_label": "Biophysics",
        "evidence_cols": [],
        "headline_col": None,
        "omit_if_empty": True,
        "interpretation_hint": (
            "S2 — whole-protein biophysical shift, isoform vs canonical. The scored "
            "value keys off the gravy/fraction_charged/disorder whole-protein deltas "
            "(|isoform − canonical| ≥ cutoff, any one firing → shifted). The "
            "unique/shared/ratio columns are extra region-vs-core context; the scored "
            "call itself is the whole-protein delta."
        ),
    },
    "S3_sae": {
        "axis": "F",
        "label": "SAE features",
        "short_label": "SAE",
        "evidence_cols": [],
        "headline_col": None,
        "omit_if_empty": True,
        "interpretation_hint": (
            "S3 — sparse-autoencoder (ESM-C) interpretability features that differ "
            "between the isoform and canonical protein. The scored value is a "
            "MAGNITUDE check: True when the strongest shared-feature activation "
            "shift, max(|top_gained_delta_max|, |top_lost_delta_max|), meets the "
            "threshold; False when a differential exists but nothing shifted that "
            "strongly. The gained/lost counts are context only — two proteins of "
            "different length always differ in hundreds of features, so their "
            "presence says nothing about magnitude. "
            "Treat this as a PRESENCE/ABSENCE signal only. Feature labels are "
            "auto-generated and provisional, not curated annotation: many carry no "
            "content (hollow labels are withheld, leaving the feature identified by "
            "index alone), and many others merely describe the feature's own "
            "activation pattern or restate sequence composition already scored by "
            "the whole-protein biophysical shift in this same category. Do NOT name, "
            "quote or interpret a feature label in the reasoning, and do not let a "
            "label move the verdict — cite only that features differ, and how many."
        ),
    },
}


# ──────────────────────────────────────────────────────────────────────────
# CDLMPS evidence categories — the single source of truth for how the scored
# criteria group into the six category boxes shown on the site. The LLM
# interpretation runs one read per category (see ``slice_category`` + the
# ``category`` pass in ``llm.py``); the website derives its ``CARD_GROUPS`` from
# this list so UI and LLM never drift.
#
# ``members`` are all keys in ``CRITERIA`` — every member (including S2 biophysics
# and S3 SAE) is a first-class scored criterion sliced through ``slice_criterion``.
# All 15 criteria — including P2 — are covered exactly once; there is no
# LLM-excluded criterion.
CATEGORIES: list[dict[str, Any]] = [
    {
        "letter": "C",
        "name": "Conservation",
        "members": [
            "C1_primate_conservation",
            "C2_mammalian_conservation",
            "C3_phylop_coding_selection",
        ],
    },
    {
        "letter": "D",
        "name": "Detection",
        "members": [
            "D1_multi_cell_line",
            "D2_initiation_efficiency",
            "D3_mass_spec",
        ],
    },
    {
        "letter": "L",
        "name": "Localization",
        "members": ["L1_localization_change", "L2_targeting_change"],
    },
    {
        "letter": "M",
        "name": "Mutation Landscape",
        "members": ["M1_pathogenic_variant_enrichment", "M2_clinical_variant_overlap"],
    },
    {
        "letter": "P",
        "name": "Predicted Structure",
        "members": [
            "P1_structured_extension",
            "P2_shared_structural_change",
            "P3_secondary_structure",
        ],
    },
    {
        "letter": "S",
        "name": "Structural Characteristics",
        "members": ["S1_domain_change", "S2_biophysics", "S3_sae"],
    },
]

# Biophysical properties fed into the biophysics evidence builder — (label, key)
# over the ``cmp_biophysics_<key>_{unique,shared,ratio}`` differential columns.
_BIOPHYSICS_FEATURES: list[tuple[str, str]] = [
    ("Isoelectric point (pI)", "pI"),
    ("Hydropathy (GRAVY)", "gravy"),
    ("Fraction charged", "fraction_charged"),
    ("Disorder fraction", "disorder"),
    ("Disorder-promoting", "fraction_disorder_promoting"),
    ("Low-complexity fraction", "fraction_lcr"),
    ("Prion-like fraction", "prionlike_fraction"),
    ("LLPS score", "llps_score"),
    ("π–π propensity", "pipi_propensity"),
    ("Aromaticity", "aromaticity"),
    ("Instability index", "instability_index"),
    ("Shannon entropy", "shannon_entropy"),
    ("Normalized complexity", "normalized_complexity"),
]


# ──────────────────────────────────────────────────────────────────────────
# Human-readable metric labels + formatters for the expanded evidence tile.
# Keys are raw parquet column names; values describe how to render them.
# ──────────────────────────────────────────────────────────────────────────

CRITERIA_METRIC_LABELS: dict[str, dict[str, str]] = {
    # E1/E2 — conservation_frame primate + mammalian
    "isoform_conservation_frame_primate_frac_intact": {
        "label": "Fraction of primates with intact ORF",
        "format": "percent",
    },
    "isoform_conservation_frame_primate_start_codon_conserved": {
        "label": "Primates with conserved start codon",
        "format": "percent",
    },
    "isoform_conservation_frame_primate_n_species_aligned": {
        "label": "Primate species aligned",
        "format": "int",
    },
    "isoform_conservation_frame_primate_n_species_intact_frame": {
        "label": "Primate species with intact frame",
        "format": "int",
    },
    "isoform_conservation_frame_primate_mean_pident": {
        "label": "Mean % identity to primate orthologs",
        "format": "percent",
    },
    "isoform_conservation_frame_primate_deepest_species": {
        "label": "Deepest primate with intact frame",
        "format": "str",
    },
    "isoform_conservation_frame_primate_max_depth": {
        "label": "Max evolutionary depth (primates)",
        "format": "int",
    },
    "isoform_conservation_frame_mammalian_frac_intact": {
        "label": "Fraction of mammals with intact ORF",
        "format": "percent",
    },
    "isoform_conservation_frame_mammalian_start_codon_conserved": {
        "label": "Mammals with conserved start codon",
        "format": "percent",
    },
    "isoform_conservation_frame_mammalian_n_species_aligned": {
        "label": "Mammalian species aligned",
        "format": "int",
    },
    "isoform_conservation_frame_mammalian_n_species_intact_frame": {
        "label": "Mammalian species with intact frame",
        "format": "int",
    },
    "isoform_conservation_frame_mammalian_mean_pident": {
        "label": "Mean % identity to mammalian orthologs",
        "format": "percent",
    },
    "isoform_conservation_frame_mammalian_deepest_species": {
        "label": "Deepest mammal with intact frame",
        "format": "str",
    },
    "isoform_conservation_frame_mammalian_max_depth": {
        "label": "Max evolutionary depth (mammals)",
        "format": "int",
    },
    # E1/E2 — canonical (within-gene baseline) twins
    "isoform_conservation_frame_primate_canonical_mean_pident": {
        "label": "Canonical mean % identity to primate orthologs",
        "format": "percent",
    },
    "isoform_conservation_frame_primate_canonical_frac_intact": {
        "label": "Canonical fraction of primates with intact ORF",
        "format": "percent",
    },
    "isoform_conservation_frame_primate_canonical_n_species_aligned": {
        "label": "Canonical primate species aligned",
        "format": "int",
    },
    "isoform_conservation_frame_mammalian_canonical_mean_pident": {
        "label": "Canonical mean % identity to mammalian orthologs",
        "format": "percent",
    },
    "isoform_conservation_frame_mammalian_canonical_frac_intact": {
        "label": "Canonical fraction of mammals with intact ORF",
        "format": "percent",
    },
    "isoform_conservation_frame_mammalian_canonical_n_species_aligned": {
        "label": "Canonical mammalian species aligned",
        "format": "int",
    },
    # E3 — phyloP / phastCons
    "isoform_conservation_phylop_at_tis": {
        "label": "PhyloP score at TIS",
        "format": "float3",
    },
    "isoform_conservation_phylop_unique_region_mean": {
        "label": "PhyloP mean over unique region",
        "format": "float3",
    },
    "isoform_conservation_phylop_shared_region_mean": {
        "label": "PhyloP mean over shared region",
        "format": "float3",
    },
    "isoform_conservation_phylop_enrichment": {
        "label": "PhyloP enrichment (unique vs shared)",
        "format": "float3",
    },
    "isoform_conservation_phylop_kozak_mean": {
        "label": "PhyloP mean over Kozak window",
        "format": "float3",
    },
    "isoform_conservation_phastcons_unique_region_mean": {
        "label": "phastCons mean over unique region",
        "format": "float3",
    },
    "isoform_conservation_phastcons_shared_region_mean": {
        "label": "phastCons mean over shared region",
        "format": "float3",
    },
    "isoform_conservation_phastcons_at_tis": {
        "label": "phastCons at TIS",
        "format": "float3",
    },
    "isoform_conservation_phastcons_kozak_mean": {
        "label": "phastCons mean over Kozak window",
        "format": "float3",
    },
    # E5 — initiation efficiency stats
    "ribo_pvalue": {"label": "Ribo-TISH p-value", "format": "sci"},
    "tis_pvalue": {"label": "TIS detection p-value", "format": "sci"},
    "fisher_qvalue": {"label": "Fisher combined q-value", "format": "sci"},
    # E6 — mass spec
    "isoform_massspec_summary": {"label": "PepQuery2 summary", "format": "json"},
    "cmp_massspec_n_hits_in_diff_region": {
        "label": "MS peptides in diff region",
        "format": "int",
    },
    # F1 — structure pLDDT
    "isoform_structure_status": {"label": "Structure prediction status", "format": "str"},
    "isoform_structure_plddt_canonical_mean": {
        "label": "Mean pLDDT (canonical)",
        "format": "float3",
    },
    "isoform_structure_plddt_isoform_mean": {
        "label": "Mean pLDDT (isoform)",
        "format": "float3",
    },
    "isoform_structure_plddt_diffregion_mean": {
        "label": "Mean pLDDT (differential region)",
        "format": "float3",
    },
    "isoform_structure_plddt_diffregion_std": {
        "label": "pLDDT std (differential region)",
        "format": "float3",
    },
    "isoform_structure_plddt_delta_shared": {
        "label": "Δ pLDDT (isoform vs canonical, shared region)",
        "format": "float3",
    },
    # P — global fold trust (pTM) + PAE region blocks
    "isoform_structure_ptm_isoform": {"label": "pTM (isoform fold)", "format": "float3"},
    "isoform_structure_ptm_canonical": {"label": "pTM (canonical fold)", "format": "float3"},
    "isoform_structure_pae_diff_vs_diff": {
        "label": "Mean PAE within differential region (Å)",
        "format": "float2",
    },
    "isoform_structure_pae_body_vs_body": {
        "label": "Mean PAE within fold body (Å)",
        "format": "float2",
    },
    "isoform_structure_pae_diff_vs_body": {
        "label": "Mean PAE differential↔body (Å; high = dangling)",
        "format": "float2",
    },
    "isoform_structure_pae_status": {"label": "PAE availability", "format": "str"},
    "isoform_structure_extension_contacts": {
        "label": "Extension↔body Cα contacts (<8 Å)",
        "format": "int",
    },
    # F1 — biophysical distinctness (unique vs shared region)
    "cmp_biophysics_gravy_delta": {"label": "Δ GRAVY (isoform − canonical)", "format": "float3"},
    "cmp_biophysics_fraction_charged_delta": {
        "label": "Δ fraction charged (isoform − canonical)",
        "format": "float3",
    },
    "cmp_biophysics_disorder_delta": {
        "label": "Δ disorder (isoform − canonical)",
        "format": "float3",
    },
    "cmp_biophysics_pI_unique": {"label": "pI over unique region", "format": "float3"},
    "cmp_biophysics_pI_shared": {"label": "pI over shared region", "format": "float3"},
    "cmp_biophysics_pI_ratio": {"label": "pI unique/shared ratio", "format": "float3"},
    "cmp_biophysics_gravy_unique": {"label": "GRAVY over unique region", "format": "float3"},
    "cmp_biophysics_gravy_shared": {"label": "GRAVY over shared region", "format": "float3"},
    "cmp_biophysics_gravy_ratio": {"label": "GRAVY unique/shared ratio", "format": "float3"},
    "cmp_biophysics_disorder_unique": {"label": "Disorder over unique region", "format": "float3"},
    "cmp_biophysics_disorder_shared": {"label": "Disorder over shared region", "format": "float3"},
    "cmp_biophysics_disorder_ratio": {"label": "Disorder unique/shared ratio", "format": "float3"},
    "cmp_biophysics_fraction_charged_unique": {
        "label": "Fraction charged over unique region",
        "format": "float3",
    },
    "cmp_biophysics_fraction_charged_shared": {
        "label": "Fraction charged over shared region",
        "format": "float3",
    },
    "cmp_biophysics_fraction_charged_ratio": {
        "label": "Fraction charged unique/shared ratio",
        "format": "float3",
    },
    "cmp_biophysics_fraction_disorder_promoting_unique": {
        "label": "Disorder-promoting fraction over unique region",
        "format": "float3",
    },
    "cmp_biophysics_fraction_disorder_promoting_shared": {
        "label": "Disorder-promoting fraction over shared region",
        "format": "float3",
    },
    "cmp_biophysics_fraction_disorder_promoting_ratio": {
        "label": "Disorder-promoting fraction unique/shared ratio",
        "format": "float3",
    },
    # F2/F4 — localization
    "canonical_localization_deeploc_prediction": {
        "label": "DeepLoc prediction (canonical)",
        "format": "str",
    },
    "isoform_localization_deeploc_prediction": {
        "label": "DeepLoc prediction (isoform)",
        "format": "str",
    },
    "cmp_localization_deeploc_prediction_changed": {
        "label": "Predicted localization changes",
        "format": "bool",
    },
    "cmp_localization_deeploc_prediction_canonical": {
        "label": "Canonical prediction",
        "format": "str",
    },
    "cmp_localization_deeploc_prediction_isoform": {
        "label": "Isoform prediction",
        "format": "str",
    },
    "canonical_localization_deeploc_signals": {
        "label": "DeepLoc signals (canonical)",
        "format": "str",
    },
    "isoform_localization_deeploc_signals": {
        "label": "DeepLoc signals (isoform)",
        "format": "str",
    },
    "canonical_localization_deeploc_membrane": {
        "label": "Membrane state (canonical)",
        "format": "str",
    },
    "isoform_localization_deeploc_membrane": {
        "label": "Membrane state (isoform)",
        "format": "str",
    },
    "cmp_localization_deeploc_signals_changed": {
        "label": "Sorting signals change",
        "format": "bool",
    },
    "cmp_localization_deeploc_signals_canonical": {
        "label": "Canonical sorting signals",
        "format": "str",
    },
    "cmp_localization_deeploc_signals_isoform": {
        "label": "Isoform sorting signals",
        "format": "str",
    },
    "cmp_localization_deeploc_membrane_changed": {
        "label": "Membrane association changes",
        "format": "bool",
    },
    "cmp_localization_deeploc_membrane_canonical": {
        "label": "Canonical membrane state",
        "format": "str",
    },
    "cmp_localization_deeploc_membrane_isoform": {
        "label": "Isoform membrane state",
        "format": "str",
    },
    # F4 — SignalP / TargetP N-terminal sorting signals
    "cmp_signalp_signalp_prediction_changed": {
        "label": "Signal-peptide prediction changes",
        "format": "bool",
    },
    "cmp_signalp_signalp_prediction_canonical": {
        "label": "Canonical SignalP prediction",
        "format": "str",
    },
    "cmp_signalp_signalp_prediction_isoform": {
        "label": "Isoform SignalP prediction",
        "format": "str",
    },
    "cmp_signalp_signalp_probability_delta": {
        "label": "Δ SignalP probability (isoform − canonical)",
        "format": "float3",
    },
    "cmp_signalp_signalp_cleavage_site_changed": {
        "label": "Signal-peptide cleavage site changes",
        "format": "bool",
    },
    "cmp_signalp_signalp_cleavage_site_canonical": {
        "label": "Canonical SignalP cleavage site",
        "format": "str",
    },
    "cmp_signalp_signalp_cleavage_site_isoform": {
        "label": "Isoform SignalP cleavage site",
        "format": "str",
    },
    "canonical_signalp_signalp_prediction": {
        "label": "SignalP prediction (canonical)",
        "format": "str",
    },
    "isoform_signalp_signalp_prediction": {
        "label": "SignalP prediction (isoform)",
        "format": "str",
    },
    "cmp_targetp_targetp_prediction_changed": {
        "label": "Transit-peptide prediction changes",
        "format": "bool",
    },
    "cmp_targetp_targetp_prediction_canonical": {
        "label": "Canonical TargetP prediction",
        "format": "str",
    },
    "cmp_targetp_targetp_prediction_isoform": {
        "label": "Isoform TargetP prediction",
        "format": "str",
    },
    "cmp_targetp_targetp_probability_delta": {
        "label": "Δ TargetP probability (isoform − canonical)",
        "format": "float3",
    },
    "cmp_targetp_targetp_sp_prob_delta": {
        "label": "Δ TargetP signal-peptide prob",
        "format": "float3",
    },
    "cmp_targetp_targetp_mtp_prob_delta": {
        "label": "Δ TargetP mitochondrial-transit prob",
        "format": "float3",
    },
    "cmp_targetp_targetp_cleavage_site_changed": {
        "label": "Transit-peptide cleavage site changes",
        "format": "bool",
    },
    "canonical_targetp_targetp_prediction": {
        "label": "TargetP prediction (canonical)",
        "format": "str",
    },
    "isoform_targetp_targetp_prediction": {
        "label": "TargetP prediction (isoform)",
        "format": "str",
    },
    # F3 — InterProScan domains
    "isoform_interproscan_summary": {"label": "InterProScan summary", "format": "json"},
    "cmp_interproscan_n_hits_in_diff_region": {
        "label": "Domains in diff region",
        "format": "int",
    },
    # F5/F6 — variant intersection + variant effect
    "isoform_variant_intersection_gnomad_depletion_ratio": {
        "label": "gnomAD depletion ratio (unique/shared density)",
        "format": "float3",
    },
    "isoform_variant_intersection_disease_enrichment_ratio": {
        "label": "Disease enrichment ratio (unique/shared density)",
        "format": "float3",
    },
    "isoform_variant_intersection_unique_region_nt": {
        "label": "Unique region length (nt)",
        "format": "int",
    },
    "isoform_variant_intersection_shared_region_nt": {
        "label": "Shared region length (nt)",
        "format": "int",
    },
    "isoform_variant_intersection_n_gnomad_in_unique_region": {
        "label": "gnomAD variants in unique region",
        "format": "int",
    },
    "isoform_variant_intersection_n_gnomad_in_shared_region": {
        "label": "gnomAD variants in shared region",
        "format": "int",
    },
    # F5 — ESM-C (PLM VEP) constraint
    "isoform_plm_vep_status": {"label": "PLM VEP status", "format": "str"},
    "isoform_plm_vep_constraint_enrichment": {
        "label": "ESM-C constraint enrichment (unique vs shared)",
        "format": "float3",
    },
    "isoform_plm_vep_mean_llr_unique_region": {
        "label": "Mean ESM-C LLR over unique region",
        "format": "float3",
    },
    "isoform_plm_vep_mean_llr_shared_region": {
        "label": "Mean ESM-C LLR over shared region",
        "format": "float3",
    },
    "isoform_plm_vep_n_constrained_positions_unique": {
        "label": "ESM-C constrained positions (unique)",
        "format": "int",
    },
    "isoform_plm_vep_n_constrained_positions_shared": {
        "label": "ESM-C constrained positions (shared)",
        "format": "int",
    },
    "isoform_variant_intersection_n_total": {
        "label": "Total clinical variants over isoform",
        "format": "int",
    },
    "isoform_variant_intersection_n_in_unique_region": {
        "label": "Variants in unique region",
        "format": "int",
    },
    "isoform_variant_intersection_n_in_shared_region": {
        "label": "Variants in shared region",
        "format": "int",
    },
    "isoform_variant_intersection_n_disease_in_unique_region": {
        "label": "Disease variants in unique region",
        "format": "int",
    },
    "isoform_variant_intersection_n_disease_in_shared_region": {
        "label": "Disease variants in shared region",
        "format": "int",
    },
    "isoform_variant_intersection_n_pathogenic_in_unique_region": {
        "label": "Pathogenic in unique region",
        "format": "int",
    },
    "isoform_variant_intersection_n_pathogenic_in_shared_region": {
        "label": "Pathogenic in shared region",
        "format": "int",
    },
    "isoform_variant_intersection_n_dropped_outside_coding": {
        "label": "Variants dropped outside coding",
        "format": "int",
    },
    "isoform_varianteffect_n_scorable_in_unique_gnomad": {
        "label": "Scorable germline variants (unique)",
        "format": "int",
    },
    "isoform_varianteffect_n_damaging_in_unique_gnomad": {
        "label": "Damaging germline predictions (AM+PLM)",
        "format": "int",
    },
    "isoform_varianteffect_n_lof_in_unique_gnomad": {
        "label": "LoF germline variants (unique)",
        "format": "int",
    },
    "isoform_varianteffect_mean_delta_llr_unique_gnomad": {
        "label": "Mean ESM-C ΔLLR, germline (unique)",
        "format": "float3",
    },
    "isoform_varianteffect_min_delta_llr_unique_gnomad": {
        "label": "Min ESM-C ΔLLR, germline (unique)",
        "format": "float3",
    },
    "isoform_varianteffect_mean_am_pathogenicity_unique_gnomad": {
        "label": "Mean AlphaMissense pathogenicity, germline (unique)",
        "format": "float3",
    },
    # F7 — shared-region structural change (structure)
    "isoform_structure_rmsd_shared": {
        "label": "Shared-region Cα RMSD",
        "format": "angstrom",
    },
    "isoform_structure_tm_score_shared": {
        "label": "Shared-region TM-score",
        "format": "float3",
    },
    "isoform_structure_shared_region_len": {
        "label": "Shared region length (aa)",
        "format": "int",
    },
    "isoform_structure_rmsd_shared_status": {
        "label": "Shared-region RMSD status",
        "format": "str",
    },
    "isoform_structure_plddt_shared_mean_isoform": {
        "label": "Mean shared-region pLDDT (isoform)",
        "format": "float3",
    },
    "isoform_structure_plddt_shared_mean_canonical": {
        "label": "Mean shared-region pLDDT (canonical)",
        "format": "float3",
    },
    "isoform_structure_rmsd_global": {
        "label": "Global Cα RMSD (tm-align)",
        "format": "angstrom",
    },
    "isoform_structure_tm_score": {
        "label": "Global TM-score",
        "format": "float3",
    },
}

# E4/E5 — per-cell-line expression columns (generated programmatically)
for _sample in ("HeLa", "K562", "U2OS", "RPE1_Async", "RPE1_Que", "RPE1_Sen"):
    _display = _sample.replace("_", " ")
    CRITERIA_METRIC_LABELS[f"expr_{_sample}_initiation_efficiency"] = {
        "label": f"{_display}: initiation efficiency",
        "format": "float3",
    }
    CRITERIA_METRIC_LABELS[f"expr_{_sample}_p_value"] = {
        "label": f"{_display}: p-value",
        "format": "sci",
    }
    CRITERIA_METRIC_LABELS[f"expr_{_sample}_cpm"] = {
        "label": f"{_display}: CPM",
        "format": "float3",
    }
    CRITERIA_METRIC_LABELS[f"expr_{_sample}_raw_count"] = {
        "label": f"{_display}: raw count",
        "format": "int",
    }
    CRITERIA_METRIC_LABELS[f"canonical_expr_{_sample}_initiation_efficiency"] = {
        "label": f"{_display}: canonical initiation efficiency",
        "format": "float3",
    }

# L1 — DeepLoc per-compartment probabilities + top-class confidence (both sides)
for _side in ("isoform", "canonical"):
    CRITERIA_METRIC_LABELS[f"{_side}_localization_deeploc_top_prob"] = {
        "label": f"DeepLoc top-class probability ({_side})",
        "format": "float3",
    }
    for _compartment, _suffix in (
        ("Cytoplasm", "cytoplasm"),
        ("Nucleus", "nucleus"),
        ("Extracellular", "extracellular"),
        ("Cell membrane", "cell_membrane"),
        ("Mitochondrion", "mitochondrion"),
        ("Endoplasmic reticulum", "endoplasmic_reticulum"),
        ("Golgi apparatus", "golgi_apparatus"),
        ("Lysosome/Vacuole", "lysosome_vacuole"),
        ("Peroxisome", "peroxisome"),
    ):
        CRITERIA_METRIC_LABELS[f"{_side}_localization_deeploc_prob_{_suffix}"] = {
            "label": f"DeepLoc P({_compartment}) ({_side})",
            "format": "float3",
        }


def format_metric(value: Any, fmt: str) -> str:
    """Format a metric value for display per a small spec of format codes.

    Args:
        value: Raw scalar value from the parquet row.
        fmt: One of ``"percent"``, ``"int"``, ``"float3"``, ``"angstrom"``,
            ``"sci"``, ``"bool"``, ``"json"``, or ``"str"`` (default).

    Returns:
        Display string. ``None`` / NaN values render as an em dash.
    """
    if value is None:
        return "—"
    try:
        if isinstance(value, float) and math.isnan(value):
            return "—"
    except (TypeError, ValueError):
        pass
    if fmt == "percent":
        try:
            return f"{float(value) * 100:.1f}%"
        except (TypeError, ValueError):
            return str(value)
    if fmt == "int":
        try:
            return f"{int(value)}"
        except (TypeError, ValueError):
            return str(value)
    if fmt == "float3":
        try:
            return f"{float(value):.3g}"
        except (TypeError, ValueError):
            return str(value)
    if fmt == "angstrom":
        try:
            return f"{float(value):.2f} Å"
        except (TypeError, ValueError):
            return str(value)
    if fmt == "sci":
        try:
            return f"{float(value):.2e}"
        except (TypeError, ValueError):
            return str(value)
    if fmt == "bool":
        if isinstance(value, bool):
            return "Yes" if value else "No"
        return str(value)
    if fmt == "json":
        try:
            s = json.dumps(value, default=str)
        except (TypeError, ValueError):
            s = str(value)
        return s[:120] + ("…" if len(s) > 120 else "")
    return str(value)


def slice_criterion(isoform_record: dict[str, Any], criterion_id: str) -> dict[str, Any]:
    """Build the per-(isoform, criterion) slice for V2 UI tiles + future LLM passes.

    Args:
        isoform_record: A full per-isoform record (one entry from
            ``build_gene_record(...)["isoforms"]``) with the ``"_raw"`` mirror
            of the parquet row.
        criterion_id: One of the keys in ``CRITERIA``.

    Returns:
        Dict with:
          - ``criterion_id`` and ``axis``, ``label``, ``short_label``, ``interpretation_hint``
          - ``isoform`` identity block (tis_id, gene_name, orf_type, etc.)
          - ``value`` (True/False/None from isoform_scoring_criteria[criterion_id])
          - ``reason`` (string from isoform_scoring_reasons[criterion_id])
          - ``headline`` (the value at headline_col, or None)
          - ``headline_fmt`` (format code for rendering ``headline`` via ``format_metric``)
          - ``evidence`` (dict of {col_name: value} for all evidence_cols)
          - ``hits`` (list of dicts from evidence_hits_col if present, else [])
    """
    if criterion_id not in CRITERIA:
        raise KeyError(f"Unknown criterion: {criterion_id!r}")
    cfg = CRITERIA[criterion_id]
    raw = isoform_record.get("_raw") or {}
    scoring = isoform_record.get("scoring") or {}
    criteria_map = scoring.get("criteria") or {}
    criterion_entry = criteria_map.get(criterion_id, {}) or {}

    iso_block = {
        "tis_id": isoform_record.get("tis_id"),
        "gene_name": (isoform_record.get("gene") or {}).get("name"),
        "orf_type": isoform_record.get("orf_type"),
        "differential_sequence": isoform_record.get("differential_sequence"),
        "diff_space": isoform_record.get("diff_space"),
        "isoform_length_aa": isoform_record.get("isoform_length_aa"),
        "canonical_length_aa": isoform_record.get("canonical_length_aa"),
    }

    # Criteria whose evidence is nested (S2 biophysics table / S3 SAE features)
    # delegate evidence construction to their ``evidence_builder`` hook and skip
    # the flat evidence_cols + headline_col machinery. The return shape is
    # identical to a normal criterion so every downstream consumer is uniform;
    # ``evidence`` is empty (→ omitted by ``slice_category`` when ``omit_if_empty``)
    # when the builder has no data.
    builder = cfg.get("evidence_builder")
    if builder is not None:
        built = builder(isoform_record) or {}
        return {
            "criterion_id": criterion_id,
            "axis": cfg["axis"],
            "label": cfg["label"],
            "short_label": cfg["short_label"],
            "interpretation_hint": cfg["interpretation_hint"],
            "isoform": iso_block,
            "value": criterion_entry.get("value"),
            "reason": criterion_entry.get("reason"),
            "headline": built.get("headline"),
            "headline_fmt": "str",
            "headline_segments": built.get("headline_segments"),
            "evidence": built.get("evidence", {}),
            "hits": [],
            "n_hits_total": 0,
            "n_hits_shown": 0,
        }

    evidence = {col: raw.get(col) for col in cfg["evidence_cols"]}
    headline_col = cfg.get("headline_col")
    # Styled headline: a list of {"t", "strong"} segments so composed taglines can
    # mute the label parts and bold only the important value(s). None → plain path.
    headline_segments = None
    if headline_col == _START_SITE_USAGE:
        # "alt used {r}× vs canonical": how hard the alternative start is used
        # relative to the canonical start (both = TIS counts / gene RNA-seq
        # counts, normalized per cell line), taken in the cell line where the
        # alt is used most AND a canonical reference is measured. Falls back to
        # the absolute max alt efficiency when no cell line has a canonical ref.
        def _num(x: Any) -> float | None:
            return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)

        max_alt = None  # overall highest alt efficiency (fallback)
        best = None  # (alt_eff, ratio) at highest alt among cells with a canonical ref
        for s in _INITIATION_EFFICIENCY_SAMPLES:
            a = _num(raw.get(f"expr_{s}_initiation_efficiency"))
            if a is None:
                continue
            if max_alt is None or a > max_alt:
                max_alt = a
            c = _num(raw.get(f"canonical_expr_{s}_initiation_efficiency"))
            if c is not None and c > 0 and (best is None or a > best[0]):
                best = (a, a / c)
        if best is not None:
            headline = f"alt used {best[1]:.2g}× vs canonical"
            headline_segments = [
                {"t": "alt used ", "strong": False},
                {"t": f"{best[1]:.2g}×", "strong": True},
                {"t": " vs canonical", "strong": False},
            ]
        elif max_alt is not None:
            headline = f"Max Initiation Efficiency: {max_alt:.3g}"
            headline_segments = [
                {"t": "Max Initiation Efficiency: ", "strong": False},
                {"t": f"{max_alt:.3g}", "strong": True},
            ]
        else:
            headline = None
        headline_fmt = "str"
    elif headline_col == _SSE_HEADLINE:
        # "17-res helix (pLDDT 0.75)" — the longest CONFIDENT element, which is
        # what P3 scores on. Falls back to naming the longest element and its
        # shortfall so a near-miss is visible rather than reading as "nothing".
        # `or []` would call __bool__ on a numpy object array and raise; the hits
        # branch below avoids the same trap with an explicit None check.
        _all = raw.get("isoform_structure_sse_diff_elements")
        els = [e for e in (_all if _all is not None else []) if isinstance(e, dict)]
        if not els:
            ok = raw.get("isoform_structure_sse_status") == "ok"
            headline = "No helix or strand in region" if ok else None
        else:
            # Must agree with P3, which requires BOTH length and confidence. A
            # headline naming a sub-threshold element reads as a finding while
            # the criterion scores False — the tile and the verdict then tell a
            # reader opposite things.
            ok = [
                e for e in els
                if (e.get("length") or 0) >= _P3_MIN_LEN
                and (e.get("plddt_mean") or 0) >= _P3_MIN_PLDDT
            ]
            if ok:
                best = max(ok, key=lambda e: e.get("length") or 0)
                n, kind = best.get("length"), best.get("type")
                pl = best.get("plddt_mean")
                headline = f"{n}-res {kind}" + (f" (pLDDT {pl:.2f})" if pl is not None else "")
                headline_segments = [
                    {"t": f"{n}-res {kind}", "strong": True},
                    {"t": f" (pLDDT {pl:.2f})" if pl is not None else "", "strong": False},
                ]
            else:
                # Keep the near-miss visible rather than reading as "nothing".
                longest = max(els, key=lambda e: e.get("length") or 0)
                n, kind = longest.get("length"), longest.get("type")
                headline = f"longest {kind} {n} aa — below threshold"
                headline_segments = [
                    {"t": f"longest {kind} {n} aa", "strong": True},
                    {"t": " — below threshold", "strong": False},
                ]
        headline_fmt = "str"
    elif headline_col == _CODING_SELECTION:
        # Mean PhyloP over the isoform-unique region + a selection call: PhyloP
        # measures deviation from neutral evolution, so >0 = conserved (purifying
        # selection), ~0 = neutral, <0 = accelerated. Neutral band = ±1.0.
        x = raw.get("isoform_conservation_phylop_unique_region_mean")
        if x is None or (isinstance(x, float) and math.isnan(x)):
            headline = None
        else:
            x = float(x)
            call = "purifying" if x >= 1.0 else "accelerated" if x <= -1.0 else "neutral"
            headline = f"Unique region PhyloP: {x:.2f} — {call} selection"
            headline_segments = [
                {"t": "Unique region PhyloP: ", "strong": False},
                {"t": f"{x:.2f}", "strong": True},
                {"t": " — ", "strong": False},
                {"t": f"{call} selection", "strong": True},
            ]
        headline_fmt = "str"
    elif headline_col == _N_CELL_LINES_DETECTED:
        # "detected in n/m cell lines": n = cell lines with an expression record
        # for this TIS (present per-sample column, mirrors the E4 scorer's
        # ``len(site.expression)``); m = the full cell-line panel.
        n = 0
        for s in _INITIATION_EFFICIENCY_SAMPLES:
            v = raw.get(f"expr_{s}_p_value")
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                n += 1
        m = len(_INITIATION_EFFICIENCY_SAMPLES)
        headline = f"detected in {n}/{m} cell lines"
        headline_segments = [
            {"t": "detected in ", "strong": False},
            {"t": f"{n}/{m}", "strong": True},
            {"t": " cell lines", "strong": False},
        ]
        headline_fmt = "str"
    elif headline_col == _MASSPEC_VALIDATED:
        # "{v}/{u} isoform-unique peptides validated": v = isoform-unique tryptic
        # peptides matched to public MS spectra by PepQuery2 (the E6 score
        # numerator), u = total isoform-unique peptides searched. The isoform
        # digest is scoped to unique peptides, so ``validated_peptides`` == the
        # validated-unique count. Unscored (—) until PepQuery has run.
        summary = raw.get("isoform_massspec_summary")
        if isinstance(summary, dict) and summary.get("pepquery_run"):
            v = summary.get("validated_peptides") or 0
            u = summary.get("unique_peptides")
            if u is None:
                u = summary.get("total_peptides") or 0
            headline = f"{v}/{u} isoform-unique peptides validated"
            headline_segments = [
                {"t": f"{v}/{u}", "strong": True},
                {"t": " isoform-unique peptides validated", "strong": False},
            ]
        else:
            headline = None
        headline_fmt = "str"
    elif headline_col == _DIVERGING_DOMAINS:
        # Curated count of real domains changed in the differential region — the
        # same value the F3 score uses (>=1 passes). Rendered "n diverging domains".
        n = raw.get("cmp_interproscan_n_real_domains_changed_in_diff_region")
        if n is None or (isinstance(n, float) and math.isnan(n)):
            headline = None
        else:
            n = int(n)
            noun = "domain" if n == 1 else "domains"
            count = "No" if n == 0 else str(n)
            headline = f"{count} diverging {noun}"
            headline_segments = [
                {"t": count, "strong": True},
                {"t": f" diverging {noun}", "strong": False},
            ]
        headline_fmt = "str"
    elif headline_col in _VARIANT_FOLD_HEADLINES:
        # Fold-change of variant density in the unique region vs baseline, shown
        # as "{noun} {fold}x more/less in unique region — {call}". gnomAD
        # (germline): fewer = constrained, more = tolerant. Disease
        # (ClinVar/COSMIC): more = enriched, fewer = depleted. Within 1.1x
        # either way = neutral.
        spec = _VARIANT_FOLD_HEADLINES[headline_col]
        r = raw.get(spec["col"])
        if r is None or (isinstance(r, float) and math.isnan(r)) or float(r) <= 0:
            headline = None
        else:
            r = float(r)
            fold = r if r >= 1 else 1.0 / r
            if fold < 1.1:
                headline = f"{spec['noun']} comparable in unique region — neutral"
                headline_segments = [
                    {"t": f"{spec['noun']} ", "strong": False},
                    {"t": "comparable", "strong": True},
                    {"t": " in unique region — ", "strong": False},
                    {"t": "neutral", "strong": True},
                ]
            else:
                direction = "more" if r > 1 else "less"
                call = spec["high"] if r > 1 else spec["low"]
                headline = f"{spec['noun']} {fold:.2f}× {direction} in unique region — {call}"
                headline_segments = [
                    {"t": f"{spec['noun']} ", "strong": False},
                    {"t": f"{fold:.2f}× {direction}", "strong": True},
                    {"t": " in unique region — ", "strong": False},
                    {"t": call, "strong": True},
                ]
        headline_fmt = "str"
    elif headline_col in _SIMILARITY_HEADLINES:
        # Mean AA % identity of the differential (unique) region to orthologs
        # across the clade. Rendered as "Unique region {x}% similar across {clade}"
        # — neutral wording that fits both extensions (isoform-unique gained
        # region) and truncations (lost canonical region).
        col, clade = _SIMILARITY_HEADLINES[headline_col]
        x = raw.get(col)
        if x is None or (isinstance(x, float) and math.isnan(x)):
            headline = None
        else:
            pct = float(x) * 100
            headline = f"Unique region {pct:.1f}% similar across {clade}"
            headline_segments = [
                {"t": "Unique region ", "strong": False},
                {"t": f"{pct:.1f}%", "strong": True},
                {"t": f" similar across {clade}", "strong": False},
            ]
        headline_fmt = "str"
    elif headline_col in _ISO_CANON_HEADLINES:
        iso_col, canon_col = _ISO_CANON_HEADLINES[headline_col]

        def _pred(x: Any) -> str:
            if x is None or (isinstance(x, float) and math.isnan(x)):
                return "—"
            # DeepLoc joins multiple locations with "|"; render as "a, b" so the
            # "|" reads only as the iso/canon separator.
            return str(x).replace("|", ", ")

        iso_v = _pred(raw.get(iso_col))
        canon_v = _pred(raw.get(canon_col))
        headline = f"iso: {iso_v} | canon: {canon_v}"
        headline_segments = [
            {"t": "iso: ", "strong": False},
            {"t": iso_v, "strong": True},
            {"t": " | canon: ", "strong": False},
            {"t": canon_v, "strong": True},
        ]
        headline_fmt = "str"
    else:
        headline = raw.get(headline_col) if headline_col else None
        headline_fmt = CRITERIA_METRIC_LABELS.get(headline_col, {}).get("format", "str")

    hits: list[dict[str, Any]] = []
    n_hits_total = 0
    hits_col = cfg.get("evidence_hits_col")
    if hits_col:
        raw_hits = raw.get(hits_col)
        if raw_hits is not None:
            try:
                # Handle numpy arrays + lists
                all_hits = [h for h in raw_hits if isinstance(h, dict)]
                n_hits_total = len(all_hits)
                # Cap at MAX_HITS to keep the LLM prompt under the 200k token limit.
                # F5/F6 are unique-region claims, so prioritise unique-region hits
                # first (then pathogenic/damaging within that), ensuring the
                # truncated view always surfaces the region the criterion is about.
                MAX_HITS = 30
                # Criteria whose claim is about the isoform-unique region.
                unique_region_criteria = {
                    "M1_pathogenic_variant_enrichment",
                    "M2_clinical_variant_overlap",
                }
                prioritize_unique = criterion_id in unique_region_criteria
                if n_hits_total > MAX_HITS:

                    def _priority(h: dict[str, Any]) -> tuple[int, int]:
                        # Lead with unique-region membership for unique-region
                        # criteria so those hits survive truncation; clinical
                        # significance is the secondary sort within each bucket.
                        in_unique = 0 if (prioritize_unique and h.get("in_isoform_unique")) else 1
                        return (in_unique, clinsig_rank(h))

                    hits = sorted(all_hits, key=_priority)[:MAX_HITS]
                else:
                    hits = all_hits
            except TypeError:
                hits = []

    return {
        "criterion_id": criterion_id,
        "axis": cfg["axis"],
        "label": cfg["label"],
        "short_label": cfg["short_label"],
        "interpretation_hint": cfg["interpretation_hint"],
        "isoform": iso_block,
        "value": criterion_entry.get("value"),
        "reason": criterion_entry.get("reason"),
        "headline": headline,
        "headline_fmt": headline_fmt,
        "headline_segments": headline_segments,
        "evidence": evidence,
        "hits": hits,
        "n_hits_total": n_hits_total,
        "n_hits_shown": len(hits),
    }


def _diff_region_location(orf_type: Any) -> str | None:
    """Explicit N-terminal directionality fact for the identity block.

    Alt-TIS isoforms differ ONLY at the N-terminus (the start codon moves; the
    C-terminus and stop are invariant). Stated as a hard input fact so the LLM
    cannot mislabel the differential region as C-terminal from a training prior
    (e.g. CDC34's well-known C-terminal tail). Returns None when there is no
    differential region (annotated) or the orf_type is unknown.
    """
    if orf_type is None:
        return None
    s = str(orf_type).strip().lower()
    if s == "truncated":
        return (
            "N-terminal: the differential region is the N-terminal segment of the "
            "CANONICAL protein that this isoform REMOVES (the alt start codon is "
            "downstream); the shared C-terminal portion is retained unchanged. The "
            "removed region is NOT C-terminal."
        )
    if s == "extended":
        return (
            "N-terminal: the differential region is the N-terminal segment this "
            "isoform ADDS ahead of the canonical start; the canonical protein is "
            "retained unchanged downstream. The added region is NOT C-terminal."
        )
    if s in {"uorf", "uoorf", "internal_oof", "3utr_orf", "alt_orf"}:
        return (
            "Separate ORF: the entire isoform sequence is the differential region "
            "and does not share reading frame with the canonical CDS; there is no "
            "shared region."
        )
    return None


def _iso_identity_block(isoform_record: dict[str, Any]) -> dict[str, Any]:
    """Shared isoform-identity block used by criterion + category slices."""
    return {
        "tis_id": isoform_record.get("tis_id"),
        "gene_name": (isoform_record.get("gene") or {}).get("name"),
        "orf_type": isoform_record.get("orf_type"),
        "differential_region_location": _diff_region_location(isoform_record.get("orf_type")),
        "differential_sequence": isoform_record.get("differential_sequence"),
        "diff_space": isoform_record.get("diff_space"),
        "isoform_length_aa": isoform_record.get("isoform_length_aa"),
        "canonical_length_aa": isoform_record.get("canonical_length_aa"),
    }


def _biophysics_evidence(isoform_record: dict[str, Any]) -> dict[str, Any] | None:
    """S2 biophysics evidence builder — the ``evidence_builder`` hook for CRITERIA.

    Reads the ``cmp_biophysics_<feat>_{unique,shared,ratio}`` differential columns
    (plus the three GRAVY/charge/disorder whole-protein deltas) into a compact
    numeric ``evidence`` dict. Numbers only — no HTML, no UI formatting. The
    surrounding ``slice_criterion`` supplies the criterion identity + scored
    value/reason; this builds only the nested evidence the flat ``evidence_cols``
    model cannot express. Returns ``None`` when no biophysics comparison columns
    are present (→ S2 slice carries empty evidence and is omitted from the
    category display via ``omit_if_empty``).
    """
    raw = isoform_record.get("_raw") or {}
    evidence: dict[str, Any] = {}
    for label, feat in _BIOPHYSICS_FEATURES:
        vals = {
            "unique": _to_native(raw.get(f"cmp_biophysics_{feat}_unique")),
            "shared": _to_native(raw.get(f"cmp_biophysics_{feat}_shared")),
            "ratio": _to_native(raw.get(f"cmp_biophysics_{feat}_ratio")),
            "enriched": _to_native(raw.get(f"cmp_biophysics_{feat}_enriched")),
        }
        if all(v is None for v in (vals["unique"], vals["shared"], vals["ratio"])):
            continue
        evidence[label] = vals
    # The three whole-protein deltas (context alongside the region-vs-core levers).
    for feat in ("gravy_delta", "fraction_charged_delta", "disorder_delta"):
        v = _to_native(raw.get(f"cmp_biophysics_{feat}"))
        if v is not None:
            evidence[feat] = v
    if not evidence:
        return None
    return {"evidence": evidence}


def _strip_hollow_label(rec: dict[str, Any]) -> dict[str, Any]:
    """Drop an SAE feature label (and its description) when it carries no content.

    Roughly half of top SAE features are labelled "Unknown generic feature" by the
    ESM-Atlas dictionary. Handing the model a hollow noun invites it to reason from
    a label that means "we don't know what this is", so the label and its
    auto-generated description are removed and the feature is left identified by
    index + magnitude only. Labels with content are passed through unchanged.
    """
    label = rec.get("label")
    if isinstance(label, str) and "unknown" in label.lower():
        rec = {k: v for k, v in rec.items() if k not in ("label", "description")}
    return rec


def _sae_evidence(isoform_record: dict[str, Any]) -> dict[str, Any] | None:
    """S3 SAE-feature evidence builder — the ``evidence_builder`` hook for CRITERIA.

    Reads the ``isoform_sae_*`` columns: interpretable-feature counts, the top
    gained/lost features, and the isoform-unique-region features (capped). The
    surrounding ``slice_criterion`` supplies the criterion identity + scored
    value/reason; this builds only the nested evidence. Returns ``None`` when the
    SAE step did not run (status != "ok") (→ S3 slice carries empty evidence and
    is omitted from the category display via ``omit_if_empty``).
    """
    raw = isoform_record.get("_raw") or {}
    if raw.get("isoform_sae_status") != "ok":
        return None

    def _records(value: Any, cap: int = 15) -> list[dict[str, Any]]:
        native = _to_native(value)
        if not isinstance(native, list):
            return []
        recs = [_strip_hollow_label(r) for r in native if isinstance(r, dict)]
        return recs[:cap]

    def _top(prefix: str) -> dict[str, Any] | None:
        idx = _to_native(raw.get(f"isoform_sae_top_{prefix}_feature_index"))
        if idx is None:
            return None
        rec = {
            "feature_index": idx,
            "label": _to_native(raw.get(f"isoform_sae_top_{prefix}_feature_label")),
            "delta_max": _to_native(raw.get(f"isoform_sae_top_{prefix}_delta_max")),
        }
        return _strip_hollow_label(rec)

    evidence = {
        "counts": {
            "isoform_only": _to_native(raw.get("isoform_sae_n_isoform_only")),
            "canonical_only": _to_native(raw.get("isoform_sae_n_canonical_only")),
            "shared": _to_native(raw.get("isoform_sae_n_shared")),
        },
        "mean_abs_delta_shared": _to_native(raw.get("isoform_sae_mean_abs_delta_shared")),
        "unique_region_space": _to_native(raw.get("isoform_sae_unique_region_space")),
        "n_unique_region_features": _to_native(raw.get("isoform_sae_n_unique_region_features")),
        "top_gained": _top("gained"),
        "top_lost": _top("lost"),
        "unique_region_features": _records(raw.get("isoform_sae_unique_region_top_features")),
    }
    return {"evidence": evidence}


# Attach the evidence-builder hooks now that the builders are defined. S2/S3 carry
# nested evidence (biophysics property table / SAE feature records) that the flat
# ``evidence_cols`` model cannot express, so ``slice_criterion`` delegates their
# evidence construction to these. Set here (not in the CRITERIA literal) because
# the builders are defined after the dict.
CRITERIA["S2_biophysics"]["evidence_builder"] = _biophysics_evidence
CRITERIA["S3_sae"]["evidence_builder"] = _sae_evidence


def slice_category(isoform_record: dict[str, Any], category: dict[str, Any]) -> dict[str, Any]:
    """Bundle every member of one CDLMPS category into a single LLM-input slice.

    Args:
        isoform_record: A full per-isoform record (as passed to ``slice_criterion``).
        category: One entry from ``CATEGORIES`` (``{letter, name, members}``).

    Returns:
        Dict with the category identity (``letter``, ``name``), the shared isoform
        identity block, and ``members`` — a list of per-member slices. Every member
        is a first-class criterion sliced through ``slice_criterion`` (all tagged
        ``kind="criterion"``); a member whose ``CRITERIA`` entry sets
        ``omit_if_empty`` and produced no evidence (e.g. S2/S3 when biophysics/SAE
        did not run) is dropped from the display.

        Each member's own ``isoform`` block is DROPPED here: the category-level
        block above carries the same identity (plus ``differential_region_location``),
        so keeping the per-member copies repeated it once per member — four
        near-identical blocks in a 3-member category, 5-10% of the payload.
        ``slice_criterion`` still returns the block for standalone callers (the
        website UI tiles, the synthesis pass), which have no outer block to
        inherit from; only the bundled form drops it.
    """
    members: list[dict[str, Any]] = []
    for member in category["members"]:
        if member not in CRITERIA:  # pragma: no cover - guards a stale CATEGORIES entry
            raise KeyError(f"Unknown category member: {member!r}")
        entry = slice_criterion(isoform_record, member)
        if CRITERIA[member].get("omit_if_empty") and not entry["evidence"]:
            continue
        entry["kind"] = "criterion"
        entry.pop("isoform", None)
        members.append(entry)

    return {
        "category": category["letter"],
        "name": category["name"],
        "isoform": _iso_identity_block(isoform_record),
        "members": members,
    }
