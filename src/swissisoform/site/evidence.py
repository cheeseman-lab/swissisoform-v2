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
_MAX_INITIATION_EFFICIENCY = "__max_initiation_efficiency__"

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


def _diff_space_from_orf_type(orf_type: Any) -> str | None:
    if orf_type is None:
        return None
    s = str(orf_type).strip().lower()
    if not s:
        return None
    return "canonical" if s == "truncated" else "isoform"


def _build_scoring(row: pd.Series) -> dict[str, Any]:
    """Pack E1–E6 / F1–F6 criteria + reasons into the spec's nested shape."""
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
            "function": _scalar_or_none(head, "generef_uniprot_function"),
            "subcellular_location": _scalar_or_none(head, "generef_subcellular_location"),
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
    "E1_primate_conservation": {
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
            "isoform_conservation_frame_primate_canonical_frac_intact",
            "isoform_conservation_frame_primate_canonical_n_species_aligned",
        ],
        "headline_col": "isoform_conservation_frame_primate_mean_pident",
        "interpretation_hint": (
            "Is the alternative reading frame conserved across primates? Score on "
            "mean_pident (mean amino-acid % identity to primate orthologs); "
            "frac_intact and start_codon_conserved are context. Compare to the "
            "_canonical_ twins for a within-gene baseline."
        ),
    },
    "E2_mammalian_conservation": {
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
            "isoform_conservation_frame_mammalian_canonical_frac_intact",
            "isoform_conservation_frame_mammalian_canonical_n_species_aligned",
        ],
        "headline_col": "isoform_conservation_frame_mammalian_mean_pident",
        "interpretation_hint": (
            "Is the alternative reading frame conserved deeper in mammals? Score on "
            "mean_pident (mean amino-acid % identity to mammalian orthologs); "
            "frac_intact is context. Compare to the _canonical_ twins for a "
            "within-gene baseline."
        ),
    },
    "E3_phylop_coding_selection": {
        "axis": "E",
        "label": "PhyloP coding selection",
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
        "headline_col": "isoform_conservation_phylop_unique_region_mean",
        "interpretation_hint": (
            "Does the unique coding region show purifying selection by phyloP "
            "(absolute mean ≥ ~2 indicates strong constraint)? The shared region "
            "and enrichment ratio are context only, not the basis for the call."
        ),
    },
    "E4_multi_cell_line": {
        "axis": "E",
        "label": "Multi cell line expression",
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
        "headline_col": None,  # multi-sample — UI shows a small table
        "interpretation_hint": (
            "Is the TIS reproducibly detected across multiple cell lines? Count "
            "samples with significant p-values."
        ),
    },
    "E5_initiation_efficiency": {
        "axis": "E",
        "label": "Initiation efficiency",
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
        "headline_col": _MAX_INITIATION_EFFICIENCY,
        "interpretation_hint": (
            "How efficiently is this TIS initiated? Each per-cell-line value is the "
            "TIS read counts / gene RNA-seq counts ratio; the headline is the max "
            "across cell lines. Compare to the canonical_ twins for a within-gene "
            "baseline."
        ),
    },
    "E6_mass_spec": {
        "axis": "E",
        "label": "Mass spec",
        "short_label": "MS",
        "evidence_cols": [
            "isoform_massspec_summary",
            "cmp_massspec_n_hits_in_diff_region",
        ],
        "evidence_hits_col": "cmp_massspec_hits_in_diff_region",
        "headline_col": "cmp_massspec_n_hits_in_diff_region",
        "interpretation_hint": (
            "Are there PepQuery2 validated peptides in the isoform's unique region?"
        ),
    },
    "F1_structured_extension": {
        "axis": "F",
        "label": "Structured extension",
        "short_label": "Folding",
        "evidence_cols": [
            "isoform_structure_status",
            "isoform_structure_plddt_canonical_mean",
            "isoform_structure_plddt_isoform_mean",
            "isoform_structure_plddt_diffregion_mean",
            "isoform_structure_plddt_diffregion_std",
            "isoform_structure_plddt_delta_shared",
            "cmp_biophysics_pI_unique",
            "cmp_biophysics_pI_shared",
            "cmp_biophysics_pI_ratio",
            "cmp_biophysics_gravy_unique",
            "cmp_biophysics_gravy_shared",
            "cmp_biophysics_gravy_ratio",
            "cmp_biophysics_disorder_unique",
            "cmp_biophysics_disorder_shared",
            "cmp_biophysics_disorder_ratio",
            "cmp_biophysics_fraction_charged_unique",
            "cmp_biophysics_fraction_charged_shared",
            "cmp_biophysics_fraction_charged_ratio",
            "cmp_biophysics_fraction_disorder_promoting_unique",
            "cmp_biophysics_fraction_disorder_promoting_shared",
            "cmp_biophysics_fraction_disorder_promoting_ratio",
            "cmp_biophysics_gravy_delta",
            "cmp_biophysics_fraction_charged_delta",
            "cmp_biophysics_disorder_delta",
        ],
        "headline_col": "isoform_structure_plddt_diffregion_mean",
        "interpretation_hint": (
            "Does the unique region fold confidently (Boltz pLDDT) AND look "
            "biophysically distinct from the canonical core? Higher diffregion_mean "
            "means more structured; the cmp_biophysics unique-vs-shared deltas and "
            "ratios (GRAVY, fraction_charged, disorder) report distinctness."
        ),
    },
    "F2_localization_change": {
        "axis": "F",
        "label": "Localization change",
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
        ],
        "headline_col": "cmp_localization_deeploc_prediction_changed",
        "interpretation_hint": (
            "Do the isoform's localization features change vs canonical "
            "(DeepLoc prediction / sorting signals / membrane association)?"
        ),
    },
    "F3_domain_change": {
        "axis": "F",
        "label": "Domain change",
        "short_label": "Domains",
        "evidence_cols": [
            "isoform_interproscan_summary",
            "cmp_interproscan_n_hits_in_diff_region",
        ],
        "evidence_hits_col": "cmp_interproscan_hits_in_diff_region",
        "headline_col": "cmp_interproscan_n_hits_in_diff_region",
        "interpretation_hint": ("Does the differential region overlap with InterProScan domains?"),
    },
    "F4_targeting_change": {
        "axis": "F",
        "label": "Targeting change",
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
        "headline_col": "cmp_signalp_signalp_prediction_changed",
        "interpretation_hint": (
            "Do N-terminal sorting signals differ between canonical and isoform — a "
            "secretory signal peptide (SignalP) or a mitochondrial/chloroplast "
            "transit peptide (TargetP)?"
        ),
    },
    "F5_pathogenic_variant_enrichment": {
        "axis": "F",
        "label": "Germline tolerance",
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
        "headline_col": "isoform_variant_intersection_gnomad_depletion_ratio",
        "interpretation_hint": (
            "Is the unique region under germline constraint? Two independent "
            "signals: (1) gnomad_depletion_ratio < 1 means germline variation "
            "AVOIDS the unique region (density-normalized vs shared core); "
            "(2) ESM-2 constraint_enrichment high means residues there are "
            "predicted intolerant to substitution. This measures tolerance/"
            "constraint, not damaging-variant burden."
        ),
    },
    "F6_clinical_variant_overlap": {
        "axis": "F",
        "label": "Disease variant overlap",
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
        "headline_col": "isoform_variant_intersection_disease_enrichment_ratio",
        "interpretation_hint": (
            "Do disease variants (ClinVar + COSMIC) CONCENTRATE in the isoform's "
            "unique coding region? disease_enrichment_ratio > 1 means the unique "
            "region carries a higher disease-variant density than the shared core; "
            "the raw unique/shared disease and pathogenic counts are context."
        ),
    },
}


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
    # F5 — ESM-2 (PLM VEP) constraint
    "isoform_plm_vep_status": {"label": "PLM VEP status", "format": "str"},
    "isoform_plm_vep_constraint_enrichment": {
        "label": "ESM-2 constraint enrichment (unique vs shared)",
        "format": "float3",
    },
    "isoform_plm_vep_mean_llr_unique_region": {
        "label": "Mean ESM-2 LLR over unique region",
        "format": "float3",
    },
    "isoform_plm_vep_mean_llr_shared_region": {
        "label": "Mean ESM-2 LLR over shared region",
        "format": "float3",
    },
    "isoform_plm_vep_n_constrained_positions_unique": {
        "label": "ESM-2 constrained positions (unique)",
        "format": "int",
    },
    "isoform_plm_vep_n_constrained_positions_shared": {
        "label": "ESM-2 constrained positions (shared)",
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
        "label": "Mean ESM-2 ΔLLR, germline (unique)",
        "format": "float3",
    },
    "isoform_varianteffect_min_delta_llr_unique_gnomad": {
        "label": "Min ESM-2 ΔLLR, germline (unique)",
        "format": "float3",
    },
    "isoform_varianteffect_mean_am_pathogenicity_unique_gnomad": {
        "label": "Mean AlphaMissense pathogenicity, germline (unique)",
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


def format_metric(value: Any, fmt: str) -> str:
    """Format a metric value for display per a small spec of format codes.

    Args:
        value: Raw scalar value from the parquet row.
        fmt: One of ``"percent"``, ``"int"``, ``"float3"``, ``"sci"``, ``"bool"``,
            ``"json"``, or ``"str"`` (default).

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
        criterion_id: One of the 12 keys in ``CRITERIA``.

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

    evidence = {col: raw.get(col) for col in cfg["evidence_cols"]}
    headline_col = cfg.get("headline_col")
    if headline_col == _MAX_INITIATION_EFFICIENCY:
        # Max per-cell-line initiation efficiency (TIS counts / gene RNA-seq
        # counts ratio) across the six samples; None if no sample is scorable.
        vals = [
            raw.get(f"expr_{s}_initiation_efficiency") for s in _INITIATION_EFFICIENCY_SAMPLES
        ]
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        headline = max(vals) if vals else None
        headline_fmt = "float3"
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
                    "F5_pathogenic_variant_enrichment",
                    "F6_clinical_variant_overlap",
                }
                prioritize_unique = criterion_id in unique_region_criteria
                if n_hits_total > MAX_HITS:

                    def _clinsig_rank(h: dict[str, Any]) -> int:
                        sig = str(h.get("clinical_significance") or "").lower()
                        if "pathogenic" in sig and "likely" not in sig:
                            return 0  # Pathogenic
                        if "likely_pathogenic" in sig or "likely pathogenic" in sig:
                            return 1
                        if h.get("effect_damaging") is True:
                            return 2
                        if "uncertain" in sig:
                            return 4
                        if "benign" in sig:
                            return 5
                        return 3  # other / unknown

                    def _priority(h: dict[str, Any]) -> tuple[int, int]:
                        # Lead with unique-region membership for unique-region
                        # criteria so those hits survive truncation; clinical
                        # significance is the secondary sort within each bucket.
                        in_unique = 0 if (prioritize_unique and h.get("in_isoform_unique")) else 1
                        return (in_unique, _clinsig_rank(h))

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
        "evidence": evidence,
        "hits": hits,
        "n_hits_total": n_hits_total,
        "n_hits_shown": len(hits),
    }
