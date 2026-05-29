"""Build per-gene LLM evidence-record JSON from the paired parquet.

Consumes `data/output/cheeseman_12gene/all_paired.parquet` and emits one
`{gene}.json` per gene matching the "Per-gene evidence record" schema in
`docs/site_and_llm_plan.md`.

This is pure DataFrame → dict conversion: no LLM calls, no network.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PATHOGENIC_CLINSIG_TOKENS = ("pathogenic", "likely_pathogenic", "likely pathogenic")


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
        # V2: raw columns the modality slicer reaches into; kept under _raw so
        # the existing per-gene JSON files surface identical user-facing keys.
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


def _summarise(out_dir: Path) -> None:
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--parquet",
        type=Path,
        required=True,
        help="Path to all_paired.parquet",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for per-gene JSON files",
    )
    p.add_argument(
        "--variants-long-out",
        type=Path,
        default=None,
        help="If set, also write a flat variants_long parquet to this path.",
    )
    args = p.parse_args()

    counts = write_evidence_records(args.parquet, args.out)
    print(f"Wrote {counts['genes']} gene files ({counts['isoforms']} isoforms) to {args.out}")
    _summarise(args.out)

    if args.variants_long_out is not None:
        n = write_variants_long(args.parquet, args.variants_long_out)
        print(f"Wrote {n} variant rows → {args.variants_long_out}")


# ──────────────────────────────────────────────────────────────────────────
# V2 modality slicer — turns a full isoform record into the slim shape the
# reading-pass LLM consumes. Pure function; no I/O.
#
# Criterion names come from `src/swissisoform/modules/scoring.py`
# (EXISTENCE_CRITERIA = E1..E6, FUNCTIONAL_CRITERIA = F1..F6). Axis is strict:
# E* lives on the existence axis, F* on the functional axis. The "variants"
# modality is a special case — both F5_pathogenic_variant_enrichment and
# F6_clinical_variant_overlap are functional, so its existence axis is empty.
# ──────────────────────────────────────────────────────────────────────────

MODALITY_CRITERIA: dict[str, dict[str, list[str]]] = {
    "cell_lines": {
        "existence": ["E4_multi_cell_line", "E5_initiation_efficiency"],
        "functional": [],
    },
    "conservation": {
        "existence": [
            "E1_primate_conservation",
            "E2_mammalian_conservation",
            "E3_phylop_coding_selection",
        ],
        "functional": [],
    },
    "variants": {
        "existence": [],
        "functional": [
            "F5_pathogenic_variant_enrichment",
            "F6_clinical_variant_overlap",
        ],
    },
    "mass_spec": {
        "existence": ["E6_mass_spec"],
        "functional": [],
    },
    "structure": {
        "existence": [],
        "functional": ["F1_structured_extension"],
    },
    "domains_motifs": {
        "existence": [],
        "functional": ["F3_domain_change"],
    },
    "localization": {
        "existence": [],
        "functional": ["F2_localization_change", "F4_targeting_change"],
    },
}

# Modality → list of raw parquet column prefixes whose values feed the LLM as evidence.
# We match by prefix so renames within a module don't break the slicer immediately.
MODALITY_RAW_PREFIXES: dict[str, tuple[str, ...]] = {
    "cell_lines": ("expr_", "ribo_pvalue", "fisher_qvalue", "tis_pvalue"),
    "conservation": (
        "isoform_conservation_phylop_",
        "isoform_conservation_phastcons_",
        "isoform_conservation_frame_",
    ),
    "variants": (
        "isoform_variant_intersection_n_",
        "isoform_varianteffect_n_",
        "isoform_varianteffect_mean_",
    ),
    "mass_spec": ("isoform_massspec_", "cmp_massspec_"),
    "structure": ("isoform_structure_", "cmp_structure_"),
    "domains_motifs": (
        "isoform_interproscan_",
        "isoform_motifs_",
        "cmp_interproscan_",
        "cmp_motifs_",
    ),
    "localization": ("canonical_localization_", "isoform_localization_", "cmp_localization_"),
}


def _pick_criteria(scoring: dict[str, Any], names: list[str]) -> list[dict[str, Any]]:
    criteria = (scoring or {}).get("criteria") or {}
    out: list[dict[str, Any]] = []
    for name in names:
        entry = criteria.get(name)
        if entry is None:
            continue
        out.append(
            {
                "name": name,
                "value": entry.get("value"),
                "reason": entry.get("reason"),
            }
        )
    return out


def _pick_raw(raw: dict[str, Any], prefixes: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (raw or {}).items():
        if any(k.startswith(p) for p in prefixes):
            out[k] = v
    return out


def slice_modality(isoform_record: dict[str, Any], modality: str, *, axis: str) -> dict[str, Any]:
    """Build the per-(isoform, modality, axis) LLM input record.

    Args:
        isoform_record: A full per-isoform record (one entry from
            ``build_gene_record(...)["isoforms"]``) with an additional
            ``"_raw"`` dict mirroring the parquet row.
        modality: One of ``MODALITY_CRITERIA``.
        axis: ``"existence"`` or ``"functional"``.

    Returns:
        A slim dict carrying isoform identity + the cited criteria for this
        axis + the modality-relevant raw evidence + per-variant data for the
        variants modality. ``axis_unavailable`` is True when the modality has
        no criteria on the requested axis.
    """
    if modality not in MODALITY_CRITERIA:
        raise KeyError(f"Unknown modality: {modality!r}")
    if axis not in ("existence", "functional"):
        raise ValueError(f"axis must be existence|functional, got {axis!r}")

    iso_block = {
        "tis_id": isoform_record.get("tis_id"),
        "gene_name": (isoform_record.get("gene") or {}).get("name"),
        "orf_type": isoform_record.get("orf_type"),
        "differential_sequence": isoform_record.get("differential_sequence"),
        "diff_space": isoform_record.get("diff_space"),
        "isoform_length_aa": isoform_record.get("isoform_length_aa"),
        "canonical_length_aa": isoform_record.get("canonical_length_aa"),
    }

    names = MODALITY_CRITERIA[modality][axis]
    criteria = _pick_criteria(isoform_record.get("scoring") or {}, names)
    evidence = _pick_raw(isoform_record.get("_raw") or {}, MODALITY_RAW_PREFIXES[modality])

    # Variants modality also gets the pathogenic list, regardless of axis.
    if modality == "variants":
        evidence["pathogenic_variants_in_unique"] = list(
            isoform_record.get("pathogenic_variants_in_unique") or []
        )

    out: dict[str, Any] = {
        "modality": modality,
        "axis": axis,
        "isoform": iso_block,
        "criteria": criteria,
        "evidence": evidence,
    }
    if not names:
        out["axis_unavailable"] = True
    return out


# ──────────────────────────────────────────────────────────────────────────
# V2 per-criterion config — replaces MODALITY_CRITERIA. Each entry drives
# (a) the slicer (what raw cols to pull) and (b) the UI tile (label, headline).
# Verified against the real cheeseman_12gene parquet column schema.
# ──────────────────────────────────────────────────────────────────────────

CRITERIA: dict[str, dict[str, Any]] = {
    "E1_primate_conservation": {
        "axis": "E",
        "label": "Primate conservation",
        "short_label": "Primates",
        "evidence_cols": [
            "isoform_conservation_frame_primate_frac_intact",
            "isoform_conservation_frame_primate_start_codon_conserved",
            "isoform_conservation_frame_primate_n_species_aligned",
            "isoform_conservation_frame_primate_n_species_intact_frame",
            "isoform_conservation_frame_primate_mean_pident",
            "isoform_conservation_frame_primate_deepest_species",
            "isoform_conservation_frame_primate_max_depth",
        ],
        "headline_col": "isoform_conservation_frame_primate_frac_intact",
        "interpretation_hint": (
            "Is the alternative reading frame preserved across primates? Look at "
            "frac_intact (fraction of aligned species with intact ORF) and "
            "start_codon_conserved."
        ),
    },
    "E2_mammalian_conservation": {
        "axis": "E",
        "label": "Mammalian conservation",
        "short_label": "Mammals",
        "evidence_cols": [
            "isoform_conservation_frame_mammalian_frac_intact",
            "isoform_conservation_frame_mammalian_start_codon_conserved",
            "isoform_conservation_frame_mammalian_n_species_aligned",
            "isoform_conservation_frame_mammalian_n_species_intact_frame",
            "isoform_conservation_frame_mammalian_mean_pident",
            "isoform_conservation_frame_mammalian_deepest_species",
            "isoform_conservation_frame_mammalian_max_depth",
        ],
        "headline_col": "isoform_conservation_frame_mammalian_frac_intact",
        "interpretation_hint": ("Is the alternative reading frame preserved deeper in mammals?"),
    },
    "E3_phylop_coding_selection": {
        "axis": "E",
        "label": "PhyloP coding selection",
        "short_label": "PhyloP",
        "evidence_cols": [
            "isoform_conservation_phylop_at_tis",
            "isoform_conservation_phylop_unique_region_mean",
            "isoform_conservation_phylop_shared_region_mean",
            "isoform_conservation_phylop_enrichment",
            "isoform_conservation_phylop_kozak_mean",
            "isoform_conservation_phastcons_unique_region_mean",
            "isoform_conservation_phastcons_at_tis",
        ],
        "headline_col": "isoform_conservation_phylop_unique_region_mean",
        "interpretation_hint": (
            "Does the unique region show purifying selection by phyloP? Compare "
            "to shared region for enrichment."
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
            "ribo_pvalue",
            "tis_pvalue",
            "fisher_qvalue",
            "expr_HeLa_initiation_efficiency",
            "expr_K562_initiation_efficiency",
            "expr_U2OS_initiation_efficiency",
            "expr_RPE1_Async_initiation_efficiency",
            "expr_RPE1_Que_initiation_efficiency",
            "expr_RPE1_Sen_initiation_efficiency",
        ],
        "headline_col": "fisher_qvalue",
        "interpretation_hint": (
            "Is the Ribo-TISH signal strong enough that this TIS is reproducibly "
            "initiating? Lower fisher_qvalue is better."
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
        ],
        "headline_col": "isoform_structure_plddt_diffregion_mean",
        "interpretation_hint": (
            "Does the unique region fold confidently (Boltz pLDDT)? Higher "
            "diffregion_mean means more structured."
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
        ],
        "headline_col": "cmp_localization_deeploc_prediction_changed",
        "interpretation_hint": (
            "Does DeepLoc predict a different subcellular location for the isoform vs canonical?"
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
            "canonical_localization_deeploc_signals",
            "isoform_localization_deeploc_signals",
            "canonical_localization_deeploc_membrane",
            "isoform_localization_deeploc_membrane",
        ],
        "headline_col": None,
        "interpretation_hint": (
            "Do targeting signals (NLS, signal peptide, etc.) differ between canonical and isoform?"
        ),
    },
    "F5_pathogenic_variant_enrichment": {
        "axis": "F",
        "label": "Pathogenic variant enrichment",
        "short_label": "Pathogenic",
        "evidence_cols": [
            "isoform_variant_intersection_n_pathogenic_in_unique_region",
            "isoform_variant_intersection_n_pathogenic_in_shared_region",
            "isoform_varianteffect_n_damaging_in_unique",
            "isoform_varianteffect_mean_delta_llr_unique",
            "isoform_varianteffect_mean_am_pathogenicity_unique",
        ],
        "evidence_hits_col": "isoform_variant_intersection_hits",
        "headline_col": "isoform_variant_intersection_n_pathogenic_in_unique_region",
        "interpretation_hint": (
            "Are clinical variants enriched in the unique region "
            "(vs shared region or genome average)?"
        ),
    },
    "F6_clinical_variant_overlap": {
        "axis": "F",
        "label": "Clinical variant overlap",
        "short_label": "Variants",
        "evidence_cols": [
            "isoform_variant_intersection_n_total",
            "isoform_variant_intersection_n_in_unique_region",
            "isoform_variant_intersection_n_in_shared_region",
            "isoform_variant_intersection_n_dropped_outside_coding",
        ],
        "headline_col": "isoform_variant_intersection_n_in_unique_region",
        "interpretation_hint": (
            "How many clinical variants overlap the isoform's coding region at all?"
        ),
    },
}


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
    headline = raw.get(headline_col) if headline_col else None

    hits: list[dict[str, Any]] = []
    hits_col = cfg.get("evidence_hits_col")
    if hits_col:
        raw_hits = raw.get(hits_col)
        if raw_hits is not None:
            try:
                # Handle numpy arrays + lists
                hits = [h for h in raw_hits if isinstance(h, dict)]
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
        "evidence": evidence,
        "hits": hits,
    }


if __name__ == "__main__":
    main()
