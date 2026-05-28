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
    "pos",
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
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


if __name__ == "__main__":
    main()
