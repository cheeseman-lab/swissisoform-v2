"""Parquet serialization for TIS and Gene domain objects.

Converts between rich domain objects and flat DataFrames suitable for
Parquet storage, enabling efficient columnar I/O for large TIS datasets.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from swissisoform.models import (
    CellLineExpression,
    Gene,
    ORFType,
    TranslationInitiationSite,
)

# Columns that map directly to TIS identity/protein fields
_IDENTITY_COLS = [
    "tis_id",
    "gene_name",
    "transcript_id",
    "chrom",
    "position",
    "strand",
    "start_codon",
    "orf_type",
]

_PROTEIN_COLS = [
    "canonical_protein",
    "isoform_protein",
    "differential_sequence",
    "shared_sequence",
    "kozak_context",
]

_EXPR_METRICS = ["raw_count", "cpm", "p_value", "initiation_efficiency"]

_EXPR_PATTERN = re.compile(r"^expr_(.+?)_(raw_count|cpm|p_value|initiation_efficiency)$")


def tis_to_dataframe(sites: list[TranslationInitiationSite]) -> pd.DataFrame:
    """Serialize a list of TIS objects to a flat DataFrame.

    Args:
        sites: Translation initiation sites to serialize.

    Returns:
        DataFrame with one row per TIS, expression as wide columns
        (``expr_{cell_line}_{metric}``), and annotations as prefixed
        columns (``{module_name}_{key}``).
    """
    rows: list[dict[str, Any]] = []
    for site in sites:
        row: dict[str, Any] = {
            "tis_id": site.tis_id,
            "gene_name": site.gene_name,
            "transcript_id": site.transcript_id,
            "chrom": site.chrom,
            "position": site.position,
            "strand": site.strand,
            "start_codon": site.start_codon,
            "orf_type": site.orf_type.value,
            "canonical_protein": site.canonical_protein,
            "isoform_protein": site.isoform_protein,
            "differential_sequence": site.differential_sequence,
            "shared_sequence": site.shared_sequence,
            "kozak_context": site.kozak_context,
        }

        # Expression -> wide columns
        for cell_line, expr in site.expression.items():
            row[f"expr_{cell_line}_raw_count"] = expr.raw_count
            row[f"expr_{cell_line}_cpm"] = expr.cpm
            row[f"expr_{cell_line}_p_value"] = expr.p_value
            row[f"expr_{cell_line}_initiation_efficiency"] = expr.initiation_efficiency

        # Annotations -> prefixed columns
        for module_name, annot_dict in site.annotations.items():
            for key, value in annot_dict.items():
                row[f"{module_name}_{key}"] = value

        rows.append(row)

    return pd.DataFrame(rows)


def dataframe_to_tis(df: pd.DataFrame) -> list[TranslationInitiationSite]:
    """Reconstruct TIS objects from a flat DataFrame.

    Discovers cell lines from ``expr_*`` column patterns and annotation
    modules from remaining non-identity, non-expression columns by
    splitting on the first underscore.

    Args:
        df: DataFrame produced by ``tis_to_dataframe``.

    Returns:
        List of reconstructed TranslationInitiationSite objects.
    """
    # Build ORFType lookup: value -> enum member
    orf_lookup = {member.value: member for member in ORFType}

    # Discover expression columns grouped by cell line
    expr_cols: dict[str, dict[str, str]] = {}  # cell_line -> {metric: col_name}
    known_cols = set(_IDENTITY_COLS) | set(_PROTEIN_COLS)

    for col in df.columns:
        m = _EXPR_PATTERN.match(col)
        if m:
            cell_line, metric = m.group(1), m.group(2)
            expr_cols.setdefault(cell_line, {})[metric] = col
            known_cols.add(col)

    # Remaining columns are annotations: split on first "_" to get module name
    annotation_cols: dict[str, list[tuple[str, str]]] = {}  # module -> [(key, col)]
    for col in df.columns:
        if col not in known_cols:
            parts = col.split("_", 1)
            if len(parts) == 2:
                module_name, key = parts
                annotation_cols.setdefault(module_name, []).append((key, col))
            # Single-word columns with no underscore are ignored

    sites: list[TranslationInitiationSite] = []
    for _, row in df.iterrows():
        # Reconstruct expression
        expression: dict[str, CellLineExpression] = {}
        for cell_line, metrics in expr_cols.items():
            ie_val = row.get(metrics.get("initiation_efficiency", ""))
            expression[cell_line] = CellLineExpression(
                raw_count=int(row[metrics["raw_count"]]),
                cpm=float(row[metrics["cpm"]]),
                p_value=float(row[metrics["p_value"]]),
                initiation_efficiency=(
                    None if pd.isna(ie_val) else float(ie_val)  # type: ignore[arg-type]
                ),
            )

        # Reconstruct annotations
        annotations: dict[str, dict[str, Any]] = {}
        for module_name, keys in annotation_cols.items():
            mod_dict: dict[str, Any] = {}
            for key, col in keys:
                val = row[col]
                # Convert NaN back to None
                if isinstance(val, float) and pd.isna(val):
                    val = None
                mod_dict[key] = val
            annotations[module_name] = mod_dict

        site = TranslationInitiationSite(
            tis_id=row["tis_id"],
            gene_name=row["gene_name"],
            transcript_id=row["transcript_id"],
            chrom=row["chrom"],
            position=int(row["position"]),
            strand=row["strand"],
            start_codon=row["start_codon"],
            orf_type=orf_lookup[row["orf_type"]],
            canonical_protein=row.get("canonical_protein", ""),
            isoform_protein=row.get("isoform_protein", ""),
            differential_sequence=row.get("differential_sequence", ""),
            shared_sequence=row.get("shared_sequence", ""),
            kozak_context=row.get("kozak_context"),
            expression=expression,
            annotations=annotations,
        )
        sites.append(site)

    return sites


def genes_to_dataframe(genes: list[Gene]) -> pd.DataFrame:
    """Serialize a list of Gene objects to a flat DataFrame.

    Args:
        genes: Gene objects to serialize.

    Returns:
        DataFrame with one row per gene. Gene annotations are flattened
        as ``{annotation_key}_{sub_key}`` columns.
    """
    rows: list[dict[str, Any]] = []
    for gene in genes:
        row: dict[str, Any] = {
            "gene_name": gene.gene_name,
            "gene_id": gene.gene_id,
            "canonical_transcript_id": gene.canonical_transcript_id,
            "canonical_protein": gene.canonical_protein,
        }
        for annot_key, annot_val in gene.gene_annotations.items():
            if isinstance(annot_val, dict):
                for sub_key, sub_val in annot_val.items():
                    row[f"{annot_key}_{sub_key}"] = sub_val
            else:
                row[annot_key] = annot_val
        rows.append(row)

    return pd.DataFrame(rows)
