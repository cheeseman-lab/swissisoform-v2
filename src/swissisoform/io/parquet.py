"""Parquet serialization for TIS and Gene domain objects.

Converts between rich domain objects and flat DataFrames suitable for
Parquet storage, enabling efficient columnar I/O for large TIS datasets.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from swissisoform.models import (
    CellLineExpression,
    DifferentialRegion,
    Gene,
    ORFType,
    TranslationInitiationSite,
    is_separate_orf,
)

# The clinical summaries hold Counters over open key sets — variant sources and
# VEP consequence terms — so pyarrow infers a DIFFERENT struct for every frame:
# 9-13 distinct `by_consequence` fields across the genome-wide shards, and
# `struct<>` when a frame has no clinical hits at all, which Parquet cannot write
# ("no child field"). Declaring them as maps fixes both: a map's type does not
# depend on which keys are present, and an empty map is representable.
#
# Note this makes the column read back as list-of-(key, value) rather than dict.
# Nothing consumes it today (ClinicalModule writes it; the site reads
# `variants_long.parquet` and the `cmp_clinical_*` columns instead).
_COUNT_MAP = pa.map_(pa.string(), pa.int64())
CLINICAL_SUMMARY_TYPE = pa.struct([
    ("total_variants", pa.int64()),
    ("by_source", _COUNT_MAP),
    ("by_consequence", _COUNT_MAP),
    ("pathogenic_count", pa.int64()),
])
CLINICAL_SUMMARY_COLUMNS = ("canonical_clinical_summary", "isoform_clinical_summary")


def paired_schema(df: pd.DataFrame, overrides: dict[str, pa.DataType] | None = None) -> pa.Schema:
    """Arrow schema for a paired frame, with the unstable columns declared.

    Inference is per-frame, so it cannot be relied on for a sharded run: two
    shards infer incompatible types for the same column, and a shard whose rows
    are all empty infers a type Parquet refuses to write. Everything else is
    inferred as before; only the columns known to degenerate are pinned.

    Args:
        df: The paired frame about to be written.
        overrides: Extra ``{column: type}`` pins, for columns whose type is known
            from a declaration rather than from the data. The tag structs are the
            live case: a shard where one tag happens to be null on every row would
            infer that field as ``null`` and disagree with every other shard.

    Returns:
        The inferred schema with :data:`CLINICAL_SUMMARY_COLUMNS` and *overrides*
        applied.
    """
    schema = pa.Table.from_pandas(df, preserve_index=False).schema
    pins: dict[str, pa.DataType] = {n: CLINICAL_SUMMARY_TYPE for n in CLINICAL_SUMMARY_COLUMNS}
    pins.update(overrides or {})
    for name, dtype in pins.items():
        idx = schema.get_field_index(name)
        if idx >= 0:
            schema = schema.set(idx, pa.field(name, dtype))
    return schema


# The map-typed fields of the clinical summaries, as dotted paths. Their values are
# counts over an open key set, so a key a PRESENT map lacks is a count of 0 — only
# a missing summary is not-evaluable.
COUNT_MAP_PATHS: frozenset[str] = frozenset(
    f"{col}.{CLINICAL_SUMMARY_TYPE.field(i).name}"
    for col in CLINICAL_SUMMARY_COLUMNS
    for i in range(CLINICAL_SUMMARY_TYPE.num_fields)
    if pa.types.is_map(CLINICAL_SUMMARY_TYPE.field(i).type)
)


def paired_table(df: pd.DataFrame, overrides: dict[str, pa.DataType] | None = None) -> pa.Table:
    """*df* as an Arrow table, inferred except for the pinned columns.

    The table :func:`paired_schema` describes, built in ONE conversion: the
    unpinned columns are inferred, and the pinned ones (the clinical summaries
    plus *overrides*) are converted straight to their declared types. Converting
    to infer a schema and again to apply it holds two Arrow copies of a
    genome-wide frame at once; and a pinned map read back from Parquet arrives as
    ``(key, value)`` pairs that inference cannot type, but its declared map type
    accepts as they are.
    """
    pins: dict[str, pa.DataType] = {n: CLINICAL_SUMMARY_TYPE for n in CLINICAL_SUMMARY_COLUMNS}
    pins.update(overrides or {})
    pinned = [c for c in df.columns if c in pins]
    table = pa.Table.from_pandas(df.drop(columns=pinned), preserve_index=False)
    for col in pinned:  # df order, so each insert lands at its original position
        array = pa.array(df[col], type=pins[col], from_pandas=True)
        table = table.add_column(df.columns.get_loc(col), pa.field(col, pins[col]), array)
    return table


def write_table_atomic(table: pa.Table, path: Path) -> None:
    """Write *table* to *path* so that it is never observable half-finished.

    A same-directory temp file plus ``os.replace`` means a reader sees either no
    file or the complete one — a plain write into the final path leaves a
    truncated file behind when the job is killed mid-write (Slurm timeout, OOM).
    """
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def mapping_of(value: Any) -> dict[str, Any] | None:
    """A struct or map value as a dict, or None when it is neither.

    A map read back from Parquet through pandas is a list (or object ndarray) of
    ``(key, value)`` tuples, while the in-memory frame holds a dict; a struct is a
    dict either way. Only tuples count as pairs: a list of 2-element lists is a
    list, not a map, and a scalar (numpy included) is never a mapping.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple, np.ndarray)):
        pairs = list(value)
        if all(isinstance(p, tuple) and len(p) == 2 for p in pairs):
            return {str(k): v for k, v in pairs}
    return None


def flatten_structs(table: pa.Table) -> tuple[pd.DataFrame, list[str]]:
    """*table* with its structs flattened to dotted columns, and the map columns left.

    Maps stay one column each until :func:`expand_maps`, which needs every file's
    rows at once: a key one file never saw is still a 0 there, not a gap.
    """
    while any(pa.types.is_struct(f.type) for f in table.schema):
        table = table.flatten()
    maps = [f.name for f in table.schema if pa.types.is_map(f.type)]
    return table.to_pandas(), maps


def expand_maps(df: pd.DataFrame, maps: list[str]) -> pd.DataFrame:
    """Replace each map column in *maps* with one dotted column per key seen.

    The names are those the struct-typed summaries produced before
    :data:`CLINICAL_SUMMARY_TYPE` pinned them to maps. A row whose map is null
    stays null on every key (not evaluable). A row whose map is present but lacks
    a key reads 0 under a count map (:data:`COUNT_MAP_PATHS`), null otherwise.
    """
    blocks: dict[str, list[Any]] = {}
    for name in maps:
        if name not in df.columns:
            continue
        rows = [mapping_of(v) for v in df.pop(name)]
        absent = 0 if name in COUNT_MAP_PATHS else None
        for key in sorted({k for r in rows if r for k in r}):
            blocks[f"{name}.{key}"] = [None if r is None else r.get(key, absent) for r in rows]
    if not blocks:
        return df
    return pd.concat([df, pd.DataFrame(blocks, index=df.index)], axis=1)


def flat_frame(table: pa.Table) -> pd.DataFrame:
    """*table* as a frame with every struct and map expanded into dotted columns.

    One table's worth of :func:`flatten_structs` + :func:`expand_maps`. A reader
    over several files flattens each and expands once over the concatenation.
    """
    df, maps = flatten_structs(table)
    return expand_maps(df, maps)


def schema_paths(schema: pa.Schema) -> set[str]:
    """Every name a flattened read of *schema* can produce, for name-level checks.

    Top-level columns, every dotted struct leaf, and ``<map path>.*`` for each
    map — whose keys are data, so the schema can only say "a key under here".
    """
    out: set[str] = set()

    def walk(name: str, t: pa.DataType) -> None:
        out.add(name)
        if pa.types.is_struct(t):
            for i in range(t.num_fields):
                walk(f"{name}.{t.field(i).name}", t.field(i).type)
        elif pa.types.is_map(t):
            out.add(f"{name}.*")

    for field in schema:
        walk(field.name, field.type)
    return out


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
    "diff_region_confidence",
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
            "differential_sequence": site.diff_region.sequence if site.diff_region else "",
            "diff_region_confidence": (
                site.diff_region.confidence if site.diff_region else "exact"
            ),
            "kozak_context": site.kozak_context,
        }

        # Expression -> wide columns
        for cell_line, expr in site.expression.items():
            row[f"expr_{cell_line}_raw_count"] = expr.raw_count
            row[f"expr_{cell_line}_cpm"] = expr.cpm
            row[f"expr_{cell_line}_p_value"] = expr.p_value
            row[f"expr_{cell_line}_initiation_efficiency"] = expr.initiation_efficiency

        # Annotations -> prefixed columns
        for module_name, annot_dict in site.isoform_annotations.items():
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
            diff_region=DifferentialRegion(
                sequence=row.get("differential_sequence", ""),
                confidence=row.get("diff_region_confidence", "exact"),
            ),
            kozak_context=row.get("kozak_context"),
            expression=expression,
            isoform_annotations=annotations,
        )
        sites.append(site)

    return sites


def genes_to_dataframe(genes: list[Gene]) -> pd.DataFrame:
    """Serialize a list of Gene objects to a flat DataFrame.

    Args:
        genes: Gene objects to serialize.

    Returns:
        DataFrame with one row per gene. Gene-level annotations flatten as
        ``{annotation_key}_{sub_key}`` and canonical protein annotations
        flatten as ``canonical_{module}_{key}``.
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
        row.update(_flatten_annotations("canonical", gene.canonical_annotations))
        rows.append(row)

    return pd.DataFrame(rows)


def _flatten_annotations(prefix: str, annotations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for module_name, annot_dict in annotations.items():
        for key, value in annot_dict.items():
            out[f"{prefix}_{module_name}_{key}"] = value
    return out


def _genomic_unique_shared(
    orf_type: Any,
    orf_exons: list[tuple[int, int]] | None,
    canonical_orf_exons: list[tuple[int, int]] | None,
    *,
    separate_orf: bool = False,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    r"""Compute the genomic unique and shared interval sets per ORF type.

    Mirrors the ORF-type-aware unique definition used by
    :class:`VariantIntersectionModule` / conservation / conservation_frame —
    extensions / uORFs / altORFs contribute new isoform sequence
    (``isoform_orf \ canonical_orf``); truncations *lose* canonical
    sequence (``canonical_orf \ isoform_orf``). The shared set is the
    intersection in both cases. Empty lists are returned when either ORF
    skeleton is missing.

    Surfacing these in the parquet lets downstream codebases — genome
    language models, motif callers, off-target tools — pull DNA over
    arbitrary windows around the *biologically meaningful* region of each
    TIS without re-running the assembly layer.
    """
    from swissisoform.coords import unique_shared_intervals
    from swissisoform.models import ORFType

    if not orf_exons or not canonical_orf_exons:
        return [], []
    return unique_shared_intervals(
        orf_type == ORFType.TRUNCATED, orf_exons, canonical_orf_exons, separate_orf=separate_orf
    )


def _diff_location(diff: Any) -> tuple[int | None, int | None, str | None]:
    """Collapse a DifferentialRegion to a single (start, end, space) triple.

    The unique region exists in exactly one protein coordinate space per ORF
    type: extensions/uORFs carry isoform-space coords, truncations carry
    canonical-space coords. This returns whichever pair is populated plus the
    space label (``"isoform"`` / ``"canonical"``), or ``(None, None, None)``
    when there is no differential region (Annotated). Downstream consumers read
    ``site.diff_region`` directly and branch on ORF type; this is purely the
    serialized, always-populated view for the paired DataFrame.
    """
    if diff is None:
        return None, None, None
    if diff.isoform_start is not None:
        return diff.isoform_start, diff.isoform_end, "isoform"
    if diff.canonical_start is not None:
        return diff.canonical_start, diff.canonical_end, "canonical"
    return None, None, None


def paired_tis_dataframe(genes: list[Gene]) -> pd.DataFrame:
    """One row per TIS with paired canonical + isoform annotations.

    Each row carries both ``canonical_{module}_{key}`` columns (from the
    gene's canonical protein annotations) and ``isoform_{module}_{key}``
    columns (from the TIS's own isoform annotations), so the comparator
    can diff them without another pass over ``Gene`` objects.

    Args:
        genes: Gene objects with both ``canonical_annotations`` and per-TIS
            ``isoform_annotations`` populated by ``AnnotationPipeline``.

    Returns:
        DataFrame with identity + expression wide cols + paired annotation
        cols, one row per TIS across all input genes.
    """
    rows: list[dict[str, Any]] = []
    for gene in genes:
        canonical_cols = _flatten_annotations("canonical", gene.canonical_annotations)
        canonical_len = len(gene.canonical_protein.rstrip("*"))
        # Gene-level reference annotations (e.g. generef) repeat on every TIS row.
        gene_annot_cols: dict[str, Any] = {}
        for annot_key, annot_val in gene.gene_annotations.items():
            if isinstance(annot_val, dict):
                for sub_key, sub_val in annot_val.items():
                    gene_annot_cols[f"{annot_key}_{sub_key}"] = sub_val
            else:
                gene_annot_cols[annot_key] = annot_val
        for site in gene.tis_sites:
            diff = site.diff_region
            diff_start, diff_end, diff_space = _diff_location(diff)
            # Per-Tid canonical can differ from gene-level (the assembly layer
            # picks each TIS's canonical from its own transcript's Annotated row;
            # gene.canonical_protein is the gene-level representative). When
            # they differ, cmp_*_length-family deltas (computed against
            # gene-level) and diff_*_length (computed against per-Tid) describe
            # different reference proteins in the same row — surface this so
            # consumers can detect it instead of being silently inconsistent.
            per_tid_len = len(site.canonical_protein.rstrip("*"))
            # Serialize the per-TIS ORF exon intervals + the derived genomic
            # unique / shared sets so downstream codebases (genome language
            # models, motif scanners, off-target callers) can extract DNA over
            # arbitrary windows around the isoform without re-running our
            # assembly layer. All intervals are 0-based half-open plus-strand
            # genomic coords; strand is in the ``strand`` column.
            orf_exons = [list(t) for t in (site.orf_exons or [])]
            canonical_orf_exons = [list(t) for t in (site.canonical_orf_exons or [])]
            unique_genomic_intervals, shared_genomic_intervals = _genomic_unique_shared(
                site.orf_type,
                site.orf_exons,
                site.canonical_orf_exons,
                separate_orf=is_separate_orf(site),
            )
            row: dict[str, Any] = {
                "gene_name": gene.gene_name,
                "gene_id": gene.gene_id,
                "canonical_transcript_id": gene.canonical_transcript_id,
                "tis_id": site.tis_id,
                "transcript_id": site.transcript_id,
                "chrom": site.chrom,
                "position": site.position,
                "strand": site.strand,
                "start_codon": site.start_codon,
                "canonical_start_codon": site.canonical_start_codon,
                "canonical_per_tid_length": per_tid_len,
                "canonical_per_tid_differs_from_gene": per_tid_len != canonical_len,
                "orf_type": site.orf_type.value,
                "aa_len": site.aa_len,
                "canonical_len": canonical_len,
                "isoform_len": len(site.isoform_protein.rstrip("*")),
                "differential_sequence": diff.sequence if diff else "",
                "diff_region_confidence": diff.confidence if diff else "exact",
                "diff_start": diff_start,
                "diff_end": diff_end,
                "diff_space": diff_space,
                "orf_exons": orf_exons,
                "canonical_orf_exons": canonical_orf_exons,
                "unique_genomic_intervals": [list(t) for t in unique_genomic_intervals],
                "shared_genomic_intervals": [list(t) for t in shared_genomic_intervals],
                "kozak_context": site.kozak_context,
                "canonical_kozak_context": site.canonical_kozak_context,
                "tis_pvalue": site.tis_pvalue,
                "ribo_pvalue": site.ribo_pvalue,
                "fisher_qvalue": site.fisher_qvalue,
            }
            for cell_line, expr in site.expression.items():
                row[f"expr_{cell_line}_raw_count"] = expr.raw_count
                row[f"expr_{cell_line}_cpm"] = expr.cpm
                row[f"expr_{cell_line}_p_value"] = expr.p_value
                row[f"expr_{cell_line}_initiation_efficiency"] = expr.initiation_efficiency
            # Canonical (Annotated) start's expression per cell line, under a
            # distinct ``canonical_expr_`` prefix so the alt-TIS ``expr_``
            # discovery regex never picks it up. Lets the viewer put
            # canonical-start vs alt-start initiation efficiency side by side.
            for cell_line, expr in site.canonical_expression.items():
                row[f"canonical_expr_{cell_line}_raw_count"] = expr.raw_count
                row[f"canonical_expr_{cell_line}_cpm"] = expr.cpm
                row[f"canonical_expr_{cell_line}_p_value"] = expr.p_value
                row[f"canonical_expr_{cell_line}_initiation_efficiency"] = (
                    expr.initiation_efficiency
                )
            row.update(gene_annot_cols)
            row.update(canonical_cols)
            # Per-TIS canonical-start site annotations (conservation /
            # initiation_context at the canonical start) → canonical_<module>_*,
            # the symmetric twin of isoform_<module>_*. Distinct module names
            # from the gene-level canonical_cols, so no key collision.
            row.update(_flatten_annotations("canonical", site.canonical_annotations))
            row.update(_flatten_annotations("isoform", site.isoform_annotations))
            row.update(_flatten_annotations("diff", site.diff_annotations))
            row.update(_flatten_annotations("cmp", site.comparison))
            rows.append(row)

    return pd.DataFrame(rows)
