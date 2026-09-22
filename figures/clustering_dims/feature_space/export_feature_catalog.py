#!/usr/bin/env python
"""Enumerate every feature the pipeline produces, and say which can be a plot dimension.

The scoring layer thresholds ~19 numbers; ``all_paired.parquet`` carries 724
scalar leaves once structs are flattened. To place isoforms in a high-dimensional
space that reflects the real continuum of protein variation — rather than the
16-bit corner-count the CDLMPS criterion booleans collapse to — we first need the
full inventory. This script writes it.

One row per candidate feature, with:

  * ``include_in_plot`` — usable as a numeric dimension as-is,
  * ``encodable_as``    — how to opt a *non*-numeric feature in (binary, one-hot,
                          count, text embedding), so nothing is silently lost,
  * ``category``        — which of the six CDLMPS categories it belongs to.

Category comes from three tiers, most authoritative first: the curated
``CRITERIA[*]["evidence_cols"]`` registry in ``swissisoform.site.evidence``, then
a module→category map derived from the plumbing table in
``swissisoform.evidence.__init__``, then ``-`` for identity/coordinate fields
that belong to no category.

Everything empirical (fill rate, cardinality, duplicate detection, the ORF-type
null pattern) is measured against the run, not asserted — a feature that is
all-null or constant in practice is excluded with that reason recorded.

Outputs (alongside this script):
  - feature_catalog.csv       one row per feature

Usage:
    python figures/clustering_dims/export_feature_catalog.py
    python figures/clustering_dims/export_feature_catalog.py --parquet <path>
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from swissisoform import metrics
from swissisoform.site.evidence import CRITERIA, CRITERIA_METRIC_LABELS

HERE = Path(__file__).resolve().parent


def _repo_root(start: Path) -> Path:
    """Walk up until the repo root is found, rather than counting directories.

    These figure scripts get moved between folders; a fixed ``parent.parent``
    silently resolves to the wrong place when the nesting depth changes.
    """
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise SystemExit(f"could not locate the repo root above {start}")


ROOT = _repo_root(HERE)

DEFAULT_PARQUET = ROOT / "data" / "output" / "full_catalog" / "all_paired.parquet"
SHARD_GLOB = str(ROOT / "data" / "output" / "full_catalog_shard_*" / "all_paired.parquet")

OUT_CSV = HERE / "feature_catalog.csv"

SAMPLES = metrics.SAMPLES

# Separate-ORF types have no shared region at all, so every unique-vs-shared
# feature is null for them by construction (comparator.py `_shared_annotations`).
SEPARATE_ORF_TYPES = ("uorf", "uoorf", "internal_oof", "3utr_orf", "alt_orf")

MIN_FILL = 0.05
MIN_UNIQUE = 3

# ---------------------------------------------------------------------------
# Module identification
# ---------------------------------------------------------------------------

# Longest-first so `conservation_frame` wins over `conservation`.
MODULES = (
    "conservation_frame",
    "variant_intersection",
    "initiation_context",
    "varianteffect",
    "interproscan",
    "core_identity",
    "localization",
    "conservation",
    "biophysics",
    "structure",
    "massspec",
    "clinical",
    "scoring",
    "signalp",
    "targetp",
    "plm_vep",
    "motifs",
    "generef",
    "sae",
)

# ``diff_`` is the Scope-A pane: the differential region annotated in its own
# right (comparator.py re-runs the biophysics module on ``diff_region.sequence``).
PANE_PREFIXES = ("canonical_", "isoform_", "cmp_", "diff_")

# Module -> CDLMPS category, from the plumbing table in
# src/swissisoform/evidence/__init__.py. Two judgement calls are flagged in the
# summary: `motifs` -> S (short linear interaction/PTM motifs sit closest to S1
# domains) and `initiation_context` -> D (start-site strength is detection-side,
# alongside D2).
MODULE_CATEGORY = {
    "conservation": "C",
    "conservation_frame": "C",
    "expr": "D",
    "massspec": "D",
    "initiation_context": "D",
    "localization": "L",
    "signalp": "L",
    "targetp": "L",
    "clinical": "M",
    "variant_intersection": "M",
    "varianteffect": "M",
    "plm_vep": "M",
    "structure": "P",
    "biophysics": "S",
    "interproscan": "S",
    "sae": "S",
    "motifs": "S",
    # No CDLMPS category — identity, coordinates, reference context, roll-ups.
    "core_identity": "-",
    "generef": "-",
    "scoring": "-",
    "site": "-",
}

CATEGORY_ORDER = ("C", "D", "L", "M", "P", "S", "-")

# ---------------------------------------------------------------------------
# The scored levers — the ~19 numbers the evidence layer actually thresholds
# ---------------------------------------------------------------------------
#
# Curated from the per-criterion score() functions in
# src/swissisoform/evidence/<bucket>/__init__.py. Some criteria threshold a
# quantity DERIVED from a list column rather than a stored scalar (D1 counts
# cell lines with a detection, D3 counts hits passing two flags, P3 scans the
# SSE element list); those name the columns the derivation reads, so the lever
# is still traceable to storage.

SCORED_LEVERS: dict[str, list[str]] = {
    "C1_primate_conservation": ["isoform_conservation_frame_primate_mean_pident"],
    "C2_mammalian_conservation": ["isoform_conservation_frame_mammalian_mean_pident"],
    "C3_phylop_coding_selection": ["isoform_conservation_phylop_unique_region_mean"],
    # Derived: count of cell lines with a detection.
    "D1_multi_cell_line": [f"expr_{s}_raw_count" for s in SAMPLES],
    "D2_initiation_efficiency": [f"expr_{s}_initiation_efficiency" for s in SAMPLES],
    # Derived: count of hits with unique_to_isoform AND validated.
    "D3_mass_spec": ["isoform_massspec_hits"],
    "L1_localization_change": [
        "cmp_localization_deeploc_prediction_changed",
        "cmp_localization_deeploc_signals_changed",
        "cmp_localization_deeploc_membrane_changed",
    ],
    "L2_targeting_change": [
        "cmp_signalp_signalp_prediction_changed",
        "cmp_signalp_signalp_cleavage_site_changed",
        "cmp_targetp_targetp_prediction_changed",
        "cmp_targetp_targetp_ctp_prob_changed",
        "cmp_targetp_targetp_cleavage_site_changed",
    ],
    "M1_pathogenic_variant_enrichment": [
        "isoform_plm_vep_constraint_delta",
        # The full_catalog run predates the rename; carry both so a re-run
        # doesn't silently drop the dimension.
        "isoform_plm_vep_constraint_enrichment",
        "isoform_variant_intersection_gnomad_depletion_ratio",
    ],
    "M2_clinical_variant_overlap": ["isoform_variant_intersection_disease_enrichment_ratio"],
    "P1_structured_extension": ["isoform_structure_plddt_diffregion_mean"],
    "P2_shared_structural_change": [
        "isoform_structure_rmsd_shared",
        "isoform_structure_shared_region_len",
        "isoform_structure_plddt_shared_mean_isoform",
        "isoform_structure_plddt_shared_mean_canonical",
    ],
    # Derived: longest qualifying element in the SSE list.
    "P3_secondary_structure": ["isoform_structure_sse_all_elements"],
    "S1_domain_change": ["cmp_interproscan_n_real_domains_changed_in_diff_region"],
    "S2_biophysics": [
        "cmp_biophysics_gravy_delta",
        "cmp_biophysics_fraction_charged_delta",
        "cmp_biophysics_disorder_delta",
    ],
    "S3_sae": [
        "isoform_sae_top_gained_delta_max",
        "isoform_sae_top_lost_delta_max",
    ],
}

# ---------------------------------------------------------------------------
# Name-based exclusions
# ---------------------------------------------------------------------------

IDENTIFIER_PATTERNS = (
    r"(^|_)gene_id$",
    r"(^|_)gene_name$",
    r"(^|_)tis_id$",
    r"(^|_)transcript_id$",
    r"(^|_)variant_id$",
    r"_hash$",
    r"^chrom$",
    r"^position$",
    r"^strand$",
    r"uniprot_id$",
)

FREE_TEXT_PATTERNS = (
    r"^generef_function$",
    r"^generef_keywords$",
    r"^generef_subcellular_location$",
    r"^isoform_scoring_reasons\.",
    r"_sequence$",
    r"kozak_context$",
    r"start_codon$",
    r"deepest_species$",
    r"_label$",
    r"_description$",
)

STATUS_PATTERNS = (
    r"_status$",
    r"\.status$",
    r"unique_space$",
    r"hal_path$",
    r"_bigwig$",
    r"^diff_space$",
    r"^diff_region_confidence$",
    r"backend$",
)

# Scoring roll-ups are computed FROM these features; as dimensions they would
# double-count and leak the label. Kept in the catalog as stratifiers.
DERIVED_PATTERNS = (r"^isoform_scoring_",)

# ---------------------------------------------------------------------------
# Schema flattening
# ---------------------------------------------------------------------------


def leaves(field: pa.Field, prefix: str = "") -> Iterator[tuple[str, pa.DataType]]:
    """Yield ``(dotted_name, type)`` for every scalar-or-list leaf under *field*.

    Structs recurse (``isoform_motifs_summary`` -> its 32 children — where most
    of the unused signal lives). Lists stop at the list itself: their element
    structs are per-hit records, not per-isoform features.
    """
    name = prefix + field.name
    t = field.type
    if pa.types.is_struct(t):
        for i in range(t.num_fields):
            yield from leaves(t.field(i), name + ".")
    else:
        yield name, t


def kind_of(t: pa.DataType) -> str:
    """Map an Arrow type onto the catalog's coarse dtype vocabulary."""
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        return "list"
    if pa.types.is_null(t):
        return "null"
    if pa.types.is_boolean(t):
        return "bool"
    if pa.types.is_integer(t):
        return "int"
    if pa.types.is_floating(t) or pa.types.is_decimal(t):
        return "float"
    return "str"


def module_of(name: str) -> tuple[str, str]:
    """Return ``(module, pane)`` for a flattened feature name.

    A pane prefix only counts when what follows names a real module — otherwise
    ``diff_start`` / ``diff_space`` would be read as the Scope-A pane when they
    are plain per-site coordinate fields.
    """
    if name.startswith("expr_"):
        return "expr", "isoform"
    for pfx in PANE_PREFIXES:
        if not name.startswith(pfx):
            continue
        rest = name[len(pfx) :]
        pane = pfx.rstrip("_")
        if rest.startswith("expr_"):
            return "expr", pane
        for m in MODULES:
            if rest.startswith(m):
                return m, pane
    for m in MODULES:
        if name.startswith(m):
            return m, "site"
    return "site", "site"


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


def resolve_parquet(explicit: str | None) -> list[Path]:
    """Return the parquet path(s) to profile — merged file, else the shards."""
    if explicit:
        files = sorted(Path(p) for p in glob.glob(explicit))
        if not files:
            raise SystemExit(f"no parquet matched {explicit!r}")
        return files
    if DEFAULT_PARQUET.exists():
        return [DEFAULT_PARQUET]
    files = sorted(Path(p) for p in glob.glob(SHARD_GLOB))
    if not files:
        raise SystemExit(
            f"no genome-wide run found at {DEFAULT_PARQUET} or {SHARD_GLOB}\n"
            "Point --parquet at an all_paired.parquet."
        )
    return files


def read_flat(files: list[Path]) -> tuple[pd.DataFrame, dict[str, pa.DataType]]:
    """Read the non-list columns, fully flattened, plus the leaf type map.

    List columns are handled separately (:func:`list_lengths`) so the big hit
    lists are never materialised into pandas.
    """
    schema = pq.ParquetFile(files[0]).schema_arrow
    types = {n: t for fld in schema for n, t in leaves(fld)}
    non_list = [
        fld.name
        for fld in schema
        if not (pa.types.is_list(fld.type) or pa.types.is_large_list(fld.type))
    ]

    frames = []
    for path in files:
        present = {f.name for f in pq.ParquetFile(path).schema_arrow}
        table = pq.read_table(path, columns=[c for c in non_list if c in present])
        while any(pa.types.is_struct(f.type) for f in table.schema):
            table = table.flatten()
        frames.append(table.to_pandas())
    return pd.concat(frames, ignore_index=True), types


def list_lengths(files: list[Path], schema: pa.Schema) -> dict[str, pd.Series]:
    """Return per-list-column element counts, one row per run row, in file order.

    Streamed in batches, one column at a time: the hit lists dominate the file
    (``canonical_clinical_hits`` alone is 876 MB compressed) and we only want
    their lengths, so nothing larger than a batch is ever held.

    A file lacking the column contributes NaN for each of its rows rather than
    being skipped. ``build_catalog`` assigns these into the flat frame and then
    stratifies them by ``orf_type``, so a skipped file would shift every later
    shard's counts onto earlier shards' isoforms and corrupt ``null_pattern`` —
    which is what decides a tag's ``valid_for``.
    """
    out: dict[str, pd.Series] = {}
    counts = [pq.ParquetFile(p).metadata.num_rows for p in files]
    for fld in schema:
        if not (pa.types.is_list(fld.type) or pa.types.is_large_list(fld.type)):
            continue
        parts: list[pd.Series] = []
        seen = False
        for path, n_rows in zip(files, counts):
            pf = pq.ParquetFile(path)
            if fld.name not in {f.name for f in pf.schema_arrow}:
                parts.append(pd.Series([float("nan")] * n_rows, dtype="float64"))
                continue
            seen = True
            for batch in pf.iter_batches(batch_size=512, columns=[fld.name]):
                parts.append(pc.list_value_length(batch.column(0)).to_pandas())
        if seen:
            out[fld.name] = pd.concat(parts, ignore_index=True)
    return out


def profile(series: pd.Series) -> tuple[float, int]:
    """Return ``(fill_rate, n_unique)`` for one column."""
    fill = float(series.notna().mean()) if len(series) else 0.0
    try:
        n_unique = int(series.nunique(dropna=True))
    except TypeError:  # unhashable (nested) values
        n_unique = -1
    return round(fill, 4), n_unique


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def matches(name: str, patterns: tuple[str, ...]) -> bool:
    """True when *name* matches any regex in *patterns*."""
    return any(re.search(p, name) for p in patterns)


def classify(name: str, kind: str, fill: float, n_unique: int) -> tuple[bool, str, str]:
    """Return ``(include_in_plot, exclude_reason, encodable_as)``.

    A feature is a plot dimension when it is numeric, present often enough to
    carry information, and varies. Everything else records *why* it is out and
    *how* it could be brought in.
    """
    if kind == "null" or fill == 0.0:
        return False, "all_null", "-"
    if kind == "list":
        return False, "list", "count"
    if matches(name, DERIVED_PATTERNS):
        return False, "derived_from_features", "-"
    if kind == "bool":
        return False, "binary", "binary_0_1"
    if kind == "str":
        if matches(name, IDENTIFIER_PATTERNS):
            return False, "identifier", "-"
        if matches(name, FREE_TEXT_PATTERNS):
            return False, "free_text", "text_embedding"
        if matches(name, STATUS_PATTERNS):
            return False, "status_string", "one_hot" if n_unique > 1 else "-"
        if n_unique <= 1:
            return False, "constant", "-"
        return False, "categorical", "one_hot" if n_unique <= 20 else "-"

    # Numeric from here.
    if matches(name, IDENTIFIER_PATTERNS):
        return False, "identifier", "-"
    if n_unique <= 1:
        return False, "constant", "-"
    if n_unique < MIN_UNIQUE:
        return False, "binary", "binary_0_1"
    if fill < MIN_FILL:
        return False, "sparse", "numeric"
    return True, "", "numeric"


def transform_of(name: str, kind: str) -> str:
    """Suggested scaling for the plot, from the feature's name and dtype."""
    if kind not in ("int", "float"):
        return "-"
    base = name.split(".")[-1]
    if re.search(r"(p_value|pvalue|qvalue)$", base):
        return "neglog10"
    if re.search(r"(_ratio|_enrichment|_delta|_delta_max|_depletion_ratio)$", base):
        return "signed_log"
    if re.search(r"^(n_|total_)|(_count|_cpm|_psms|_nt|_len|_length|per100aa)$", base):
        return "log1p"
    return "linear"


def build_category_index(
    known: list[str],
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Map parquet column -> criterion ids, from the curated evidence registry.

    A curated entry that names a *struct* (``isoform_massspec_summary``) claims
    every leaf beneath it — the registry was written against top-level columns,
    while this catalog works in flattened leaves.
    """
    col_criteria: dict[str, list[str]] = {}
    for cid, spec in CRITERIA.items():
        for col in spec.get("evidence_cols", []) or []:
            targets = [col] if col in known else [n for n in known if n.startswith(col + ".")]
            for target in targets or [col]:
                col_criteria.setdefault(target, []).append(cid)
    for cols in col_criteria.values():
        cols.sort(key=lambda c: CATEGORY_ORDER.index(c[0]) if c[0] in CATEGORY_ORDER else 99)
    lever_criterion = {col: cid for cid, cols in SCORED_LEVERS.items() for col in cols}
    return col_criteria, lever_criterion


def null_pattern(name: str, df: pd.DataFrame, orf_type: pd.Series | None) -> tuple[str, float]:
    """Report how a feature's availability splits by ORF type.

    Separate ORFs (uORF, uoORF, internal-OOF, 3'UTR-ORF) have no *sequence*
    shared with the canonical protein, so protein-space unique-vs-shared
    features are missing for them entirely. Genomic-space ones are not: a uoORF
    or internal-OOF ORF can still overlap the canonical CDS on the genome, which
    is why ``phylop_shared_region_mean`` is ~29% present for that stratum rather
    than 0.

    Returns ``(flag, fill_separate_orfs)``. The measured stratum fill is
    reported alongside the flag so a downstream embedding can apply its own
    tolerance instead of inheriting the cutoff chosen here.
    """
    if orf_type is None or name not in df.columns:
        return "", float("nan")
    sep = orf_type.isin(SEPARATE_ORF_TYPES)
    if not sep.any() or sep.all():
        return "", float("nan")
    col = df[name]
    fill_sep = round(float(col[sep].notna().mean()), 4)
    fill_paired = float(col[~sep].notna().mean())
    if fill_paired < 0.5:
        return "", fill_sep
    if fill_sep < 0.01:
        return "absent_for_separate_orfs", fill_sep
    if fill_sep < fill_paired / 2:
        return "depleted_for_separate_orfs", fill_sep
    return "", fill_sep


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_catalog(files: list[Path]) -> pd.DataFrame:
    """Profile the run and classify every feature into the catalog frame."""
    df, types = read_flat(files)
    schema = pq.ParquetFile(files[0]).schema_arrow
    lengths = list_lengths(files, schema)
    orf_type = df["orf_type"] if "orf_type" in df.columns else None

    col_criteria, lever_criterion = build_category_index(list(types))

    rows: list[dict[str, Any]] = []
    for name, arrow_type in types.items():
        kind = kind_of(arrow_type)
        if kind == "list":
            series = lengths.get(name)
            fill, n_unique = profile(series) if series is not None else (0.0, 0)
        elif name in df.columns:
            fill, n_unique = profile(df[name])
        else:
            fill, n_unique = 0.0, 0
        parent = _row(name, kind, fill, n_unique, col_criteria, lever_criterion)
        rows.append(parent)

        # Derived per-isoform count for each list column. It belongs to the same
        # module and category as the list it counts — ``n_isoform_massspec_hits``
        # is a D feature, not an uncategorized one. Redundancy against an
        # existing count column is detected numerically below, not guessed.
        if kind == "list" and name in lengths:
            derived = f"n_{name}"
            d_fill, d_unique = profile(lengths[name])
            row = _row(derived, "int", d_fill, d_unique, col_criteria, lever_criterion)
            for key in ("module", "pane", "category", "category_source", "criteria"):
                row[key] = parent[key]
            row["derived_from"] = name
            rows.append(row)
            df[derived] = lengths[name]

    catalog = pd.DataFrame(rows)
    patterns = [
        null_pattern(n, df, orf_type) if inc else ("", float("nan"))
        for n, inc in zip(catalog["feature"], catalog["include_in_plot"])
    ]
    catalog["null_pattern"] = [p for p, _ in patterns]
    catalog["fill_separate_orfs"] = [f for _, f in patterns]
    _mark_duplicates(catalog, df)
    return catalog


def _row(
    name: str,
    kind: str,
    fill: float,
    n_unique: int,
    col_criteria: dict[str, list[str]],
    lever_criterion: dict[str, str],
) -> dict[str, Any]:
    """Assemble one catalog row."""
    module, pane = module_of(name)
    include, reason, encodable = classify(name, kind, fill, n_unique)

    criteria = col_criteria.get(name, [])
    lever = lever_criterion.get(name)
    if lever and lever not in criteria:
        criteria = [*criteria, lever]

    if criteria:
        category, source = criteria[0][0], "evidence_cols"
    else:
        category, source = MODULE_CATEGORY.get(module, "-"), "module_map"
        if category == "-":
            source = "none"

    label = CRITERIA_METRIC_LABELS.get(name, {}).get("label", "")
    return {
        "feature": name,
        "module": module,
        "pane": pane,
        "category": category,
        "category_source": source,
        "criteria": ";".join(criteria),
        "scored": bool(lever),
        "dtype": kind,
        "fill_rate": fill,
        "n_unique": n_unique,
        "include_in_plot": include,
        "exclude_reason": reason,
        "encodable_as": encodable,
        "transform": transform_of(name, kind),
        "label": label,
        "derived_from": "",
        "duplicate_of": "",
    }


def _mark_duplicates(catalog: pd.DataFrame, df: pd.DataFrame) -> None:
    """Demote exact value-duplicates among included features to ``redundant``.

    Two features carrying identical values (same numbers, same null pattern) are
    one dimension counted twice — e.g. ``diff_end`` and
    ``isoform_core_identity_differential_length_aa``.

    Which of the pair survives matters: the representative should be the one a
    reader can interpret. Prefer a curated category over an inferred one, a
    stored column over a derived count, a categorized feature over an
    uncategorized one, then the shorter name. That keeps
    ``isoform_biophysics_length`` (S) over the bare ``aa_len``.
    """

    def preference(idx: int) -> tuple:
        row = catalog.loc[idx]
        return (
            0 if row["category_source"] == "evidence_cols" else 1,
            0 if not row["derived_from"] else 1,
            0 if row["category"] != "-" else 1,
            len(row["feature"]),
        )

    seen: dict[tuple, str] = {}
    for idx in sorted(catalog.index[catalog["include_in_plot"]], key=preference):
        name = catalog.at[idx, "feature"]
        if name not in df.columns:
            continue
        col = df[name]
        try:
            key = (tuple(pd.util.hash_pandas_object(col, index=False)),)
        except TypeError:
            continue
        if key in seen:
            catalog.at[idx, "include_in_plot"] = False
            catalog.at[idx, "exclude_reason"] = "redundant"
            catalog.at[idx, "duplicate_of"] = seen[key]
        else:
            seen[key] = name


def verify(catalog: pd.DataFrame) -> list[str]:
    """Return human-readable warnings about the catalog's internal consistency."""
    warnings: list[str] = []
    known = set(catalog["feature"])

    for cid, cols in SCORED_LEVERS.items():
        missing = [c for c in cols if c not in known]
        if len(missing) == len(cols):
            warnings.append(f"{cid}: none of its levers are in this run ({missing})")
        elif missing:
            warnings.append(f"{cid}: levers absent from this run: {missing}")

    scored = catalog[catalog["scored"]]
    excluded_levers = scored[~scored["include_in_plot"]]
    if len(excluded_levers):
        detail = ", ".join(
            f"{r.feature} ({r.exclude_reason})" for r in excluded_levers.itertuples()
        )
        warnings.append(
            f"{len(excluded_levers)} scored levers are not plot dimensions: {detail}. "
            "Expected for L1/L2 (boolean change flags) and D3/P3 (list-derived)."
        )

    uncategorised = catalog[(catalog["category"] == "-") & catalog["include_in_plot"]]
    if len(uncategorised):
        warnings.append(
            f"{len(uncategorised)} included features have no CDLMPS category "
            "(identity / coordinate fields — expected): "
            f"{', '.join(uncategorised['feature'])}"
        )

    # A curated entry is satisfied by the column itself or, when it names a
    # struct, by any leaf flattened out of it.
    curated = {
        c for cols in (CRITERIA[k].get("evidence_cols") or [] for k in CRITERIA) for c in cols
    }
    unmatched = sorted(
        c for c in curated if c not in known and not any(n.startswith(c + ".") for n in known)
    )
    if unmatched:
        warnings.append(f"{len(unmatched)} curated evidence_cols absent from this run: {unmatched}")
    return warnings


def main() -> None:
    """Profile the run, classify every feature, write the catalog."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--parquet",
        default=None,
        help="all_paired.parquet path or glob (default: merged full_catalog, else shards)",
    )
    args = ap.parse_args()

    files = resolve_parquet(args.parquet)
    print(f"profiling {files[0]}" + (f" (+{len(files) - 1} more)" if len(files) > 1 else ""))

    catalog = build_catalog(files)

    catalog = catalog.sort_values(
        ["category", "module", "pane", "feature"],
        key=lambda s: s.map(CATEGORY_ORDER.index) if s.name == "category" else s,
    )
    catalog.to_csv(OUT_CSV, index=False)

    for w in verify(catalog):
        print(f"NOTE: {w}", file=sys.stderr)
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
