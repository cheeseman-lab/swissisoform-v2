"""Build the frozen metric distributions from a paired-TIS run (setup-time).

Profiles **every** metric in ``all_paired.parquet`` — scored and unscored alike —
stratified by ``orf_type``, and freezes the result as provisioned reference data
under ``data/reference/distributions/<version>/``. The runtime read-side
(``percentile`` / ``value_at``) lives in :mod:`swissisoform.distributions`.

Why: issue #30 replaces the interesting/neutral/not_interesting verdict axis with
per-category tags whose cutoffs must be *distribution-referenced* — an absolute
test whose number is recorded as a percentile of a stable population, rather than
a hand-picked constant. The distribution has to exist and be frozen before any tag
can be defined against it.

Four tables per version:

- ``numeric.parquet``     one row per (metric, stratum): quantile grid + histogram
- ``categorical.parquet`` one row per (metric, stratum, value): count + frac
- ``criteria.parquet``    one row per (criterion, stratum): as-run tri-state rates
- ``distributions_summary.csv``  the scalar columns of ``numeric``, for reading

The metric registry is an **input**, not re-derived here: the feature catalog
(``figures/clustering_dims/feature_space/feature_catalog.csv``) already names every
flattened leaf with its module / pane / CDLMPS category, so "what is a metric" stays
answered in one place. Its sha256 goes into the provenance sidecar.

Driven by the thin CLI ``scripts/setup/build_distributions.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from swissisoform import metrics
from swissisoform.distributions import (
    CATEGORICAL_FILE,
    CRITERIA_FILE,
    HIST_BINS,
    NAMED_PERCENTILES,
    NUMERIC_FILE,
    OTHER_VALUE,
    QUANTILE_GRID,
    SEPARATE_ORF_TYPES,
    SIDECAR_FILE,
    STRATUM_ALL,
    STRATUM_SEPARATE,
    SUMMARY_FILE,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG = ROOT / "figures" / "clustering_dims" / "feature_space" / "feature_catalog.csv"
CRITERIA_STRUCT = "isoform_scoring_criteria"

# Catalog dtypes routed to each table. `bool` goes to the categorical side: its
# useful summary is a rate, not a quantile grid.
NUMERIC_DTYPES = frozenset({"float", "int"})
CATEGORICAL_DTYPES = frozenset({"str", "bool"})

# Distinct values kept per categorical metric; the tail aggregates into __other__.
TOP_K = 20

CATALOG_COLUMNS = ("feature", "module", "pane", "category", "dtype")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _rel(path: Path) -> str:
    """Repo-relative path when it is inside the repo, else absolute.

    Inputs are routinely outside the tree (a scratch parquet, a tmp_path fixture),
    so a bare ``relative_to`` would fail the build on the provenance write.
    """
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_catalog(path: Path) -> pd.DataFrame:
    """Read the feature catalog that names every metric.

    Raises:
        SystemExit: The catalog is missing or lacks a required column — naming the
            command that regenerates it, since it is an untracked build input.
    """
    if not path.exists():
        raise SystemExit(
            f"feature catalog not found: {path}\n"
            "Regenerate it with:\n"
            "  python figures/clustering_dims/feature_space/export_feature_catalog.py"
        )
    cat = pd.read_csv(path)
    missing = [c for c in CATALOG_COLUMNS if c not in cat.columns]
    if missing:
        raise SystemExit(f"{path} is missing required column(s): {missing}")
    return cat


def read_flat(files: list[Path]) -> pd.DataFrame:
    """Read every non-list column, structs fully flattened to dotted names.

    List columns are excluded here and handled by :func:`list_lengths` — the hit
    lists dominate the file (``canonical_clinical_hits`` alone is ~876 MB
    compressed) and only their lengths are wanted.
    """
    frames = []
    for path in files:
        schema = pq.ParquetFile(path).schema_arrow
        non_list = [
            f.name
            for f in schema
            if not (pa.types.is_list(f.type) or pa.types.is_large_list(f.type))
        ]
        table = pq.read_table(path, columns=non_list)
        while any(pa.types.is_struct(f.type) for f in table.schema):
            table = table.flatten()
        frames.append(table.to_pandas())
    return pd.concat(frames, ignore_index=True)


def read_list_columns(files: list[Path], names: Iterable[str]) -> dict[str, pd.Series]:
    """Materialise named list columns in full, for transforms that need the hits.

    :func:`read_flat` drops every list column because they dominate the file, but
    a few derived metrics need the elements themselves — D3 counts hits flagged
    both unique and validated, a conjunction the summary struct does not carry.
    """
    wanted = list(names)
    out: dict[str, list[pd.Series]] = {}
    for path in files:
        present = {f.name for f in pq.ParquetFile(path).schema_arrow}
        for name in wanted:
            if name not in present:
                continue
            col = pq.read_table(path, columns=[name]).column(0).to_pandas()
            out.setdefault(name, []).append(col)
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}


def list_lengths(files: list[Path]) -> dict[str, pd.Series]:
    """Per-list-column element counts, streamed one column at a time.

    A hit-list length is a real per-isoform metric (how many variants, domains,
    peptides) and would otherwise be lost entirely.
    """
    out: dict[str, list[pd.Series]] = {}
    for path in files:
        pf = pq.ParquetFile(path)
        for fld in pf.schema_arrow:
            if not (pa.types.is_list(fld.type) or pa.types.is_large_list(fld.type)):
                continue
            for batch in pf.iter_batches(batch_size=512, columns=[fld.name]):
                out.setdefault(fld.name, []).append(
                    pc.list_value_length(batch.column(0)).to_pandas()
                )
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}


def load_run(files: list[Path]) -> pd.DataFrame:
    """The canonical frame a run is profiled and swept over.

    Flattened scalars, plus ``<col>__len`` for every list column, plus the few
    list columns some transform needs whole. Both the profiler and the candidate
    sweep go through here: a cutoff derived from one frame and evaluated against
    a differently-loaded one would silently disagree (D3 is the live example — it
    needs the mass-spec hit list, which the flattened read drops).
    """
    df = read_flat(files)
    for col, series in list_lengths(files).items():
        df[f"{col}__len"] = series.reindex(df.index)
    for col, series in read_list_columns(files, metrics.required_list_columns()).items():
        df[col] = series.reindex(df.index)
    return df


# ---------------------------------------------------------------------------
# Strata
# ---------------------------------------------------------------------------


def build_strata(orf_type: pd.Series) -> dict[str, pd.Series]:
    """Boolean masks for every stratum, in report order.

    ``all`` first, then each observed ``orf_type``, then the ``separate`` roll-up
    of the no-shared-region types. The roll-up exists because those types are
    individually far too small for percentiles (uoORF n=21 on full_catalog).
    """
    strata: dict[str, pd.Series] = {STRATUM_ALL: pd.Series(True, index=orf_type.index)}
    for value in orf_type.dropna().unique():
        strata[str(value)] = orf_type == value
    sep = orf_type.isin(SEPARATE_ORF_TYPES)
    if sep.any():
        strata[STRATUM_SEPARATE] = sep
    return strata


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


def numeric_row(values: pd.Series, meta: dict[str, Any], stratum: str) -> dict[str, Any]:
    """Profile one metric within one stratum.

    Quantiles and the histogram are computed over non-null values only; the null
    count is carried separately as ``n_missing`` so a sparse metric is never
    mistaken for a narrow one.
    """
    v = pd.to_numeric(values, errors="coerce")
    v = v[np.isfinite(v)]
    n = int(len(v))
    row: dict[str, Any] = {
        **meta,
        "stratum": stratum,
        "n": n,
        "n_missing": int(len(values) - n),
        "fill_rate": round(n / len(values), 6) if len(values) else 0.0,
    }
    if n == 0:
        empty = [float("nan")] * len(QUANTILE_GRID)
        row.update(
            {k: None for k in ("mean", "std", "min", "max")}
            | {f"p{p:02d}": None for p in NAMED_PERCENTILES}
            | {"q_grid": empty, "hist_edges": [], "hist_counts": []}
        )
        return row

    grid = np.quantile(v.to_numpy(dtype=float), QUANTILE_GRID)
    row.update(
        {
            "mean": float(v.mean()),
            "std": float(v.std()) if n > 1 else 0.0,
            "min": float(grid[0]),
            "max": float(grid[-1]),
            **{f"p{p:02d}": float(grid[p]) for p in NAMED_PERCENTILES},
            "q_grid": [float(x) for x in grid],
        }
    )
    # Histogram spans p01..p99 so a single outlier cannot collapse every value
    # into one bin; a degenerate (constant) metric gets a unit-wide span instead.
    lo, hi = float(grid[1]), float(grid[99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(grid[0]), float(grid[0]) + 1.0
    counts, edges = np.histogram(v.to_numpy(dtype=float), bins=HIST_BINS, range=(lo, hi))
    row["hist_edges"] = [float(x) for x in edges]
    row["hist_counts"] = [int(x) for x in counts]
    return row


def categorical_rows(
    values: pd.Series, meta: dict[str, Any], stratum: str, top_k: int = TOP_K
) -> list[dict[str, Any]]:
    """Value frequencies for one categorical metric within one stratum.

    Localization compartments and targeting calls have no percentile but do have a
    rate, and L's tags are categorical-change tags — without this the category has
    no substrate at all.
    """
    counts = values.dropna().astype(str).value_counts()
    total = int(counts.sum())
    if total == 0:
        return []
    head = counts.head(top_k)
    rows = [
        {
            **meta,
            "stratum": stratum,
            "value": str(val),
            "count": int(cnt),
            "frac": round(int(cnt) / total, 6),
        }
        for val, cnt in head.items()
    ]
    tail = total - int(head.sum())
    if tail > 0:
        rows.append(
            {
                **meta,
                "stratum": stratum,
                "value": OTHER_VALUE,
                "count": tail,
                "frac": round(tail / total, 6),
            }
        )
    return rows


def criteria_rows(df: pd.DataFrame, strata: dict[str, pd.Series]) -> list[dict[str, Any]]:
    """As-run tri-state rates per scored criterion, per stratum.

    Null means "could not evaluate", which is exactly the state issue #30 wants
    represented — and a ``frac_none`` gap confined to one stratum is how the
    M1-on-extensions confound shows up.
    """
    prefix = f"{CRITERIA_STRUCT}."
    cols = [c for c in df.columns if c.startswith(prefix)]
    rows: list[dict[str, Any]] = []
    for col in sorted(cols):
        name = col[len(prefix) :]
        for stratum, mask in strata.items():
            sub = df.loc[mask, col]
            n_true = int((sub == True).sum())  # noqa: E712 — nullable bool, `is` fails
            n_false = int((sub == False).sum())  # noqa: E712
            n_none = int(len(sub) - n_true - n_false)
            evaluable = n_true + n_false
            rows.append(
                {
                    "criterion": name,
                    "category": name[0] if name else "-",
                    "stratum": stratum,
                    "n": int(len(sub)),
                    "n_true": n_true,
                    "n_false": n_false,
                    "n_none": n_none,
                    "frac_true": round(n_true / evaluable, 6) if evaluable else None,
                    "frac_evaluable": round(evaluable / len(sub), 6) if len(sub) else None,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _metric_meta(cat_row: pd.Series, feature: str) -> dict[str, Any]:
    def _s(key: str, default: str = "") -> str:
        val = cat_row.get(key)
        return default if pd.isna(val) else str(val)

    return {
        "metric": feature,
        "module": _s("module"),
        "pane": _s("pane"),
        "category": _s("category", "-"),
        "scored": bool(cat_row.get("scored", False)),
        "include_in_plot": bool(cat_row.get("include_in_plot", False)),
        "exclude_reason": _s("exclude_reason"),
    }


def build(
    parquet_files: list[Path],
    catalog_csv: Path,
    out_dir: Path,
    *,
    source_label: str = "",
) -> dict[str, int]:
    """Profile a run and write one frozen distributions version.

    Args:
        parquet_files: One or more ``all_paired.parquet`` paths (merged file, or
            the per-shard files of one campaign).
        catalog_csv: The feature catalog naming every metric.
        out_dir: Version directory to create (must not already hold tables unless
            the caller cleared it).
        source_label: Run name recorded in provenance.

    Returns:
        Counts of rows written per table.
    """
    catalog = load_catalog(catalog_csv)
    df = load_run(parquet_files)
    if "orf_type" not in df.columns:
        raise SystemExit("orf_type column absent — cannot stratify; is this a paired parquet?")

    strata = build_strata(df["orf_type"])
    present = set(df.columns)

    numeric: list[dict[str, Any]] = []
    categorical: list[dict[str, Any]] = []
    n_skipped = 0

    for _, cat_row in catalog.iterrows():
        feature = str(cat_row["feature"])
        dtype = str(cat_row.get("dtype", ""))
        # A list column contributes its length metric, inheriting the parent's
        # module/pane/category — the list itself has no distribution.
        targets: list[tuple[str, str]] = (
            [(f"{feature}__len", "int")] if dtype == "list" else [(feature, dtype)]
        )
        for name, kind in targets:
            if name not in present:
                n_skipped += 1
                continue
            meta = _metric_meta(cat_row, name)
            for stratum, mask in strata.items():
                values = df.loc[mask, name]
                if kind in NUMERIC_DTYPES:
                    numeric.append(numeric_row(values, meta, stratum))
                elif kind in CATEGORICAL_DTYPES:
                    categorical.extend(categorical_rows(values, meta, stratum))

    # Magnitudes of signed columns. A tag on a *_delta asks whether a property
    # changed appreciably (|delta| over a bar), not which direction it moved, so
    # the magnitude needs its own frozen distribution to cut against.
    n_magnitudes = 0
    for _, cat_row in catalog.iterrows():
        feature = str(cat_row["feature"])
        if (
            "_delta" not in feature
            or str(cat_row.get("dtype", "")) not in NUMERIC_DTYPES
            or feature not in present
            or metrics.is_magnitude(feature)
        ):
            continue
        meta = _metric_meta(cat_row, metrics.magnitude_of(feature))
        values = df[feature].abs()
        for stratum, mask in strata.items():
            numeric.append(numeric_row(values[mask], meta, stratum))
        n_magnitudes += 1

    # Derived metrics. Five criteria (D1/D2/D3/P2's gate/S3) and S2's magnitudes
    # score a computed quantity rather than a column, so without this pass they
    # would have no frozen distribution to reference a cutoff against.
    n_transforms = 0
    for tx in metrics.available(present):
        values = tx.fn(df)
        meta = {
            "metric": tx.metric,
            "module": "transform",
            "pane": "isoform",
            "category": tx.category,
            "scored": True,
            "include_in_plot": True,
            "exclude_reason": "",
        }
        for stratum, mask in strata.items():
            numeric.append(numeric_row(values[mask], meta, stratum))
        n_transforms += 1

    numeric_df = pd.DataFrame(numeric)
    categorical_df = pd.DataFrame(categorical)
    criteria_df = pd.DataFrame(criteria_rows(df, strata))

    out_dir.mkdir(parents=True, exist_ok=True)
    numeric_df.to_parquet(out_dir / NUMERIC_FILE, index=False)
    if len(categorical_df):
        categorical_df.to_parquet(out_dir / CATEGORICAL_FILE, index=False)
    if len(criteria_df):
        criteria_df.to_parquet(out_dir / CRITERIA_FILE, index=False)

    # Human-readable view: the scalar columns only. The grid and histogram are
    # for code; a CSV carrying 101+65+64 numbers per row is unreadable.
    drop = [c for c in ("q_grid", "hist_edges", "hist_counts") if c in numeric_df.columns]
    numeric_df.drop(columns=drop).to_csv(out_dir / SUMMARY_FILE, index=False)

    write_sidecar(
        out_dir,
        parquet_files=parquet_files,
        catalog_csv=catalog_csv,
        source_label=source_label,
        n_rows=int(len(df)),
        strata={k: int(v.sum()) for k, v in strata.items()},
        counts={
            "numeric": len(numeric_df),
            "categorical": len(categorical_df),
            "criteria": len(criteria_df),
            "metrics_not_in_run": n_skipped,
            "transforms": n_transforms,
            "magnitudes": n_magnitudes,
        },
    )
    return {
        "numeric": len(numeric_df),
        "categorical": len(categorical_df),
        "criteria": len(criteria_df),
        "isoforms": int(len(df)),
    }


def write_sidecar(
    out_dir: Path,
    *,
    parquet_files: list[Path],
    catalog_csv: Path,
    source_label: str,
    n_rows: int,
    strata: dict[str, int],
    counts: dict[str, int],
) -> None:
    """Write ``_setup.json`` — provenance, plus the caveats a reader must know.

    The population caveat is load-bearing: the genome-wide campaign runs with
    ``--drop-unsupported-tis``, so it is HeLa long-read-supported TIS only.
    Cross-cell-line criteria are biased by construction, and a version built from
    a different population is not comparable to this one.
    """
    payload: dict[str, Any] = {
        "artifact": "metric distributions (frozen percentile reference)",
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_run": source_label,
        "source_parquet": [_rel(p) for p in parquet_files],
        "source_parquet_sha256": [_sha256(p) for p in parquet_files],
        "feature_catalog": _rel(catalog_csv),
        "feature_catalog_sha256": _sha256(catalog_csv),
        "n_isoforms": n_rows,
        "strata_n": strata,
        "row_counts": counts,
        "quantile_grid": f"{len(QUANTILE_GRID)} points, 0-100 by 1",
        "hist_bins": HIST_BINS,
        "caveats": [
            "Quantiles are over non-null values only; n_missing is carried separately.",
            "Stratified by orf_type; separate-ORF types roll up into 'separate'.",
            "The full_catalog campaign runs with --drop-unsupported-tis, so the "
            "population is HeLa long-read-supported TIS only. Cross-cell-line "
            "criteria (D1) are biased by construction, and a version built from an "
            "unrestricted population is not comparable to this one.",
        ],
    }
    (out_dir / SIDECAR_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def resolve_parquet(run: str | None, parquet: Path | None) -> tuple[list[Path], str]:
    """Return ``(files, label)`` for a run name or an explicit parquet path."""
    if parquet is not None:
        return [parquet], parquet.parent.name
    run_dir = ROOT / "data" / "output" / (run or "")
    merged = run_dir / "all_paired.parquet"
    if merged.exists():
        return [merged], run or ""
    shards = sorted((ROOT / "data" / "output").glob(f"{run}_shard_*/all_paired.parquet"))
    if shards:
        return shards, run or ""
    raise SystemExit(f"no all_paired.parquet under {run_dir} or {run}_shard_*/")


def main(argv: Iterable[str] | None = None) -> int:
    """Build one frozen distributions version. Refuses to clobber without --force."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="Run name under data/output/")
    g.add_argument("--parquet", type=Path, help="Explicit all_paired.parquet path")
    p.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG, help="Feature catalog CSV")
    p.add_argument("--version", default="v1", help="Version directory name (default: v1)")
    p.add_argument("--out", type=Path, default=None, help="Override the output directory")
    p.add_argument("--force", action="store_true", help="Overwrite an existing version")
    args = p.parse_args(list(argv) if argv is not None else None)

    files, label = resolve_parquet(args.run, args.parquet)
    out_dir = args.out or (ROOT / "data" / "reference" / "distributions" / args.version)
    if (out_dir / NUMERIC_FILE).exists() and not args.force:
        raise SystemExit(
            f"{out_dir} already holds a built version. The distributions are frozen on "
            "purpose — bump --version for a new one, or pass --force to rebuild in place."
        )

    counts = build(files, args.catalog, out_dir, source_label=label)
    print(
        f"wrote {out_dir}  ({counts['isoforms']} isoforms → "
        f"{counts['numeric']} numeric, {counts['categorical']} categorical, "
        f"{counts['criteria']} criteria rows)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
