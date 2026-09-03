"""Frozen metric distributions — runtime read-side (percentile lookup).

Places a metric value against the distribution of that metric over a frozen
reference population, so a tag cutoff can be *derived* once from a percentile and
then recorded as a plain scalar with its provenance ("p75 among truncations, v1").

The distributions are provisioned reference data under
``data/reference/distributions/<version>/``, built by
:mod:`swissisoform.setup.distributions` (CLI:
``scripts/setup/build_distributions.py``). This module is import-safe, network-free
and does no computation over the pipeline output — it only reads the frozen tables.

Why frozen: percentiles recomputed over a rolling corpus would make an isoform's
tags change when unrelated genes are added. A version is immutable; re-versioning
is deliberate.

Stratification is by ``orf_type``. Several metrics have different baselines — or no
validity at all — for extensions vs truncations, so pooled percentiles mislead.
Separate-ORF types are individually too small for percentiles and roll up into a
single ``separate`` stratum.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REF_DIR = ROOT / "data" / "reference" / "distributions"
DEFAULT_VERSION = "v1"

# 101-point grid: one quantile per whole percentile, so percentile rank resolves
# to 1 point. Denormalised p01/p05/... columns are read off this same grid.
QUANTILE_GRID: tuple[float, ...] = tuple(i / 100 for i in range(101))
NAMED_PERCENTILES: tuple[int, ...] = (1, 5, 10, 25, 50, 75, 90, 95, 99)

# Fixed bin count for the shape histogram. Raw counts, deliberately: step 0
# describes shape, the tag-proposal step decides whether a break is real.
HIST_BINS = 64

STRATUM_ALL = "all"
STRATUM_SEPARATE = "separate"

# Separate-ORF types have no shared region at all, so every unique-vs-shared
# feature is null for them by construction (see comparator `_shared_annotations`).
SEPARATE_ORF_TYPES: tuple[str, ...] = ("uorf", "uoorf", "internal_oof", "3utr_orf", "alt_orf")

# Below this a stratum cannot support a percentile-derived cutoff. On the
# full_catalog population only `uoorf` (n=21) falls short; the `separate` roll-up
# exists so those isoforms still have a baseline.
MIN_STRATUM_N = 30

# Sentinel row in the categorical table holding the tail beyond the top-K values.
OTHER_VALUE = "__other__"

NUMERIC_FILE = "numeric.parquet"
CATEGORICAL_FILE = "categorical.parquet"
CRITERIA_FILE = "criteria.parquet"
SUMMARY_FILE = "distributions_summary.csv"
SIDECAR_FILE = "_setup.json"


def stratum_for(orf_type: Any) -> str:
    """Return the stratum an ``orf_type`` belongs to, besides ``all``.

    Separate-ORF types map to the ``separate`` roll-up rather than to themselves;
    callers wanting the per-type stratum should pass the ``orf_type`` directly,
    which is also a valid stratum name when its ``n`` clears the floor.
    """
    if orf_type in SEPARATE_ORF_TYPES:
        return STRATUM_SEPARATE
    return str(orf_type) if orf_type is not None else STRATUM_ALL


def _none_if_nan(value: Any) -> Any:
    """Map a non-finite float to None, leaving everything else untouched."""
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class DistributionsError(RuntimeError):
    """A distributions version is missing, malformed, or too small to use."""


@dataclass(frozen=True)
class Distributions:
    """One frozen version of the metric distributions.

    Attributes:
        version: Version directory name (``"v1"``).
        numeric: One row per ``(metric, stratum)`` with quantile grid + histogram.
        categorical: One row per ``(metric, stratum, value)`` with count + frac.
        criteria: One row per ``(criterion, stratum)`` with tri-state counts.
        provenance: The ``_setup.json`` payload — source run, caveats, build stamp.
    """

    version: str
    numeric: pd.DataFrame
    categorical: pd.DataFrame
    criteria: pd.DataFrame
    provenance: dict[str, Any]

    # ── Lookup ────────────────────────────────────────────────────────────

    def summary(self, metric: str, stratum: str = STRATUM_ALL) -> dict[str, Any] | None:
        """Return the scalar summary for one ``(metric, stratum)``, or None.

        Includes ``n`` — always check it before trusting a percentile from a rare
        stratum; the floor is only enforced by :meth:`value_at`.
        """
        row = self._numeric_row(metric, stratum)
        if row is None:
            return None
        drop = {"q_grid", "hist_edges", "hist_counts"}
        # An empty stratum writes null quantiles, which parquet returns as NaN.
        # Hand back None so `is None` works and no NaN propagates into arithmetic.
        return {k: _none_if_nan(v) for k, v in row.items() if k not in drop}

    def percentile(
        self, metric: str, value: float | None, stratum: str = STRATUM_ALL
    ) -> float | None:
        """Percentile rank of *value* within ``(metric, stratum)``, 0-100.

        Resolution is one percentile point (the grid is 101 wide). Ties resolve to
        the top of the tied run, so a value equal to a spike reports the whole
        spike as below it — which is the honest reading for "how extreme is this".

        Returns None when the metric, the stratum, or *value* is missing. Does NOT
        enforce :data:`MIN_STRATUM_N`: this is a display path, and the caller can
        read ``n`` from :meth:`summary`.
        """
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return None
        row = self._numeric_row(metric, stratum)
        if row is None or row.get("n", 0) == 0:
            return None
        grid = np.asarray(row["q_grid"], dtype=float)
        return float(np.clip(np.searchsorted(grid, float(value), side="right"), 0, 100))

    def value_at(
        self,
        metric: str,
        pctile: float,
        stratum: str = STRATUM_ALL,
        *,
        allow_small: bool = False,
    ) -> float | None:
        """The metric value at *pctile* (0-100) within ``(metric, stratum)``.

        The inverse of :meth:`percentile`, and the one that matters for tag
        definition: it turns "p75 among truncations" into the scalar that gets
        frozen into the tag registry. Interpolates linearly between grid points.

        Raises:
            DistributionsError: The stratum holds fewer than
                :data:`MIN_STRATUM_N` values, so a cutoff derived from it would be
                noise. Pass ``allow_small=True`` to override deliberately.
        """
        if not 0 <= pctile <= 100:
            raise ValueError(f"pctile must be in [0, 100], got {pctile}")
        row = self._numeric_row(metric, stratum)
        if row is None or row.get("n", 0) == 0:
            return None
        n = int(row["n"])
        if n < MIN_STRATUM_N and not allow_small:
            hint = (
                f"Use the {STRATUM_SEPARATE!r} roll-up"
                if stratum in SEPARATE_ORF_TYPES
                else f"Use the {STRATUM_ALL!r} stratum"
            )
            raise DistributionsError(
                f"stratum {stratum!r} has n={n} for {metric!r}, below MIN_STRATUM_N="
                f"{MIN_STRATUM_N}; a cutoff derived from it would be noise. "
                f"{hint}, or pass allow_small=True."
            )
        grid = np.asarray(row["q_grid"], dtype=float)
        return float(np.interp(pctile, np.arange(len(grid)), grid))

    def histogram(
        self, metric: str, stratum: str = STRATUM_ALL
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return ``(edges, counts)`` for the shape histogram, or None."""
        row = self._numeric_row(metric, stratum)
        if row is None or row.get("n", 0) == 0:
            return None
        return np.asarray(row["hist_edges"], float), np.asarray(row["hist_counts"], float)

    def rate(self, metric: str, value: Any, stratum: str = STRATUM_ALL) -> float | None:
        """Fraction of ``(metric, stratum)`` taking *value*, or None if unknown.

        A value outside the stored top-K returns None rather than 0 — it is
        genuinely unknown how often it occurs, and the ``__other__`` row only
        gives the aggregate tail.
        """
        sub = self.categorical
        hit = sub[
            (sub["metric"] == metric)
            & (sub["stratum"] == stratum)
            & (sub["value"] == ("" if value is None else str(value)))
        ]
        return float(hit["frac"].iloc[0]) if len(hit) else None

    def criterion_rates(self, criterion: str, stratum: str = STRATUM_ALL) -> dict[str, Any] | None:
        """Tri-state as-run rates for one scored criterion, or None."""
        sub = self.criteria
        hit = sub[(sub["criterion"] == criterion) & (sub["stratum"] == stratum)]
        return hit.iloc[0].to_dict() if len(hit) else None

    def strata(self, metric: str) -> list[str]:
        """Strata that carry at least one non-null value for *metric*."""
        sub = self.numeric
        hit = sub[(sub["metric"] == metric) & (sub["n"] > 0)]
        return list(hit["stratum"])

    def metrics(self, category: str | None = None) -> list[str]:
        """Every profiled numeric metric, optionally filtered to one CDLMPS letter."""
        sub = self.numeric
        if category is not None:
            sub = sub[sub["category"] == category]
        return sorted(sub["metric"].unique())

    # ── Internals ─────────────────────────────────────────────────────────

    def _numeric_row(self, metric: str, stratum: str) -> dict[str, Any] | None:
        sub = self.numeric
        hit = sub[(sub["metric"] == metric) & (sub["stratum"] == stratum)]
        return hit.iloc[0].to_dict() if len(hit) else None


def version_dir(version: str = DEFAULT_VERSION, root: Path | None = None) -> Path:
    """Path to one distributions version directory."""
    base = (root / "data" / "reference" / "distributions") if root else REF_DIR
    return base / version


@lru_cache(maxsize=4)
def load(version: str = DEFAULT_VERSION, root: Path | None = None) -> Distributions:
    """Load a frozen distributions version.

    Cached, since the tables are read repeatedly during tag definition and site
    rendering and never change within a version.

    Raises:
        DistributionsError: The version directory or a required table is missing.
    """
    vdir = version_dir(version, root)
    if not vdir.is_dir():
        raise DistributionsError(
            f"no distributions version at {vdir}. Build it with:\n"
            f"  python scripts/setup/build_distributions.py --run full_catalog "
            f"--version {version}"
        )
    numeric_path = vdir / NUMERIC_FILE
    if not numeric_path.exists():
        raise DistributionsError(f"{numeric_path} is missing — the version is incomplete")

    def _opt(name: str, columns: list[str]) -> pd.DataFrame:
        path = vdir / name
        return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=columns)

    sidecar = vdir / SIDECAR_FILE
    return Distributions(
        version=version,
        numeric=pd.read_parquet(numeric_path),
        categorical=_opt(CATEGORICAL_FILE, ["metric", "stratum", "value", "count", "frac"]),
        criteria=_opt(CRITERIA_FILE, ["criterion", "stratum", "n_true", "n_false", "n_none"]),
        provenance=json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {},
    )
