"""Metric resolution — how a named quantity is computed from a paired-TIS frame.

Five of the sixteen scored criteria do not threshold a parquet column directly.
D1 counts cell lines, D2 takes a max across them, D3 counts validated peptides
under a strict two-flag test, P2's confidence gate takes the weaker of two pLDDT
means, and S3 takes the strongest of two signed activation shifts. The three S2
deltas are scored as magnitudes.

This module sits beside :mod:`swissisoform.distributions` rather than under
``tags/`` because it answers "what is a metric", which both layers need and
neither owns: :mod:`swissisoform.setup.distributions` profiles these to freeze
their distributions, and the tag layer resolves them at firing time. A quantity
computed one way for its cutoff and another way for its test would silently
mis-fire, so :func:`resolve` is the single point both go through.

Lifted from ``figures/scoring_cutoffs/plot_cutoff_distributions.py:63-138``, whose
``SERIES`` registry remains the reference for how each criterion is plotted.

Metric names are prefixed ``tx:`` so a derived quantity is never mistaken for a
column that exists in the parquet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from swissisoform.config import CELL_LINES

PREFIX = "tx:"

# Magnitude of a signed column: `abs:<column>`. A tag on a *_delta asks whether a
# property changed appreciably, which is |delta| over a bar — not sign(delta),
# which fires on the direction of arbitrarily small changes. The three S2 criteria
# already score magnitudes (`tx:abs_*_delta`); this generalises that to the sweep.
ABS_PREFIX = "abs:"

# Element count of a list column: `<column>__len`. The profiler synthesizes one
# per list column (`setup.distributions.list_lengths`) because a hit-list length
# is a real per-isoform quantity — how many variants, domains, peptides. It is
# a suffix rather than a prefix only because that is how the profiler already
# named it; both sides read this constant so the two cannot drift.
LEN_SUFFIX = "__len"

# Cell lines the expression columns are emitted for, in report order.
SAMPLES = CELL_LINES


def _num(name: str) -> Callable[[pd.DataFrame], pd.Series]:
    """Read one numeric column, coercing non-numeric entries to NaN."""
    return lambda df: pd.to_numeric(df[name], errors="coerce")


def _abs_num(name: str) -> Callable[[pd.DataFrame], pd.Series]:
    """Read one numeric column as a magnitude (S2 scores |delta|)."""
    return lambda df: pd.to_numeric(df[name], errors="coerce").abs()


def n_cell_lines(df: pd.DataFrame) -> pd.Series:
    """Count cell lines with a detection for this TIS (D1's ``len(site.expression)``)."""
    cols = [c for c in (f"expr_{s}_raw_count" for s in SAMPLES) if c in df.columns]
    if not cols:
        return pd.Series(float("nan"), index=df.index)
    return df[cols].notna().sum(axis=1).astype(float)


def max_initiation_efficiency(df: pd.DataFrame) -> pd.Series:
    """Best initiation efficiency across cell lines (D2 scores the max)."""
    cols = [c for c in (f"expr_{s}_initiation_efficiency" for s in SAMPLES) if c in df.columns]
    if not cols:
        return pd.Series(float("nan"), index=df.index)
    return df[cols].apply(pd.to_numeric, errors="coerce").max(axis=1)


def _summary_field(df: pd.DataFrame, summary: str, field: str) -> pd.Series | None:
    """One key of a summary struct, whether the frame is flattened or not.

    The profiling frame flattens every struct (``read_flat`` calls
    ``table.flatten()``) but the runtime frame keeps one level nested, so a
    metric named off a nested key has to accept both spellings or it resolves
    to None for the entire run while still profiling fine.
    """
    flat = df.get(f"{summary}.{field}")
    if flat is not None:
        return flat
    nested = df.get(summary)
    if nested is None:
        return None
    return nested.map(lambda v: v.get(field) if isinstance(v, dict) else None)


def n_validated_unique_peptides(df: pd.DataFrame) -> pd.Series:
    """Count isoform-unique PepQuery-validated peptides (D3).

    Mirrors ``evidence/d3_mass_spec``: strict ``is True`` on both flags, so an
    unknown (None) never counts. NaN when PepQuery did not run for the gene —
    that is D3's not-evaluable case, not a zero.

    The conjunction (unique AND validated) is not in ``massspec_summary``, which
    carries ``unique_peptides`` and ``validated_peptides`` separately, so this has
    to walk the hit list.
    """
    ran = _summary_field(df, "isoform_massspec_summary", "pepquery_run")
    hits_col = df.get("isoform_massspec_hits")
    if ran is None or hits_col is None:
        return pd.Series(float("nan"), index=df.index)
    out: list[float] = []
    for hits, pepquery_run in zip(hits_col, ran):
        if pepquery_run is not True:
            out.append(float("nan"))
        elif hits is None or len(hits) == 0:
            out.append(0.0)
        else:
            out.append(
                float(
                    sum(
                        1
                        for h in hits
                        if h.get("unique_to_isoform") is True and h.get("validated") is True
                    )
                )
            )
    return pd.Series(out, index=df.index)


def min_shared_plddt(df: pd.DataFrame) -> pd.Series:
    """The weaker of the two shared-region pLDDT means (P2's confidence gate).

    P2 is only meaningful when the shared region is confidently folded in BOTH
    structures, so the gate is the minimum, not either one alone — hence
    ``skipna=False``, which is also what makes one missing side read as unknown
    rather than as the other side's value. ``reindex`` keeps a frame that has
    only one of the two columns from raising: ``Transform.available`` admits it.
    """
    pair = df.reindex(
        columns=[
            "isoform_structure_plddt_shared_mean_isoform",
            "isoform_structure_plddt_shared_mean_canonical",
        ]
    ).apply(pd.to_numeric, errors="coerce")
    return pair.min(axis=1, skipna=False)


def sae_top_delta(df: pd.DataFrame) -> pd.Series:
    """The strongest shared-feature activation shift (S3), as a magnitude.

    ``reindex`` rather than ``df[[...]]``: ``Transform.available`` admits a frame
    carrying only one of the two columns, and the max of the one present is still
    the answer to "strongest shift".
    """
    pair = df.reindex(
        columns=["isoform_sae_top_gained_delta_max", "isoform_sae_top_lost_delta_max"]
    ).apply(pd.to_numeric, errors="coerce")
    return pair.abs().max(axis=1)


@dataclass(frozen=True)
class Transform:
    """One derived metric, with the metadata a profiled metric needs.

    Attributes:
        name: Short id; the profiled metric is ``tx:<name>``.
        fn: ``(df) -> Series`` over a flattened paired-TIS frame.
        category: CDLMPS letter, matching the catalog's ``category``.
        label: Human-readable name for the review table.
        requires: Flat columns the transform reads. A transform whose columns are
            all absent is skipped rather than producing an all-NaN metric.
        requires_lists: List columns it needs. These are excluded from the default
            flattened read (the hit lists dominate the file), so a profiler must
            fetch them explicitly.
    """

    name: str
    fn: Callable[[pd.DataFrame], pd.Series]
    category: str
    label: str
    requires: tuple[str, ...] = field(default=())
    requires_lists: tuple[str, ...] = field(default=())

    @property
    def metric(self) -> str:
        """The profiled metric name."""
        return f"{PREFIX}{self.name}"

    def available(self, columns: set[str]) -> bool:
        """True when at least one required column is present.

        Deliberately ``any``: the per-sample transforms are defined over whatever
        subset of cell lines a run carries. The obligation that buys is on ``fn``,
        which must be total over any frame this admits — so a transform reads its
        columns through ``reindex``/``get`` and returns NaN, never ``KeyError``.
        """
        return not self.requires or any(c in columns for c in self.requires)


TRANSFORMS: tuple[Transform, ...] = (
    Transform(
        "n_cell_lines", n_cell_lines, "D", "Cell lines detecting the TIS",
        tuple(f"expr_{s}_raw_count" for s in SAMPLES),
    ),
    Transform(
        "max_initiation_efficiency", max_initiation_efficiency, "D",
        "Best initiation efficiency across cell lines",
        tuple(f"expr_{s}_initiation_efficiency" for s in SAMPLES),
    ),
    Transform(
        "n_validated_unique_peptides", n_validated_unique_peptides, "D",
        "Validated isoform-unique peptides",
        # Both spellings: flattened in the profiling frame, nested at runtime.
        ("isoform_massspec_summary.pepquery_run", "isoform_massspec_summary"),
        requires_lists=("isoform_massspec_hits",),
    ),
    Transform(
        "min_shared_plddt", min_shared_plddt, "P", "Weaker shared-region pLDDT mean",
        ("isoform_structure_plddt_shared_mean_isoform",
         "isoform_structure_plddt_shared_mean_canonical"),
    ),
    Transform(
        "sae_top_delta", sae_top_delta, "S", "Strongest shared-feature activation shift",
        ("isoform_sae_top_gained_delta_max", "isoform_sae_top_lost_delta_max"),
    ),
    Transform(
        "abs_gravy_delta", _abs_num("cmp_biophysics_gravy_delta"), "S",
        "|GRAVY delta| (isoform vs canonical)", ("cmp_biophysics_gravy_delta",),
    ),
    Transform(
        "abs_fraction_charged_delta", _abs_num("cmp_biophysics_fraction_charged_delta"), "S",
        "|fraction-charged delta|", ("cmp_biophysics_fraction_charged_delta",),
    ),
    Transform(
        "abs_disorder_delta", _abs_num("cmp_biophysics_disorder_delta"), "S",
        "|disorder delta|", ("cmp_biophysics_disorder_delta",),
    ),
)

BY_NAME: dict[str, Transform] = {t.name: t for t in TRANSFORMS}
BY_METRIC: dict[str, Transform] = {t.metric: t for t in TRANSFORMS}


def is_magnitude(metric: str) -> bool:
    """True for a metric that is a magnitude, so its sign carries no information.

    Two families, both matched on where the name *starts*: the generic
    ``abs:<column>``, and the three S2 transforms named ``tx:abs_*_delta``.
    Matching ``"abs_"`` anywhere in the name also swallowed the real signed
    column ``isoform_sae_mean_abs_delta_shared`` — which ``candidates.propose``
    then skipped before any ``funnel.drop``, dropping an S metric from the
    vocabulary with no record of it in the review table.
    """
    if metric.startswith(ABS_PREFIX):
        return True
    return metric.startswith(PREFIX) and metric[len(PREFIX) :].startswith("abs_")


def magnitude_of(column: str) -> str:
    """Metric name for the magnitude of a signed column."""
    return f"{ABS_PREFIX}{column}"


def resolve(metric: str, df: pd.DataFrame) -> pd.Series | None:
    """Return the values for *metric*, whether it is a raw column or a transform.

    The single resolution point shared by the profiler and the tag evaluator, so
    a derived quantity cannot be computed one way for its cutoff and another way
    for its test.
    """
    if metric.startswith(ABS_PREFIX):
        column = metric[len(ABS_PREFIX):]
        if column not in df.columns:
            return None
        return pd.to_numeric(df[column], errors="coerce").abs()
    if metric.endswith(LEN_SUFFIX) and metric not in df.columns:
        # Only the profiler materialises `<col>__len` as a column; the runtime
        # frame carries the list itself, so a tag cut on a list length has to be
        # counted here or it resolves to None for the whole run.
        column = metric[: -len(LEN_SUFFIX)]
        if column not in df.columns:
            return None
        return df[column].map(lambda v: float("nan") if v is None else float(len(v)))
    tx = BY_METRIC.get(metric)
    if tx is not None:
        return tx.fn(df) if tx.available(set(df.columns)) else None
    if metric in df.columns:
        return pd.to_numeric(df[metric], errors="coerce")
    return None


def resolvable(metric: str, columns: set[str]) -> bool:
    """Whether :func:`resolve` could produce values for *metric* from *columns*.

    Name-level only, so a builder can reject a tag whose metric no run can see
    without reading any data. This is the guard that would have caught
    ``cmp_motifs_hits_in_diff_region__len`` reaching a frozen registry: the
    profiler invents `__len` columns, so the sweep cut a real cutoff against a
    name that exists nowhere at firing time.
    """
    if metric.startswith(ABS_PREFIX):
        return metric[len(ABS_PREFIX):] in columns
    if metric in columns:
        return True
    if metric.endswith(LEN_SUFFIX):
        return metric[: -len(LEN_SUFFIX)] in columns
    tx = BY_METRIC.get(metric)
    return tx is not None and tx.available(columns)


def available(df_columns: set[str]) -> tuple[Transform, ...]:
    """Transforms whose source columns are present in a run."""
    return tuple(t for t in TRANSFORMS if t.available(df_columns))


def required_list_columns() -> tuple[str, ...]:
    """List columns some transform needs, which a flattened read would omit."""
    return tuple(sorted({c for t in TRANSFORMS for c in t.requires_lists}))


__all__: list[str] = [
    "PREFIX",
    "ABS_PREFIX",
    "is_magnitude",
    "magnitude_of",
    "SAMPLES",
    "Transform",
    "TRANSFORMS",
    "BY_NAME",
    "BY_METRIC",
    "resolve",
    "available",
    "required_list_columns",
]
