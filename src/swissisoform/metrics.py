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

Metric names are prefixed ``tx:`` so a derived quantity is never mistaken for a
column that exists in the parquet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
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

# Unique-vs-shared region ratios (`cmp_biophysics_<prop>_ratio`). ``ratio >= c``
# only means "unique exceeds shared" when the shared denominator is positive: for
# a property that crosses zero (GRAVY, Top-IDP disorder, instability index) a
# negative denominator flips the inequality, so unique -0.6 / shared -0.3 reads
# 2.0 — "more hydrophobic" for a region that is more hydrophilic. The comparator
# already refuses ``enriched`` in that case (compare/paired.py); :func:`resolve`
# applies the same guard to the ratio itself.
RATIO_SUFFIX = "_ratio"
SHARED_SUFFIX = "_shared"
UNIQUE_SUFFIX = "_unique"

# Properties whose region ratio is not sign-safe, carried instead as the signed
# difference ``unique - shared`` (`tx:<prop>_unique_minus_shared`). Zero is then
# a real null and the sign is the direction.
SIGNED_REGION_PROPERTIES: tuple[str, ...] = ("gravy", "disorder", "instability_index")

# Cell lines the expression columns are emitted for, in report order.
SAMPLES = CELL_LINES


def _num(name: str) -> Callable[[pd.DataFrame], pd.Series]:
    """Read one numeric column, coercing non-numeric entries to NaN."""
    return lambda df: pd.to_numeric(df[name], errors="coerce")


def _abs_num(name: str) -> Callable[[pd.DataFrame], pd.Series]:
    """Read one numeric column as a magnitude (S2 scores |delta|)."""
    return lambda df: pd.to_numeric(df[name], errors="coerce").abs()


def _region_difference(prop: str) -> Callable[[pd.DataFrame], pd.Series]:
    """``unique - shared`` for one biophysical property, NaN where either is missing."""

    def fn(df: pd.DataFrame) -> pd.Series:
        stem = f"cmp_biophysics_{prop}"
        pair = df.reindex(columns=[f"{stem}{UNIQUE_SUFFIX}", f"{stem}{SHARED_SUFFIX}"]).apply(
            pd.to_numeric, errors="coerce"
        )
        return pair.iloc[:, 0] - pair.iloc[:, 1]

    return fn


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
    *(
        Transform(
            f"{prop}_unique_minus_shared", _region_difference(prop), "S",
            f"{prop} (unique region minus shared region)",
            (f"cmp_biophysics_{prop}{UNIQUE_SUFFIX}", f"cmp_biophysics_{prop}{SHARED_SUFFIX}"),
        )
        for prop in SIGNED_REGION_PROPERTIES
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
        # Anything but a sequence is a missing list. In the runtime frame that is
        # NaN, not None (a row whose module the comparator skipped), and
        # `len(nan)` raised — which `_attach_tags` swallows by dropping every tag.
        return df[column].map(
            lambda v: float(len(v)) if isinstance(v, (list, tuple, np.ndarray)) else float("nan")
        )
    tx = BY_METRIC.get(metric)
    if tx is not None:
        return tx.fn(df) if tx.available(set(df.columns)) else None
    if metric in df.columns:
        values = pd.to_numeric(df[metric], errors="coerce")
        shared = f"{metric[: -len(RATIO_SUFFIX)]}{SHARED_SUFFIX}"
        if metric.endswith(RATIO_SUFFIX) and shared in df.columns:
            # Not-evaluable rather than inverted: see RATIO_SUFFIX.
            values = values.where(pd.to_numeric(df[shared], errors="coerce") > 0)
        return values
    return None


CHANGED_SUFFIX = "_changed"


def _missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value))


def changed_state(metric: str, df: pd.DataFrame) -> pd.Series | None:
    """Tri-state ``cmp_<module>_<field>_changed``, reading a one-sided None as a change.

    The comparator emits ``None`` whenever exactly one side is missing
    (compare/comparator.py ``_categorical_changes``), on the theory that the
    module may have failed on one pane. For a categorical *call* that is usually
    wrong: SignalP reports no cleavage site because the protein has no signal
    peptide, so canonical ``None`` / isoform ``"CS pos: 23-24"`` is a signal
    peptide gained, not an unknown. On cheeseman50 the cleavage-site tag fired on
    0 of 48 evaluable rows while the prediction tag fired on the 2 isoforms that
    gained one.

    This re-derives the flag from the ``_canonical`` / ``_isoform`` columns the
    comparator writes beside it:

    - When the module's own ``<tool>_prediction`` pair is present, it says whether
      the predictor ran on each side. Ran on both: ``None`` is "absent", so one
      side present is ``True`` and neither is ``False``. Did not run on a side:
      not-evaluable.
    - Without that pair, ``True`` when exactly one side is present, not-evaluable
      when both are missing.

    Returns None when the frame lacks the value columns, so the caller can fall
    back to the comparator's flag.
    """
    if not metric.endswith(CHANGED_SUFFIX):
        return None
    base = metric[: -len(CHANGED_SUFFIX)]
    can_col, iso_col = f"{base}_canonical", f"{base}_isoform"
    if can_col not in df.columns or iso_col not in df.columns:
        return None
    parts = base.split("_", 2)
    ran_cols = None
    if len(parts) == 3:
        ran_base = f"{parts[0]}_{parts[1]}_{parts[2].split('_', 1)[0]}_prediction"
        if f"{ran_base}_canonical" in df.columns and f"{ran_base}_isoform" in df.columns:
            ran_cols = (f"{ran_base}_canonical", f"{ran_base}_isoform")

    out: list[bool | None] = []
    for i in range(len(df)):
        can, iso = df[can_col].iloc[i], df[iso_col].iloc[i]
        can_absent, iso_absent = _missing(can), _missing(iso)
        if ran_cols is not None:
            if _missing(df[ran_cols[0]].iloc[i]) or _missing(df[ran_cols[1]].iloc[i]):
                out.append(None)
                continue
        elif can_absent and iso_absent:
            out.append(None)
            continue
        if can_absent or iso_absent:
            out.append(can_absent != iso_absent)
        else:
            out.append(bool(can != iso))
    return pd.Series(out, index=df.index, dtype="object")


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
    "changed_state",
    "available",
    "required_list_columns",
]
