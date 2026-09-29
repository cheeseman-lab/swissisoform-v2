"""Fire a frozen tag registry against a run.

One entry point, :func:`fire`, producing two aligned frames: the tri-state per
tag, and the single number each fired tag rests on.

Tri-state is the point. ``True`` and ``False`` are both findings — "tested and
absent" is evidence — while ``NA`` means the tag could not be evaluated at all,
for either of two reasons the caller should not have to distinguish:

- **Declared**: the tag is undefined for this ORF type (``valid_for``). A uORF has
  no shared region, so a shared-region tag is undefined, not false.
- **Observed**: the metric is null for this row. A data hole.

Both surface as ``NA``. Collapsing either into ``False`` is the failure issue #30
names — a criterion that "cannot evaluate" reading as "evidence absent" is how M1
came to look negative on every extension.

Threshold and boolean tags evaluate vectorized over the run frame, so a cutoff
derived from the frozen distributions is applied to exactly the column it was
derived from. Derived tags call their criterion's scorer per site (see
:mod:`swissisoform.tags.derived`), which needs the site objects — hence
:func:`fire` taking both, aligned positionally.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import numpy as np
import pandas as pd

from swissisoform import metrics
from swissisoform.config import ScoringConfig
from swissisoform.models import TranslationInitiationSite
from swissisoform.tags import derived as derived_tags
from swissisoform.tags.registry import (
    KIND_BOOL,
    KIND_DERIVED,
    KIND_THRESHOLD,
    Tag,
    TagRegistry,
)

logger = logging.getLogger(__name__)

STATE_DTYPE = "boolean"  # pandas nullable bool: True / False / pd.NA


class TagEvaluationError(RuntimeError):
    """Raised when the registry and the run cannot be reconciled."""


def _validity_mask(df: pd.DataFrame, tag: Tag) -> np.ndarray:
    """True where *tag* is defined for the row's ORF type.

    An empty ``valid_for`` means "everywhere" rather than "nowhere": a registry row
    that forgot to declare validity should not silently blank the whole column.
    """
    if not tag.valid_for:
        return np.ones(len(df), dtype=bool)
    return df["orf_type"].astype("string").isin(tag.valid_for).to_numpy(dtype=bool)


def _threshold_state(df: pd.DataFrame, tag: Tag) -> tuple[pd.Series, pd.Series] | None:
    """``(state, citation)`` for a threshold tag, or None if unresolvable here."""
    values = metrics.resolve(tag.metric, df)
    if values is None or tag.cutoff is None:
        return None
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    with np.errstate(invalid="ignore"):
        hit = arr >= tag.cutoff if tag.direction == ">=" else arr < tag.cutoff
    state = pd.array(hit, dtype=STATE_DTYPE)
    state[np.isnan(arr)] = pd.NA
    return pd.Series(state, index=df.index), pd.Series(arr, index=df.index)


def _bool_state(df: pd.DataFrame, tag: Tag) -> tuple[pd.Series, pd.Series] | None:
    """``(state, citation)`` for a boolean-column tag.

    The citation is empty: a boolean has no number behind it, and inventing
    1.0 / 0.0 would read as one.
    """
    if tag.metric not in df.columns:
        return None
    state = (
        df[tag.metric]
        .astype("object")
        .map(lambda v: pd.NA if v is None or (isinstance(v, float) and pd.isna(v)) else bool(v))
    )
    return (
        pd.Series(pd.array(state.to_numpy(), dtype=STATE_DTYPE), index=df.index),
        pd.Series(np.full(len(df), np.nan), index=df.index),
    )


def _derived_state(
    df: pd.DataFrame,
    sites: list[TranslationInitiationSite],
    tag: Tag,
    cfg: ScoringConfig,
) -> tuple[pd.Series, pd.Series]:
    """``(state, citation)`` for a derived tag, by calling its criterion's scorer.

    The citation is the tag's headline metric where it has exactly one. Either-or
    criteria (M1 over two inputs, S2 over three) have no single number, so they
    get none rather than an arbitrary branch's — their ``reason`` string, which
    names whichever branch decided, already travels in ``isoform_scoring_reasons``.
    """
    values = [derived_tags.score_criterion(tag.criterion_id, site, cfg).value for site in sites]
    state = pd.array([pd.NA if v is None else bool(v) for v in values], dtype=STATE_DTYPE)
    citation = None if not tag.metric else metrics.resolve(tag.metric, df)
    return (
        pd.Series(state, index=df.index),
        pd.Series(np.full(len(df), np.nan), index=df.index)
        if citation is None
        else pd.to_numeric(citation, errors="coerce").astype("float64").set_axis(df.index),
    )


def effective_scoring(reg: TagRegistry, base: ScoringConfig) -> ScoringConfig:
    """*base* with every ``cutoff_overrides`` entry in *reg* applied.

    This is what makes a calibrated registry usable without loss: the criterion
    scorers keep their gates (status checks, ORF-type validity, either-or
    roll-ups) and only their numbers move. Building the config here rather than
    at the call site means a run can never fire tags at one set of cutoffs while
    believing it used another.

    **Overrides are partial, deliberately.** A criterion may consult several
    thresholds while the sweep only ever cuts its headline metric — P2 also reads
    ``p2_min_shared_len`` and ``p2_plddt_min``, P3 also reads ``p3_min_sse_plddt``.
    Those are *gates*, not the criterion's cutoff, and calibrating a gate against
    the distribution of the quantity it does not gate would be meaningless. They
    stay at their ``base`` values under every registry version.

    Values are coerced to the field's declared type, so an ``int`` threshold
    stored as a parquet double comes back an ``int``.

    Raises:
        TagEvaluationError: An override names a field ``ScoringConfig`` does not
            have — a renamed threshold, which would otherwise be silently ignored
            and leave the tag firing at the old default.
    """
    types = {f.name: f.type for f in dataclasses.fields(ScoringConfig)}
    overrides: dict[str, Any] = {}
    for tag in reg:
        for field_name, value in tag.cutoff_overrides.items():
            if field_name not in types:
                raise TagEvaluationError(
                    f"registry {reg.version} tag {tag.tag_id!r} overrides "
                    f"{field_name!r}, which is not a ScoringConfig field"
                )
            overrides[field_name] = int(value) if types[field_name] == "int" else float(value)
    if not overrides:
        return base
    return dataclasses.replace(base, **overrides)


def fire(
    df: pd.DataFrame,
    sites: list[TranslationInitiationSite],
    reg: TagRegistry,
    cfg: ScoringConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every code-fired tag in *reg* over *df*.

    Args:
        df: One row per TIS, as produced by ``paired_tis_dataframe`` — the same
            column names the registry's cutoffs were derived against.
        sites: The TIS objects for those rows, in row order. Required by derived
            tags; may be empty only if the registry has none.
        reg: The frozen registry.
        cfg: Base scoring thresholds. The registry's ``cutoff_overrides`` are
            applied on top (see :func:`effective_scoring`), so the numbers the
            scorers use are the frozen ones. Defaults to ``ScoringConfig()``.

    Returns:
        ``(states, citations)``. ``states`` is nullable-boolean, ``citations`` is
        float, both one column per code-fired tag in registry order and indexed
        like *df*.

    Raises:
        TagEvaluationError: ``sites`` does not align with *df*, or the run lacks
            ``orf_type``.
    """
    if "orf_type" not in df.columns:
        raise TagEvaluationError("run frame has no orf_type column; tags cannot be scoped")
    tags = reg.code_fired()
    needs_sites = any(t.kind == KIND_DERIVED for t in tags)
    if needs_sites and len(sites) != len(df):
        raise TagEvaluationError(
            f"{len(sites)} sites for {len(df)} rows — derived tags need the TIS objects "
            "in row order (build them as [s for g in genes for s in g.tis_sites])"
        )
    cfg = effective_scoring(reg, cfg or ScoringConfig())

    states: dict[str, pd.Series] = {}
    citations: dict[str, pd.Series] = {}
    unresolved: list[str] = []
    for tag in tags:
        if tag.kind == KIND_THRESHOLD:
            result = _threshold_state(df, tag)
        elif tag.kind == KIND_BOOL:
            result = _bool_state(df, tag)
        elif tag.kind == KIND_DERIVED:
            result = _derived_state(df, sites, tag, cfg)
        else:  # pragma: no cover - code_fired() filters llm and blocked
            continue
        if result is None:
            unresolved.append(tag.tag_id)
            # The column still exists, all-NA: a tag the run cannot evaluate is
            # not-evaluable, and dropping it would make the emitted struct's
            # fields depend on the run rather than on the registry version.
            states[tag.tag_id] = pd.Series(
                pd.array([pd.NA] * len(df), dtype=STATE_DTYPE), index=df.index
            )
            citations[tag.tag_id] = pd.Series(np.full(len(df), np.nan), index=df.index)
            continue
        state, citation = result
        valid = _validity_mask(df, tag)
        state = state.mask(~valid, pd.NA)
        citations[tag.tag_id] = citation.mask(~valid | state.isna())
        states[tag.tag_id] = state

    if unresolved:
        logger.warning(
            "%d tag(s) not evaluable against this run (missing metric or column), "
            "emitted as all-null: %s",
            len(unresolved),
            ", ".join(unresolved[:8]) + ("..." if len(unresolved) > 8 else ""),
        )
    order = list(reg.state_columns())
    return (
        pd.DataFrame(states, index=df.index)[order],
        pd.DataFrame(citations, index=df.index)[order],
    )


def to_structs(states: pd.DataFrame, citations: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Row-wise dicts, the shape ``_flatten_annotations`` turns into a parquet struct.

    ``None`` rather than ``pd.NA`` so pyarrow writes a real null instead of
    inferring an object column.
    """
    state_rows = [
        {c: (None if pd.isna(v) else bool(v)) for c, v in row.items()}
        for row in states.to_dict(orient="records")
    ]
    citation_rows = [
        {c: (None if pd.isna(v) else float(v)) for c, v in row.items()}
        for row in citations.to_dict(orient="records")
    ]
    return state_rows, citation_rows


__all__ = ["STATE_DTYPE", "TagEvaluationError", "effective_scoring", "fire", "to_structs"]
