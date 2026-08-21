"""Collapse a tagged TIS table to one mRNA per resolved initiation site.

The source-resolution cascade (:mod:`swissisoform.sourceresolve.resolve`) is
*tag-only*: it labels every row with a verdict but never drops anything, so the
per-sample filtered tables and the combined catalog stay full and auditable
(the divergence diagnostic re-runs off them).

This module is the *consumer* of that verdict. It runs at the assembly boundary
(``runner.prepare`` before ``assemble_genes``) and reduces the in-memory frame
to what should advance to annotation:

  - **all Annotated rows** are kept — canonical selection
    (``assembly._select_canonical`` / ``_build_canonical_by_tid``) needs them,
    and every resolved source transcript is guaranteed an Annotated row by
    upstream imputation;
  - each **resolved** initiation site keeps exactly the **source-transcript**
    row (one mRNA per TIS);
  - non-resolved sites (``no_support`` long-read drop-outs and ``unresolved``
    threshold-failing divergent sites) contribute no alternative-TIS rows — they
    do not advance, but they remain in the saved tables.

Collapse is **gated to rows a long-read sample actually scored** (production
default, ``keep_unevaluated=True``). In the combined catalog a TIS called only in
samples without long-read data (e.g. K562 / U2OS / RPE1 when only HeLa has
IsoQuant) has ``NaN`` in every ``{sample}_resolved`` column — it was never
evaluated by the cascade, so it **passes through unchanged** rather than being
dropped as a non-source candidate. This keeps the full cross-sample TIS set alive
for downstream ``min_cell_lines`` scoring during a single-long-read-sample phase.

Passing ``keep_unevaluated=False`` (the CLI ``--drop-unsupported-tis`` opt-in)
reverts to the pre-gate behavior: un-evaluated rows are dropped too, so only
Annotated rows and resolved source transcripts survive. That is **non-production**
— it exists for long-read-only timing tests where you want just the
long-read-supported TIS to advance.

When the verdict columns are absent (cascade skipped / never ran) the frame is
returned unchanged, so the collapse is a transparent no-op.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _is_true(value: object) -> bool:
    """Truthy check robust to bool / ``"True"`` strings / missing (CSV round-trips).

    Covers every missing marker, not just ``float('nan')``: the combined catalog
    stores these columns as ``object``/``None`` today, but a nullable-boolean
    column would yield ``pd.NA``, whose ``bool()`` raises rather than returning
    False — a crash on the step that decides the run's population.
    """
    if isinstance(value, str):
        return value.strip().lower() == "true"
    if value is None or pd.isna(value):
        return False
    return bool(value)


def resolution_columns(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Return ``(resolved_col, source_col)`` pairs present in *df*.

    Handles both the per-sample filtered shape (bare ``resolved`` /
    ``source_transcript``) and the combined-catalog shape
    (``{sample}_resolved`` / ``{sample}_source_transcript`` from
    :mod:`swissisoform.combine`). Empty when neither shape is present.
    """
    pairs: list[tuple[str, str]] = []
    if "resolved" in df.columns and "source_transcript" in df.columns:
        pairs.append(("resolved", "source_transcript"))
    for col in df.columns:
        if col.endswith("_source_transcript"):
            prefix = col[: -len("_source_transcript")]
            resolved_col = f"{prefix}_resolved"
            if resolved_col in df.columns:
                pairs.append((resolved_col, col))
    return pairs


def collapse_to_source(df: pd.DataFrame, *, keep_unevaluated: bool = True) -> pd.DataFrame:
    """Keep Annotated rows + one source-transcript row per resolved TIS.

    Args:
        df: A source-resolution-tagged TIS table (per-sample filtered frame or
            the combined catalog). Must carry ``Tid`` and ``TisType``.
        keep_unevaluated: When ``True`` (the production default), rows no
            long-read sample scored (``NaN`` in every ``{sample}_resolved``
            column) pass through unchanged — a TIS is never dropped merely
            because its cell line lacks long-read support. When ``False``, those
            un-evaluated rows are dropped too (the pre-gate behavior): only
            Annotated rows and resolved source transcripts survive, so a run
            keeps only long-read-supported TIS. Since only HeLa currently has
            IsoQuant data, that drops every TIS called solely in another cell
            line — about half the catalog, a coverage artifact that shrinks as
            long-read data lands for the other lines. Selected by
            ``--drop-unsupported-tis``, recorded in the run's ``population.json``.

    Returns:
        The collapsed frame (a copy). If the verdict columns are absent the
        input is returned unchanged.
    """
    pairs = resolution_columns(df)
    if not pairs:
        # Population-defining step quietly doing nothing is how a run ends up
        # advertising long-read filtering it never applied. Callers that
        # *require* the collapse should check _resolution_columns themselves.
        logger.warning(
            "collapse_to_source: no source-resolution verdict columns — skipping "
            "collapse (no-op). Rebuild the combined catalog (--rebuild-combined) "
            "if the long-read cascade was meant to have run."
        )
        return df
    if "TisType" not in df.columns or "Tid" not in df.columns:
        logger.warning(
            "collapse_to_source: missing TisType/Tid — skipping collapse (no-op)"
        )
        return df

    tid = df["Tid"].astype(str)
    annotated = df["TisType"].astype(str).str.startswith("Annotated")

    # A row is a source row if, for any resolved sample, that sample's chosen
    # source transcript equals the row's own Tid. The verdict is broadcast to
    # every row of an init_site, so the source transcript's own row carries
    # ``source_transcript == Tid``; non-source candidate rows carry a different
    # Tid and are dropped.
    is_source_row = pd.Series(False, index=df.index)
    evaluated = pd.Series(False, index=df.index)
    for resolved_col, source_col in pairs:
        resolved = df[resolved_col].map(_is_true)
        src = df[source_col].astype("string")
        is_source_row |= resolved & src.notna() & (tid == src)
        # A sample "evaluated" a row iff it wrote a verdict for it. In the
        # combined catalog a TIS called only in samples without long-read data
        # has NaN in every {sample}_resolved column — it was never evaluated, so
        # collapse must not drop it as a non-source candidate.
        evaluated |= df[resolved_col].notna()

    # Un-evaluated rows (no long-read sample scored them) pass through by default
    # so a TIS is never dropped just for lacking long-read support. With
    # keep_unevaluated=False they are dropped too — only Annotated rows and
    # resolved source transcripts survive (long-read-supported TIS only).
    keep = annotated | is_source_row
    if keep_unevaluated:
        keep = keep | ~evaluated
    out = df[keep].reset_index(drop=True)

    n_alt_in = int((~annotated).sum())
    n_alt_out = int((~out["TisType"].astype(str).str.startswith("Annotated")).sum())
    n_unevaluated = int((~annotated & ~is_source_row & ~evaluated).sum())
    logger.info(
        "collapse_to_source: %d rows → %d (alt TIS %d → %d; %d Annotated kept; "
        "%d un-evaluated rows %s)",
        len(df),
        len(out),
        n_alt_in,
        n_alt_out,
        int(annotated.sum()),
        n_unevaluated,
        "passed through" if keep_unevaluated else "dropped",
    )
    return out
