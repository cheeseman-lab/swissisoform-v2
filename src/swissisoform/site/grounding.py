"""Alternative groundings for the category LLM slice.

Four ways to tell a model what an isoform's evidence *is*, so the question can be
measured rather than argued:

``criteria``
    Today's framing — each scored criterion's ``value`` / ``reason`` plus the
    curated ``evidence_cols`` behind it. Built by :mod:`swissisoform.site.evidence`
    itself, so this module returns ``None`` for it and no hook is installed.
``raw``
    No verdicts at all: every column the feature catalog assigns to the category.
``tags``
    No verdicts: the fired tags off ``isoform_tags_states``, each with the one
    number it rests on — and, where a tag restates a scored criterion, that
    criterion's full ``evidence_cols`` and its interpretation hint. Plus the
    judgment tags the M/P tool loops answer.
``dist``
    No verdicts and no tags: every profiled numeric field with its value and its
    percentile in the frozen reference population.

``tags`` and ``dist`` are deliberately separate. Bundled — tags *plus* percentile
context, as first sketched — a win would be uninterpretable, because population
context is information no other grounding has in any form.

**This module is unstaged on purpose.** ``site/evidence.py`` ships to the website
deploy with only ``config.py`` for company (``website/prepare_deploy.sh:43-49``),
so it cannot import pandas-heavy machinery. It stores a *callable* instead, and
everything that needs the distributions, the tag registry or the feature catalog
lives here and is injected through ``evidence.use_category_body``.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from swissisoform import distributions as dist_mod
from swissisoform.site import evidence as ev
from swissisoform.tags import registry as reg_mod
from swissisoform.tags import seeds

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG = ROOT / "figures" / "clustering_dims" / "feature_space" / "feature_catalog.csv"
DEFAULT_DIST_VERSION = "v3"
DEFAULT_TAG_VERSION = "v1"

GROUNDINGS: tuple[str, ...] = ("criteria", "raw", "tags", "dist")

# Rows kept from a list column before it is truncated. Matches the cap
# ``slice_criterion`` already applies to hit lists, so no grounding shows more
# evidence rows than the criteria framing does.
MAX_LIST_ROWS = 30

# Columns a grounding must never carry, whatever the catalog says. The scoring
# family is the verdict this experiment is measuring the model *without*; the tag
# family is the verdict a different arm supplies. `cmp_biophysics_*_enriched` is a
# judgement call: it is `ratio > 1.0` hardcoded in the comparator, i.e. a threshold
# output wearing a boolean's clothes, so it leaks a cutoff into the arms that are
# supposed to have none.
LEAK_PREFIXES: tuple[str, ...] = ("isoform_scoring_", "isoform_tags_", "canonical_scoring_")
LEAK_SUFFIXES: tuple[str, ...] = ("_enriched",)

# Categories whose row-level data reaches the model through tools rather than the
# payload. Mirrors ``llm.STRIP_HITS_FOR_TOOLS``, and for the same reason: the
# criteria path strips M's hit rows to 5.8k chars because ``query_variants`` reads
# them instead. Measured on cheeseman50, M's raw body is **100% hit lists** —
# 123,167 of 123,676 chars, against 509 chars of scalars — so leaving them in
# would hand the raw arm a 20x larger opening context than the criteria arm on the
# one category that re-sends it every turn. That is a payload difference, not a
# grounding difference, and it would dominate M's comparison.
STRIP_LISTS_FOR: frozenset[str] = frozenset({"M"})


class GroundingError(RuntimeError):
    """Raised when a grounding cannot be built for the records in hand."""


def _leaks(column: str) -> bool:
    """Whether *column* would hand a grounding a verdict it is meant to lack."""
    return column.startswith(LEAK_PREFIXES) or column.endswith(LEAK_SUFFIXES)


def _scrub(value: Any) -> Any:
    """Drop leaking keys at every depth of a nested evidence block.

    ``_leaks`` is name-based, so it cannot see a leak nested under a display
    label — and S2's builder emits exactly that, a ``cmp_biophysics_<f>_enriched``
    per feature keyed by "Hydropathy (GRAVY)".
    """
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items() if not _leaks(str(k))}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value


def _is_list(value: Any) -> bool:
    """Whether *value* is a row list — pandas hands these back as ndarrays."""
    return isinstance(value, (list, tuple)) or hasattr(value, "tolist")


def _clean(value: Any) -> Any:
    """JSON-safe scalar: NaN and pandas NA become None."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


# ---------------------------------------------------------------------------
# Column universe
# ---------------------------------------------------------------------------


def category_columns(catalog: pd.DataFrame) -> dict[str, list[str]]:
    """``{letter: [column, ...]}`` — the feature catalog's CDLMPS assignment.

    The canonical pane is excluded because it describes the gene, not the
    isoform's change. Nothing stricter: the differential metrics live on the
    ``isoform_`` pane, so a tighter pane filter starves C and P.
    """
    sel = catalog[(catalog["category"].isin(list("CDLMPS"))) & (catalog["pane"] != "canonical")]
    out: dict[str, list[str]] = {}
    for letter, sub in sel.groupby("category"):
        out[str(letter)] = [c for c in sub["feature"].astype(str) if not _leaks(c)]
    return out


def numeric_category_columns(
    catalog: pd.DataFrame, dist: dist_mod.Distributions
) -> dict[str, list[str]]:
    """``{letter: [metric, ...]}`` — the subset that has a frozen distribution.

    A percentile is only meaningful for a metric the reference population
    profiled, so ``dist`` is scoped to the intersection rather than reporting
    ``None`` for the rest.
    """
    profiled = set(dist.numeric.loc[dist.numeric["stratum"] == dist_mod.STRATUM_ALL, "metric"])
    sel = catalog[
        (catalog["category"].isin(list("CDLMPS")))
        & (catalog["pane"] != "canonical")
        & (catalog["dtype"].isin(["float", "int"]))
    ]
    out: dict[str, list[str]] = {}
    for letter, sub in sel.groupby("category"):
        out[str(letter)] = [
            c for c in sub["feature"].astype(str) if c in profiled and not _leaks(c)
        ]
    return out


# ---------------------------------------------------------------------------
# The three alternative bodies
# ---------------------------------------------------------------------------


def _raw_body(
    columns: dict[str, list[str]],
    strip_lists_for: frozenset[str] = STRIP_LISTS_FOR,
) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    """Every catalog column for the category, no verdicts.

    *Not* the "more information" arm it sounds like: measured per category, this
    body is **smaller** than the criteria slice for C, L and P and enormous for M
    and S. The list cap is therefore part of the arm's definition, not an
    implementation detail, so the payload states it.
    """

    def build(record: dict[str, Any], category: dict[str, Any]) -> dict[str, Any]:
        raw = record.get("_raw") or {}
        letter = category["letter"]
        stripped = letter in strip_lists_for
        evidence: dict[str, Any] = {}
        dropped: dict[str, int] = {}
        row_counts: dict[str, int] = {}
        for column in columns.get(letter, []):
            if column not in raw:
                continue
            value = raw[column]
            if _is_list(value):
                rows = list(value)
                if stripped:
                    row_counts[column] = len(rows)
                    continue
                kept, n_dropped = rows[:MAX_LIST_ROWS], max(0, len(rows) - MAX_LIST_ROWS)
                evidence[column] = kept
                if n_dropped:
                    dropped[column] = n_dropped
                continue
            evidence[column] = _clean(value)
        body: dict[str, Any] = {"evidence": evidence}
        if row_counts:
            body["hits_note"] = {
                "n_rows": row_counts,
                "note": (
                    "Row-level data is not in this payload — read it with the tools. "
                    "The counts above are exact."
                ),
            }
        if dropped:
            body["truncated"] = {
                "rows_kept_per_list": MAX_LIST_ROWS,
                "rows_dropped": dropped,
                "note": "Long hit lists are a sample, not the full set; counts above are exact.",
            }
        return body

    return build


def _tags_body(
    reg: reg_mod.TagRegistry,
    *,
    hints: bool = True,
) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    """Fired tags with their cited number, plus the judgment tags for M and P.

    Every *evaluated* tag is carried, not only the fired ones. ``off`` is evidence
    — "tested and absent" — and ``not_evaluable`` is the distinction the whole tag
    layer exists to preserve; dropping either would leave the model unable to tell
    a negative result from an untestable one.

    A tag that restates a scored criterion additionally carries that criterion's
    ``evidence_cols`` — the same numbers the ``criteria`` arm sees — because the
    single cited value is one input to a verdict that may rest on nine. Sweep
    tags stay lean: their value *is* their metric, so there is nothing withheld.

    ``hints`` gates the per-tag ``means`` only. ``interpretation_hint`` is the
    hint axis itself (``variants.py`` documents ``-hint`` as stripping it from
    ``criteria``), so carrying it unconditionally would hand ``tags_nohint`` the
    guidance the axis exists to remove.

    Raises:
        GroundingError: A tag names a ``criterion_id`` absent from ``CRITERIA``
            — registry/criteria drift, which must fail at arm setup rather than
            silently degrade to a lean payload partway through a paid run.
    """
    by_category: dict[str, list[reg_mod.Tag]] = {}
    for tag in reg:
        by_category.setdefault(tag.category, []).append(tag)

    # Resolved once, not per record: `TagRegistry.get` rebuilds its index on
    # every call, and CRITERIA is a module-level dict either way.
    criterion_cfg: dict[str, dict[str, Any]] = {}
    for tag in reg:
        if not tag.criterion_id:
            continue
        cfg = ev.CRITERIA.get(tag.criterion_id)
        if cfg is None:
            raise GroundingError(
                f"tag {tag.tag_id!r} names criterion {tag.criterion_id!r}, which "
                "site.evidence.CRITERIA does not define"
            )
        criterion_cfg[tag.tag_id] = cfg

    def metrics_for(tag: reg_mod.Tag, record: dict[str, Any]) -> dict[str, Any] | None:
        """The criterion's supporting numbers, or None when it has none here."""
        cfg = criterion_cfg.get(tag.tag_id)
        if cfg is None:
            return None
        builder = cfg.get("evidence_builder")
        if builder is not None:
            # S2/S3 express their evidence as a nested dict no flat column list
            # can hold, so the criteria arm builds it on the fly; do the same
            # rather than reporting them as having no metrics.
            built = builder(record) or {}
            return _scrub(built.get("evidence")) or None
        raw = record.get("_raw") or {}
        return {col: _clean(raw.get(col)) for col in cfg.get("evidence_cols", ())} or None

    def build(record: dict[str, Any], category: dict[str, Any]) -> dict[str, Any]:
        raw = record.get("_raw") or {}
        states = raw.get("isoform_tags_states") or {}
        citations = raw.get("isoform_tags_citations") or {}
        if not states:
            raise GroundingError(
                "record carries no isoform_tags_states — the run predates the tag "
                "layer, or was built with --skip-modules tags"
            )
        letter = category["letter"]
        fired: list[dict[str, Any]] = []
        questions: list[dict[str, Any]] = []
        for tag in by_category.get(letter, []):
            if tag.kind == reg_mod.KIND_LLM:
                questions.append(_open_question(tag, record))
                continue
            if tag.tag_id not in states:
                continue
            state = states[tag.tag_id]
            entry: dict[str, Any] = {
                "tag_id": tag.tag_id,
                "label": tag.label,
                "state": "on" if state is True else "off" if state is False else "not_evaluable",
            }
            citation = _clean(citations.get(tag.tag_id))
            if citation is not None:
                entry["value"] = citation
            if tag.cutoff is not None:
                entry["cutoff"] = tag.cutoff
            # `Tag.test` renders a derived tag as "derived predicate for X", which
            # reads as a non-sequitur beside a cutoff. Where the tag has a single
            # metric — true of every threshold tag and of the derived criteria that
            # are not either-or roll-ups — state the comparison it actually makes.
            if tag.metric and tag.direction and tag.cutoff is not None:
                entry["test"] = f"{tag.metric} {tag.direction} {tag.cutoff:.6g}"
            if tag.note:
                entry["note"] = tag.note
            if tag.criterion_id:
                entry["criterion_id"] = tag.criterion_id
                metrics = metrics_for(tag, record)
                if metrics:
                    entry["metrics"] = metrics
                if hints:
                    entry["means"] = criterion_cfg[tag.tag_id]["interpretation_hint"]
            fired.append(entry)
        body: dict[str, Any] = {"tags": fired, "tags_registry_version": reg.version}
        if questions:
            body["open_questions"] = questions
        return body

    return build


def _open_question(tag: reg_mod.Tag, record: dict[str, Any]) -> dict[str, Any]:
    """One judgment tag, posed rather than answered.

    ``unanswered`` is not ``off``. An LLM tag is *undetermined* until the loop
    reads the data, and rendering it as off would be the same
    absence-reads-as-negative failure the tri-state exists to prevent.
    """
    seed = next((s for s in seeds.LLM_SEEDS if s.tag_id == tag.tag_id), None)
    orf_type = str(record.get("orf_type") or "")
    evaluable = not tag.valid_for or orf_type in tag.valid_for
    entry: dict[str, Any] = {
        "tag_id": tag.tag_id,
        "label": tag.label,
        "state": "unanswered" if evaluable else "not_evaluable",
    }
    if seed is not None:
        entry["question"] = seed.question
        entry["reader"] = seed.reader
        entry["citation"] = seed.citation
    if not evaluable:
        entry["why"] = f"defined only for {'/'.join(tag.valid_for)}; this is {orf_type}"
    return entry


def _dist_body(
    metrics_by_category: dict[str, list[str]], dist: dist_mod.Distributions
) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    """Every profiled numeric field with its value and its population percentile.

    Two traps in the distributions read-side, both handled here:

    - ``percentile`` deliberately does **not** enforce ``MIN_STRATUM_N``
      (``distributions.py:145-147``), so it will happily return an
      authoritative-looking rank off n=3. The stratum is checked and falls back to
      ``all``, and the payload records which one it quoted.
    - ``_numeric_row`` is an O(n) mask scan over ~4,900 rows per call, and this
      body asks for up to 157 fields per isoform per category. The summaries are
      resolved once, here, into a dict.
    """
    summaries: dict[tuple[str, str], dict[str, Any] | None] = {}

    def summary(metric: str, stratum: str) -> dict[str, Any] | None:
        key = (metric, stratum)
        if key not in summaries:
            summaries[key] = dist.summary(metric, stratum)
        return summaries[key]

    def build(record: dict[str, Any], category: dict[str, Any]) -> dict[str, Any]:
        raw = record.get("_raw") or {}
        wanted = dist_mod.stratum_for(record.get("orf_type"))
        fields: dict[str, Any] = {}
        for metric in metrics_by_category.get(category["letter"], []):
            value = _clean(raw.get(metric))
            if value is None:
                continue
            stratum = wanted
            info = summary(metric, stratum)
            if info is None or (info.get("n") or 0) < dist_mod.MIN_STRATUM_N:
                stratum = dist_mod.STRATUM_ALL
                info = summary(metric, stratum)
            if info is None:
                continue
            entry: dict[str, Any] = {
                "value": value,
                "pctile": dist.percentile(metric, float(value), stratum),
                "stratum": stratum,
                "n": info.get("n"),
            }
            for point in ("p05", "p25", "p50", "p75", "p95"):
                entry[point] = info.get(point)
            fields[metric] = entry
        return {
            "fields": fields,
            "reference_population": {
                "version": dist.version,
                "source_run": (dist.provenance.get("source_run") or "unknown"),
                "n_isoforms": dist.provenance.get("n_isoforms"),
                "note": (
                    "Percentiles are against the frozen genome-wide population, not "
                    "against the isoforms in this corpus."
                ),
            },
        }

    return build


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build(
    mode: str,
    *,
    catalog_csv: Path = DEFAULT_CATALOG,
    dist_version: str = DEFAULT_DIST_VERSION,
    tag_version: str = DEFAULT_TAG_VERSION,
    strip_lists_for: frozenset[str] = STRIP_LISTS_FOR,
    hints: bool = True,
) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None:
    """The body builder for one grounding, or ``None`` for ``criteria``.

    ``None`` is not a failure: it means "install no hook", so the status-quo arm
    runs the untouched code path rather than a reimplementation of it.

    ``hints`` reaches only ``tags``, whose criterion-backed entries carry an
    ``interpretation_hint``. The other two alternative bodies have no per-member
    hint to strip, which is why ``strip_hints`` is a ``criteria``-only path.

    Raises:
        GroundingError: Unknown mode.
    """
    if mode not in GROUNDINGS:
        raise GroundingError(f"unknown grounding {mode!r}; expected one of {GROUNDINGS}")
    if mode == "criteria":
        return None
    if mode == "tags":
        return _tags_body(reg_mod.load(tag_version), hints=hints)
    catalog = pd.read_csv(catalog_csv)
    if mode == "raw":
        return _raw_body(category_columns(catalog), strip_lists_for)
    dist = dist_mod.load(dist_version)
    return _dist_body(numeric_category_columns(catalog, dist), dist)


def strip_hints(record: dict[str, Any]) -> dict[str, Any]:
    """Remove every ``interpretation_hint`` from a built category slice.

    Applied to the ``criteria`` grounding only — the other three carry no
    per-member hints, which is itself a reason the hint axis is not orthogonal to
    the grounding axis and has to be reported that way.
    """
    out = dict(record)
    members = out.get("members")
    if isinstance(members, list):
        out["members"] = [
            {k: v for k, v in m.items() if k != "interpretation_hint"} for m in members
        ]
    return out


def llm_tag_ids(reg: reg_mod.TagRegistry, letter: str) -> list[str]:
    """The judgment tag ids for one category — the enum for ``tags_fired``."""
    return [t.tag_id for t in reg.by_category(letter) if t.kind == reg_mod.KIND_LLM]


def verdict_extra_fields(reg: reg_mod.TagRegistry, letter: str) -> dict[str, Any]:
    """The ``tags_fired`` property to splice into a tool-loop terminal schema.

    Empty when the category has no judgment tags, so a caller can splice
    unconditionally.
    """
    ids = llm_tag_ids(reg, letter)
    if not ids:
        return {}
    return {
        "tags_fired": {
            "type": "array",
            "items": {"type": "string", "enum": ids},
            "description": (
                "Which of the judgment tags in `open_questions` fired. Include a tag "
                "id only if the data you read supports it, and cite that tag's number "
                "in `reasoning`. Omit the field or send [] if none fired."
            ),
        }
    }


def install_verdict_extras(
    tools: list[dict[str, Any]], extras: dict[str, Any], *, required: bool = True
) -> Callable[[], None]:
    """Splice extra properties into a tool list's ``emit_verdict``; return a restore fn.

    The terminal tool is declared ``"strict": True`` with
    ``"additionalProperties": False``, which constrains sampling to the schema —
    so without this the model *cannot* emit a tag decision at all, and the
    judgment tags would be answerable only as prose. ``_tool_schemas``
    (``llm.py:1878``) hands back ``M_TOOLS`` / ``P_TOOLS`` by reference, so
    replacing the entry in place is picked up with no change to ``llm.py``.

    The extras **are** added to ``required`` by default, and that is the point.
    Left optional, 3 of 4 tool loops in the smoke test omitted the field entirely
    rather than sending ``[]`` — which makes "judged, nothing fired"
    indistinguishable from "never answered the question", across every M and P
    record in the run. Requiring it forces an *answer*, not a firing: ``[]`` is the
    answer when nothing fired. The shared ``category_read.json`` keeps it optional,
    so the single-shot categories are unaffected and a tool-loop payload still
    validates against it.

    Returns a callable that puts the original entry back. Call it in a ``finally``
    — these are module-level constants, and leaving one patched would silently
    change the next arm.
    """
    if not extras:
        return lambda: None
    index = next((i for i, t in enumerate(tools) if t.get("name") == "emit_verdict"), None)
    if index is None:  # pragma: no cover - both tool lists define one
        raise GroundingError("tool list has no emit_verdict entry to extend")
    original = tools[index]
    patched = copy.deepcopy(original)
    patched["input_schema"]["properties"].update(extras)
    if required:
        req = patched["input_schema"].setdefault("required", [])
        req.extend(k for k in extras if k not in req)
    tools[index] = patched

    def restore() -> None:
        tools[index] = original

    return restore


def dump(record: dict[str, Any]) -> str:
    """Stable JSON for a built slice — used by the tests and the capture corpus."""
    return json.dumps(record, indent=2, sort_keys=False, default=str)


__all__ = [
    "GROUNDINGS",
    "MAX_LIST_ROWS",
    "STRIP_LISTS_FOR",
    "GroundingError",
    "build",
    "category_columns",
    "dump",
    "install_verdict_extras",
    "llm_tag_ids",
    "numeric_category_columns",
    "strip_hints",
    "verdict_extra_fields",
]
