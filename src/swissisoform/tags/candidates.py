"""The candidate sweep — frozen distributions in, reviewable tag table out.

Proposes every mechanically-derivable tag, cuts each one against the frozen
distribution, measures what it would actually fire on, drops the ones that cannot
work, and writes what is left for a human to accept or reject. It decides nothing:
the output is a CSV with an empty ``decision`` column.

Three things are deliberately separated:

- **Cutoffs come from the frozen distribution** (:mod:`swissisoform.distributions`),
  so a tag's number is reproducible and stamped with where it came from.
- **Rates come from the run itself**, because a cutoff's fire rate, its
  not-evaluable profile, and any two tags' overlap are properties of the data, and
  a marginal distribution cannot express the last of those at all.
- **Validity is declared** (:mod:`.seeds`), never inferred from nulls — "no shared
  region, so this is undefined" is a fact about the isoform; a low fill rate is a
  fact about the pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from swissisoform import metrics
from swissisoform.distributions import (
    MIN_STRATUM_N,
    STRATUM_ALL,
    STRATUM_SEPARATE,
    Distributions,
)
from swissisoform.tags import seeds

# Issue #30's usefulness band: a tag firing on <10% is a curiosity, one firing on
# >60% does not partition the corpus. Out-of-band candidates become chips.
DEFAULT_BAND: tuple[float, float] = (10.0, 60.0)

# Percentiles swept when no anchor or break yields an in-band cutoff.
SWEEP_PERCENTILES: tuple[int, ...] = (40, 50, 60, 70, 75, 80, 90)

# Two tags overlapping this much are the same tag wearing two names.
JACCARD_MAX = 0.8

# A metric must be populated this densely to carry a tag at all.
MIN_FILL = 0.8

# Catalog exclusions that also disqualify a tag. `binary` and `categorical` are
# deliberately absent — they are excluded from the embedding for having little
# variance, which is no argument at all against being a checkbox.
DISQUALIFYING_EXCLUSIONS = frozenset(
    {"identifier", "derived_from_features", "free_text", "status_string"}
)

# A trough must be this deep, relative to the lower of its flanking peaks, before
# it counts as structure rather than histogram noise.
MIN_BREAK_PROMINENCE = 0.25

# An anchor landing inside this percentile band is rejected. A principled
# zero-point is only informative if the corpus actually sits to one side of it;
# when the null falls at the median the TYPICAL isoform is at the null, so
# "above it" is a coin flip. Measured: the biophysics enrichment ratios put 1.0
# at p41-p54 (a coin flip), while the anchors that work sit off-centre — gnomAD
# depletion 0.8 at p12, M2 disease enrichment 1.0 at p64.
ANCHOR_DEAD_ZONE: tuple[float, float] = (40.0, 60.0)

# Names that mark a metric as being *about the difference* rather than about the
# isoform in absolute terms. A sort key for review, never a filter: the isoform_
# pane carries both kinds and the reviewer should meet the differential ones first.
DIFFERENTIAL_RE = re.compile(
    r"unique_region|shared_region|enrichment|diffregion|diff_vs|_delta|in_diff_region|_changed"
)

STRATA_REPORTED: tuple[str, ...] = ("extended", "truncated", STRATUM_SEPARATE)


@dataclass(frozen=True)
class Candidate:
    """One proposed tag, before or after cutoff selection.

    Attributes:
        tag_id: Stable slug, unique within the table.
        category: CDLMPS letter.
        label: Proposed human label — a placeholder the reviewer rewrites.
        metric: Parquet column or ``tx:<name>``; empty for LLM tags.
        kind: ``code`` (threshold), ``bool`` (already tri-state), ``llm`` (judged).
        direction: ``">="`` or ``"<"``; empty for bool/llm.
        valid_for: ORF types the tag is defined for; everything else is
            not-evaluable by declaration.
        source: Which stream proposed it.
        cutoff / cutoff_source / cutoff_pctile / break_depth: filled by
            :func:`choose_cutoff`.
        blocked: Non-empty when the candidate cannot be proposed, carrying why.
    """

    tag_id: str
    category: str
    label: str
    metric: str
    kind: str
    direction: str
    valid_for: tuple[str, ...]
    source: str
    criterion_id: str = ""
    note: str = ""
    blocked: str = ""
    cutoff: float | None = None
    cutoff_source: str = ""
    cutoff_pctile: float | None = None
    break_depth: float | None = None
    current_cutoff: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def differential(self) -> bool:
        """Whether the metric name marks it as being about the difference."""
        return bool(DIFFERENTIAL_RE.search(self.metric))

    @property
    def test(self) -> str:
        """One-line plain-English statement of what fires this tag."""
        if self.kind == "llm":
            return f"judged by the {self.category} tool loop"
        if self.kind == "bool":
            return f"{self.metric} is true"
        if self.cutoff is None:
            return f"{self.metric} {self.direction} (no cutoff)"
        return f"{self.metric} {self.direction} {self.cutoff:.6g}"


# ---------------------------------------------------------------------------
# Proposal
# ---------------------------------------------------------------------------


def _slug(text: str) -> str:
    """Filesystem- and column-safe slug for a tag id."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _catalog_index(catalog: pd.DataFrame) -> dict[str, pd.Series]:
    return {str(r["feature"]): r for _, r in catalog.iterrows()}


def propose(
    catalog: pd.DataFrame, dist: Distributions, columns: set[str]
) -> list[Candidate]:
    """Enumerate every candidate from all four streams, before cutoff selection.

    Args:
        catalog: The feature catalog (metric registry).
        dist: The frozen distributions, used for which metrics were profiled.
        columns: Column names present in the run, for boolean availability.

    Returns:
        Candidates in stream order: criteria, numeric sweep, booleans, LLM.
    """
    by_feature = _catalog_index(catalog)
    profiled = set(dist.numeric["metric"])
    out: list[Candidate] = []

    # Stream 1 — the sixteen scored criteria.
    for seed in seeds.CRITERION_SEEDS:
        if seed.metric is None:
            continue  # categorical or list-scored; the boolean stream covers these
        null_pattern = _null_pattern(by_feature, seed.metric)
        out.append(
            Candidate(
                tag_id=_slug(f"{seed.criterion_id}__{seed.metric}"),
                category=seed.criterion_id[0],
                label=seed.label,
                metric=seed.metric,
                kind="code",
                direction=seed.direction,
                valid_for=seeds.validity_for(seed.metric, null_pattern),
                source="criterion",
                criterion_id=seed.criterion_id,
                note=seed.note,
                blocked=seed.blocked
                or ("" if seed.metric in profiled else "metric not profiled in this version"),
                current_cutoff=_current_cutoff(seed.config_field, seed.literal),
                extra={"config_field": seed.config_field or ""},
            )
        )

    # Stream 2a — numeric sweep, both directions. Canonical-pane metrics describe
    # the gene, not the isoform's change, so they are excluded; everything else on
    # the isoform/cmp/diff/site panes is fair game (the differential metrics live
    # on the isoform_ pane, so a stricter pane filter would starve C and P).
    seeded = {c.metric for c in out}
    numeric_all = dist.numeric[dist.numeric["stratum"] == STRATUM_ALL]
    for _, row in numeric_all.iterrows():
        metric = str(row["metric"])
        if metric in seeded or str(row["pane"]) == "canonical":
            continue
        # Magnitudes are proposed from their signed parent below, one direction
        # only. Sweeping them here as ordinary metrics would re-add the second
        # direction — "|delta| < x", i.e. "the property barely changed", which is
        # absence of change dressed up as a finding.
        if metrics.is_magnitude(metric):
            continue
        null_pattern = _null_pattern(by_feature, metric)
        valid = seeds.validity_for(metric, null_pattern)
        # A signed delta is proposed as its MAGNITUDE, one direction only. The
        # tag worth having asks "did this property change appreciably?" — which is
        # |delta| over a bar. Thresholding the signed value instead tests only the
        # DIRECTION of the change, which fires on ~50% of the corpus at shifts far
        # below what the pipeline treats as real (measured on the S2 properties:
        # ~87% of firings fall under S2's own magnitude bar).
        if "_delta" in metric and not metrics.is_magnitude(metric):
            mag = metrics.magnitude_of(metric)
            if mag not in profiled:
                continue
            out.append(
                Candidate(
                    tag_id=_slug(f"{metric}_magnitude"),
                    category=str(row["category"]) or "-",
                    label=f"{metric} changes appreciably",
                    metric=mag,
                    kind="code",
                    direction=">=",
                    valid_for=valid,
                    source="sweep",
                    note="magnitude of a signed delta — direction carries no information here",
                )
            )
            continue
        for direction in (">=", "<"):
            out.append(
                Candidate(
                    tag_id=_slug(f"{metric}_{'hi' if direction == '>=' else 'lo'}"),
                    category=str(row["category"]) or "-",
                    label=f"{metric} {'high' if direction == '>=' else 'low'}",
                    metric=metric,
                    kind="code",
                    direction=direction,
                    valid_for=valid,
                    source="sweep",
                )
            )

    # Stream 2b — booleans. A boolean column is already a tri-state tag; there is
    # no cutoff to pick. `include_in_plot` excludes these (it was built for an
    # embedding, where a binary carries almost no variance) which is exactly the
    # wrong gate here.
    for feature, row in by_feature.items():
        if str(row.get("dtype")) != "bool" or str(row.get("pane")) == "canonical":
            continue
        if str(row.get("exclude_reason")) != "binary" or feature not in columns:
            continue
        # `*_enriched` is not an observation, it is `ratio > 1.0` hardcoded in
        # the comparator (compare/paired.py:81) — a direction test with no
        # magnitude gate, and 1.0 sits at the median for every biophysical
        # property. Propose the underlying ratio so the cutoff comes from the
        # frozen distribution like every other tag's.
        if feature.endswith("_enriched"):
            ratio = feature[: -len("_enriched")] + "_ratio"
            if ratio in profiled:
                out.append(
                    Candidate(
                        tag_id=_slug(f"{ratio}_hi"),
                        category=str(row.get("category") or "-"),
                        label=f"{ratio} high",
                        metric=ratio,
                        kind="code",
                        direction=">=",
                        valid_for=seeds.validity_for(ratio, _null_pattern(by_feature, ratio)),
                        source="enriched",
                        note="re-derived from the *_enriched boolean, whose 1.0 cutoff is "
                        "hardcoded upstream and sits at the median",
                    )
                )
                continue
        out.append(
            Candidate(
                tag_id=_slug(feature),
                category=str(row.get("category") or "-"),
                label=feature,
                metric=feature,
                kind="bool",
                direction="",
                valid_for=seeds.validity_for(feature, _null_pattern(by_feature, feature)),
                source="bool",
            )
        )

    # Stream 3 — judgment tags. No cutoff, no measurable rate until the loop runs.
    for llm in seeds.LLM_SEEDS:
        out.append(
            Candidate(
                tag_id=llm.tag_id,
                category=llm.category,
                label=llm.label,
                metric="",
                kind="llm",
                direction="",
                valid_for=llm.valid_for,
                source="llm",
                note=f"{llm.question} Reader: {llm.reader}. Cites: {llm.citation}.",
            )
        )
    return out


def _current_cutoff(config_field: str | None, literal: float | None = None) -> float | None:
    """Today's live threshold for a criterion, read off ``ScoringConfig``.

    Carried beside the proposal so a reviewer sees what changes: several criteria
    fire on >90% at their shipped cutoff, which is the whole reason #30 wants them
    re-cut or demoted to a chip.
    """
    if not config_field:
        return literal
    from swissisoform.config import ScoringConfig

    value = getattr(ScoringConfig(), config_field, None)
    return float(value) if isinstance(value, (int, float)) else literal


def _null_pattern(by_feature: dict[str, pd.Series], metric: str) -> str | None:
    row = by_feature.get(metric)
    if row is None and metric.startswith(metrics.ABS_PREFIX):
        row = by_feature.get(metric[len(metrics.ABS_PREFIX):])
    if row is None:
        return None
    val = row.get("null_pattern")
    return None if pd.isna(val) else str(val)


# ---------------------------------------------------------------------------
# Cutoff selection
# ---------------------------------------------------------------------------


def find_break(edges: np.ndarray, counts: np.ndarray) -> tuple[float, float] | None:
    """Deepest interior trough in a histogram, as ``(value, prominence)``.

    Prominence is the trough's depth below the lower of its two flanking peaks,
    as a fraction of that peak — so a shallow dip in a unimodal spread scores near
    zero and a genuine valley between two modes scores near one. A metric with no
    prominent trough is unimodal, and a boolean cut of it is arbitrary wherever it
    is placed.
    """
    if len(counts) < 5:
        return None
    best: tuple[float, float] | None = None
    for i in range(1, len(counts) - 1):
        left, right = counts[:i].max(initial=0), counts[i + 1 :].max(initial=0)
        peak = min(left, right)
        if peak <= 0 or counts[i] >= peak:
            continue
        prominence = float((peak - counts[i]) / peak)
        if best is None or prominence > best[1]:
            best = (float((edges[i] + edges[i + 1]) / 2), prominence)
    if best is None or best[1] < MIN_BREAK_PROMINENCE:
        return None
    return best


def _rate_from_grid(dist: Distributions, metric: str, cutoff: float, direction: str,
                    stratum: str) -> float | None:
    """Approximate fire rate at *cutoff*, read off the frozen quantile grid."""
    pct = dist.percentile(metric, cutoff, stratum)
    if pct is None:
        return None
    return 100.0 - pct if direction == ">=" else pct


def choose_cutoff(
    cand: Candidate, dist: Distributions, band: tuple[float, float] = DEFAULT_BAND
) -> Candidate:
    """Pick a cutoff for one candidate: anchor, then break, then percentile.

    Ordered by how much the number means. An anchor is a boundary in the biology
    (a ratio's 1.0, a delta's 0.0). A break is a boundary in the data. A bare
    percentile is neither — it *sets* the fire rate rather than discovering it,
    which is why it comes last and is recorded as such.

    Scored against the stratum the tag is mostly about: ``all`` unless the tag is
    only valid for paired ORF types, in which case the larger of those.
    """
    if cand.kind != "code" or cand.blocked:
        return cand
    stratum = _scoring_stratum(cand, dist)
    lo, hi = band

    anchor = seeds.anchor_for(cand.metric)
    if anchor is not None:
        value, reason = anchor
        rate = _rate_from_grid(dist, cand.metric, value, cand.direction, stratum)
        at = dist.percentile(cand.metric, value, stratum)
        centred = at is not None and ANCHOR_DEAD_ZONE[0] <= at <= ANCHOR_DEAD_ZONE[1]
        if rate is not None and lo <= rate <= hi and not centred:
            return replace(
                cand, cutoff=value, cutoff_source="anchor",
                cutoff_pctile=dist.percentile(cand.metric, value, stratum),
                extra={**cand.extra, "anchor_reason": reason},
            )

    hist = dist.histogram(cand.metric, stratum)
    if hist is not None:
        found = find_break(*hist)
        if found is not None:
            value, prominence = found
            rate = _rate_from_grid(dist, cand.metric, value, cand.direction, stratum)
            if rate is not None and lo <= rate <= hi:
                return replace(
                    cand, cutoff=value, cutoff_source="break", break_depth=prominence,
                    cutoff_pctile=dist.percentile(cand.metric, value, stratum),
                )

    # Percentile fallback: the one landing nearest the middle of the band.
    target = (lo + hi) / 2
    best: tuple[float, int] | None = None
    for pct in SWEEP_PERCENTILES:
        rate = 100.0 - pct if cand.direction == ">=" else float(pct)
        if not (lo <= rate <= hi):
            continue
        if best is None or abs(rate - target) < abs(best[0] - target):
            best = (rate, pct)
    if best is None:
        return cand
    try:
        value = dist.value_at(cand.metric, best[1], stratum)
    except Exception:
        return cand
    if value is None:
        return cand
    return replace(cand, cutoff=value, cutoff_source="percentile", cutoff_pctile=float(best[1]))


def _scoring_stratum(cand: Candidate, dist: Distributions) -> str:
    """The stratum a cutoff is derived from.

    ``all`` for a tag valid everywhere. For a paired-only tag, pooling would mix in
    isoforms the tag can never fire on, so it uses the larger paired stratum that
    clears the floor.
    """
    if set(cand.valid_for) >= set(seeds.ALL_ORF_TYPES):
        return STRATUM_ALL
    best, best_n = STRATUM_ALL, -1
    for stratum in cand.valid_for:
        summary = dist.summary(cand.metric, stratum)
        n = int(summary["n"]) if summary else 0
        if n >= MIN_STRATUM_N and n > best_n:
            best, best_n = stratum, n
    return best


# ---------------------------------------------------------------------------
# Evaluation on the run
# ---------------------------------------------------------------------------


def evaluate(df: pd.DataFrame, cands: Iterable[Candidate]) -> dict[str, np.ndarray]:
    """Tri-state firing per candidate: 1 on, 0 off, -1 not-evaluable.

    Not-evaluable comes first and for two distinct reasons: the tag is undefined
    for this ORF type (declared), or the metric is missing for this row (a data
    hole). Both read as -1 here; the table reports them together as the tag's
    not-evaluable rate, which is what a reviewer needs to judge coverage.
    """
    orf = df["orf_type"].astype("string")
    out: dict[str, np.ndarray] = {}
    for cand in cands:
        if cand.kind == "llm" or cand.blocked:
            continue
        values = metrics.resolve(cand.metric, df) if cand.kind == "code" else None
        if cand.kind == "bool":
            raw = df[cand.metric] if cand.metric in df.columns else None
            if raw is None:
                continue
            fired = raw.map({True: 1, False: 0}).to_numpy(dtype="float64", na_value=np.nan)
        else:
            if values is None or (cand.cutoff is None):
                continue
            arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
            with np.errstate(invalid="ignore"):
                hit = arr >= cand.cutoff if cand.direction == ">=" else arr < cand.cutoff
            fired = np.where(np.isnan(arr), np.nan, hit.astype("float64"))
        state = np.where(np.isnan(fired), -1.0, fired)
        state = np.where(~orf.isin(cand.valid_for).to_numpy(), -1.0, state)
        out[cand.tag_id] = state.astype("int8")
    return out


def rates(state: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float]:
    """Return ``(fire_pct, not_evaluable_pct)`` for one tri-state vector.

    ``fire_pct`` is over the **evaluable** rows — the fraction of isoforms the tag
    could have fired on that it did. Reporting it over all rows would let a tag
    look selective purely because it is undefined for most of the corpus.
    """
    sub = state if mask is None else state[mask]
    if len(sub) == 0:
        return float("nan"), float("nan")
    evaluable = sub >= 0
    n_eval = int(evaluable.sum())
    not_eval = 100.0 * (len(sub) - n_eval) / len(sub)
    fire = 100.0 * float((sub == 1).sum()) / n_eval if n_eval else float("nan")
    return fire, not_eval


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard overlap of two tags' True sets, over rows where both are evaluable.

    Restricting to jointly-evaluable rows is what keeps a pair from looking
    independent merely because they are undefined on disjoint strata.
    """
    both = (a >= 0) & (b >= 0)
    if not both.any():
        return 0.0
    x, y = (a[both] == 1), (b[both] == 1)
    union = int((x | y).sum())
    return float((x & y).sum()) / union if union else 0.0


# ---------------------------------------------------------------------------
# Filters and the review table
# ---------------------------------------------------------------------------


@dataclass
class Funnel:
    """What each filter removed, so the sweep can report itself."""

    proposed: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    chips: int = 0
    kept: int = 0

    def drop(self, reason: str, n: int = 1) -> None:
        """Record *n* candidates removed for *reason*."""
        self.dropped[reason] = self.dropped.get(reason, 0) + n


def _catalog_flag(by_feature: dict[str, pd.Series], metric: str, key: str) -> Any:
    """Catalog field for a metric, resolving a magnitude back to its signed column."""
    row = by_feature.get(metric)
    if row is None and metric.startswith(metrics.ABS_PREFIX):
        row = by_feature.get(metric[len(metrics.ABS_PREFIX):])
    return None if row is None else row.get(key)


def apply_filters(
    cands: list[Candidate],
    dist: Distributions,
    by_feature: dict[str, pd.Series],
    funnel: Funnel,
) -> list[Candidate]:
    """Structural filters — everything decidable without touching the run.

    Ordered cheapest-first, and every drop is counted so the funnel can be
    reported rather than inferred.
    """
    kept: list[Candidate] = []
    for cand in cands:
        if cand.kind == "llm" or cand.blocked or cand.source == "criterion":
            kept.append(cand)  # judged, explained, or already in production
            continue
        summary = dist.summary(cand.metric, STRATUM_ALL) if cand.metric else None
        if cand.kind == "code":
            if summary is None:
                funnel.drop("not profiled")
                continue
            if (summary.get("fill_rate") or 0.0) < MIN_FILL:
                funnel.drop("sparse (fill < 0.8)")
                continue
            lo, hi = summary.get("min"), summary.get("max")
            if lo is None or hi is None or lo == hi:
                funnel.drop("constant")
                continue
        # A tag belongs to a category. Metrics the catalog cannot assign a CDLMPS
        # letter to are coordinates and identity fields (`position`, `orf_exons`),
        # which cannot be any category's checkbox.
        if cand.category in ("", "-", "nan", "None"):
            funnel.drop("no CDLMPS category")
            continue
        reason = _catalog_flag(by_feature, cand.metric, "exclude_reason")
        if isinstance(reason, str) and reason in DISQUALIFYING_EXCLUSIONS:
            funnel.drop(f"catalog exclusion: {reason}")
            continue
        dup = _catalog_flag(by_feature, cand.metric, "duplicate_of")
        if dup is not None and not pd.isna(dup):
            funnel.drop("duplicate of another column")
            continue
        if not any(
            (dist.summary(cand.metric, s) or {}).get("n", 0) >= MIN_STRATUM_N
            for s in cand.valid_for
        ) and cand.kind == "code":
            funnel.drop("no stratum clears the size floor")
            continue
        kept.append(cand)
    return kept


def _examples(
    df: pd.DataFrame, state: np.ndarray, coreset_ids: set[str], n: int = 3
) -> tuple[str, str]:
    """Gene names from the coreset that fire / do not fire, for spot-checking."""
    in_coreset = df["tis_id"].isin(coreset_ids).to_numpy()
    genes = df["gene_name"].to_numpy()
    on = genes[in_coreset & (state == 1)][:n]
    off = genes[in_coreset & (state == 0)][:n]
    return ", ".join(map(str, on)), ", ".join(map(str, off))


def build_table(
    df: pd.DataFrame,
    catalog: pd.DataFrame,
    dist: Distributions,
    *,
    coreset: pd.DataFrame | None = None,
    band: tuple[float, float] = DEFAULT_BAND,
    jaccard_max: float = JACCARD_MAX,
) -> tuple[pd.DataFrame, pd.DataFrame, Funnel]:
    """Run the whole sweep and return ``(candidates, chips, funnel)``.

    Args:
        df: The flattened run, with ``orf_type`` / ``gene_name`` / ``tis_id``.
        catalog: Feature catalog (metric registry).
        dist: Frozen distributions to cut cutoffs against.
        coreset: Optional cheeseman50 table, for worked examples.
        band: Acceptable fire-rate window; out-of-band becomes a chip.
        jaccard_max: Overlap above which the later candidate is redundant.
    """
    by_feature = _catalog_index(catalog)
    funnel = Funnel()

    cands = propose(catalog, dist, set(df.columns))
    funnel.proposed = len(cands)
    cands = apply_filters(cands, dist, by_feature, funnel)
    cands = [choose_cutoff(c, dist, band) for c in cands]

    # Band split, on two grounds.
    #
    # No cutoff at all: no in-band cut exists anywhere in the distribution.
    #
    # A bare *percentile* cutoff: no anchor, no break — the metric is unimodal, so
    # the boolean is arbitrary wherever it lands, and the fire rate was *set* by
    # the percentile rather than discovered. Such a metric is a range filter, not
    # a checkbox.
    #
    # Two exemptions, both already in the vocabulary and so not facing a
    # tag-or-chip choice: existing criteria (human-blessed, and the reviewer must
    # see all sixteen), and the re-derived `*_enriched` tags, which shipped with a
    # hardcoded 1.0 — for those the real choice is "distribution percentile or
    # that constant", and the percentile is strictly better even at p50.
    def _is_chip(c: Candidate) -> bool:
        if c.kind != "code" or c.blocked:
            return False
        if c.cutoff is None:
            return True
        return c.cutoff_source == "percentile" and c.source not in ("criterion", "enriched")

    chips = [c for c in cands if _is_chip(c)]
    chip_ids = {id(c) for c in chips}
    cands = [c for c in cands if id(c) not in chip_ids]
    funnel.drop("unimodal — percentile cutoff only", len(chips))
    funnel.chips = len(chips)

    # One direction per metric. `>= x` and `< x` are exact complements, so their
    # True sets are disjoint and Jaccard can never catch the pair — but as filter
    # checkboxes one is the negation of the other. Keep whichever sits nearer the
    # middle of the band; the other is recoverable by flipping the direction.
    cands = _one_direction_per_metric(cands, dist, band, funnel)

    state = evaluate(df, cands)
    lo, hi = band
    scored: list[tuple[Candidate, np.ndarray, float, float]] = []
    for cand in cands:
        vec = state.get(cand.tag_id)
        if vec is None:
            if cand.kind == "llm" or cand.blocked:
                scored.append((cand, np.zeros(len(df), dtype="int8"), float("nan"), float("nan")))
            else:
                funnel.drop("could not be evaluated on the run")
            continue
        fire, not_eval = rates(vec)
        # Measured on the run, which can disagree with the grid estimate the
        # cutoff was chosen by (ties, missing rows). The measurement wins.
        if cand.kind == "code" and not (lo <= fire <= hi):
            chips.append(cand)
            funnel.chips += 1
            continue
        scored.append((cand, vec, fire, not_eval))

    scored.sort(key=lambda t: (t[0].category, -int(t[0].source == "criterion"), t[0].tag_id))
    rows: list[dict[str, Any]] = []
    accepted: list[tuple[str, np.ndarray]] = []
    coreset_ids = set(coreset["tis_id"]) if coreset is not None else set()

    for cand, vec, fire, not_eval in scored:
        max_j, nearest = 0.0, ""
        if cand.kind != "llm" and not cand.blocked:
            for tag_id, other in accepted:
                j = jaccard(vec, other)
                if j > max_j:
                    max_j, nearest = j, tag_id
            if max_j >= jaccard_max:
                funnel.drop(f"redundant (Jaccard >= {jaccard_max})")
                continue
            accepted.append((cand.tag_id, vec))
        on, off = _examples(df, vec, coreset_ids) if coreset_ids else ("", "")
        row: dict[str, Any] = {
            "category": cand.category,
            "tag_id": cand.tag_id,
            "proposed_label": seeds.label_for(
                cand.metric, cand.direction or "bool", fallback=cand.label
            )
            if cand.kind != "llm"
            else cand.label,
            "test": cand.test,
            "source_metric": cand.metric,
            "kind": cand.kind,
            "stream": cand.source,
            "criterion_id": cand.criterion_id,
            "differential_hint": cand.differential,
            "cutoff": cand.cutoff,
            "current_cutoff": cand.current_cutoff,
            "current_fire_pct": _current_fire(df, cand),
            "cutoff_source": cand.cutoff_source,
            "cutoff_pctile": cand.cutoff_pctile,
            "break_depth": cand.break_depth,
            "valid_for": "|".join(cand.valid_for),
            "fire_pct": round(fire, 2) if fire == fire else None,
            "not_evaluable_pct": round(not_eval, 2) if not_eval == not_eval else None,
        }
        for stratum in STRATA_REPORTED:
            mask = _stratum_mask(df, stratum)
            f, ne = rates(vec, mask)
            row[f"fire_pct_{stratum}"] = round(f, 2) if f == f else None
            row[f"not_evaluable_pct_{stratum}"] = round(ne, 2) if ne == ne else None
        row.update(
            {
                "max_jaccard": round(max_j, 3),
                "nearest_tag": nearest,
                "examples_on": on,
                "examples_off": off,
                "blocked": cand.blocked,
                "note": cand.note,
                "decision": "",
                "reviewer_notes": "",
            }
        )
        rows.append(row)

    funnel.kept = len(rows)
    table = pd.DataFrame(rows)
    if len(table):
        table = table.sort_values(
            ["category", "max_jaccard", "tag_id"], ascending=[True, True, True]
        ).reset_index(drop=True)
    # A metric kept as a tag must not also appear as a chip: only one of its
    # two directions may have failed the band, and offering both readings
    # would show the reviewer the same metric twice with opposite advice.
    kept_metrics = set(table["source_metric"]) if len(table) else set()
    return table, _chip_table(chips, dist, exclude=kept_metrics), funnel


def _one_direction_per_metric(
    cands: list[Candidate], dist: Distributions, band: tuple[float, float], funnel: Funnel
) -> list[Candidate]:
    """Collapse complementary ``>=`` / ``<`` proposals to one per metric."""
    target = sum(band) / 2
    best: dict[str, Candidate] = {}
    order: list[Candidate] = []
    for cand in cands:
        if cand.kind != "code" or cand.source == "criterion" or cand.blocked:
            order.append(cand)
            continue
        prev = best.get(cand.metric)
        if prev is None:
            best[cand.metric] = cand
            order.append(cand)
            continue
        stratum = _scoring_stratum(cand, dist)

        def _gap(c: Candidate) -> float:
            rate = _rate_from_grid(dist, c.metric, c.cutoff, c.direction, stratum)
            return abs((rate if rate is not None else 0.0) - target)

        funnel.drop("complementary direction of the same metric")
        if _gap(cand) < _gap(prev):
            order[order.index(prev)] = cand
            best[cand.metric] = cand
    return order


def _current_fire(df: pd.DataFrame, cand: Candidate) -> float | None:
    """What the criterion fires on **today**, at its shipped cutoff."""
    if cand.current_cutoff is None or cand.kind != "code" or cand.blocked:
        return None
    values = metrics.resolve(cand.metric, df)
    if values is None:
        return None
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    with np.errstate(invalid="ignore"):
        hit = arr >= cand.current_cutoff if cand.direction == ">=" else arr < cand.current_cutoff
    fired = np.where(np.isnan(arr), np.nan, hit.astype("float64"))
    state = np.where(np.isnan(fired), -1.0, fired)
    state = np.where(
        ~df["orf_type"].astype("string").isin(cand.valid_for).to_numpy(), -1.0, state
    )
    fire, _ = rates(state.astype("int8"))
    return round(fire, 2) if fire == fire else None


def _stratum_mask(df: pd.DataFrame, stratum: str) -> np.ndarray:
    if stratum == STRATUM_SEPARATE:
        return df["orf_type"].isin(seeds.SEPARATE_ORF_TYPES).to_numpy()
    return (df["orf_type"] == stratum).to_numpy()


def _chip_table(
    chips: list[Candidate], dist: Distributions, exclude: set[str] | None = None
) -> pd.DataFrame:
    """Demoted candidates, with the shape evidence for why a boolean was wrong."""
    exclude = exclude or set()
    rows = []
    for cand in chips:
        if cand.metric in exclude:
            continue
        summary = dist.summary(cand.metric, STRATUM_ALL) or {}
        hist = dist.histogram(cand.metric, STRATUM_ALL)
        found = find_break(*hist) if hist is not None else None
        rows.append(
            {
                "category": cand.category,
                "metric": cand.metric,
                "label": cand.label,
                "direction": cand.direction,
                "n": summary.get("n"),
                "p05": summary.get("p05"),
                "p25": summary.get("p25"),
                "p50": summary.get("p50"),
                "p75": summary.get("p75"),
                "p95": summary.get("p95"),
                "best_break_depth": round(found[1], 3) if found else None,
                "reason": "no in-band cutoff — continuous, range-filter it instead",
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out = out.drop_duplicates(subset=["metric"]).sort_values(["category", "metric"])
    return out.reset_index(drop=True)


REVIEW_COLUMNS = ("decision", "reviewer_notes")
SUPERSEDED = "[superseded] no longer proposed by the current sweep. "


def merge_decisions(new: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    """Carry a reviewer's decisions across a re-sweep, keyed by ``tag_id``.

    Two rules, both about not destroying review work:

    - A tag that still exists keeps whatever decision it was given.
    - A tag the new sweep no longer proposes is **kept as a row** rather than
      dropped, marked ``superseded``. Its note is the record of why it was
      rejected; deleting it would erase that finding from the audit trail.
    """
    if not len(previous) or "decision" not in previous:
        return new
    prior = previous.set_index("tag_id")
    out = new.copy()
    for col in REVIEW_COLUMNS:
        if col in prior:
            carried = out["tag_id"].map(prior[col])
            out[col] = carried.where(carried.notna() & (carried != ""), out.get(col, ""))

    retired = prior.index.difference(out["tag_id"])
    if len(retired):
        rows = previous[previous["tag_id"].isin(retired)].copy()
        rows["stream"] = "retired"
        # Idempotent: a row already retired by an earlier sweep keeps one marker,
        # not one per sweep.
        rows["reviewer_notes"] = rows["reviewer_notes"].fillna("").astype(str)
        fresh = ~rows["reviewer_notes"].str.startswith(SUPERSEDED)
        rows.loc[fresh, "reviewer_notes"] = SUPERSEDED + rows.loc[fresh, "reviewer_notes"]
        rows["decision"] = rows["decision"].replace("", "remove").fillna("remove")
        out = pd.concat([out, rows], ignore_index=True)
    return out


def write_outputs(
    table: pd.DataFrame, chips: pd.DataFrame, funnel: Funnel, out_dir: Path, *, version: str
) -> None:
    """Write the candidate table, the chip table, and the review summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = out_dir / "tag_candidates.csv"
    if existing.exists():
        table = merge_decisions(table, pd.read_csv(existing).fillna(""))
    table.to_csv(existing, index=False)
    chips.to_csv(out_dir / "percentile_chips.csv", index=False)

    lines = [
        "# Tag candidate review",
        "",
        f"Swept against distributions `{version}`. "
        f"{funnel.proposed} proposals → **{funnel.kept} candidates** "
        f"+ {len(chips)} percentile chips. Counts in the funnel are *proposals* "
        f"(a metric is proposed in both directions); the chip table is per metric.",
        "",
        "Fill in `decision` (keep / drop / reword / merge-into) and rewrite",
        "`proposed_label` in `tag_candidates.csv`. Accepted rows become the registry.",
        "",
        "## Funnel",
        "",
        "| stage | n |",
        "|---|---|",
        f"| proposed | {funnel.proposed} |",
    ]
    for reason, n in sorted(funnel.dropped.items(), key=lambda kv: -kv[1]):
        lines.append(f"| dropped — {reason} | {n} |")
    lines += [
        f"| demoted to chip | {funnel.chips} proposals → {len(chips)} metrics |",
        f"| **kept for review** | **{funnel.kept}** |",
        "",
        "## Per category",
        "",
    ]
    if len(table):
        counts = table.groupby("category").size()
        lines += ["| category | candidates |", "|---|---|"]
        lines += [f"| {cat} | {n} |" for cat, n in counts.items()]
    blocked = table[table["blocked"].astype(bool)] if len(table) else table
    if len(blocked):
        lines += ["", "## Blocked", ""]
        for _, r in blocked.iterrows():
            lines.append(f"- **{r['tag_id']}** ({r['source_metric']}) — {r['blocked']}")
    lines += [
        "",
        "## Deliberately not swept",
        "",
        "- **Categorical string columns.** The `cmp_*_changed` booleans already",
        "  encode 'the compartment / targeting call differs'; a raw compartment",
        "  string would need pairing logic that reproduces them.",
        "- **Canonical-pane metrics.** They describe the gene, not the isoform's",
        "  change. Note this excludes only the `canonical_` pane — the differential",
        "  metrics live on `isoform_`, so a stricter pane filter would starve C and P.",
        "",
    ]
    (out_dir / "tag_review.md").write_text("\n".join(lines), encoding="utf-8")
