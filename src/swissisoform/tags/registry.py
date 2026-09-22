"""The frozen tag registry — runtime read-side.

A registry version is the tag vocabulary plus the number each tag cuts at,
frozen as provisioned reference data under ``data/reference/tags/<version>/`` and
built by :mod:`swissisoform.setup.tags` (CLI:
``scripts/setup/build_tag_registry.py``). This module only reads it: import-safe,
network-free, no computation over pipeline output.

Why frozen and versioned: a tag's cutoff is the whole content of the tag. Letting
it drift with the code — as the ``ScoringConfig`` thresholds do today, eight of
them still carrying ``# CALIBRATE ON GENOME-WIDE RUN — provisional`` — means an
isoform's tags change for reasons nothing records. A version is immutable, every
parquet stamps the version that fired it, and re-versioning is deliberate.

Four kinds of tag, which is the whole taxonomy:

``threshold``
    ``metric ⋈ cutoff``, where the metric resolves through
    :mod:`swissisoform.metrics` and the cutoff came from the frozen
    distributions or from ``ScoringConfig``. Pure data — adding one is adding a
    row, never a function.
``bool``
    An existing boolean column, read as tri-state.
``derived``
    Runs a scored criterion's own function (see :mod:`swissisoform.tags.derived`),
    with that criterion's cutoffs taken from ``cutoff_overrides``. This is how all
    sixteen CDLMPS criteria enter the vocabulary, because a bare ``metric ⋈
    cutoff`` demonstrably cannot reproduce them — measured on cheeseman50, 5 of 13
    disagreed with the scorer, every one of them a gate the threshold form drops
    (a fold status of ``too_long`` with the pLDDT column still populated; a
    criterion undefined for separate ORFs; an either-or over two inputs). Carrying
    the cutoffs separately is what lets a calibrated registry move a criterion's
    number *without* also discarding the gates around it.
``llm``
    A judgment call for the M/P tool loop. Never fired by code; carried here so
    the vocabulary is complete and so a consumer can render it as *unanswered*
    rather than absent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
REF_DIR = ROOT / "data" / "reference" / "tags"
# v3 is the current vocabulary. Its table is byte-identical to v2's 44 tags —
# what changed is the code underneath: `metrics.resolve` could not read a
# `<col>__len` metric, so v2's `cmp_motifs_hits_in_diff_region__len` tag was
# null on every row of every run. Fixing that without a new version would have
# altered what v2 fires while leaving v2's bytes alone, which is exactly the
# drift the freeze exists to prevent, so the fix gets its own version.
#
# v1 predates the review pass: it was built from an 89-row candidate table and
# ships the eight retired tags plus S2/S3, which `seeds.UNCALIBRATED_CRITERIA`
# now excludes. It is also no longer reproducible — building `--version v1`
# against today's table yields the reviewed 44 tags under the v1 name — so the
# on-disk v1 is a historical artifact, not a rebuildable one.
DEFAULT_VERSION = "v3"

REGISTRY_FILE = "registry.parquet"
SIDECAR_FILE = "_setup.json"

# Kinds, in the order the evaluator dispatches them.
KIND_THRESHOLD = "threshold"
KIND_BOOL = "bool"
KIND_DERIVED = "derived"
KIND_LLM = "llm"
CODE_FIRED_KINDS = frozenset({KIND_THRESHOLD, KIND_BOOL, KIND_DERIVED})

# Existence = Conservation + Detection; functional = the rest. Mirrors
# EXISTENCE_CRITERIA / FUNCTIONAL_CRITERIA in swissisoform.evidence.
EXISTENCE_CATEGORIES = frozenset({"C", "D"})

REGISTRY_COLUMNS: tuple[str, ...] = (
    "tag_id",
    "category",
    "axis",
    "label",
    "kind",
    "metric",
    "direction",
    "cutoff",
    "cutoff_source",
    "cutoff_pctile",
    "cutoff_overrides",
    "valid_for",
    "criterion_id",
    "source",
    "blocked",
    "note",
)


class TagRegistryError(RuntimeError):
    """Raised when a registry version is missing or unreadable."""


def axis_for(category: str) -> str:
    """``"E"`` for Conservation/Detection, ``"F"`` for everything else."""
    return "E" if category in EXISTENCE_CATEGORIES else "F"


@dataclass(frozen=True)
class Tag:
    """One frozen tag definition.

    ``valid_for`` is a declaration, not an inference: "this ORF type has no shared
    region, so the tag is undefined" is a fact about the isoform, whereas a null
    metric is a fact about the pipeline. The evaluator reports both as
    not-evaluable but only the first is knowable ahead of the data.
    """

    tag_id: str
    category: str
    axis: str
    label: str
    kind: str
    metric: str
    direction: str
    cutoff: float | None
    cutoff_source: str
    cutoff_pctile: float | None
    cutoff_overrides: dict[str, float]
    valid_for: tuple[str, ...]
    criterion_id: str
    source: str
    blocked: str
    note: str

    @property
    def code_fired(self) -> bool:
        """Whether the evaluator produces a state for this tag."""
        return self.kind in CODE_FIRED_KINDS and not self.blocked

    @property
    def test(self) -> str:
        """One-line plain-English statement of what fires this tag."""
        if self.kind == KIND_LLM:
            return f"judged by the {self.category} tool loop"
        if self.kind == KIND_DERIVED:
            return f"derived predicate for {self.criterion_id}"
        if self.kind == KIND_BOOL:
            return f"{self.metric} is true"
        if self.cutoff is None:
            return f"{self.metric} {self.direction} (no cutoff)"
        return f"{self.metric} {self.direction} {self.cutoff:.6g}"


@dataclass(frozen=True)
class TagRegistry:
    """A frozen tag vocabulary, in registry-file order."""

    version: str
    tags: tuple[Tag, ...]
    provenance: dict[str, Any]

    def __len__(self) -> int:
        """Number of tags, code-fired or not."""
        return len(self.tags)

    def __iter__(self):
        """Iterate the tags in registry-file order."""
        return iter(self.tags)

    def get(self, tag_id: str) -> Tag | None:
        """The tag with this id, or None."""
        return self._by_id.get(tag_id)

    @property
    def _by_id(self) -> dict[str, Tag]:
        return {t.tag_id: t for t in self.tags}

    def code_fired(self) -> tuple[Tag, ...]:
        """Tags the evaluator produces a state for (excludes ``llm`` and blocked)."""
        return tuple(t for t in self.tags if t.code_fired)

    def by_category(self, category: str) -> tuple[Tag, ...]:
        """Every tag in one CDLMPS category, code-fired or not."""
        return tuple(t for t in self.tags if t.category == category)

    def by_kind(self, kind: str) -> tuple[Tag, ...]:
        """Every tag of one kind."""
        return tuple(t for t in self.tags if t.kind == kind)

    def state_columns(self) -> tuple[str, ...]:
        """Tag ids, in order — the field order of the emitted state struct."""
        return tuple(t.tag_id for t in self.code_fired())


def _tuple_from_pipe(value: Any) -> tuple[str, ...]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ()
    return tuple(p for p in str(value).split("|") if p)


def _str(value: Any) -> str:
    return "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)


def _overrides(value: Any) -> dict[str, float]:
    """Parse the JSON ``{config_field: cutoff}`` a derived tag carries.

    A derived tag runs its criterion's scorer, so its cutoffs live in
    ``ScoringConfig`` fields rather than in one ``cutoff`` scalar — an
    either-or criterion has several. Recording them here is what lets a
    calibrated registry move a criterion's number without losing the gates
    the scorer applies around it.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return {}
    return {str(k): float(v) for k, v in json.loads(str(value)).items()}


def _float_or_none(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return float(value)


def from_frame(version: str, frame: pd.DataFrame, provenance: dict[str, Any]) -> TagRegistry:
    """Build a :class:`TagRegistry` from a registry table.

    Split out from :func:`load` so tests (and the builder's own round-trip check)
    can construct one without touching the filesystem.
    """
    missing = [c for c in REGISTRY_COLUMNS if c not in frame.columns]
    if missing:
        raise TagRegistryError(f"registry is missing columns: {', '.join(missing)}")
    tags = tuple(
        Tag(
            tag_id=str(row["tag_id"]),
            category=str(row["category"]),
            axis=str(row["axis"]),
            label=_str(row["label"]),
            kind=str(row["kind"]),
            metric=_str(row["metric"]),
            direction=_str(row["direction"]),
            cutoff=_float_or_none(row["cutoff"]),
            cutoff_source=_str(row["cutoff_source"]),
            cutoff_pctile=_float_or_none(row["cutoff_pctile"]),
            cutoff_overrides=_overrides(row["cutoff_overrides"]),
            valid_for=_tuple_from_pipe(row["valid_for"]),
            criterion_id=_str(row["criterion_id"]),
            source=_str(row["source"]),
            blocked=_str(row["blocked"]),
            note=_str(row["note"]),
        )
        for _, row in frame.iterrows()
    )
    duplicates = {t.tag_id for t in tags if sum(1 for o in tags if o.tag_id == t.tag_id) > 1}
    if duplicates:
        raise TagRegistryError(f"duplicate tag_id in registry: {', '.join(sorted(duplicates))}")
    return TagRegistry(version=version, tags=tags, provenance=provenance)


def version_dir(version: str = DEFAULT_VERSION, root: Path | None = None) -> Path:
    """Path to one tag-registry version directory."""
    base = (root / "data" / "reference" / "tags") if root else REF_DIR
    return base / version


@lru_cache(maxsize=4)
def load(version: str = DEFAULT_VERSION, root: Path | None = None) -> TagRegistry:
    """Load a frozen tag registry version.

    Cached: the table is read once per run and never changes within a version.

    Raises:
        TagRegistryError: The version directory or the registry table is missing.
    """
    vdir = version_dir(version, root)
    path = vdir / REGISTRY_FILE
    if not path.exists():
        raise TagRegistryError(
            f"no tag registry at {path}. Build it with:\n"
            f"  python scripts/setup/build_tag_registry.py --version {version}"
        )
    sidecar = vdir / SIDECAR_FILE
    return from_frame(
        version,
        pd.read_parquet(path),
        json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {},
    )


__all__ = [
    "CODE_FIRED_KINDS",
    "DEFAULT_VERSION",
    "KIND_BOOL",
    "KIND_DERIVED",
    "KIND_LLM",
    "KIND_THRESHOLD",
    "REGISTRY_COLUMNS",
    "REGISTRY_FILE",
    "SIDECAR_FILE",
    "Tag",
    "TagRegistry",
    "TagRegistryError",
    "axis_for",
    "from_frame",
    "load",
    "version_dir",
]
