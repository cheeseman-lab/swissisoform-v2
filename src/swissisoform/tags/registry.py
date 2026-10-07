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
    Runs a scored criterion's own function (see :mod:`swissisoform.tags.derived`)
    at the run's ``ScoringConfig``, so it equals the criterion. ``cutoff_overrides``
    records the cutoffs the build proposed; they are reported, not applied. This is how all
    sixteen CDLMPS criteria enter the vocabulary, because a bare ``metric ⋈
    cutoff`` demonstrably cannot reproduce them — measured on cheeseman50, 5 of 13
    disagreed with the scorer, every one of them a gate the threshold form drops
    (a fold status of ``too_long`` with the pLDDT column still populated; a
    criterion undefined for separate ORFs; an either-or over two inputs).
``llm``
    A judgment call for the M/P tool loop. Never fired by code; carried here so
    the vocabulary is complete and so a consumer can render it as *unanswered*
    rather than absent.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
REF_DIR = ROOT / "data" / "reference" / "tags"
# v3 is the current vocabulary: v2's 44 tags plus S2_biophysics and S3_sae,
# readmitted after the judge study showed their absence was what the tags arm
# was being marked down for — 85% of its losses in Structural Characteristics
# cite the biophysical or SAE evidence those two carry. The on-disk v3 carries a
# swept S3 cutoff (11.376) in `cutoff_overrides`; the evaluator reports it and
# fires S3 at the scoring config, like every derived tag.
#
# v3 also fixes a dead tag: `metrics.resolve` could not read a `<col>__len`
# metric, so v2's `cmp_motifs_hits_in_diff_region__len` was null on every row of
# every run.
#
# v1 predates the review pass: it was built from an 89-row candidate table and
# ships eight tags later retired for asserting absence. It is also no longer
# reproducible — building `--version v1` against today's table yields today's
# vocabulary under the v1 name — so the on-disk v1 is a historical artifact, not
# a rebuildable one.
DEFAULT_VERSION = "v3"
# v3 is cut from the provisional v3 distributions (see
# `distributions.PROVISIONAL_VERSIONS`); rebuild it after the re-freeze.
PROVISIONAL_VERSIONS: dict[str, str] = {
    "v3": (
        "cut from the provisional v3 distributions (Aug-12 full_catalog); "
        "rebuild after the next genome-wide run and distributions re-freeze"
    ),
}

# How a label names the unique region; a truncation reads it as the lost region
# (``Tag.label_for``).
_UNIQUE_REGION = re.compile(r"\b[Uu]nique region\b")

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

# Columns a registry may carry but need not: absent from versions built before
# they existed, so `from_frame` reads them with a default instead of refusing.
#
# ``cutoff_by_stratum`` is JSON ``{stratum: cutoff}`` for a threshold tag whose
# number should differ by ORF type — a stratum is an ``orf_type`` or the
# ``separate`` roll-up (``distributions.stratum_for``). A row's own ``orf_type``
# wins over its roll-up, and a stratum with no entry falls back to ``cutoff``.
# One pooled cutoff makes `length_ratio_hi` fire on 0.2% of extensions and 28.5%
# of truncations; this is the format that lets a build fix that.
OPTIONAL_COLUMNS: tuple[str, ...] = ("cutoff_by_stratum",)


def threshold_test(
    metric: str, direction: str, cutoff: float | None, by_stratum: dict[str, float]
) -> str:
    """One-line statement of a threshold test, per-stratum cutoffs included.

    Shared by :attr:`Tag.test` and the sweep's ``Candidate.test``, so the
    candidate table and the registry describe a cutoff the same way.
    """
    if by_stratum:
        per = ", ".join(f"{k} {v:.6g}" for k, v in sorted(by_stratum.items()))
        rest = "" if cutoff is None else f"; else {cutoff:.6g}"
        return f"{metric} {direction} by stratum ({per}{rest})"
    if cutoff is None:
        return f"{metric} {direction} (no cutoff)"
    return f"{metric} {direction} {cutoff:.6g}"


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
    cutoff_by_stratum: dict[str, float] = field(default_factory=dict)

    @property
    def code_fired(self) -> bool:
        """Whether the evaluator produces a state for this tag."""
        return self.kind in CODE_FIRED_KINDS and not self.blocked

    def label_for(self, orf_type: str | None) -> str:
        """The label as it reads for one isoform.

        A tag about the unique region describes sequence the isoform *adds* on an
        extension and sequence it *lost* on a truncation — the canonical stretch
        ahead of the truncated start (``diff_region`` is canonical-space there).
        The label table is written for the extension reading, so for a
        truncation:

        - "unique region" reads "lost region" ("Unique region more basic" →
          "Lost region more basic", "Long unique region" → "Long lost region");
        - a trailing " gained" reads " lost" ("Constrained residues gained" →
          "Constrained residues lost").

        Separate ORFs are wholly unique and keep the label as written. SAE tags
        are left alone: their "gained" is a feature the isoform gained, compared
        across both proteins, not something about the region.
        """
        if orf_type != "truncated" or "_sae_" in self.metric:
            return self.label
        label = _UNIQUE_REGION.sub(
            lambda m: "Lost region" if m.group(0)[0] == "U" else "lost region", self.label
        )
        if label.endswith(" gained"):
            label = label[: -len(" gained")] + " lost"
        return label

    @property
    def test(self) -> str:
        """One-line plain-English statement of what fires this tag."""
        if self.kind == KIND_LLM:
            return f"judged by the {self.category} tool loop"
        if self.kind == KIND_DERIVED:
            return f"derived predicate for {self.criterion_id}"
        if self.kind == KIND_BOOL:
            return f"{self.metric} is true"
        return threshold_test(self.metric, self.direction, self.cutoff, self.cutoff_by_stratum)


@dataclass(frozen=True)
class TagRegistry:
    """A frozen tag vocabulary, in registry-file order."""

    version: str
    tags: tuple[Tag, ...]
    provenance: dict[str, Any]

    @property
    def sha256(self) -> str:
        """Content hash of the tag definitions — what fires, not when it was built.

        A version name alone cannot tell two builds apart (v3 was rebuilt in place
        once, 44 -> 46 tags), so every parquet stamps this beside the name. Hashed
        over the parsed :class:`Tag` fields rather than the parquet bytes, so it
        is stable across pyarrow versions and changes exactly when a label,
        metric, direction, cutoff or validity does.
        """
        payload = json.dumps([asdict(t) for t in self.tags], sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def provisional(self) -> str:
        """Why this version must be rebuilt, or ``""`` for a settled one."""
        return PROVISIONAL_VERSIONS.get(self.version, "")

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
            cutoff_by_stratum=_overrides(row.get("cutoff_by_stratum")),
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
    "PROVISIONAL_VERSIONS",
    "KIND_BOOL",
    "KIND_DERIVED",
    "KIND_LLM",
    "KIND_THRESHOLD",
    "OPTIONAL_COLUMNS",
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
