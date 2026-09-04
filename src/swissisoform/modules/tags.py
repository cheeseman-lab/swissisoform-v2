"""Tag annotation — fire a frozen registry over a finished run.

Additive by design. ``EvidenceScoringModule`` is untouched and keeps emitting its
nine ``isoform_scoring_*`` columns; this adds three more beside them, so both the
criterion axis and the tag axis travel in the same parquet and every consumer
picks the one it wants. Retiring either is a later, separate decision.

Unlike the annotation modules, this one runs over the **finished paired frame**
rather than over ``Gene`` objects alone. That is deliberate: a threshold tag's
cutoff was derived from a named parquet column, and evaluating it against a
column rebuilt by different code is how the two silently disagree. Reading the
frame that is about to be written removes the possibility.

Derived tags still need the ``TranslationInitiationSite`` objects, because they
call their criterion's scorer rather than reimplementing it — see
:mod:`swissisoform.tags.derived`. Hence :meth:`TagModule.annotate_frame` takes
both, aligned by row order.

Output columns, named so they match what ``_flatten_annotations`` would produce
from ``isoform_annotations["tags"]`` — and the module writes that annotation too,
so the object graph and the frame cannot drift:

``isoform_tags_states``
    ``struct<n x bool>``, one field per code-fired tag. ``True`` / ``False`` /
    null, where null is not-evaluable.
``isoform_tags_citations``
    ``struct<n x double>`` — the single number each fired tag rests on. Null for
    boolean and derived tags, which have no one number behind them.
``isoform_tags_registry_version``
    Flat string. A struct's *fields* change with the vocabulary, so two parquets
    built under different registry versions have different schemas; this is what
    lets a merge detect that instead of silently unioning two vocabularies.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import pyarrow as pa

from swissisoform.config import PipelineConfig, ScoringConfig
from swissisoform.models import TranslationInitiationSite
from swissisoform.tags import evaluate as tag_eval
from swissisoform.tags.registry import TagRegistry, TagRegistryError
from swissisoform.tags.registry import load as load_registry

logger = logging.getLogger(__name__)

STATES_COLUMN = "isoform_tags_states"
CITATIONS_COLUMN = "isoform_tags_citations"
VERSION_COLUMN = "isoform_tags_registry_version"


class TagModule:
    """Fire a frozen tag registry over a run's paired frame.

    Attributes:
        MODULE_NAME: ``"tags"``.
        OUTPUT_COLUMNS: Column names produced per TIS (already prefixed, since
            this module writes them onto the frame directly).
        SCOPE: ``"C"`` (per-candidate).
    """

    MODULE_NAME: str = "tags"
    OUTPUT_COLUMNS: list[str] = [STATES_COLUMN, CITATIONS_COLUMN, VERSION_COLUMN]
    SCOPE: str = "C"

    def __init__(self, registry: TagRegistry, config: PipelineConfig | None = None) -> None:
        """Hold the registry and the thresholds the derived scorers need."""
        self._registry = registry
        self._scoring: ScoringConfig = (
            config.scoring if config is not None and config.scoring else ScoringConfig()
        )

    @property
    def registry(self) -> TagRegistry:
        """The frozen registry this module fires."""
        return self._registry

    def annotate_frame(
        self,
        df: pd.DataFrame,
        sites: list[TranslationInitiationSite],
    ) -> pd.DataFrame:
        """Return *df* with the three tag columns appended.

        Also writes ``site.isoform_annotations["tags"]`` for each site, so code
        working from ``Gene`` objects sees the same values the parquet carries.

        Args:
            df: The paired frame, one row per TIS.
            sites: The TIS objects for those rows, in row order —
                ``[s for g in genes for s in g.tis_sites]``.
        """
        if not self._registry.code_fired():
            # An empty struct is a type Parquet cannot represent — the same hazard
            # paired_schema pins the clinical summaries against. A vocabulary with
            # nothing code-fired (all llm, all blocked) gets no columns at all.
            logger.warning(
                "Tags: registry %s has no code-fired tags; emitting no tag columns",
                self._registry.version,
            )
            return df
        states, citations = tag_eval.fire(df, sites, self._registry, self._scoring)
        state_rows, citation_rows = tag_eval.to_structs(states, citations)

        out = df.copy()
        out[STATES_COLUMN] = state_rows
        out[CITATIONS_COLUMN] = citation_rows
        out[VERSION_COLUMN] = self._registry.version

        for site, state, citation in zip(sites, state_rows, citation_rows, strict=False):
            site.isoform_annotations[self.MODULE_NAME] = {
                "states": state,
                "citations": citation,
                "registry_version": self._registry.version,
            }

        fired = sum(1 for row in state_rows for v in row.values() if v is True)
        evaluable = sum(1 for row in state_rows for v in row.values() if v is not None)
        logger.info(
            "Tags: registry %s, %d code-fired tags over %d isoforms "
            "(%d/%d evaluable tag-slots fired)",
            self._registry.version,
            len(self._registry.code_fired()),
            len(df),
            fired,
            evaluable,
        )
        return out

    def summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """Per-tag fire and not-evaluable counts, for logging or a spot check."""
        if STATES_COLUMN not in df.columns:
            return {}
        rows = list(df[STATES_COLUMN])
        out: dict[str, Any] = {}
        for tag in self._registry.code_fired():
            values = [r.get(tag.tag_id) for r in rows]
            out[tag.tag_id] = {
                "on": sum(1 for v in values if v is True),
                "off": sum(1 for v in values if v is False),
                "not_evaluable": sum(1 for v in values if v is None),
            }
        return out


def schema_overrides(df: pd.DataFrame) -> dict[str, pa.DataType]:
    """Arrow types for the tag structs, declared rather than inferred.

    A shard where one tag is null on every row would infer that field as ``null``
    and disagree with every other shard of the same campaign — the failure
    ``paired_schema`` already pins the clinical summaries against. The field names
    come from the registry the frame stamps, so the type is a property of the
    vocabulary rather than of the rows that happened to land here.

    Returns an empty dict when the frame carries no tags, so a run without a
    registry needs no special casing at the call site.
    """
    if VERSION_COLUMN not in df.columns or df.empty:
        return {}
    version = str(df[VERSION_COLUMN].iloc[0])
    try:
        reg = load_registry(version)
    except TagRegistryError:  # pragma: no cover - the frame was written with it
        logger.warning("Tags: registry %s vanished; leaving the struct types inferred", version)
        return {}
    names = reg.state_columns()
    return {
        STATES_COLUMN: pa.struct([pa.field(n, pa.bool_()) for n in names]),
        CITATIONS_COLUMN: pa.struct([pa.field(n, pa.float64()) for n in names]),
    }


__all__ = [
    "CITATIONS_COLUMN",
    "STATES_COLUMN",
    "VERSION_COLUMN",
    "TagModule",
    "schema_overrides",
]
