"""Loading the arm corpus as cells.

One place that knows where an arm's outputs live and how to iterate them, so the
pre-checks, the request builder and the analysis cannot disagree about what a cell
is or which isoforms are in it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from swissisoform.judge import (
    ALL_ARMS,
    CATEGORY_NAMES,
    DEFAULT_CORPUS,
    SYNTHESIS_UNIT,
    UNITS,
)

ROOT = Path(__file__).resolve().parents[3]


class CorpusError(RuntimeError):
    """The corpus on disk is not what the analysis assumes."""


@dataclass(frozen=True)
class Output:
    """One arm's output for one cell."""

    arm: str
    slug: str
    unit: str
    verdict: str | None
    reasoning: str
    payload: dict[str, Any]

    @property
    def text(self) -> str:
        """What the judge is shown as the response.

        Verdict and reasoning only. ``evidence_used`` is excluded deliberately:
        it is non-empty in only ~40% of outputs in *every* arm, so including it
        would add noise that cannot separate arms.
        """
        if self.unit == SYNTHESIS_UNIT:
            parts = [
                f"headline: {self.payload.get('headline', '')}",
                f"divergence_hypothesis: {self.payload.get('divergence_hypothesis', '')}",
                f"function_relevance: {self.payload.get('function_relevance', '')}",
                f"tags: {', '.join(self.payload.get('tags') or []) or '(none)'}",
                f"confidence: {self.payload.get('confidence', '')}",
            ]
            if self.payload.get("caveats"):
                parts.append(f"caveats: {self.payload['caveats']}")
            return "\n".join(parts)
        return f"verdict: {self.verdict}\n\nreasoning: {self.reasoning}"


@dataclass
class Corpus:
    """Every arm's outputs, indexed by ``(arm, slug, unit)``."""

    name: str
    outputs: dict[tuple[str, str, str], Output]
    slugs: tuple[str, ...]
    arms: tuple[str, ...]
    gene_by_slug: dict[str, str] = field(default_factory=dict)
    tis_id_by_slug: dict[str, str] = field(default_factory=dict)

    def get(self, arm: str, slug: str, unit: str) -> Output | None:
        """One output, or None when that arm never produced it."""
        return self.outputs.get((arm, slug, unit))

    def cells(self) -> Iterator[tuple[str, str]]:
        """Every ``(slug, unit)`` cell, in a stable order."""
        for slug in self.slugs:
            for unit in UNITS:
                yield slug, unit

    def arms_for(self, slug: str, unit: str) -> list[str]:
        """Arms that produced an output in this cell.

        A cell missing an arm is dropped from that cell's comparisons rather than
        imputed -- a missing verdict is not a neutral one.
        """
        return [a for a in self.arms if (a, slug, unit) in self.outputs]


def arm_dir(arm: str, corpus: str = DEFAULT_CORPUS) -> Path:
    """Where one arm's per-isoform outputs live."""
    return ROOT / "data" / "output" / f"{corpus}_{arm}" / "llm"


def load_corpus(
    corpus: str = DEFAULT_CORPUS,
    arms: tuple[str, ...] = ALL_ARMS,
    *,
    require_complete: bool = True,
) -> Corpus:
    """Load every arm's outputs.

    Args:
        corpus: Run name under ``data/output/``.
        arms: Arm ids to load.
        require_complete: Raise when an arm is missing entirely. Individual
            missing cells are always tolerated (and reported by
            :func:`completeness`), because dropping one cell is recoverable while
            silently comparing 9 arms on 8 arms' data is not.

    Raises:
        CorpusError: An arm directory is absent, or no isoform was found at all.
    """
    outputs: dict[tuple[str, str, str], Output] = {}
    gene_by_slug: dict[str, str] = {}
    tis_by_slug: dict[str, str] = {}
    seen: set[str] = set()

    for arm in arms:
        base = arm_dir(arm, corpus)
        if not base.is_dir():
            if require_complete:
                raise CorpusError(
                    f"no outputs for arm {arm!r} at {base}. Run it, or pass a narrower `arms=`."
                )
            continue
        for iso_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            slug = iso_dir.name
            cats = iso_dir / "categories.json"
            if cats.exists():
                data = json.loads(cats.read_text(encoding="utf-8"))
                for name, entry in data.items():
                    letter = {v: k for k, v in CATEGORY_NAMES.items()}.get(name)
                    if letter is None:
                        continue
                    seen.add(slug)
                    outputs[(arm, slug, letter)] = Output(
                        arm=arm,
                        slug=slug,
                        unit=letter,
                        verdict=entry.get("verdict"),
                        reasoning=entry.get("reasoning") or "",
                        payload=entry,
                    )
            syn = iso_dir / "synthesis.json"
            if syn.exists():
                data = json.loads(syn.read_text(encoding="utf-8"))
                seen.add(slug)
                outputs[(arm, slug, SYNTHESIS_UNIT)] = Output(
                    arm=arm,
                    slug=slug,
                    unit=SYNTHESIS_UNIT,
                    verdict=data.get("confidence"),
                    reasoning=data.get("divergence_hypothesis") or "",
                    payload=data,
                )
                if data.get("tis_id"):
                    tis_by_slug[slug] = data["tis_id"]

    if not seen:
        raise CorpusError(f"no isoform outputs found for corpus {corpus!r} under {arms}")

    return Corpus(
        name=corpus,
        outputs=outputs,
        slugs=tuple(sorted(seen)),
        arms=tuple(a for a in arms if arm_dir(a, corpus).is_dir()),
        gene_by_slug=gene_by_slug,
        tis_id_by_slug=tis_by_slug,
    )


def completeness(corpus: Corpus) -> dict[str, Any]:
    """Which cells are short of arms, so a gap is stated rather than absorbed."""
    holes: list[dict[str, Any]] = []
    for slug, unit in corpus.cells():
        present = corpus.arms_for(slug, unit)
        if len(present) != len(corpus.arms):
            holes.append(
                {
                    "slug": slug,
                    "unit": unit,
                    "missing": sorted(set(corpus.arms) - set(present)),
                }
            )
    return {
        "n_arms": len(corpus.arms),
        "n_isoforms": len(corpus.slugs),
        "n_cells": len(corpus.slugs) * len(UNITS),
        "n_outputs": len(corpus.outputs),
        "n_expected": len(corpus.arms) * len(corpus.slugs) * len(UNITS),
        "incomplete_cells": holes,
    }
