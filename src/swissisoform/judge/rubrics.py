"""The scoring rubric, loaded from its text file rather than written in Python.

``scripts/judge/prompts/pairwise.txt`` *is* what the model sees. Keeping it as
text means a wording change is reviewable as a diff of the words, with no Python
quoting or line-wrapping in the way -- the same reason ``scripts/site/prompts/``
holds the interpretation prompts.

One axis, and every question it asks has a referent the judge can locate in the
payload: is this number present, is this field marked not_evaluable, do these
sentences relate two measurements or merely list them. The file opens with "Judge
on substance, NOT style or length" and carries a tiebreaker making overreach a
defect rather than richness.

It is gated on the hand-written terse/verbose pairs in ``pairwise_anchors/``,
equally supported by construction so that only economy can decide them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "scripts" / "judge" / "prompts"

PAIRWISE_ID = "PW_overall"

# A guard, not a parser: relative grading emits A or B, so a file carrying
# ``Score 1:``..``Score 5:`` lines was written for a different kind of grading.
_SCORE_LINE = re.compile(r"^Score ([1-5]):\s*(.+)$", re.M)


class RubricError(RuntimeError):
    """A rubric file is missing or malformed."""


@dataclass(frozen=True)
class Rubric:
    """One scoring axis, as loaded from its file."""

    id: str
    criterion: str
    path: Path

    def render(self) -> str:
        """The rubric exactly as the judge should receive it.

        Read back from the file rather than reassembled, so what a reviewer reads
        in the diff is what the model gets.
        """
        return self.path.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=None)
def load(rubric_id: str, prompts_dir: Path | None = None) -> Rubric:
    """Load one rubric by id.

    Raises:
        RubricError: The file is absent, or it declares ``Score 1:``..``Score 5:``
            lines, which relative grading has no use for.
    """
    root = prompts_dir or PROMPTS_DIR
    if rubric_id != PAIRWISE_ID:
        raise RubricError(f"unknown rubric {rubric_id!r}; only {PAIRWISE_ID!r} exists")
    path = root / "pairwise.txt"
    if not path.exists():
        raise RubricError(f"no rubric file for {rubric_id!r} at {path}")

    text = path.read_text(encoding="utf-8").strip()
    if _SCORE_LINE.search(text):
        raise RubricError(
            f"{path} declares Score lines, but relative grading emits A or B, not a 1-5 score"
        )
    return Rubric(id=rubric_id, criterion=text, path=path)


def pairwise(prompts_dir: Path | None = None) -> Rubric:
    """The single composite axis used for relative grading."""
    return load(PAIRWISE_ID, prompts_dir)


def by_id(rubric_id: str, prompts_dir: Path | None = None) -> Rubric:
    """One rubric by id, for reading an existing results file."""
    return load(rubric_id, prompts_dir)


def all_ids() -> tuple[str, ...]:
    """Every active rubric id."""
    return (PAIRWISE_ID,)
