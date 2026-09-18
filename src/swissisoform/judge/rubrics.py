"""Score rubrics, loaded from text files rather than written in Python.

``scripts/judge/prompts/`` holds one file per axis. The file *is* what the model
sees: criterion prose, then ``Score 1:``..``Score 5:`` lines for the absolute
axes. Keeping them as text means a wording change is reviewable as a diff of the
words, with no Python quoting or line-wrapping in the way -- the same reason
``scripts/site/prompts/`` holds the interpretation prompts.

**Why the wording is shaped the way it is.** The first version of these rubrics
could not fail an obviously bad answer. Measured against verdicts built to be
wrong in exactly one way:

  - a calibration axis scored 4/5 a verdict calling a marginal metric "decisive
    beyond reasonable doubt" while declining to weigh the conflicting inputs
  - a directionality axis scored 4, then 4, then 5/5 across three wordings a
    verdict calling a uoORF a C-terminal truncation

Both were removed. The surviving axes ask questions with a referent the judge can
locate in the payload -- is this number present, is this field marked
not_evaluable -- and every file now opens with "Judge on substance, NOT style or
length", because the failures all rewarded fluent, confident prose. The pairwise
rubric additionally carries a tiebreaker making overreach a defect rather than
richness, which is the specific thing the anchors caught it rewarding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from swissisoform.judge import SYNTHESIS_UNIT

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "scripts" / "judge" / "prompts"

# Active axes, in the order they are scored. Two category axes, not four -- see
# the module docstring for what was removed and why.
CATEGORY_IDS: tuple[str, ...] = ("R1_evidential_support", "R2_not_evaluable_discipline")
SYNTHESIS_IDS: tuple[str, ...] = (
    "SY1_coherence",
    "SY2_no_new_claims",
    "SY3_hypothesis_quality",
)
PAIRWISE_ID = "PW_overall"

_SCORE_LINE = re.compile(r"^Score ([1-5]):\s*(.+)$", re.M)


class RubricError(RuntimeError):
    """A rubric file is missing or malformed."""


@dataclass(frozen=True)
class Rubric:
    """One scoring axis, as loaded from its file."""

    id: str
    criterion: str
    levels: tuple[str, ...]
    path: Path

    def render(self) -> str:
        """The rubric exactly as the judge should receive it.

        For an absolute axis this is the file's own text: criterion then the five
        score lines. Rendering from the file rather than reassembling it means what
        a reviewer reads in the diff is what the model gets.
        """
        return self.path.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=None)
def load(rubric_id: str, prompts_dir: Path | None = None) -> Rubric:
    """Load one rubric by id.

    Raises:
        RubricError: The file is absent, or an absolute rubric does not declare
            exactly five score levels. Checked because a rubric silently missing a
            level still produces scores -- just on a scale the levels do not
            describe.
    """
    root = prompts_dir or PROMPTS_DIR
    if rubric_id == PAIRWISE_ID:
        path = root / "pairwise.txt"
    else:
        path = root / "absolute" / f"{rubric_id}.txt"
    if not path.exists():
        raise RubricError(f"no rubric file for {rubric_id!r} at {path}")

    text = path.read_text(encoding="utf-8").strip()
    found = dict(_SCORE_LINE.findall(text))
    criterion = _SCORE_LINE.split(text)[0].strip() if found else text

    if rubric_id == PAIRWISE_ID:
        if found:
            raise RubricError(
                f"{path} declares Score lines, but relative grading emits A or B, not a 1-5 score"
            )
        return Rubric(id=rubric_id, criterion=criterion, levels=(), path=path)

    missing = [str(i) for i in range(1, 6) if str(i) not in found]
    if missing:
        raise RubricError(
            f"{path} is missing Score line(s) {missing}. A rubric short of a level "
            f"still returns scores, on a scale its own text does not describe."
        )
    return Rubric(
        id=rubric_id,
        criterion=criterion,
        levels=tuple(found[str(i)].strip() for i in range(1, 6)),
        path=path,
    )


def category_rubrics(prompts_dir: Path | None = None) -> tuple[Rubric, ...]:
    """The active axes for a category verdict."""
    return tuple(load(rid, prompts_dir) for rid in CATEGORY_IDS)


def synthesis_rubrics(prompts_dir: Path | None = None) -> tuple[Rubric, ...]:
    """The active axes for a synthesis output."""
    return tuple(load(rid, prompts_dir) for rid in SYNTHESIS_IDS)


def pairwise(prompts_dir: Path | None = None) -> Rubric:
    """The single composite axis used for relative grading."""
    return load(PAIRWISE_ID, prompts_dir)


def rubrics_for(unit: str, prompts_dir: Path | None = None) -> tuple[Rubric, ...]:
    """The absolute axes that apply to one output unit."""
    return (
        synthesis_rubrics(prompts_dir) if unit == SYNTHESIS_UNIT else category_rubrics(prompts_dir)
    )


def by_id(rubric_id: str, prompts_dir: Path | None = None) -> Rubric:
    """One rubric by id, for reading an existing results file."""
    return load(rubric_id, prompts_dir)


def all_ids() -> tuple[str, ...]:
    """Every active rubric id."""
    return (*CATEGORY_IDS, *SYNTHESIS_IDS, PAIRWISE_ID)
