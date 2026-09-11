"""The arm matrix, as data.

Four groundings x two hint levels. Kept in one place so the driver and anything
that later reads the corpora cannot disagree about what an arm was.

The two axes are not independent, and the write-up has to say so: in ``criteria``,
``-hint`` removes the system Directionality block *and* ~18 per-member
``interpretation_hint`` strings, while in the other three groundings there are no
per-member hints to remove, so only the system block goes. That asymmetry is a
property of the framings, not a bug — but it means "the hint effect" is not one
number across the matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

GROUNDINGS: tuple[str, ...] = ("criteria", "raw", "tags", "dist")


@dataclass(frozen=True)
class Variant:
    """One arm of the matrix."""

    arm_id: str
    grounding: str
    hints: bool
    note: str = ""

    @property
    def out_run(self) -> str:
        """Run-directory suffix, so each arm's outputs are isolated."""
        return self.arm_id

    def capture_dir(self, corpus: str) -> str:
        """Prompt-capture subdirectory name.

        Namespaced by corpus as well as arm: ``_default_prompt_dir`` would derive
        it from the run alone, so two corpora would collide and the second would
        truncate the first's ``index_category.json``.
        """
        return f"{corpus}__{self.arm_id}"


VARIANTS: tuple[Variant, ...] = tuple(
    Variant(
        arm_id=f"{g}_{'hint' if h else 'nohint'}",
        grounding=g,
        hints=h,
        note="status quo" if (g == "criteria" and h) else "",
    )
    for g in GROUNDINGS
    for h in (True, False)
)

BY_ID: dict[str, Variant] = {v.arm_id: v for v in VARIANTS}


def select(names: list[str] | None) -> list[Variant]:
    """The named arms, or every arm when *names* is empty.

    Raises:
        KeyError: An unknown arm id — a typo would otherwise silently run nothing.
    """
    if not names:
        return list(VARIANTS)
    unknown = [n for n in names if n not in BY_ID]
    if unknown:
        raise KeyError(f"unknown arm(s) {unknown}; known: {sorted(BY_ID)}")
    return [BY_ID[n] for n in names]


__all__ = ["BY_ID", "GROUNDINGS", "VARIANTS", "Variant", "select"]
