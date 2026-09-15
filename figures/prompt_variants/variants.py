"""The arm matrix, as data.

Four groundings x two hint levels. Kept in one place so the driver and anything
that later reads the corpora cannot disagree about what an arm was.

The two axes are not independent, and the write-up has to say so. ``-hint``
removes the system Directionality block everywhere; on top of that it removes
per-member interpretation hints in ``criteria`` (~18 strings) and in ``tags``
(one ``means`` per criterion-backed tag, 14), while ``raw`` and ``dist`` have
none to remove. That asymmetry is a property of the framings, not a bug — but
it means "the hint effect" is not one number across the matrix, and it is not
even the same *kind* of difference in the two arms that have one.
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

# The noise-floor arm: `criteria_hint` re-run under an identical framing, so its
# disagreement with `criteria_hint` measures sampling variance alone. `--temperature`
# is silently dropped on claude-sonnet-5 and thinking is force-disabled, so there is
# no knob for this — the only way to get the number is to run the arm twice.
#
# Kept out of `VARIANTS` on purpose: `select(None)` returns `VARIANTS`, so appending
# here would silently turn every existing "run all arms" command into a 9-arm run.
# It is reachable only by name.
REPLICATES: tuple[Variant, ...] = (
    Variant(
        arm_id="criteria_hint_rep",
        grounding="criteria",
        hints=True,
        note="noise-floor replicate of criteria_hint",
    ),
)

BY_ID: dict[str, Variant] = {v.arm_id: v for v in VARIANTS + REPLICATES}


def select(names: list[str] | None) -> list[Variant]:
    """The named arms, or every arm of the matrix when *names* is empty.

    "Every arm" is `VARIANTS` — the 8 real arms — never the replicates.

    Raises:
        KeyError: An unknown arm id — a typo would otherwise silently run nothing.
    """
    if not names:
        return list(VARIANTS)
    unknown = [n for n in names if n not in BY_ID]
    if unknown:
        raise KeyError(f"unknown arm(s) {unknown}; known: {sorted(BY_ID)}")
    return [BY_ID[n] for n in names]


__all__ = ["BY_ID", "GROUNDINGS", "REPLICATES", "VARIANTS", "Variant", "select"]
