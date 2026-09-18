"""Figure describing the judge corpus and what is measured on it.

Four panels, each answering a question the design rests on:

  A  What exists      -- 9 arms x 50 isoforms x 7 units = 3,150 outputs.
  B  What it costs    -- 108 judge calls per cell, 37,800 total.
  C  What can be seen -- observed framing effects against each category's own
                         noise floor. This is the panel that reorders the question.
  D  What fits        -- reference payload size per unit against the 32k context.

Run (needs matplotlib, so the dev env):
    python figures/judge_design/export_judge_design.py

Writes judge_design.png + judge_design.csv beside this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swissisoform.judge import ARMS, REPLICATE, UNITS  # noqa: E402

# Measured on cheeseman50: verdict disagreement between criteria_hint and its
# bit-identical replicate, per category. The 5x spread is the point -- a single
# pooled threshold would be wrong in both directions.
FLOOR = {"C": 4.0, "D": 6.0, "L": 6.0, "M": 16.0, "P": 18.0, "S": 20.0}

# Disagreement with criteria_hint, same corpus.
EFFECTS = {
    "hint": {"C": 10.0, "D": 16.0, "L": 16.0, "M": 16.0, "P": 30.0, "S": 38.0},
    "raw": {"C": 28.0, "D": 32.0, "L": 26.0, "M": 12.0, "P": 16.0, "S": 56.0},
    "tags": {"C": 12.0, "D": 16.0, "L": 34.0, "M": 30.0, "P": 14.0, "S": 30.0},
    "dist": {"C": 34.0, "D": 42.0, "L": 50.0, "M": 14.0, "P": 24.0, "S": 52.0},
}

# Reference payload, worst cell per unit, estimated tokens (measured).
REFERENCE_TOKENS = {
    "C": 5356,
    "D": 5827,
    "L": 5087,
    "M": 17381,
    "P": 4837,
    "S": 17067,
    "synthesis": 24374,
}
CONTEXT = 32768
MAX_NEW = 512

N_ISOFORMS = 50
CATS = ("C", "D", "L", "M", "P", "S")


def main() -> int:
    """Draw the figure and write the tidy CSV behind it."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    _panel_a(axes[0][0])
    _panel_b(axes[0][1])
    _panel_c(axes[1][0])
    _panel_d(axes[1][1])
    fig.suptitle(
        "Judging 9 prompt-variant arms with Prometheus-8x7B  |  corpus: cheeseman50",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = HERE / "judge_design.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    rows = []
    for cat in CATS:
        for name, series in EFFECTS.items():
            rows.append(
                {
                    "category": cat,
                    "contrast": name,
                    "disagreement_pct": series[cat],
                    "floor_pct": FLOOR[cat],
                    "floor_units": round(series[cat] / FLOOR[cat], 2),
                    "resolvable": series[cat] / FLOOR[cat] > 1.0,
                }
            )
    csv = HERE / "judge_design.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"wrote {csv}")
    return 0


def _panel_a(ax) -> None:
    """The corpus: every arm produced every unit for every isoform."""
    arms = [*ARMS, REPLICATE]
    grid = pd.DataFrame(N_ISOFORMS, index=arms, columns=list(UNITS))
    ax.imshow(grid.values, cmap="Blues", vmin=0, vmax=60, aspect="auto")
    ax.set_xticks(range(len(UNITS)))
    ax.set_xticklabels(UNITS)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels(arms, fontsize=8)
    for i in range(len(arms)):
        for j in range(len(UNITS)):
            ax.text(j, i, "50", ha="center", va="center", fontsize=7)
    ax.axhline(len(ARMS) - 0.5, color="crimson", lw=1.6)
    ax.text(
        len(UNITS) - 0.4,
        len(ARMS),
        "replicate:\nnoise floor only,\nnot a design cell",
        fontsize=7,
        color="crimson",
        va="center",
    )
    ax.set_title(
        "A. The corpus — 9 arms x 50 isoforms x 7 units = 3,150 outputs\n"
        "(complete: 0 missing cells)",
        fontsize=10,
        loc="left",
    )
    ax.set_xlabel("output unit (6 CDLMPS categories + synthesis)")


def _panel_b(ax) -> None:
    """Judge calls, and why they are counted per cell."""
    labels = ["absolute\n9 arms x 4 rubrics", "pairwise\n36 pairs x 2 orders"]
    values = [36, 72]
    bars = ax.barh(labels, values, color=["#4C72B0", "#DD8452"])
    for bar, value in zip(bars, values):
        ax.text(value + 1, bar.get_y() + bar.get_height() / 2, str(value), va="center")
    ax.set_xlim(0, 90)
    ax.set_xlabel("judge calls per cell")
    ax.set_title(
        "B. 108 calls per cell x 350 cells = 37,800\n"
        "Both presentation orders is mandatory: only pairs the judge\n"
        "decides the same way either round count as a verdict.",
        fontsize=10,
        loc="left",
    )
    ax.text(
        0.98,
        0.08,
        "a cell = (one isoform, one unit)\njudging never leaves a cell",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        bbox={"boxstyle": "round", "fc": "#f5f5f5", "ec": "grey"},
    )


def _panel_c(ax) -> None:
    """Effects in floor units. Below 1.0x nothing can be concluded."""
    width = 0.2
    positions = range(len(CATS))
    for offset, (name, series) in enumerate(EFFECTS.items()):
        ratios = [series[c] / FLOOR[c] for c in CATS]
        ax.bar(
            [p + (offset - 1.5) * width for p in positions],
            ratios,
            width,
            label=name,
        )
    ax.axhline(1.0, color="crimson", lw=1.8, ls="--")
    ax.text(
        -0.45,
        1.06,
        "1.0x = the pipeline's own run-to-run noise; below this, nothing is resolvable",
        color="crimson",
        fontsize=8,
    )
    ax.set_xticks(list(positions))
    ax.set_xticklabels([f"{c}\nfloor {FLOOR[c]:.0f}%" for c in CATS], fontsize=8)
    ax.set_ylabel("effect / that category's floor")
    ax.set_title(
        "C. What is resolvable — framing effects vs each category's own floor\n"
        "5 of 8 effects in M and P sit at or below it (the tool-loop categories).",
        fontsize=10,
        loc="left",
    )
    ax.legend(fontsize=8, ncol=4, loc="upper right")


def _panel_d(ax) -> None:
    """Reference payload against the context budget."""
    units = list(REFERENCE_TOKENS)
    values = [REFERENCE_TOKENS[u] for u in units]
    colours = ["#55A868" if v < 20000 else "#C44E52" for v in values]
    ax.bar(units, values, color=colours)
    # A pairwise call is the reference plus two responses, a rubric and the template.
    ax.axhline(CONTEXT, color="black", lw=1.6)
    ax.text(
        len(units) - 0.4,
        CONTEXT + 400,
        f"context {CONTEXT:,}",
        fontsize=8,
        ha="right",
        va="bottom",
    )
    usable = CONTEXT - MAX_NEW - 1400
    ax.axhline(usable, color="darkorange", lw=1.4, ls="--")
    ax.text(
        -0.35,
        usable - 900,
        "usable for the reference (after 2 responses + rubric + 512 new tokens)",
        fontsize=7,
        color="darkorange",
        va="top",
    )
    ax.set_ylim(0, CONTEXT * 1.1)
    for i, value in enumerate(values):
        ax.text(i, value + 700, f"{value // 1000}k", ha="center", fontsize=8)
    ax.set_ylabel("reference payload, worst cell (tokens)")
    ax.set_title(
        "D. What fits — M and synthesis needed fixing before any GPU time\n"
        "M 25.6k->17.4k (deduplicated a hit list carried twice);\n"
        "synthesis 50k->24.4k (dropped hit rows, kept n_hits_total).",
        fontsize=10,
        loc="left",
    )


if __name__ == "__main__":
    raise SystemExit(main())
