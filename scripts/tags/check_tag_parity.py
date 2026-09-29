"""Do a run's criterion tags agree with its scored criteria?

The gate that must pass before any cutoff is allowed to move. Reads one
``all_paired.parquet`` and compares, row for row, each ``derived`` tag's state
against the ``isoform_scoring_criteria`` entry for the criterion it names. Both
columns were produced by the same run from the same annotations, so a
disagreement is a wiring bug — a tag pointed at the wrong criterion, or a
``valid_for`` blanking rows the scorer could evaluate — and never a calibration
finding.

Run it with a ``--cutoffs config`` registry, where the tag layer is meant to
reproduce ``EvidenceScoringModule`` exactly. Under a ``--cutoffs distribution``
registry the criterion cutoffs have deliberately moved, so disagreement is the
*point*; ``--expect-diff`` says so and reports the movement per criterion instead
of failing on it.

Usage:
    python scripts/tags/check_tag_parity.py --run cheeseman50
    python scripts/tags/check_tag_parity.py --run cheeseman50 --expect-diff
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from swissisoform.modules.tags import STATES_COLUMN, VERSION_COLUMN
from swissisoform.tags.registry import KIND_DERIVED, load

ROOT = Path(__file__).resolve().parents[2]
CRITERIA_COLUMN = "isoform_scoring_criteria"


def _resolve(run: str | None, parquet: Path | None) -> Path:
    if parquet is not None:
        return parquet
    path = ROOT / "data" / "output" / (run or "") / "all_paired.parquet"
    if not path.exists():
        raise SystemExit(f"no parquet at {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    """Compare derived tags against scored criteria on one run."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", help="Run name under data/output/")
    p.add_argument("--parquet", type=Path, default=None, help="Explicit parquet path")
    p.add_argument(
        "--expect-diff",
        action="store_true",
        help="Report movement instead of failing (for a calibrated registry)",
    )
    args = p.parse_args(argv)

    path = _resolve(args.run, args.parquet)
    df = pd.read_parquet(path, columns=[STATES_COLUMN, CRITERIA_COLUMN, VERSION_COLUMN])
    version = str(df[VERSION_COLUMN].iloc[0])
    reg = load(version)
    states = pd.DataFrame(list(df[STATES_COLUMN]))
    criteria = pd.DataFrame(list(df[CRITERIA_COLUMN]))

    print(f"{path.parent.name}: {len(df)} isoforms, registry {version}\n")
    print(f"{'criterion':34s} {'agree':>6s} {'differ':>7s}   first difference")
    total = 0
    for tag in reg.by_kind(KIND_DERIVED):
        if tag.tag_id not in states or tag.criterion_id not in criteria:
            print(f"{tag.criterion_id:34s} {'-':>6s} {'MISSING':>7s}   {tag.tag_id} not in parquet")
            total += len(df)
            continue
        left = states[tag.tag_id].astype("boolean")
        right = criteria[tag.criterion_id].astype("boolean")
        # Null is a third value here, not "unknown": two nulls agree.
        same = (left.isna() & right.isna()) | (left == right).fillna(False)
        n_diff = int((~same).sum())
        total += n_diff
        note = ""
        if n_diff:
            i = df.index[~same][0]
            note = f"row {i}: tag={left[i]!r} criterion={right[i]!r}"
        print(f"{tag.criterion_id:34s} {int(same.sum()):6d} {n_diff:7d}   {note}")

    if total == 0:
        print("\nPARITY OK — every derived tag reproduces its criterion.")
        return 0
    if args.expect_diff:
        print(f"\n{total} differing tag-slots — expected for a calibrated registry.")
        return 0
    print(
        f"\nFAIL: {total} differing tag-slots. Both sides came from the same run, so "
        "this is a wiring bug, not a calibration result."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
