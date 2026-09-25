"""Run the deterministic pre-checks over the whole arm corpus.

No GPU, no API, no judge. Everything here is decidable by arithmetic, and the
results can change the rubrics -- so this runs before any Prometheus work.

Usage:
    python scripts/judge/run_checks.py
    python scripts/judge/run_checks.py --out data/output/judge/checks
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from swissisoform.judge import CATEGORY_LETTERS, DEFAULT_CORPUS, SYNTHESIS_UNIT  # noqa: E402
from swissisoform.judge import checks as K  # noqa: E402
from swissisoform.judge.corpus import completeness, load_corpus  # noqa: E402
from swissisoform.judge.reference import (  # noqa: E402
    ReferenceBuilder,
    isoform_records,
)
from swissisoform.site.llm import load_records  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("judge.checks")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI options."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--corpus", default=DEFAULT_CORPUS)
    p.add_argument("--out", type=Path, default=None, help="Where to write the report")
    p.add_argument("--limit", type=int, default=None, help="First N isoforms (smoke test)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run all three checks and write a JSON + TSV report."""
    args = parse_args(argv)
    out_dir = args.out or (ROOT / "data" / "output" / "judge" / args.corpus / "checks")
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus(args.corpus)
    meta = completeness(corpus)
    logger.info(
        "corpus %s: %d arms x %d isoforms, %d/%d outputs, %d incomplete cell(s)",
        args.corpus,
        meta["n_arms"],
        meta["n_isoforms"],
        meta["n_outputs"],
        meta["n_expected"],
        len(meta["incomplete_cells"]),
    )

    records = load_records(ROOT / "data" / "output" / args.corpus / "llm_evidence")
    isos = isoform_records(records)
    builder = ReferenceBuilder.build()

    slugs = list(corpus.slugs)[: args.limit] if args.limit else list(corpus.slugs)
    report = K.CheckReport()

    for i, slug in enumerate(slugs, start=1):
        record = isos.get(slug)
        if record is None:
            logger.warning("no evidence record for %s — skipping its cells", slug)
            continue
        # One reference per (slug, letter), reused across all 9 arms: the whole
        # point is that the arms are judged against the same evidence.
        refs = {letter: builder.category(record, letter) for letter in CATEGORY_LETTERS}

        for arm in corpus.arms:
            for letter in CATEGORY_LETTERS:
                out = corpus.get(arm, slug, letter)
                if out is None:
                    continue
                report.n_outputs += 1
                report.measure(arm=arm, unit=letter, reasoning=out.text)
                found = K.check_fabrication(
                    arm=arm,
                    slug=slug,
                    unit=letter,
                    reasoning=out.reasoning,
                    reference=refs[letter],
                ) + K.check_not_evaluable(
                    arm=arm,
                    slug=slug,
                    unit=letter,
                    reasoning=out.reasoning,
                    reference=refs[letter],
                )
                if found:
                    report.n_with_findings += 1
                for f in found:
                    report.add(f)

            syn = corpus.get(arm, slug, SYNTHESIS_UNIT)
            if syn is not None:
                report.n_outputs += 1
                report.measure(arm=arm, unit=SYNTHESIS_UNIT, reasoning=syn.text)
                found = K.check_synthesis_fields(arm=arm, slug=slug, payload=syn.payload)
                if found:
                    report.n_with_findings += 1
                for f in found:
                    report.add(f)

        if i % 10 == 0:
            logger.info("%d/%d isoforms", i, len(slugs))

    _write(report, meta, out_dir)
    _summarise(report, corpus.arms)
    return 0


def _write(report: K.CheckReport, meta: dict, out_dir: Path) -> None:
    """Report to JSON (machine) and TSV (eyeballing)."""
    by_check: dict[str, int] = {}
    for f in report.findings:
        by_check[f.check] = by_check.get(f.check, 0) + 1

    (out_dir / "report.json").write_text(
        json.dumps(
            {
                "corpus": meta,
                "n_outputs_checked": report.n_outputs,
                "n_outputs_with_findings": report.n_with_findings,
                "finding_rate": round(report.rate(), 4),
                "by_check": by_check,
                "by_arm": report.counts_by_arm,
                "by_unit": report.counts_by_unit,
                # The control for the economy criterion the pairwise rubric now
                # weighs: without it a BT shift cannot be told from drift.
                "words_by_arm": {
                    arm: K.word_stats(v) for arm, v in sorted(report.words_by_arm.items())
                },
                "words_by_arm_unit": {
                    k.replace("\t", "/"): K.word_stats(v)
                    for k, v in sorted(report.words_by_arm_unit.items())
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    lines = ["check\tarm\tslug\tunit\tdetail\tcontext"]
    for f in report.findings:
        detail = f.detail.replace("\t", " ")
        context = f.context.replace("\t", " ")
        lines.append(f"{f.check}\t{f.arm}\t{f.slug}\t{f.unit}\t{detail}\t{context}")
    (out_dir / "findings.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote %s", out_dir)


def _summarise(report: K.CheckReport, arms: tuple[str, ...]) -> None:
    """Print the numbers that decide whether the rubrics need changing."""
    by_check: dict[str, int] = {}
    for f in report.findings:
        by_check[f.check] = by_check.get(f.check, 0) + 1

    print(f"\n{report.n_outputs:,} outputs checked, {len(report.findings):,} finding(s)")
    print(f"{report.n_with_findings:,} output(s) with >=1 finding ({report.rate():.1%})\n")
    for check, n in sorted(by_check.items(), key=lambda kv: -kv[1]):
        print(f"  {check:26s} {n:>6,}")
    print("\nper arm:")
    for arm in arms:
        print(f"  {arm:20s} {report.counts_by_arm.get(arm, 0):>6,}")
    print("\nper unit:")
    for unit in (*CATEGORY_LETTERS, SYNTHESIS_UNIT):
        print(f"  {unit:20s} {report.counts_by_unit.get(unit, 0):>6,}")

    print("\nresponse length, words as the judge sees them (control, not a score):")
    print(f"  {'arm':20s} {'n':>6} {'median':>8} {'p90':>8}")
    for arm in arms:
        s = K.word_stats(report.words_by_arm.get(arm, []))
        print(f"  {arm:20s} {s['n']:>6,} {s['median']:>8.0f} {s['p90']:>8.0f}")


if __name__ == "__main__":
    raise SystemExit(main())
