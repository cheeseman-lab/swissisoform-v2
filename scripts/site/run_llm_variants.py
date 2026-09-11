r"""Run the category pass under each prompt-variant arm, capturing inputs + outputs.

Eight arms: four groundings (criteria / raw / tags / dist) x hints on/off. Each
gets its own prompt root, its own output directory and its own capture corpus, so
nothing an arm writes can be mistaken for another's.

Logic lives in ``swissisoform.site.grounding`` (the payload builders),
``figures/prompt_variants/assemble.py`` (the prompt splice) and
``figures/prompt_variants/variants.py`` (the matrix). This file is the driver.

``swissisoform.site.llm`` is **not** modified: ``llm.main`` already takes
``prompts_dir`` as a keyword, and the alternative payload reaches ``slice_category``
through ``evidence.use_category_body``. Both are installed here and restored in a
``finally``, so one arm cannot leak into the next.

Usage:
    # free: writes every arm's prompt corpus, makes no API calls
    python scripts/site/run_llm_variants.py --dry-run

    # one arm, one gene, live — the cheap smoke test
    python scripts/site/run_llm_variants.py --arm tags_hint --gene CBX1

    # one arm, full corpus (submit eight of these, one per Slurm job)
    python scripts/site/run_llm_variants.py --arm dist_nohint
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "figures" / "prompt_variants"))

import assemble  # noqa: E402
import variants as variants_mod  # noqa: E402

from swissisoform.site import evidence, grounding, llm  # noqa: E402
from swissisoform.tags import registry as reg_mod  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_llm_variants")

BASE_PROMPTS = ROOT / "scripts" / "site" / "prompts"
DEFAULT_CORPUS = "cheeseman50"
# Categories whose verdict is emitted through a tool loop, and so whose terminal
# tool schema must be extended for the judgment tags to be recordable at all.
TOOL_CATEGORIES: tuple[str, ...] = ("M", "P")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Driver options; everything not listed here is fixed by the matrix."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--corpus", default=DEFAULT_CORPUS, help="Run name under data/output/")
    p.add_argument(
        "--arm",
        action="append",
        dest="arms",
        help="Arm id to run; repeatable. Default: every arm.",
    )
    p.add_argument("--gene", default=None, help="Restrict to one gene (smoke test)")
    p.add_argument("--model", default=None, help="Override the model")
    p.add_argument("--dry-run", action="store_true", help="Capture prompts, make no API calls")
    p.add_argument("--batch", action="store_true", help="Message Batches API for the 4 single-shot")
    p.add_argument("--tag-registry", default=grounding.DEFAULT_TAG_VERSION)
    p.add_argument("--dist-version", default=grounding.DEFAULT_DIST_VERSION)
    p.add_argument(
        "--prompts-root",
        type=Path,
        default=ROOT / "data" / "llm_prompts",
        help="Where assembled per-arm prompt directories are written",
    )
    p.add_argument("--force", action="store_true", help="Re-run arms whose outputs exist")
    return p.parse_args(argv)


def _verdict_extras(reg: reg_mod.TagRegistry) -> dict[str, dict]:
    """``{letter: {tags_fired: schema}}`` for the tool-loop categories."""
    out: dict[str, dict] = {}
    for letter in TOOL_CATEGORIES:
        fields = grounding.verdict_extra_fields(reg, letter)
        if fields:
            out[letter] = fields
    return out


def _install_tool_extras(extras: dict[str, dict]) -> list:
    """Splice ``tags_fired`` into both tool lists; return the restore callables.

    ``emit_verdict`` is strict with ``additionalProperties: false``, so without
    this the model cannot emit a tag decision at all and the judgment tags would
    be answerable only as prose.
    """
    from swissisoform.site import structure_tools, tools

    restores = []
    for letter, tool_list in (("M", tools.M_TOOLS), ("P", structure_tools.P_TOOLS)):
        if letter in extras:
            restores.append(grounding.install_verdict_extras(tool_list, extras[letter]))
    return restores


def run_arm(variant: variants_mod.Variant, args: argparse.Namespace) -> int:
    """Assemble one arm's prompts, install its grounding, and run the pass."""
    out_dir = ROOT / "data" / "output" / f"{args.corpus}_{variant.out_run}" / "llm"
    records = ROOT / "data" / "output" / args.corpus / "llm_evidence"
    variants_long = ROOT / "data" / "output" / args.corpus / "variants_long.parquet"
    if not records.is_dir():
        raise SystemExit(
            f"no evidence records at {records}. Build them first:\n"
            f"  python scripts/site/build_evidence_records.py "
            f"--parquet data/output/{args.corpus}/all_paired.parquet "
            f"--out {records}/ --variants-long-out {variants_long}"
        )

    reg = reg_mod.load(args.tag_registry)
    extras = _verdict_extras(reg) if variant.grounding == "tags" else {}
    prompts = assemble.materialize(
        args.prompts_root / variant.arm_id,
        grounding=variant.grounding,
        hints=variant.hints,
        base_dir=BASE_PROMPTS,
        verdict_extras=extras,
    )

    builder = grounding.build(
        variant.grounding,
        dist_version=args.dist_version,
        tag_version=args.tag_registry,
    )
    if variant.grounding == "criteria" and not variant.hints:
        # The only arm with per-member hints to remove — the other three carry
        # none, which is exactly why the hint axis is not orthogonal to grounding.
        # The builder unsets the hook before recursing so it runs the real
        # criteria path rather than itself, then drops the identity keys
        # `slice_category` re-adds around whatever a builder returns.
        def builder(record, category):
            previous = evidence.use_category_body(None)
            try:
                full = evidence.slice_category(record, category)
            finally:
                evidence.use_category_body(previous)
            return {
                k: v
                for k, v in grounding.strip_hints(full).items()
                if k not in ("category", "name", "isoform")
            }

    argv = [
        "--pass",
        "category",
        "--records",
        str(records),
        "--out",
        str(out_dir),
        "--save-prompts",
        "--save-prompts-dir",
        str(llm.DEFAULT_PROMPT_DIR / variant.capture_dir(args.corpus)),
        "--variants-long",
        str(variants_long),
    ]
    if args.gene:
        argv += ["--gene", args.gene]
    if args.model:
        argv += ["--model", args.model]
    if args.dry_run:
        argv.append("--dry-run")
    if args.batch:
        argv.append("--batch")
    # A re-run after a prompt edit would otherwise skip every isoform whose
    # categories.json exists and print "0/0 successful" (llm.py:2211).
    if args.force or args.dry_run:
        argv.append("--force")

    logger.info(
        "arm %s: grounding=%s hints=%s -> %s",
        variant.arm_id,
        variant.grounding,
        variant.hints,
        out_dir,
    )
    previous_body = evidence.use_category_body(builder)
    restores = _install_tool_extras(extras)
    try:
        return llm.main(argv, prompts_dir=prompts) or 0
    finally:
        for restore in restores:
            restore()
        evidence.use_category_body(previous_body)


def main(argv: list[str] | None = None) -> int:
    """Run the selected arms in sequence."""
    args = parse_args(argv)
    try:
        selected = variants_mod.select(args.arms)
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc

    rc = 0
    for variant in selected:
        rc |= run_arm(variant, args)
    print(f"\n{len(selected)} arm(s) complete (rc={rc})")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
