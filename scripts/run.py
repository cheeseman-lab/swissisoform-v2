"""SwissIsoform v2 — unified end-to-end pipeline driver.

Single unified entry point for the SwissIsoform v2 pipeline.
Runs the full annotation + comparison + scoring pipeline on an
arbitrary gene subset, isoform subset, or the full catalog.

Usage examples:

    # Smoke test on 5 diagnostic genes (HeLa-only single sample)
    python scripts/run.py --preset 5gene

    # 12 reviewer-picked genes across all 6 cell lines
    python scripts/run.py --genes CBX1 CDC34 ... --run-name 12gene_manual

    # Restrict to specific isoforms in a parquet (join keys:
    # Tid + GenomePos + StartCodon)
    python scripts/run.py --isoforms manual_combined.parquet --run-name 12gene_isoforms

    # Full catalog
    python scripts/run.py --all --run-name full_catalog

    # Skip GPU-dependent modules cleanly when caches are empty
    python scripts/run.py --genes TP53 --skip-modules plm_vep,structure

Output is written to data/output/<run_name>/ as per-gene + combined
paired parquets.

This is a thin preset/gene-list front-end: it resolves the input mode into a
``RunSpec`` and delegates the Stage 1–7 orchestration to
``swissisoform.runner``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from swissisoform import runner
from swissisoform.references import (
    ALL_CELL_LINES,
    ALL_GENE_MODULES,
    ALL_PROTEIN_MODULES,
    ALL_SITE_MODULES,
    PRESETS,
    ROOT,
)
from swissisoform.runner import RunSpec, load_isoform_picks, restrict_to_isoforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("run")


# ── CLI ──────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SwissIsoform v2 — unified end-to-end pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--preset", choices=sorted(PRESETS),
        help="Named run from presets/<name>.toml (auto-discovered)",
    )
    mode.add_argument("--genes", nargs="+", help="Gene symbols to analyze")
    mode.add_argument("--gene-list", type=Path, help="File with one gene symbol per line")
    mode.add_argument(
        "--isoforms", type=Path,
        help="Parquet/CSV with specific isoforms (join keys: Tid, GenomePos, StartCodon)",
    )
    mode.add_argument("--all", action="store_true", help="Run on the full catalog (every gene)")

    p.add_argument(
        "--cell-lines", default=",".join(ALL_CELL_LINES),
        help=f"Comma-separated cell lines (default: all 6 = {','.join(ALL_CELL_LINES)})",
    )
    p.add_argument(
        "--single-sample", action="store_true",
        help="Single-sample mode (HeLa-only, raw predict file). "
             "Otherwise uses the multi-cell-line combined parquet.",
    )

    p.add_argument(
        "--run-name", default=None,
        help="Output subdirectory under data/output/ (default: auto-derived)",
    )

    p.add_argument(
        "--skip-modules", default="",
        help="Comma-separated module names to skip (e.g. 'plm_vep,structure')",
    )
    p.add_argument(
        "--no-gpu", action="store_true",
        help="Convenience flag: skip PLM VEP + Structure (= --skip-modules plm_vep,structure)",
    )

    p.add_argument(
        "--min-cell-lines", type=int, default=None,
        help="ScoringConfig.min_cell_lines (default: 1 single-sample, 3 multi-sample)",
    )
    p.add_argument(
        "--rebuild-combined", action="store_true",
        help="Force rebuild of the cached all_samples_combined.parquet",
    )
    p.add_argument(
        "--no-spot-check-limit", action="store_true",
        help="Print every TIS in the spot check (default caps at 5 per gene)",
    )
    p.add_argument(
        "--emit-fasta", action="store_true",
        help="Write the proteins FASTA for the selected genes and exit, before "
             "precompute/annotation (seeds the GPU jobs run_plm_embed / run_fold "
             "without a full run)",
    )
    p.add_argument(
        "--fasta-out", type=Path, default=ROOT / "data" / "cache" / "proteins.fa",
        help="Path for the deduped proteins FASTA (default data/cache/proteins.fa); "
             "set per-run so concurrent runs don't clobber each other",
    )

    return p.parse_args(argv)


def resolve_gene_selection(
    args: argparse.Namespace,
    combined: pd.DataFrame | None,
) -> tuple[list[str] | None, pd.DataFrame | None]:
    """Resolve the input mode into (gene_names, restricted_df).

    gene_names: list of HGNC symbols, or None for --all.
    restricted_df: pre-filtered combined df (when --isoforms is used), else None.
    """
    if args.preset:
        spec = PRESETS[args.preset]
        if "isoforms" in spec:
            if combined is None:
                raise RuntimeError(
                    f"preset {args.preset!r} selects isoforms but no combined "
                    "catalog is loaded (isoform presets run multi-sample)"
                )
            restricted, gene_names = restrict_to_isoforms(
                combined, load_isoform_picks(spec["isoforms"]),
            )
            return gene_names, restricted
        return spec["genes"], None
    if args.genes:
        return args.genes, None
    if args.gene_list:
        names = [
            line.strip() for line in args.gene_list.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        return names, None
    if args.isoforms:
        if combined is None:
            raise RuntimeError(
                "--isoforms requires multi-sample mode (combined parquet must exist)"
            )
        restricted, gene_names = restrict_to_isoforms(
            combined, load_isoform_picks(args.isoforms),
        )
        return gene_names, restricted
    if args.all:
        return None, None
    if "5gene" in PRESETS:
        logger.info("No input mode specified; defaulting to --preset 5gene")
        args.preset = "5gene"
        return PRESETS["5gene"]["genes"], None
    raise RuntimeError("No input mode specified and no '5gene' preset available")


def derive_run_name(args: argparse.Namespace, gene_names: list[str] | None) -> str:
    """Auto-derive a run name from the input mode."""
    if args.run_name:
        return args.run_name
    if args.preset:
        return PRESETS[args.preset].get("run_name", args.preset)
    if args.all:
        return "all_samples"
    if args.isoforms:
        return f"{args.isoforms.stem}_e2e"
    if gene_names is not None:
        n = len(gene_names)
        if n <= 3:
            return "_".join(gene_names) + "_e2e"
        return f"{n}gene_e2e"
    return "e2e"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    skip = {m.strip() for m in args.skip_modules.split(",") if m.strip()}
    if args.no_gpu:
        skip |= {"plm_vep", "structure"}
    known_modules = set(ALL_PROTEIN_MODULES) | set(ALL_SITE_MODULES) | set(ALL_GENE_MODULES)
    unknown = skip - known_modules
    if unknown:
        logger.error("Unknown module(s) in --skip-modules: %s", sorted(unknown))
        logger.error("Available: %s", sorted(known_modules))
        return 2
    if skip:
        logger.info("Skipping modules: %s", sorted(skip))

    preset = PRESETS[args.preset] if args.preset else None
    preset_is_isoform = preset is not None and "isoforms" in preset

    cell_lines = [cl.strip() for cl in args.cell_lines.split(",") if cl.strip()]
    if preset is not None and "cell_lines" in preset:
        cell_lines = preset["cell_lines"]
    elif preset_is_isoform:
        cell_lines = ALL_CELL_LINES

    # Isoform presets must run multi-sample (they restrict the combined catalog).
    single_sample = not preset_is_isoform and (
        args.single_sample or (preset is not None and len(cell_lines) == 1)
    )

    if args.min_cell_lines is not None:
        min_cl = args.min_cell_lines
    elif preset is not None and "min_cell_lines" in preset:
        min_cl = preset["min_cell_lines"]
    else:
        min_cl = 1 if single_sample else 3

    # Isoform selection requires the combined catalog up front; for those modes,
    # load it here so resolve_gene_selection can restrict against it.  Other
    # modes resolve gene_names without touching the catalog and let runner.prepare
    # load samples.
    combined: pd.DataFrame | None = None
    needs_combined = not single_sample and (
        preset_is_isoform or (not preset and bool(args.isoforms))
    )
    if needs_combined:
        ref = runner.UpstreamReference.load(
            gtf_path=runner.GTF, genome_fasta=runner.GENOME, protein_fasta=runner.PROTEIN
        )
        from swissisoform.references import build_config

        combined = runner.load_combined(
            cell_lines, ref, build_config(),
            force_rebuild=args.rebuild_combined,
        )

    gene_names, restricted_df = resolve_gene_selection(args, combined)
    run_name = derive_run_name(args, gene_names)

    spec = RunSpec(
        gene_names=gene_names,
        restricted_df=restricted_df,
        cell_lines=cell_lines,
        single_sample=single_sample,
        min_cell_lines=min_cl,
        skip=skip,
        run_name=run_name,
        fasta_out=args.fasta_out,
        emit_fasta=args.emit_fasta,
        out_dir=None,
        spot_check_limit=None if args.no_spot_check_limit else 5,
        rebuild_combined=args.rebuild_combined,
    )
    return runner.run(spec)


if __name__ == "__main__":
    sys.exit(main())
