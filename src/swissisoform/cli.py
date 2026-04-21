r"""Command-line entry point for the SwissIsoform v2 pipeline.

The CLI drives the full end-to-end flow:

    per-sample upstream → cross-cell-line combine → gene assembly
        → expensive-module precompute → annotation → comparator
        → paired parquet

Invocation::

    python -m swissisoform run \\
        --sample-manifest data/reference/ribotish_sample_manifest.csv \\
        --replicate-manifest data/reference/ribotish_replicate_manifest.csv \\
        --gtf data/reference/gencode.v49.primary_assembly.annotation.gtf \\
        --genome data/reference/Gencode_v49_GRCh38.primary_assembly.genome.fa \\
        --protein-fasta data/reference/gencode.v49.pc_translations.fa \\
        --output data/output/full_run \\
        --workers 8

The ``--workers`` flag controls two parallel sections: per-sample
upstream processing (one worker per cell line, up to ``--workers``) and
gene-level annotation after precompute (gene-parallel via
``ProcessPoolExecutor``). The expensive precompute phase (DeepLoc
batch, clinical variant prefetch) is itself single-shot
but batched across all unique proteins / genes, which is already the
bottleneck win compared to per-call invocation.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from swissisoform.assembly import assemble_genes
from swissisoform.clinical.validate import ConsequenceValidator
from swissisoform.combine import combine_filtered_samples
from swissisoform.compare import compare_genes
from swissisoform.config import (
    ClinicalConfig,
    ConservationConfig,
    PipelineConfig,
)
from swissisoform.io.parquet import paired_tis_dataframe
from swissisoform.models import Gene
from swissisoform.modules.biophysics import BiophysicsModule
from swissisoform.modules.clinical import ClinicalModule
from swissisoform.modules.conservation import ConservationModule
from swissisoform.modules.core_identity import CoreIdentityModule
from swissisoform.modules.initiation_context import InitiationContextModule
from swissisoform.modules.localization import LocalizationModule, precompute_deeploc
from swissisoform.modules.massspec import MassSpecModule
from swissisoform.modules.motifs import MotifsModule
from swissisoform.pipeline import AnnotationPipeline, UpstreamReference, run_sample

logger = logging.getLogger("swissisoform.cli")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="swissisoform",
        description="Annotate alternative translation initiation sites.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the full pipeline end-to-end.")
    run_p.add_argument(
        "--sample-manifest",
        type=Path,
        required=True,
        help="CSV with columns: sample,predict_file (one row per cell line).",
    )
    run_p.add_argument(
        "--replicate-manifest",
        type=Path,
        required=True,
        help="CSV with columns: sample,htseq_count_file (one row per replicate).",
    )
    run_p.add_argument("--gtf", type=Path, required=True, help="GENCODE GTF.")
    run_p.add_argument("--genome", type=Path, required=True, help="Primary-assembly genome FASTA.")
    run_p.add_argument(
        "--protein-fasta",
        type=Path,
        required=True,
        help="GENCODE pc_translations FASTA for canonical imputation.",
    )
    run_p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for paired parquet + logs.",
    )
    run_p.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help="Subset of samples to run (default: all samples in manifest).",
    )
    run_p.add_argument(
        "--genes",
        nargs="*",
        default=None,
        help="Subset of gene symbols to assemble (default: all).",
    )
    run_p.add_argument(
        "--gene-list",
        type=Path,
        default=None,
        help="Text file with one gene symbol per line (alternative to --genes).",
    )
    run_p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Max parallel workers for sample upstream + gene annotation.",
    )
    run_p.add_argument(
        "--phylop-bigwig",
        type=Path,
        default=None,
        help="Zoonomia 241-mammal PhyloP BigWig (enables conservation).",
    )
    run_p.add_argument(
        "--phastcons-bigwig",
        type=Path,
        default=None,
        help="Zoonomia 241-mammal PhastCons BigWig (enables conservation).",
    )
    run_p.add_argument(
        "--gnomad-db",
        type=Path,
        default=None,
        help="Local gnomAD parquet (enables offline gnomAD queries).",
    )
    run_p.add_argument(
        "--clinvar-db",
        type=Path,
        default=None,
        help="Local ClinVar parquet (enables offline ClinVar queries).",
    )
    run_p.add_argument(
        "--cosmic-db",
        type=Path,
        default=None,
        help="Local COSMIC parquet directory (enables offline COSMIC queries).",
    )
    run_p.add_argument(
        "--deeploc",
        action="store_true",
        help="Run DeepLoc2 for subcellular localization (slow; requires env).",
    )
    run_p.add_argument(
        "--no-comparator",
        action="store_true",
        help="Skip the comparator pass (emit raw paired annotations only).",
    )
    run_p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Emit DEBUG-level logs.",
    )
    return p


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------


def _resolve_path(raw: str, manifest_dir: Path) -> Path:
    """Resolve a manifest-referenced path.

    Absolute paths pass through.  Relative paths are resolved against the
    current working directory first (so a manifest listing
    ``data/reference/...`` works from the repo root), falling back to the
    manifest directory (so manifests with local-relative paths also
    work).
    """
    p = Path(raw)
    if p.is_absolute():
        return p
    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return cwd_candidate
    return manifest_dir / p


def _load_sample_manifest(path: Path) -> dict[str, Path]:
    """Read sample → predict_file mapping."""
    df = pd.read_csv(path)
    base = path.parent
    return {
        row["sample"]: _resolve_path(str(row["predict_file"]), base) for _, row in df.iterrows()
    }


def _load_replicate_manifest(path: Path) -> dict[str, list[Path]]:
    """Read sample → list-of-rnaseq-count-files mapping.

    Accepts either ``rnaseq_count_file`` (current schema) or
    ``htseq_count_file`` (legacy) column.
    """
    df = pd.read_csv(path)
    col = "rnaseq_count_file" if "rnaseq_count_file" in df.columns else "htseq_count_file"
    base = path.parent
    mapping: dict[str, list[Path]] = {}
    for _, row in df.iterrows():
        mapping.setdefault(row["sample"], []).append(_resolve_path(str(row[col]), base))
    return mapping


# ---------------------------------------------------------------------------
# Sample-level upstream (parallel across cell lines)
# ---------------------------------------------------------------------------


def _run_one_sample(
    sample: str,
    predict_file: Path,
    rnaseq_files: list[Path],
    gtf: Path,
    genome: Path,
    protein_fasta: Path,
) -> tuple[str, pd.DataFrame]:
    """Worker: run the upstream pipeline for a single sample.

    Builds its own ``UpstreamReference`` so it can be pickled across
    process boundaries without sharing file handles.
    """
    ref = UpstreamReference.load(gtf_path=gtf, genome_fasta=genome, protein_fasta=protein_fasta)
    final, _ = run_sample(predict_file, rnaseq_files, gtf, sample=sample, reference=ref)
    return sample, final


def _upstream_all_samples(
    sample_map: dict[str, Path],
    replicate_map: dict[str, list[Path]],
    gtf: Path,
    genome: Path,
    protein_fasta: Path,
    workers: int,
) -> dict[str, pd.DataFrame]:
    """Run ``run_sample`` across all samples in parallel."""
    results: dict[str, pd.DataFrame] = {}
    n_workers = min(workers, len(sample_map))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {
            ex.submit(
                _run_one_sample,
                sample,
                predict,
                replicate_map[sample],
                gtf,
                genome,
                protein_fasta,
            ): sample
            for sample, predict in sample_map.items()
        }
        for fut in as_completed(futures):
            sample = futures[fut]
            t0 = time.perf_counter()
            results[sample] = fut.result()[1]
            logger.info(
                "upstream %s: %d rows in %.1fs (total workers=%d)",
                sample,
                len(results[sample]),
                time.perf_counter() - t0,
                n_workers,
            )
    return results


# ---------------------------------------------------------------------------
# Clinical variant prefetch (thread-parallel across genes)
# ---------------------------------------------------------------------------


def _prefetch_variants(
    clinical_mod: ClinicalModule,
    gene_to_tid: dict[str, str],
) -> None:
    """Prefetch clinical variants serially and cache per gene.

    Serial because the ``ConsequenceValidator`` carries per-transcript
    coding-sequence and strand caches that are mutated during variant
    validation; threaded access would corrupt them. The dominant cost
    is parquet filter-pushdown reads (fast with PyArrow) plus codon
    translation, both CPU-bound enough that threading buys little.
    """
    for gene_name, transcript_id in gene_to_tid.items():
        clinical_mod._variant_cache[gene_name] = clinical_mod.fetch_variants(
            gene_name, transcript_id=transcript_id
        )


# ---------------------------------------------------------------------------
# Gene-level annotation (parallel after precompute)
# ---------------------------------------------------------------------------


def _annotate_genes_parallel(
    genes: list[Gene],
    cfg: PipelineConfig,
    *,
    deeploc_lookup: dict,
    clinical_prefetch: dict[str, list[dict[str, Any]]],
    cds_df: pd.DataFrame,
    genome_path: Path,
    workers: int,
) -> list[Gene]:
    """Annotate genes in parallel via ProcessPoolExecutor.

    Each worker reconstructs the module lattice from the shared
    configuration plus the precomputed lookup tables (DeepLoc, DIAMOND
    hits, clinical variants) that were passed in. Gene objects are
    pickled into the worker and the annotated result is pickled back.
    For small gene sets the process-pool overhead is not worth it; at
    that scale the annotation runs serially in the main process.
    """
    n_workers = min(workers, max(1, len(genes)))
    if n_workers <= 1 or len(genes) <= 4:
        return _annotate_genes_serial(
            genes,
            cfg,
            deeploc_lookup=deeploc_lookup,
            clinical_prefetch=clinical_prefetch,
            cds_df=cds_df,
            genome_path=genome_path,
        )

    chunks = _chunk_genes(genes, n_workers)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [
            ex.submit(
                _annotate_genes_serial,
                chunk,
                cfg,
                deeploc_lookup=deeploc_lookup,
                clinical_prefetch={
                    g.gene_name: clinical_prefetch.get(g.gene_name, []) for g in chunk
                },
                cds_df=cds_df,
                genome_path=genome_path,
            )
            for chunk in chunks
        ]
        out: list[Gene] = []
        for fut in as_completed(futures):
            out.extend(fut.result())
    # Preserve input order by gene name
    by_name = {g.gene_name: g for g in out}
    return [by_name[g.gene_name] for g in genes]


def _chunk_genes(genes: list[Gene], n: int) -> list[list[Gene]]:
    """Even split of *genes* into *n* contiguous chunks."""
    k, m = divmod(len(genes), n)
    return [genes[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def _annotate_genes_serial(
    genes: list[Gene],
    cfg: PipelineConfig,
    *,
    deeploc_lookup: dict,
    clinical_prefetch: dict[str, list[dict[str, Any]]],
    cds_df: pd.DataFrame,
    genome_path: Path,
) -> list[Gene]:
    """Annotation worker: rebuilds the pipeline from precomputed inputs."""
    validator = ConsequenceValidator(cds_df=cds_df, genome_fasta=str(genome_path))
    clinical_mod = ClinicalModule(cfg, validator=validator)
    clinical_mod._variant_cache.update(clinical_prefetch)

    pipeline = AnnotationPipeline(
        protein_modules=[
            BiophysicsModule(cfg),
            MotifsModule(cfg),
            MassSpecModule(cfg),
            clinical_mod,
            LocalizationModule(cfg, predictions=deeploc_lookup),
        ],
        site_modules=[
            CoreIdentityModule(cfg),
            InitiationContextModule(cfg),
            ConservationModule(cfg),
        ],
    )
    return pipeline.run(genes)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def _resolve_gene_list(genes: list[str] | None, gene_list: Path | None) -> list[str] | None:
    if gene_list is not None:
        with gene_list.open() as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    return genes


def _build_config(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.gtf_path = args.gtf
    cfg.genome_fasta = args.genome
    cfg.protein_fasta = args.protein_fasta
    cfg.output_dir = args.output
    cfg.clinical = ClinicalConfig(
        gnomad_db=args.gnomad_db if args.gnomad_db else None,
        clinvar_db=args.clinvar_db if args.clinvar_db else None,
        cosmic_db=args.cosmic_db if args.cosmic_db else None,
    )
    cfg.conservation = ConservationConfig(
        phylop_bigwig=args.phylop_bigwig if args.phylop_bigwig else None,
        phastcons_bigwig=args.phastcons_bigwig if args.phastcons_bigwig else None,
    )
    return cfg


def _run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    t_start = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)

    # ── 1. Manifests ────────────────────────────────────────────────────
    sample_map = _load_sample_manifest(args.sample_manifest)
    replicate_map = _load_replicate_manifest(args.replicate_manifest)
    if args.samples:
        sample_map = {s: sample_map[s] for s in args.samples if s in sample_map}
    if not sample_map:
        logger.error("No samples resolved; check --sample-manifest and --samples")
        return 2
    logger.info("Samples to run: %s", list(sample_map))

    # ── 2. Upstream per sample (parallel) ───────────────────────────────
    t0 = time.perf_counter()
    sample_dfs = _upstream_all_samples(
        sample_map,
        replicate_map,
        args.gtf,
        args.genome,
        args.protein_fasta,
        workers=args.workers,
    )
    logger.info("upstream: all samples done in %.1fs", time.perf_counter() - t0)

    # ── 3. Combine across samples ───────────────────────────────────────
    combined = combine_filtered_samples(sample_dfs)
    logger.info("combined: %d cross-sample TIS rows", len(combined))

    # ── 4. Gene assembly ───────────────────────────────────────────────
    gene_list = _resolve_gene_list(args.genes, args.gene_list)
    if gene_list:
        combined = combined[combined["Symbol"].isin(gene_list)]
    genes = assemble_genes(
        combined,
        gene_names=gene_list,
        genome_fasta=args.genome,
    )
    logger.info("assembled %d genes, %d TIS", len(genes), sum(len(g.tis_sites) for g in genes))

    # ── 5. Reference tables for validator ───────────────────────────────
    ref = UpstreamReference.load(
        gtf_path=args.gtf,
        genome_fasta=args.genome,
        protein_fasta=args.protein_fasta,
    )

    cfg = _build_config(args)

    # ── 6. Expensive-module precompute ─────────────────────────────────
    deeploc_lookup: dict = {}
    if args.deeploc:
        t0 = time.perf_counter()
        proteins = {f"canonical_{g.gene_name}": g.canonical_protein for g in genes}
        for g in genes:
            for site in g.tis_sites:
                proteins[site.tis_id] = site.isoform_protein
        deeploc_lookup = precompute_deeploc(proteins, model="Fast", device="cpu")
        logger.info(
            "deeploc precompute: %d proteins in %.1fs", len(proteins), time.perf_counter() - t0
        )

    # Conservation is now a SiteModule backed by Zoonomia BigWigs — BigWig
    # random-access is cheap, so there is no precompute step.  Each gene
    # worker opens its own pyBigWig handle via ConservationModule.__init__.

    clinical_prefetch: dict[str, list[dict[str, Any]]] = {}
    if any([cfg.clinical.gnomad_db, cfg.clinical.clinvar_db, cfg.clinical.cosmic_db]):
        t0 = time.perf_counter()
        validator = ConsequenceValidator(cds_df=ref.cds_df, genome_fasta=str(args.genome))
        clinical_mod = ClinicalModule(cfg, validator=validator)
        gene_to_tid = {g.gene_name: g.canonical_transcript_id for g in genes}
        _prefetch_variants(clinical_mod, gene_to_tid)
        clinical_prefetch = dict(clinical_mod._variant_cache)
        logger.info(
            "clinical prefetch: %d genes in %.1fs",
            len(clinical_prefetch),
            time.perf_counter() - t0,
        )

    # ── 7. Annotation (gene-parallel after precompute) ─────────────────
    t0 = time.perf_counter()
    genes = _annotate_genes_parallel(
        genes,
        cfg,
        deeploc_lookup=deeploc_lookup,
        clinical_prefetch=clinical_prefetch,
        cds_df=ref.cds_df,
        genome_path=args.genome,
        workers=args.workers,
    )
    logger.info("annotation: %d genes in %.1fs", len(genes), time.perf_counter() - t0)

    # ── 8. Comparator ──────────────────────────────────────────────────
    if not args.no_comparator:
        t0 = time.perf_counter()
        genes = compare_genes(genes, scope_a_modules=[BiophysicsModule(cfg)])
        logger.info("comparator: done in %.1fs", time.perf_counter() - t0)

    # ── 9. Paired parquet output ───────────────────────────────────────
    t0 = time.perf_counter()
    paired = paired_tis_dataframe(genes)
    out_path = args.output / "all_paired.parquet"
    paired.to_parquet(out_path, index=False)
    logger.info(
        "wrote %s: %d rows × %d cols in %.1fs",
        out_path,
        len(paired),
        len(paired.columns),
        time.perf_counter() - t0,
    )

    logger.info("swissisoform run: total wall time %.1fs", time.perf_counter() - t_start)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
