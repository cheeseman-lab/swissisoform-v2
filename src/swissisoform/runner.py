"""SwissIsoform v2 — pipeline orchestration runner.

Extracted from scripts/run.py so that multiple front-ends (run.py preset
driver, cli.py) can reuse the same Stage 1–7 orchestration.  The public API
is the ``RunSpec`` dataclass plus three functions:

    prepare(spec)  -> PreparedRun    # Stages 1–5 (refs, samples, selection,
                                       #             assembly, fasta)
    annotate(prepared, spec) -> df   # Stages 6–7 minus file writing
    run(spec)      -> int            # full orchestration + output writing

Behavior is identical to the original scripts/run.py main(); only the input
plumbing changed from argparse Namespace to RunSpec fields.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from swissisoform.assembly import assemble_genes
from swissisoform.clinical.module import ClinicalModule
from swissisoform.clinical.validate import ConsequenceValidator
from swissisoform.combine import combine_filtered_samples, dedupe_unique_proteins
from swissisoform.compare.comparator import compare_genes
from swissisoform.config import PipelineConfig
from swissisoform.conservation_frame.module import ConservationFrameModule
from swissisoform.evidence.e6_mass_spec import (
    MassSpecModule,
    collect_unique_peptides,
    precompute_pepquery,
)
from swissisoform.evidence.f2_localization import LocalizationModule, precompute_deeploc
from swissisoform.evidence.f3_domains import InterProScanModule, precompute_interproscan
from swissisoform.evidence.f4_targeting import (
    SignalPModule,
    TargetPModule,
    precompute_signalp,
    precompute_targetp,
)
from swissisoform.io.parquet import paired_tis_dataframe
from swissisoform.io.rnaseq import load_sample_manifest
from swissisoform.modules.biophysics import BiophysicsModule
from swissisoform.modules.conservation import ConservationModule
from swissisoform.modules.core_identity import CoreIdentityModule
from swissisoform.modules.generef import GeneRefModule
from swissisoform.modules.initiation_context import InitiationContextModule
from swissisoform.modules.motifs import MotifsModule
from swissisoform.modules.scoring import EvidenceScoringModule
from swissisoform.modules.variant_intersection import VariantIntersectionModule
from swissisoform.modules.varianteffect import VariantEffectModule
from swissisoform.pipeline import AnnotationPipeline, UpstreamReference, run_sample
from swissisoform.plm.module import PLMVEPModule
from swissisoform.references import (
    ALL_CELL_LINES,
    GENOME,
    GTF,
    OUT,
    PROTEIN,
    ROOT,
    build_config,
)
from swissisoform.sourceresolve import collapse_to_source
from swissisoform.structure.module import StructureModule

logger = logging.getLogger("run")

# Single-sample mode (5-gene diagnostic)
HELA_PREDICT = ROOT / "data" / "reference" / "HeLa_TIS_predict_all.txt"
HELA_RNASEQ = [
    ROOT / "data" / "reference" / "rnaseq_counts/GATCAG_8_htseqcount.txt",
    ROOT / "data" / "reference" / "rnaseq_counts/CACGGT_9_htseqcount.txt",
]

# Multi-sample mode (production)
SAMPLE_MANIFEST = ROOT / "data" / "reference" / "ribotish_sample_manifest.csv"
REPLICATE_MANIFEST = ROOT / "data" / "reference" / "ribotish_replicate_manifest.csv"
COMBINED_PARQUET = OUT / "filtered" / "all_samples_combined.parquet"
# One row per unique protein (stop-stripped AASeq), deduped from the full
# combined table — the genome-wide "what to fold / embed" set.
UNIQUE_TIS_PARQUET = OUT / "filtered" / "unique_tis_deduped.parquet"

GENEREF_JSON = ROOT / "data" / "reference" / "generef" / "generef.json"


# ── Data loading ─────────────────────────────────────────────────────────


def load_single_sample(cell_line: str, reference: UpstreamReference) -> pd.DataFrame:
    """Load + filter one cell line via run_sample.  Currently HeLa-only."""
    if cell_line != "HeLa":
        raise ValueError(
            f"Single-sample mode currently only supports HeLa; got {cell_line!r}. "
            "Use the multi-sample mode (default for non-preset runs)."
        )
    final, dropped = run_sample(
        HELA_PREDICT, HELA_RNASEQ, GTF, sample="HeLa", reference=reference,
    )
    logger.info(
        "Single-sample (HeLa): %d kept, %d dropped, %d imputed",
        len(final), len(dropped), final["Imputed"].sum(),
    )
    return final


def _write_unique_tis(combined: pd.DataFrame) -> None:
    """Materialize the unique-protein dedup of the *full* combined table."""
    unique = dedupe_unique_proteins(combined)
    UNIQUE_TIS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    unique.to_parquet(UNIQUE_TIS_PARQUET, index=False)
    logger.info(
        "Wrote unique-protein parquet: %s (%d proteins)", UNIQUE_TIS_PARQUET, len(unique)
    )


def _manifest_single_path(row: pd.Series, col: str) -> Path | None:
    """Parse a single optional manifest path cell into ``ROOT / p`` or ``None``.

    Used for the optional per-sample IsoQuant abundance table; absent / blank
    cells (samples without long-read quant) yield ``None`` so the
    source-resolution stage skips them. A configured path that does not exist on
    disk is treated as absent (warn + ``None``) so one bad/pending cell does not
    crash a multi-sample run.
    """
    val = row.get(col)
    if val is None or (isinstance(val, float) and pd.isna(val)) or str(val).strip() == "":
        return None
    path = ROOT / str(val).strip()
    if not path.exists():
        logger.warning("manifest %s path does not exist, skipping: %s", col, path)
        return None
    return path


def load_combined(
    cell_lines: list[str],
    reference: UpstreamReference,
    cfg: PipelineConfig,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Load the multi-cell-line combined parquet.

    If the cached combined parquet exists and force_rebuild is False, read
    it (and subset to requested cell lines).  Otherwise run upstream per
    sample and rebuild.
    """
    if COMBINED_PARQUET.exists() and not force_rebuild:
        logger.info("Loading cached combined parquet: %s", COMBINED_PARQUET)
        combined = pd.read_parquet(COMBINED_PARQUET)
        # Dedup is a function of the full combined set; (re)build if absent.
        if not UNIQUE_TIS_PARQUET.exists():
            _write_unique_tis(combined)
        if set(cell_lines) != set(ALL_CELL_LINES):
            present_cols = [
                f"present_{cl}" for cl in cell_lines
                if f"present_{cl}" in combined.columns
            ]
            if present_cols:
                mask = combined[present_cols].any(axis=1)
                combined = combined[mask].copy()
                logger.info("Subset to %s: %d rows", ",".join(cell_lines), len(combined))
        return combined

    logger.info("Rebuilding combined parquet from per-sample upstream runs")
    manifest = load_sample_manifest(SAMPLE_MANIFEST, REPLICATE_MANIFEST)
    per_sample: dict[str, pd.DataFrame] = {}
    for _, row in manifest.iterrows():
        sample = row["sample"]
        if sample not in cell_lines:
            continue
        predict = ROOT / row["predict_file"]
        rnaseq_files = [ROOT / f for f in row["rnaseq_count_files"]]
        t0 = time.perf_counter()
        final_df, _ = run_sample(
            predict, rnaseq_files, GTF, sample=sample, config=cfg, reference=reference,
            genome_fasta=GENOME,
            isoquant_table=_manifest_single_path(row, "isoquant_table"),
        )
        filtered_out = ROOT / row["filtered_file"]
        filtered_out.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(filtered_out, index=False)
        per_sample[sample] = final_df
        logger.info(
            "%s: %d filtered rows (%.1fs)", sample, len(final_df),
            time.perf_counter() - t0,
        )

    combined = combine_filtered_samples(per_sample)
    COMBINED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(COMBINED_PARQUET, index=False)
    logger.info("Wrote combined parquet: %s (%d rows)", COMBINED_PARQUET, len(combined))
    _write_unique_tis(combined)
    return combined


def load_isoform_picks(source: Path | str | list[dict]) -> pd.DataFrame:
    """Normalize isoform picks from a parquet/CSV path or inline preset entries.

    Inline entries (a preset TOML's ``[[isoforms]]`` array) are dicts with
    ``gene`` / ``tid`` / ``genome_pos`` / ``start_codon``.  File sources may use
    the SwissIso_* column names.  Both are normalized to the join keys
    ``Tid`` / ``GenomePos`` / ``StartCodon`` (plus ``Symbol``).
    """
    if isinstance(source, (str, Path)):
        p = Path(source)
        isos = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    else:
        isos = pd.DataFrame(
            [
                {
                    "Symbol": e.get("gene"),
                    "Tid": e["tid"],
                    "GenomePos": e["genome_pos"],
                    "StartCodon": e["start_codon"],
                }
                for e in source
            ]
        )

    rename_map = {
        "SwissIso_Tid": "Tid",
        "SwissIso_GenomePos": "GenomePos",
        "SwissIso_StartCodon": "StartCodon",
        "Gene": "Symbol",
    }
    for old, new in rename_map.items():
        if old in isos.columns and new not in isos.columns:
            isos = isos.rename(columns={old: new})

    missing = {"Tid", "GenomePos", "StartCodon"} - set(isos.columns)
    if missing:
        raise ValueError(
            f"Isoform picks missing required columns: {sorted(missing)}. Expected "
            f"Tid, GenomePos, StartCodon (or SwissIso_* / inline tid, genome_pos, "
            f"start_codon)."
        )
    return isos


def restrict_to_isoforms(
    df: pd.DataFrame, isos: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Restrict the combined catalog to specific TIS picks.

    ``isos`` is a normalized picks frame from :func:`load_isoform_picks`.

    Returns (filtered_df, gene_names).  The filter keeps:
      - All Annotated rows for the relevant genes (canonical references)
      - The specific picked TIS rows
    """
    required = ["Tid", "GenomePos", "StartCodon"]

    if "Symbol" in isos.columns and isos["Symbol"].notna().any():
        gene_names = sorted(isos["Symbol"].dropna().unique())
    else:
        merged = df.merge(isos[required], on=required, how="inner")
        gene_names = sorted(merged["Symbol"].unique())

    gene_mask = df["Symbol"].isin(gene_names)
    annotated_mask = gene_mask & (df["RecatTISType"] == "Annotated")

    iso_keys = isos[required].drop_duplicates()
    picked = df.merge(iso_keys, on=required, how="left", indicator=True)
    picked_mask = (picked["_merge"] == "both").to_numpy()

    final_mask = annotated_mask.to_numpy() | picked_mask
    restricted = df[final_mask].copy().reset_index(drop=True)

    n_isos_requested = len(isos)
    n_picked_matched = int(picked_mask.sum())
    n_annotated = int(annotated_mask.sum())
    logger.info(
        "Isoform restriction: %d requested, %d matched, %d canonical Annotated rows "
        "across %d genes (total %d rows)",
        n_isos_requested, n_picked_matched, n_annotated, len(gene_names), len(restricted),
    )
    if n_picked_matched < n_isos_requested:
        logger.warning(
            "%d of %d requested isoforms did not match the combined catalog",
            n_isos_requested - n_picked_matched, n_isos_requested,
        )

    return restricted, gene_names


# ── Precompute ───────────────────────────────────────────────────────────


def collect_all_proteins(genes) -> list[str]:
    """Return every protein sequence (canonical + isoform) across genes."""
    proteins: list[str] = []
    for g in genes:
        if g.canonical_protein:
            proteins.append(g.canonical_protein)
        for s in g.tis_sites:
            if s.isoform_protein:
                proteins.append(s.isoform_protein)
            if s.canonical_protein:
                proteins.append(s.canonical_protein)
    return proteins


def write_proteins_fasta(proteins: list[str], out_path: Path) -> int:
    """Write deduped, sha1-keyed FASTA for GPU precompute jobs."""
    from swissisoform.plm.embed import protein_hash

    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    n = 0
    with open(out_path, "w") as fh:
        for seq in proteins:
            stripped = seq.rstrip("*").upper()
            if not stripped:
                continue
            h = protein_hash(stripped)
            if h in seen:
                continue
            seen.add(h)
            fh.write(f">{h}\n{stripped}\n")
            n += 1
    return n


def run_precompute(genes, all_proteins: list[str], skip: set[str]) -> dict:
    """Run precompute steps.  `skip` controls which precomputes fire."""
    preds: dict = {}

    if "localization" not in skip:
        deeploc_input = {}
        for gene in genes:
            deeploc_input[f"canonical_{gene.gene_name}"] = gene.canonical_protein
            for site in gene.tis_sites:
                deeploc_input[site.tis_id] = site.isoform_protein
        preds["deeploc"] = precompute_deeploc(deeploc_input, model="Fast", device="cpu")
    else:
        preds["deeploc"] = {}

    if "massspec" not in skip:
        unique_peps = collect_unique_peptides(genes)
        preds["pepquery"] = precompute_pepquery(
            unique_peps,
            dataset="Deep_29_healthy_human_tissues_PXD010154,GTEx_32_Tissues_Proteome_PXD016999",
            reference_db="swissprot:human",
            cache_dir=ROOT / "data" / "cache" / "pepquery",
        )
    else:
        preds["pepquery"] = {}

    preds["signalp"] = (
        precompute_signalp(all_proteins) if "signalp" not in skip else {}
    )
    preds["targetp"] = (
        precompute_targetp(all_proteins) if "targetp" not in skip else {}
    )
    preds["interproscan"] = (
        precompute_interproscan(all_proteins) if "interproscan" not in skip else {}
    )

    # PLM VEP + Structure: cache lookup only (populate via sbatch GPU scripts).
    from swissisoform.plm.embed import precompute_plm_esm2
    from swissisoform.structure.fold import precompute_fold

    preds["plm"] = (
        precompute_plm_esm2(all_proteins, inline=False) if "plm_vep" not in skip else {}
    )
    preds["structure"] = (
        precompute_fold(all_proteins, backend="boltz", inline=False)
        if "structure" not in skip else {}
    )

    return preds


def _load_generef() -> dict[str, dict] | None:
    """Load the UniProt gene-reference table written by scripts/setup/fetch_generef.py."""
    import json

    if GENEREF_JSON.exists():
        return json.loads(GENEREF_JSON.read_text())
    return None


def _gene_locus(gene) -> tuple[str, int, int] | None:
    """``(chrom, start, end)`` bounding box over a gene's canonical + isoform ORFs.

    Covers the canonical CDS and every isoform ORF — including 5′ extensions —
    so the gnomAD fetch can pull variants by position (flag B), recovering those
    VEP attributed to an overlapping gene. ``None`` when no genomic intervals or
    chrom are available (e.g. skeleton not loaded).
    """
    chrom: str | None = None
    intervals: list[tuple[int, int]] = list(gene.canonical_orf_exons or [])
    for s in gene.tis_sites:
        if chrom is None:
            chrom = getattr(s, "chrom", None)
        intervals.extend(s.orf_exons or [])
        intervals.extend(s.canonical_orf_exons or [])
    if not chrom or not intervals:
        return None
    return chrom, min(a for a, _ in intervals), max(b for _, b in intervals)


def build_pipeline(cfg, preds, ref, genes, skip: set[str]) -> AnnotationPipeline:
    """Build the annotation pipeline, omitting modules in `skip`."""
    validator = ConsequenceValidator(cds_df=ref.cds_df, genome_fasta=str(GENOME))

    protein_mods = []
    if "biophysics" not in skip:
        protein_mods.append(BiophysicsModule(cfg))
    if "motifs" not in skip:
        protein_mods.append(MotifsModule(cfg))
    if "massspec" not in skip:
        protein_mods.append(MassSpecModule(cfg, validated_peptides=preds["pepquery"]))
    if "clinical" not in skip:
        clinical_mod = ClinicalModule(cfg, validator=validator)
        for gene in genes:
            try:
                variants = clinical_mod.fetch_variants(
                    gene.gene_name,
                    transcript_id=gene.canonical_transcript_id,
                    gene_locus=_gene_locus(gene),
                )
                clinical_mod._variant_cache[gene.gene_name] = variants
            except Exception as e:  # pragma: no cover
                logger.warning("Clinical fetch failed for %s: %s", gene.gene_name, e)
        protein_mods.append(clinical_mod)
    if "localization" not in skip:
        protein_mods.append(LocalizationModule(cfg, predictions=preds["deeploc"]))
    if "signalp" not in skip:
        protein_mods.append(SignalPModule(cfg, predictions=preds["signalp"]))
    if "targetp" not in skip:
        protein_mods.append(TargetPModule(cfg, predictions=preds["targetp"]))
    if "interproscan" not in skip:
        protein_mods.append(InterProScanModule(cfg, predictions=preds["interproscan"]))

    site_mods = []
    if "core_identity" not in skip:
        site_mods.append(CoreIdentityModule(cfg))
    if "initiation_context" not in skip:
        site_mods.append(InitiationContextModule(cfg))
    if "conservation" not in skip:
        site_mods.append(ConservationModule(cfg))
    if "conservation_frame" not in skip:
        site_mods.append(ConservationFrameModule(cfg))
    if "variant_intersection" not in skip:
        site_mods.append(VariantIntersectionModule(validator=validator))
    if "plm_vep" not in skip:
        site_mods.append(PLMVEPModule(cfg))
    if "varianteffect" not in skip:
        site_mods.append(
            VariantEffectModule(
                cfg,
                alphamissense_db=cfg.clinical.alphamissense_db if cfg.clinical else None,
            )
        )
    if "structure" not in skip:
        site_mods.append(StructureModule(cfg))

    gene_mods = []
    if "generef" not in skip:
        generef_data = _load_generef()
        if generef_data:
            gene_mods.append(GeneRefModule(cfg, gene_annotations=generef_data))
        else:
            logger.info("generef: %s missing — skipping gene-level reference", GENEREF_JSON)

    return AnnotationPipeline(
        protein_modules=protein_mods, site_modules=site_mods, gene_modules=gene_mods
    )


# ── Spot check ───────────────────────────────────────────────────────────


def print_spot_check(genes, limit: int | None = 5) -> None:
    """Print condensed per-gene / per-TIS spot check.  `limit` caps TIS per gene."""
    for gene in sorted(genes, key=lambda g: g.gene_name):
        can = gene.canonical_annotations
        bio = can.get("biophysics", {})
        clin_sum = can.get("clinical", {}).get("summary", {})
        loc = can.get("localization", {})
        n_tis = len(gene.tis_sites)
        print(
            f"\n{gene.gene_name}  ({gene.canonical_transcript_id}, "
            f"{len(gene.canonical_protein.rstrip('*'))} aa, {n_tis} TIS)"
        )
        print(
            f"  bio pI={bio.get('pI', 0):.2f} gravy={bio.get('gravy', 0):.3f}  "
            f"clin={clin_sum.get('total_variants', 0)} vars  "
            f"loc={loc.get('deeploc_prediction')}"
        )

        sites = sorted(gene.tis_sites, key=lambda s: (s.orf_type.value, s.position))
        if limit:
            sites = sites[:limit]
        for site in sites:
            ia = site.isoform_annotations
            isc = ia.get("scoring", {}) or {}
            iplm = ia.get("plm_vep", {}) or {}
            istr = ia.get("structure", {}) or {}
            ilen = len(site.isoform_protein.rstrip("*"))
            kozak = site.kozak_context or "—"
            print(
                f"    TIS {site.tis_id} ({site.orf_type.value}, {ilen} aa, kozak={kozak})"
            )
            print(
                f"      score: "
                f"existence={isc.get('existence_score')}/{isc.get('existence_evaluable')} "
                f"(hi={isc.get('existence_high_confidence')})  "
                f"functional={isc.get('functional_score')}/{isc.get('functional_evaluable')} "
                f"(hi={isc.get('functional_high_confidence')})"
            )
            print(
                f"      plm[{iplm.get('status', '—')}]  "
                f"struct[{istr.get('status', '—')}/{istr.get('backend', '—')}]"
            )
        if limit and n_tis > limit:
            print(f"    ... and {n_tis - limit} more TIS (use --no-spot-check-limit to see all)")


# ── Public API ─────────────────────────────────────────────────────────────


@dataclass
class RunSpec:
    """Declarative specification for a single pipeline run."""

    gene_names: list[str] | None          # None = all genes
    restricted_df: pd.DataFrame | None     # pre-filtered combined (isoform picks), else None
    cell_lines: list[str]
    single_sample: bool
    min_cell_lines: int
    skip: set[str]
    run_name: str
    fasta_out: Path
    emit_fasta: bool = False
    gtf: Path = GTF
    genome: Path = GENOME
    protein_fasta: Path = PROTEIN
    out_dir: Path | None = None            # default OUT / run_name
    spot_check_limit: int | None = 5
    rebuild_combined: bool = False
    # Source-transcript resolution (cascade + collapse to one mRNA per TIS).
    skip_source_resolution: bool = False
    divergence_threshold: float | None = 0.5
    window_upstream: int = 100
    window_downstream: int = 100


@dataclass
class PreparedRun:
    """Result of Stages 1–5: references + assembled genes + proteins + config."""

    ref: object          # UpstreamReference
    genes: list          # assembled Gene objects
    all_proteins: list[str]
    cfg: object          # PipelineConfig
    n_fasta_written: int = 0


class _EmptyGenes(Exception):
    """Internal signal: no genes assembled (run() maps to exit code 1)."""


class _BadCellLine(Exception):
    """Internal signal: single-sample with != 1 cell line (run() maps to exit code 2)."""


def prepare(spec: RunSpec) -> PreparedRun:
    """Stages 1–5: load references, samples, gene selection, assembly, fasta."""
    logger.info("Stage 1: loading shared references")
    ref = UpstreamReference.load(
        gtf_path=spec.gtf, genome_fasta=spec.genome, protein_fasta=spec.protein_fasta
    )

    if spec.restricted_df is not None:
        # Isoform picks already restricted the combined catalog up front.
        final = spec.restricted_df
    elif spec.single_sample:
        if len(spec.cell_lines) != 1:
            logger.error(
                "--single-sample requires exactly one cell line; got %s", spec.cell_lines
            )
            raise _BadCellLine
        logger.info("Stage 2: single-sample mode (%s)", spec.cell_lines[0])
        final = load_single_sample(spec.cell_lines[0], ref)
        if spec.gene_names is not None:
            final = final[final["Symbol"].isin(spec.gene_names)].copy()
    else:
        logger.info("Stage 2: multi-sample mode (%s)", ",".join(spec.cell_lines))
        final = load_combined(
            spec.cell_lines, ref,
            build_config(
                source_resolution=not spec.skip_source_resolution,
                divergence_threshold=spec.divergence_threshold,
                window_upstream=spec.window_upstream,
                window_downstream=spec.window_downstream,
            ),
            force_rebuild=spec.rebuild_combined,
        )
        if spec.gene_names is not None:
            final = final[final["Symbol"].isin(spec.gene_names)].copy()

    n_genes_label = f"{len(spec.gene_names)} genes" if spec.gene_names else "ALL genes"
    logger.info("Stage 3: gene selection → %s (run_name=%s)", n_genes_label, spec.run_name)

    # Collapse to one mRNA per resolved TIS before assembly. No-op when the
    # source-resolution verdict columns are absent (cascade skipped/never ran);
    # otherwise keeps Annotated rows + each resolved site's source transcript,
    # so only resolved TIS — one mRNA each — advance to annotation.
    final = collapse_to_source(final)

    logger.info("Stage 4: assembling gene + TIS domain objects")
    genes = assemble_genes(
        final,
        gene_names=spec.gene_names,
        genome_fasta=spec.genome,
        exon_skeletons=ref.exon_skeletons,
    )
    n_tis = sum(len(g.tis_sites) for g in genes)
    logger.info("Assembled %d genes, %d TIS", len(genes), n_tis)
    if not genes:
        logger.error("No genes assembled — check gene names / isoform file / cell-line filter")
        raise _EmptyGenes

    cfg = build_config(min_cell_lines=spec.min_cell_lines)
    logger.info("Stage 5: precompute (skip=%s)", sorted(spec.skip) or "none")
    all_proteins = collect_all_proteins(genes)
    n_written = write_proteins_fasta(all_proteins, spec.fasta_out)
    logger.info("proteins.fa: %d unique sequences -> %s", n_written, spec.fasta_out)

    return PreparedRun(
        ref=ref, genes=genes, all_proteins=all_proteins, cfg=cfg,
        n_fasta_written=n_written,
    )


def annotate(prepared: PreparedRun, spec: RunSpec) -> pd.DataFrame:
    """Stages 6–7 minus file writing: precompute, annotate, compare, score."""
    genes = prepared.genes
    cfg = prepared.cfg
    all_proteins = prepared.all_proteins

    preds = run_precompute(genes, all_proteins, spec.skip)
    logger.info(
        "Precompute done: deeploc=%d signalp=%d targetp=%d ips=%d plm=%d struct=%d",
        len(preds["deeploc"]), len(preds["signalp"]), len(preds["targetp"]),
        len(preds["interproscan"]), len(preds["plm"]), len(preds["structure"]),
    )

    logger.info("Stage 6: annotate + compare + score")
    pipeline = build_pipeline(cfg, preds, prepared.ref, genes, spec.skip)
    pipeline.run(genes)

    if "biophysics" not in spec.skip:
        compare_genes(genes, scope_a_modules=[BiophysicsModule(cfg)])
    else:
        compare_genes(genes, scope_a_modules=[])

    scoring_mod = EvidenceScoringModule(cfg)
    all_sites = [site for gene in genes for site in gene.tis_sites]
    scoring_mod.run(all_sites)
    logger.info("Annotation + comparison + scoring complete")

    return paired_tis_dataframe(genes)


def run(spec: RunSpec) -> int:
    """Full orchestration: Stages 1–7 + output writing.  Returns exit code."""
    t_start = time.perf_counter()

    try:
        prepared = prepare(spec)
    except _EmptyGenes:
        return 1
    except _BadCellLine:
        return 2

    if spec.emit_fasta:
        logger.info(
            "--emit-fasta: wrote %d seqs to %s; stopping before precompute/annotate",
            prepared.n_fasta_written, spec.fasta_out,
        )
        return 0

    paired = annotate(prepared, spec)

    print_spot_check(prepared.genes, limit=spec.spot_check_limit)

    out_dir = spec.out_dir or (OUT / spec.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "all_paired.parquet"
    paired.to_parquet(all_path, index=False)
    print(f"\nwrote {all_path} ({len(paired)} rows, {len(paired.columns)} cols)")
    for gene_name, sub in paired.groupby("gene_name"):
        gpath = out_dir / f"{gene_name}_paired.parquet"
        sub.to_parquet(gpath, index=False)
        if len(prepared.genes) <= 50:
            print(f"  {gene_name}: {len(sub)} rows -> {gpath.name}")
    if len(prepared.genes) > 50:
        print(f"  (per-gene parquets written for {len(prepared.genes)} genes)")

    elapsed = time.perf_counter() - t_start
    logger.info("Total wall time: %.1f min", elapsed / 60)
    return 0
