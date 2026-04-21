"""Pipeline orchestration for SwissIsoform v2.

Wires annotation modules to Gene objects, running ProteinModules symmetrically
on canonical and isoform proteins, SiteModules on each TIS, and gene-level
modules once per gene.

Context-aware dispatch: modules whose ``annotate(protein, ...)`` signature
accepts additional kwargs (``canonical_protein``, ``gene_name``) receive
them automatically.  Modules with the minimal ``annotate(protein)`` signature
are called unchanged.  This preserves the simple ProteinModule protocol for
pure-function modules while letting clinical/massspec get the context they
need — without silently degrading.
"""

from __future__ import annotations

import inspect
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from swissisoform.config import PipelineConfig
from swissisoform.filtering import filter_tis, normalize_tis_counts
from swissisoform.io.canonical import (
    get_canonical_genome_positions,
    get_protein_products,
    get_start_codons,
    get_utr_lengths,
    impute_missing_canonical_starts,
    load_reference_features,
)
from swissisoform.io.gtf import load_exon_skeletons
from swissisoform.io.ribotish import load_ribotish_predictions, recategorize_tis_type
from swissisoform.io.rnaseq import sum_replicate_counts
from swissisoform.models import Gene, TranscriptCoordinates, TranslationInitiationSite

logger = logging.getLogger(__name__)


def run_sample(
    predict_file: str | Path,
    rnaseq_count_files: list[str | Path],
    gtf_path: str | Path,
    *,
    sample: str,
    config: PipelineConfig | None = None,
    genome_fasta: str | Path | None = None,
    protein_fasta: str | Path | None = None,
    reference: "UpstreamReference | None" = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end upstream for a single sample (cell line).

    Faithful port of the upstream reference filter with canonical
    imputation:

        1. Load predict_all.txt, merge GTF (MANE_Select, TSL).
        2. Recategorize TisType → RecatTISType.
        3. Sum replicate HTSeq counts per gene.  Merge
           ``GeneRNASeqCounts`` and set ``TotalRNASeqCounts`` to the
           grand total across mapped reads.
        4. ``NormTISCounts = TISCounts / TotalRNASeqCounts * 1e6`` (true
           RPM against total mapped reads).
        5. ``filter_tis`` with reference-calibrated thresholds
           (``exempt_annotated=False`` for a faithful port — imputation
           replaces the role that exemption played in v1).
        6. ``impute_missing_canonical_starts`` — for every transcript
           left without an Annotated row after filtering, synthesise one
           from GTF CDS + start-codon + pc_translations.fa.  Imputed
           rows carry ``Imputed=True``.

    ``reference`` lets multiple samples share pre-loaded GTF / FASTA /
    protein-product tables (single-pass through GENCODE, amortised
    across the 6 cell lines).

    Returns ``(final_df, dropped_df)`` where ``final_df`` = filtered TIS
    concat'd with imputed canonical rows.  Every transcript represented
    in ``final_df`` now has an Annotated row to serve as canonical.
    """
    predict_file = Path(predict_file)
    gtf_path = Path(gtf_path)
    if not predict_file.exists():
        raise FileNotFoundError(f"Ribo-TISH file not found: {predict_file}")
    if not gtf_path.exists():
        raise FileNotFoundError(f"GTF file not found: {gtf_path}")

    cfg = (config or PipelineConfig()).filtering

    # 1 + 2: load + GTF merge + recategorize
    df = load_ribotish_predictions(predict_file, gtf_path=gtf_path)
    df = recategorize_tis_type(df)

    # 3: RNA-seq counts — gene-level + total
    gene_counts = sum_replicate_counts(list(rnaseq_count_files))
    total_reads = float(gene_counts.sum())
    df = df.merge(gene_counts, left_on="Gid", right_index=True, how="left")
    df["GeneRNASeqCounts"] = df["GeneRNASeqCounts"].fillna(0)
    df["TotalRNASeqCounts"] = total_reads
    df["Sample"] = sample

    # 4: true RPM against mapped reads
    df = normalize_tis_counts(df, total_reads=total_reads)

    logger.info(
        "run_sample[%s]: %d raw TIS, %.2e total mapped reads",
        sample,
        len(df),
        total_reads,
    )

    # 5: upstream-reference-faithful filter
    filtered_df, dropped_df = filter_tis(
        df,
        transcript_support_levels=list(cfg.transcript_support_levels),
        min_normalized_counts=cfg.min_normalized_counts,
        tis_enrichment_max_p=cfg.tis_enrichment_max_p,
        frame_test_max_p=cfg.frame_test_max_p,
        combined_test_max_q=cfg.combined_test_max_q,
        tis_distance_buffer=cfg.tis_distance_buffer,
        exempt_annotated=False,  # imputation supersedes exemption
        return_dropped=True,
    )
    filtered_df = filtered_df.assign(Imputed=False)

    # 6: impute missing canonical starts
    ref = reference or UpstreamReference.load(
        gtf_path=gtf_path,
        genome_fasta=genome_fasta,
        protein_fasta=protein_fasta,
    )
    imputed_df = impute_missing_canonical_starts(
        filtered_df,
        genome_pos=ref.genome_pos,
        utr_lengths=ref.utr_lengths,
        start_codons=ref.start_codons,
        protein_products=ref.protein_products,
    )
    if len(imputed_df) > 0:
        imputed_df["Sample"] = sample
        imputed_df["TotalRNASeqCounts"] = total_reads
        final_df = pd.concat([filtered_df, imputed_df], ignore_index=True)
    else:
        final_df = filtered_df

    # Drop transcripts that still have no canonical row — these are
    # GENCODE cds_start_NF / mRNA_start_NF cases where the canonical
    # start is unknown by definition.  Without a canonical to compare
    # against, any downstream alt-TIS annotation on these transcripts
    # is meaningless.
    has_ann = (
        final_df.assign(_ann=lambda d: d["RecatTISType"] == "Annotated")
        .groupby("Tid")["_ann"]
        .sum()
    )
    uncanonical_tids = set(has_ann[has_ann == 0].index)
    if uncanonical_tids:
        n_rows = final_df["Tid"].isin(uncanonical_tids).sum()
        logger.info(
            "run_sample[%s]: dropping %d rows from %d uncanonical transcripts "
            "(GENCODE cds_start_NF / incomplete CDS)",
            sample,
            n_rows,
            len(uncanonical_tids),
        )
        final_df = final_df[~final_df["Tid"].isin(uncanonical_tids)].reset_index(drop=True)

    logger.info(
        "run_sample[%s]: kept %d, dropped %d, imputed %d canonical rows",
        sample,
        len(filtered_df),
        len(dropped_df),
        len(imputed_df),
    )
    return final_df, dropped_df


class UpstreamReference:
    """Shared GTF + FASTA reference tables for multi-sample upstream runs.

    Loading GENCODE features takes ~30 s; the protein FASTA another ~5 s.
    ``UpstreamReference.load`` does that work once, and each sample run
    reuses the result.
    """

    def __init__(
        self,
        *,
        genome_pos: pd.DataFrame,
        utr_lengths: pd.DataFrame,
        start_codons: pd.DataFrame,
        protein_products: pd.DataFrame,
        cds_df: pd.DataFrame,
        exon_skeletons: dict[str, TranscriptCoordinates] | None = None,
    ) -> None:
        """Hold preloaded reference tables shared across upstream sample runs."""
        self.genome_pos = genome_pos
        self.utr_lengths = utr_lengths
        self.start_codons = start_codons
        self.protein_products = protein_products
        # Raw CDS feature table (chromosome, start, end, strand, gene_id,
        # transcript_id, feature_type) — consumed by ConsequenceValidator
        # to build genomic→coding position maps for variant validation.
        self.cds_df = cds_df
        # Per-transcript exon skeletons (Layer 1 of the ORF-exon infrastructure).
        # Consumed by the assembly layer to populate Gene.canonical_orf_exons
        # and TIS.orf_exons, which in turn unblocks genomic-coordinate modules
        # (conservation region means, clinical variant intersection, etc.).
        self.exon_skeletons = exon_skeletons or {}

    @classmethod
    def load(
        cls,
        *,
        gtf_path: str | Path,
        genome_fasta: str | Path | None,
        protein_fasta: str | Path | None,
    ) -> "UpstreamReference":
        """Build reference tables from GTF + FASTAs."""
        if genome_fasta is None or protein_fasta is None:
            raise ValueError(
                "genome_fasta and protein_fasta are required for canonical "
                "imputation — pass paths to download_references.sh outputs"
            )
        features = load_reference_features(gtf_path)
        return cls(
            genome_pos=get_canonical_genome_positions(features["CDS"]),
            utr_lengths=get_utr_lengths(features["CDS"], features["UTR"]),
            start_codons=get_start_codons(features["start_codon"], genome_fasta),
            protein_products=get_protein_products(protein_fasta),
            cds_df=features["CDS"],
            exon_skeletons=load_exon_skeletons(str(gtf_path)),
        )


def run_upstream(
    ribotish_path: str | Path,
    gtf_path: str | Path,
    config: PipelineConfig | None = None,
    sample: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Ribo-TISH predictions, recategorize, normalize, and filter.

    Pipeline:
        1. ``load_ribotish_predictions`` — reads the TSV, merges GTF columns
           (MANE_Select, transcript_support_level, gene_type, transcript_type).
        2. ``recategorize_tis_type`` — adds ``RecatTISType`` column.
        3. ``normalize_tis_counts`` — adds ``NormTISCounts`` (RPM).
        4. ``filter_tis`` with thresholds from ``config.filtering``, returning
           both kept and dropped rows.

    Returns two frames.  ``filtered_df`` is the set of TIS to annotate
    downstream: alternative TIS that survived filtering, PLUS Annotated rows
    from reference transcripts (which are exempt from the significance
    filter when ``FilterConfig.exempt_annotated`` is True).

    ``annotated_df`` is a separate canonical source: ALL Annotated rows from
    reference transcripts, regardless of filter outcome.  Downstream
    assembly looks up per-transcript canonical from this frame so filtering
    decisions don't strip the reference protein sequences we rely on.

    Args:
        ribotish_path: Path to a Ribo-TISH ``predict_all.txt`` file.
        gtf_path: Path to a GENCODE GTF (required — we need MANE/TSL columns
            for the reference-transcript filter).
        config: Pipeline configuration.  Defaults to ``PipelineConfig()``
            which uses upstream-reference-faithful filter thresholds.
        sample: Optional sample / cell-line label. When provided, a
            ``Sample`` column is attached to both returned frames.

    Returns:
        (filtered_df, annotated_df) tuple.

    Raises:
        FileNotFoundError: If ``ribotish_path`` or ``gtf_path`` does not exist.
    """
    ribotish_path = Path(ribotish_path)
    gtf_path = Path(gtf_path)

    if not ribotish_path.exists():
        raise FileNotFoundError(f"Ribo-TISH file not found: {ribotish_path}")
    if not gtf_path.exists():
        raise FileNotFoundError(f"GTF file not found: {gtf_path}")

    if config is None:
        config = PipelineConfig()

    fcfg = config.filtering
    logger.info("run_upstream: loading %s with GTF %s", ribotish_path, gtf_path)

    df = load_ribotish_predictions(ribotish_path, gtf_path=gtf_path)
    df = recategorize_tis_type(df)
    df = normalize_tis_counts(df)

    if sample is not None:
        df = df.copy()
        df["Sample"] = sample

    filtered_df, dropped_df = filter_tis(
        df,
        transcript_support_levels=list(fcfg.transcript_support_levels),
        min_normalized_counts=fcfg.min_normalized_counts,
        tis_enrichment_max_p=fcfg.tis_enrichment_max_p,
        frame_test_max_p=fcfg.frame_test_max_p,
        combined_test_max_q=fcfg.combined_test_max_q,
        tis_distance_buffer=fcfg.tis_distance_buffer,
        exempt_annotated=fcfg.exempt_annotated,
        return_dropped=True,
    )

    # Build the canonical-source frame: EVERY Annotated row Ribo-TISH
    # emitted, regardless of filter outcome or transcript support level.
    # Rationale: the "Annotated" label comes from GTF coordinates (it's a
    # GTF CDS call, not a prediction), so it's a reference point whether
    # or not the transcript is MANE/TSL-1-3.  The reference-transcript
    # filter applies to alt-TIS predictions we are testing for confidence;
    # it does not apply to the canonical reference itself.
    full = pd.concat([filtered_df, dropped_df], ignore_index=True)
    is_annotated = full["TisType"].astype(str).str.startswith("Annotated")
    annotated_df = full[is_annotated].copy()

    logger.info(
        "run_upstream: %d TIS after filter (from %d raw); %d Annotated rows retained",
        len(filtered_df),
        len(df),
        len(annotated_df),
    )
    return filtered_df, annotated_df


@lru_cache(maxsize=None)
def _annotate_param_names(annotate_fn) -> frozenset[str]:
    """Return the parameter names of *annotate_fn* (cached per function)."""
    return frozenset(inspect.signature(annotate_fn).parameters)


def _call_annotate(mod: Any, protein: str, **context: Any) -> dict[str, Any]:
    """Call ``mod.annotate(protein, ...)`` passing only kwargs it accepts.

    This is how the pipeline gets context (gene_name, canonical_protein) to
    modules that need it without breaking the ProteinModule protocol for
    modules that don't.  Modules opt in by declaring kwargs in their
    ``annotate()`` signature.

    Args:
        mod: A ProteinModule instance with an ``annotate`` method.
        protein: The protein sequence to annotate.
        **context: Optional context kwargs. Only those matching parameter
            names on ``mod.annotate`` are forwarded.

    Returns:
        Whatever ``mod.annotate`` returns.
    """
    param_names = _annotate_param_names(mod.annotate)
    supported = {k: v for k, v in context.items() if k in param_names}
    return mod.annotate(protein, **supported)


class AnnotationPipeline:
    """Orchestrate annotation modules over a list of genes.

    The pipeline runs three kinds of modules:
    - ProteinModules: called with `annotate(protein)`, run on both canonical
      and each isoform protein. Must have MODULE_NAME attribute.
    - SiteModules: called with `annotate_site(site)`, run once per TIS.
      Must have MODULE_NAME attribute.
    - GeneModules: called with `annotate_gene(gene)`, run once per gene.
      Must have MODULE_NAME attribute. Results stored in gene.gene_annotations.

    Attributes:
        protein_modules: Modules implementing annotate(protein: str) -> dict.
        site_modules: Modules implementing annotate_site(site: TIS) -> dict.
        gene_modules: Modules implementing annotate_gene(gene: Gene) -> dict.
    """

    def __init__(
        self,
        protein_modules: list[Any] | None = None,
        site_modules: list[Any] | None = None,
        gene_modules: list[Any] | None = None,
    ) -> None:
        """Initialize with optional module lists (defaults to empty lists)."""
        self.protein_modules = protein_modules or []
        self.site_modules = site_modules or []
        self.gene_modules = gene_modules or []

    def run(self, genes: list[Gene]) -> list[Gene]:
        """Run all annotation modules over a list of genes.

        Processing order per gene:
        1. ProteinModules on canonical protein -> gene.canonical_annotations
        2. ProteinModules on each isoform protein -> tis.isoform_annotations
        3. SiteModules on each TIS -> tis.isoform_annotations
        4. GeneModules on the gene -> gene.gene_annotations

        Args:
            genes: List of Gene objects with canonical_protein and tis_sites
                populated. Each TIS should have isoform_protein set.

        Returns:
            The same Gene objects with annotations populated.
        """
        logger.info(
            "Running annotation pipeline: %d genes, %d protein modules, "
            "%d site modules, %d gene modules",
            len(genes),
            len(self.protein_modules),
            len(self.site_modules),
            len(self.gene_modules),
        )

        for gene in genes:
            self._annotate_gene(gene)

        total_tis = sum(len(g.tis_sites) for g in genes)
        logger.info("Annotation complete: %d genes, %d TIS sites", len(genes), total_tis)
        return genes

    def _annotate_gene(self, gene: Gene) -> None:
        """Run all modules for a single gene.

        The canonical protein is annotated with ``gene_name`` as context but
        NOT ``canonical_protein`` — passing the canonical as its own
        comparison would make every massspec peptide "not unique to isoform",
        which is meaningless for canonical.  ``gene_name`` is still passed so
        clinical can look up variants.
        """
        # 1. ProteinModules on canonical protein (once per gene)
        for mod in self.protein_modules:
            name = mod.MODULE_NAME
            gene.canonical_annotations[name] = _call_annotate(
                mod,
                gene.canonical_protein,
                gene_name=gene.gene_name,
            )

        # 2-3. Per-TIS annotations
        for site in gene.tis_sites:
            self._annotate_site(site)

        # 4. Gene-level modules
        for mod in self.gene_modules:
            name = mod.MODULE_NAME
            gene.gene_annotations[name] = mod.annotate_gene(gene)

    def _annotate_site(self, site: TranslationInitiationSite) -> None:
        """Run ProteinModules on isoform + SiteModules on a single TIS.

        Isoform annotation passes full context: ``canonical_protein`` (for
        massspec uniqueness) and ``gene_name`` (for clinical variant lookup).
        Modules that don't need these kwargs receive only ``protein``.
        """
        # ProteinModules on isoform protein
        for mod in self.protein_modules:
            name = mod.MODULE_NAME
            site.isoform_annotations[name] = _call_annotate(
                mod,
                site.isoform_protein,
                canonical_protein=site.canonical_protein,
                gene_name=site.gene_name,
            )

        # SiteModules
        for mod in self.site_modules:
            name = mod.MODULE_NAME
            site.isoform_annotations[name] = mod.annotate_site(site)
