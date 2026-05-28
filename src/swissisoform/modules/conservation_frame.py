"""Module: Conservation Path 1/2 — primate + mammalian reading-frame integrity.

Implements paths 1 (primate radiation) and 2 (mammalian clades) from
``docs/reviews/conservation_path12_spec.md``. For each TIS, extracts a MAF
over the *unique* genomic region from the Zoonomia HAL, then scores each
species for:

- start codon conservation
- frameshift-free indel pattern
- absence of premature stop codons
- amino-acid percent identity vs. hg38

Complements the BigWig-backed Path 3 SiteModule
(:mod:`swissisoform.modules.conservation`). The two live as separate
SiteModules because their failure modes and data dependencies are
independent: Path 3 only needs the ~13 GB BigWig pair; Path 1/2 needs the
200–600 GB HAL + ``hal2maf``.

When the HAL or the binary isn't available the module emits
``status="not_run"`` for every output rather than crashing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from swissisoform.config import PipelineConfig
from swissisoform.conservation_frame import (
    MAMMALIAN_SPECIES,
    PRIMATE_SPECIES,
    aggregate_species_results,
    analyze_species,
    parse_maf,
)
from swissisoform.conservation_frame.hal import (
    extract_maf,
    extract_tree,
    hal2maf_available,
)
from swissisoform.conservation_frame.maf import concat_species_rows
from swissisoform.conservation_frame.tree import depth_from_reference
from swissisoform.models import TranslationInitiationSite

logger = logging.getLogger(__name__)


class ConservationFrameModule:
    """Path 1/2 SiteModule: primate + mammalian frame integrity per TIS.

    Consumes ``site.orf_exons`` and ``site.canonical_orf_exons`` produced by
    the assembly Layer-2 walker. The *unique* genomic region (isoform minus
    canonical) feeds primate/mammalian aggregates; the shared region is
    reported as a baseline so Scope-A enrichment (``unique_frac_intact /
    shared_frac_intact``) is derivable downstream.

    Attributes:
        MODULE_NAME: ``"conservation_frame"``.
        OUTPUT_COLUMNS: Prefixed column names produced per TIS.
        SCOPE: ``"C"`` (per-candidate).
    """

    MODULE_NAME: str = "conservation_frame"
    OUTPUT_COLUMNS: list[str] = [
        "conservation_primate_n_species_aligned",
        "conservation_primate_n_species_intact_frame",
        "conservation_primate_frac_intact",
        "conservation_primate_start_codon_conserved",
        "conservation_primate_mean_pident",
        "conservation_primate_deepest_species",
        "conservation_primate_max_depth",
        "conservation_mammalian_n_species_aligned",
        "conservation_mammalian_n_species_intact_frame",
        "conservation_mammalian_frac_intact",
        "conservation_mammalian_start_codon_conserved",
        "conservation_mammalian_mean_pident",
        "conservation_mammalian_deepest_species",
        "conservation_mammalian_max_depth",
        "conservation_frame_summary",
    ]
    SCOPE: str = "C"

    def __init__(self, config: PipelineConfig) -> None:
        """Resolve HAL + species lists from *config*.

        Missing HAL / missing binary is not an error: :meth:`annotate_site`
        then returns a ``not_run`` block for every TIS. This matches the
        pipeline's ability to run end-to-end on machines without the HAL.
        """
        self._config = config
        cfg = config.conservation

        self._hal_path: Path | None = None
        self._ref_genome: str = "hg38"
        self._hal_timeout: float = 300.0
        self._binary: str = "hal2maf"
        self._halstats_binary: str = "halStats"
        self._tree_newick: str | None = None
        self._primate_species: list[str] = list(PRIMATE_SPECIES)
        self._mammalian_species: list[str] = list(MAMMALIAN_SPECIES)

        if cfg is not None:
            if cfg.hal_path is not None:
                p = Path(cfg.hal_path)
                self._hal_path = p if p.exists() else None
                if cfg.hal_path and self._hal_path is None:
                    logger.warning("Zoonomia HAL file not found: %s", cfg.hal_path)
            self._ref_genome = cfg.hal_ref_genome
            self._binary = cfg.hal2maf_binary
            self._hal_timeout = float(cfg.hal_timeout)
            self._halstats_binary = cfg.halstats_binary
            self._tree_newick = cfg.hal_tree_newick
            if cfg.primate_species is not None:
                self._primate_species = list(cfg.primate_species)
            if cfg.mammalian_species is not None:
                self._mammalian_species = list(cfg.mammalian_species)

        # Restrict hal2maf to the genomes we actually score (ref + primates +
        # mammals) instead of all 241 — benchmarked ~8x faster per query (4s
        # warm vs 33s) and keeps every query well under the timeout.
        self._target_genomes: list[str] = list(
            dict.fromkeys([self._ref_genome, *self._primate_species, *self._mammalian_species])
        )

        self._available = self._hal_path is not None and hal2maf_available(self._binary)
        self._depth_map: dict[str, int] = self._load_depth_map()

        if not self._available:
            logger.info(
                "ConservationFrameModule inactive — hal_path=%s, binary_on_path=%s",
                self._hal_path,
                hal2maf_available(self._binary),
            )

    def _load_depth_map(self) -> dict[str, int]:
        """Pull the Cactus species tree and build ``{species: depth_from_ref}``.

        Priority: explicit ``cfg.hal_tree_newick`` > ``halStats --tree`` on
        the HAL > empty dict. Failing to build a depth map isn't fatal; the
        module still returns frame metrics but emits ``None`` for the
        deepest-species fields.
        """
        newick = self._tree_newick
        if newick is None and self._hal_path is not None:
            newick = extract_tree(self._hal_path, halstats_binary=self._halstats_binary)
        if not newick:
            return {}
        try:
            return depth_from_reference(newick, self._ref_genome)
        except ValueError as exc:
            logger.warning("Failed to parse Cactus species tree: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Public API (SiteModule protocol)
    # ------------------------------------------------------------------

    def annotate_site(self, site: TranslationInitiationSite) -> dict[str, Any]:
        """Extract MAF and score frame integrity for *site*.

        Returns a dict with ``primate_*`` and ``mammalian_*`` aggregates plus
        a ``summary`` sub-dict. ``summary.status`` is one of:

        - ``"ok"`` — HAL queried and at least one species aligned over the
          unique region.
        - ``"not_run"`` — HAL or binary unavailable.
        - ``"no_skeleton"`` — ``site.orf_exons`` empty (no Layer-2 data).
        - ``"no_unique_region"`` — isoform and canonical ORF exons coincide.
        - ``"no_alignment"`` — HAL queried successfully but no block covered
          the reference region.
        """
        if not self._available:
            return self._empty_result(status="not_run")

        if not site.orf_exons:
            return self._empty_result(status="no_skeleton")

        from swissisoform.coords import interval_difference
        from swissisoform.models import ORFType

        # ORF-type-aware unique region. Truncations lose canonical N-terminal
        # sequence, so their unique region is ``canonical \ isoform`` (the
        # lost intervals), not the empty ``isoform \ canonical`` direction.
        if site.orf_type == ORFType.TRUNCATED:
            unique = interval_difference(site.canonical_orf_exons, site.orf_exons)
            unique_space = "canonical"
        else:
            unique = interval_difference(site.orf_exons, site.canonical_orf_exons)
            unique_space = "isoform"
        if not unique:
            return self._empty_result(status="no_unique_region")

        ref_seq, per_species_seqs = self._fetch_alignment(site.chrom, unique, site.strand)
        if ref_seq is None:
            return self._empty_result(status="no_alignment")

        primate_results = [
            analyze_species(sp, ref_seq, per_species_seqs.get(sp, "-" * len(ref_seq)))
            for sp in self._primate_species
        ]
        mammalian_results = [
            analyze_species(sp, ref_seq, per_species_seqs.get(sp, "-" * len(ref_seq)))
            for sp in self._mammalian_species
        ]

        primate_agg = aggregate_species_results(primate_results)
        mammalian_agg = aggregate_species_results(mammalian_results)
        primate_deepest = self._deepest_intact(primate_results)
        mammalian_deepest = self._deepest_intact(mammalian_results)

        # ``start_codon_conserved`` reports conservation of the first codon of
        # the unique region. For extensions/uORFs/altORFs that IS the TIS's
        # alt-start codon, so the metric is semantically "alt-start codon
        # conserved across orthologs". For TRUNCATIONS the unique region is
        # canonical \ isoform — its first codon is the CANONICAL Met (always
        # ATG in hg38), not the isoform's alt-start. The metric would then
        # measure canonical-Met conservation, which is a different question
        # from "alt-start conserved" and would mislead a reader who took the
        # column name at face value. Emit None for truncations with an
        # explicit reason in the summary.
        if site.orf_type == ORFType.TRUNCATED:
            primate_start = None
            mammalian_start = None
            start_codon_note = (
                "not_applicable_truncation: unique region starts at the "
                "canonical Met, not the isoform alt-start"
            )
        else:
            primate_start = primate_agg["start_codon_conserved"]
            mammalian_start = mammalian_agg["start_codon_conserved"]
            start_codon_note = None

        return {
            "primate_n_species_aligned": primate_agg["n_species_aligned"],
            "primate_n_species_intact_frame": primate_agg["n_species_intact_frame"],
            "primate_frac_intact": primate_agg["frac_intact"],
            "primate_start_codon_conserved": primate_start,
            "primate_mean_pident": primate_agg["mean_pident"],
            "primate_deepest_species": primate_deepest[0],
            "primate_max_depth": primate_deepest[1],
            "mammalian_n_species_aligned": mammalian_agg["n_species_aligned"],
            "mammalian_n_species_intact_frame": mammalian_agg["n_species_intact_frame"],
            "mammalian_frac_intact": mammalian_agg["frac_intact"],
            "mammalian_start_codon_conserved": mammalian_start,
            "mammalian_mean_pident": mammalian_agg["mean_pident"],
            "mammalian_deepest_species": mammalian_deepest[0],
            "mammalian_max_depth": mammalian_deepest[1],
            "summary": {
                "status": "ok",
                "unique_space": unique_space,
                "unique_region_nt": sum(e - s for s, e in unique),
                "hal_path": str(self._hal_path),
                "tree_loaded": bool(self._depth_map),
                "start_codon_note": start_codon_note,
            },
        }

    def _deepest_intact(
        self, results: list[Any]
    ) -> tuple[str | None, int | None]:
        """Return ``(species_name, depth)`` for the deepest intact-frame species.

        Ranks by ``self._depth_map`` (built once from the Cactus tree).
        Species without a depth entry are ignored — their phylogenetic
        position is unknown, and ranking them would be misleading. When the
        depth map is empty (no tree loaded) or no species qualify, returns
        ``(None, None)``.
        """
        if not self._depth_map:
            return None, None
        best_name: str | None = None
        best_depth: int | None = None
        for res in results:
            if not (res.aligned and res.frame_intact):
                continue
            depth = self._depth_map.get(res.species)
            if depth is None:
                continue
            if best_depth is None or depth > best_depth:
                best_depth = depth
                best_name = res.species
        return best_name, best_depth

    def run(
        self,
        tis_sites: list[TranslationInitiationSite],
    ) -> list[TranslationInitiationSite]:
        """Annotate every TIS and attach to ``isoform_annotations``."""
        for site in tis_sites:
            site.isoform_annotations[self.MODULE_NAME] = self.annotate_site(site)
        return tis_sites

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fetch_alignment(
        self,
        chrom: str,
        intervals: list[tuple[int, int]],
        strand: str,
    ) -> tuple[str | None, dict[str, str]]:
        """Call ``hal2maf`` per interval, concatenate per-species aligned rows.

        For minus-strand genes, each interval's alignment is
        reverse-complemented and the intervals are concatenated in
        descending genomic order so the resulting reference sequence reads
        5'→3' in the mRNA.

        Returns ``(ref_seq, {species: seq})``. ``ref_seq`` is ``None`` when
        no MAF blocks contained the reference genome.
        """
        all_species: set[str] = set()
        per_interval_rows: list[tuple[str, dict[str, str]]] = []

        ordered = sorted(intervals, reverse=(strand == "-"))
        for start, end in ordered:
            maf_text = extract_maf(
                self._hal_path,
                chrom,
                start,
                end - start,
                ref_genome=self._ref_genome,
                hal2maf_binary=self._binary,
                timeout=self._hal_timeout,
                target_genomes=self._target_genomes,
            )
            if not maf_text:
                continue
            blocks = parse_maf(maf_text)
            if not blocks:
                continue
            rows_for_interval: dict[str, str] = {}
            # Collect every species that appears in at least one block.
            for block in blocks:
                all_species.update(block.rows)
            for species in all_species:
                extracted = concat_species_rows(blocks, species, self._ref_genome)
                if extracted is None:
                    continue
                _, sp_seq = extracted
                rows_for_interval[species] = sp_seq
            # Also stash the reference sequence for this interval.
            ref_extracted = concat_species_rows(blocks, self._ref_genome, self._ref_genome)
            ref_seq_interval = ref_extracted[0] if ref_extracted is not None else ""
            per_interval_rows.append((ref_seq_interval, rows_for_interval))

        if not per_interval_rows:
            return None, {}

        # Pad missing species per interval with '-' of the reference width.
        ref_parts: list[str] = []
        species_parts: dict[str, list[str]] = {sp: [] for sp in all_species}
        for ref_seq_interval, rows in per_interval_rows:
            width = len(ref_seq_interval)
            ref_parts.append(ref_seq_interval)
            for sp in all_species:
                species_parts[sp].append(rows.get(sp, "-" * width))

        ref_seq = "".join(ref_parts)
        per_species_seqs = {sp: "".join(parts) for sp, parts in species_parts.items()}

        if strand == "-":
            ref_seq = _revcomp_maf(ref_seq)
            per_species_seqs = {sp: _revcomp_maf(seq) for sp, seq in per_species_seqs.items()}

        return ref_seq, per_species_seqs

    @staticmethod
    def _empty_result(*, status: str) -> dict[str, Any]:
        """Shape an output dict with every metric ``None`` and a reason code."""
        return {
            "primate_n_species_aligned": None,
            "primate_n_species_intact_frame": None,
            "primate_frac_intact": None,
            "primate_start_codon_conserved": None,
            "primate_mean_pident": None,
            "primate_deepest_species": None,
            "primate_max_depth": None,
            "mammalian_n_species_aligned": None,
            "mammalian_n_species_intact_frame": None,
            "mammalian_frac_intact": None,
            "mammalian_start_codon_conserved": None,
            "mammalian_mean_pident": None,
            "mammalian_deepest_species": None,
            "mammalian_max_depth": None,
            "summary": {"status": status},
        }


_COMPLEMENT = str.maketrans("ACGTNacgtn-", "TGCANtgcan-")


def _revcomp_maf(seq: str) -> str:
    """Reverse-complement a MAF-aligned sequence, preserving gap characters."""
    return seq.translate(_COMPLEMENT)[::-1]
