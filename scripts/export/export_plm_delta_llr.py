"""Export per-isoform ESM-C delta-LLR (PLM VEP) metrics to a CSV.

Re-assembles a preset's genes (same load path as scripts/export/export_structures.py),
runs ``PLMVEPModule`` against the cached ESM-C masked-marginal LLR
(``data/cache/plm_esmc/``), and writes one row per isoform TIS with its
unique-vs-shared region constraint metrics — the "delta LLR".

The cached LLR is produced by scripts/slurm/run_plm_embed.sbatch. This script
does NO model inference; it's a pure cache read + the diff-region slicing that
the pipeline normally bakes into all_paired.parquet.

Usage:
    python scripts/export/export_plm_delta_llr.py --preset cheeseman13
    # → data/output/<run_name>/plm_vep_delta_llr.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

from swissisoform import runner  # noqa: E402, I001
from swissisoform.assembly import assemble_genes  # noqa: E402
from swissisoform.pipeline import UpstreamReference  # noqa: E402
from swissisoform.plm.embed import DEFAULT_CACHE_DIR as PLM_CACHE_DIR  # noqa: E402
from swissisoform.plm.embed import protein_hash  # noqa: E402
from swissisoform.plm.module import PLMVEPModule  # noqa: E402
from swissisoform.references import (  # noqa: E402
    ALL_CELL_LINES,
    GENOME,
    GTF,
    PRESETS,
    PROTEIN,
    build_config,
)

FIELDS = [
    "gene",
    "tis_id",
    "orf_type",
    "aa_len",
    "status",
    "mean_llr_isoform",
    "mean_llr_canonical",
    "mean_llr_unique_region",
    "mean_llr_shared_region",
    "constraint_enrichment",
    "n_constrained_positions_unique",
    "n_constrained_positions_shared",
    "isoform_protein_hash",
    "canonical_protein_hash",
]


def assemble_for_preset(preset_name: str):
    """Reproduce the runner's load + assemble for *preset_name*; return (spec, genes)."""
    spec = PRESETS[preset_name]
    ref = UpstreamReference.load(gtf_path=GTF, genome_fasta=GENOME, protein_fasta=PROTEIN)
    if "isoforms" in spec:
        combined = runner.load_combined(ALL_CELL_LINES, ref, build_config())
        final, gene_names = runner.restrict_to_isoforms(
            combined, runner.load_isoform_picks(spec["isoforms"])
        )
    else:
        final = runner.load_single_sample(spec.get("cell_lines", ["HeLa"])[0], ref)
        gene_names = spec["genes"]
    genes = assemble_genes(
        final, gene_names=gene_names, genome_fasta=GENOME, exon_skeletons=ref.exon_skeletons
    )
    return spec, genes


def _hash(seq: str | None) -> str:
    return protein_hash(seq) if seq else ""


def main(argv: list[str] | None = None) -> int:
    """Assemble the preset, run PLM VEP off the ESM-C cache, write the CSV."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--preset", default="cheeseman13", choices=sorted(PRESETS))
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=PLM_CACHE_DIR,
        help="ESM-C LLR cache dir (default data/cache/plm_esmc/).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV (default data/output/<run_name>/plm_vep_delta_llr.csv).",
    )
    p.add_argument(
        "--constraint-threshold",
        type=float,
        default=-5.0,
        help="LLR below which a residue counts as constrained (provisional for ESM-C).",
    )
    args = p.parse_args(argv)

    spec, genes = assemble_for_preset(args.preset)
    run_name = spec.get("run_name", args.preset)
    out = args.out or (ROOT / "data" / "output" / run_name / "plm_vep_delta_llr.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    module = PLMVEPModule(
        build_config(),
        cache_dir=args.cache_dir,
        constraint_threshold=args.constraint_threshold,
    )

    rows: list[dict] = []
    for gene in sorted(genes, key=lambda g: g.gene_name):
        for site in gene.tis_sites:
            if not site.isoform_protein:
                continue
            ann = module.annotate_site(site)
            rows.append(
                {
                    "gene": gene.gene_name,
                    "tis_id": site.tis_id,
                    "orf_type": site.orf_type.value if site.orf_type else "",
                    "aa_len": len(site.isoform_protein.rstrip("*")),
                    "status": ann["status"],
                    "mean_llr_isoform": ann["mean_llr_isoform"],
                    "mean_llr_canonical": ann["mean_llr_canonical"],
                    "mean_llr_unique_region": ann["mean_llr_unique_region"],
                    "mean_llr_shared_region": ann["mean_llr_shared_region"],
                    "constraint_enrichment": ann["constraint_enrichment"],
                    "n_constrained_positions_unique": ann["n_constrained_positions_unique"],
                    "n_constrained_positions_shared": ann["n_constrained_positions_shared"],
                    "isoform_protein_hash": _hash(site.isoform_protein),
                    "canonical_protein_hash": _hash(site.canonical_protein),
                }
            )

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_nocache = sum(1 for r in rows if r["status"] == "no_cache")
    print(f"wrote {len(rows)} isoform rows ({n_ok} status=ok, {n_nocache} no_cache) → {out}")
    if n_nocache:
        print(
            f"  ⚠ {n_nocache} isoforms had no cached ESM-C LLR — run "
            f"scripts/slurm/run_plm_embed.sbatch first (cache dir: {args.cache_dir})."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
