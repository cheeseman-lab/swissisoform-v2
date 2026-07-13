"""Thin CLI for the named-structure export.

Logic lives in ``swissisoform.export.structures``. Re-assembles a preset's
genes (via scripts/run.py), copies each isoform/canonical's cached Boltz
model.cif into ``data/output/<run_name>/structures/`` under human-readable
names, then precomputes the per-residue folding colour maps.

Usage:
    python scripts/export/export_structures.py --preset cheeseman13
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

from swissisoform import runner  # noqa: E402, I001
from swissisoform.assembly import assemble_genes  # noqa: E402
from swissisoform.export.folding_colors import build_folding_colors  # noqa: E402
from swissisoform.export.pae import export_pae  # noqa: E402
from swissisoform.export.structures import export_structures  # noqa: E402
from swissisoform.pipeline import UpstreamReference  # noqa: E402
from swissisoform.references import (  # noqa: E402
    ALL_CELL_LINES,
    GENOME,
    GTF,
    PRESETS,
    PROTEIN,
    build_config,
)


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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--preset", required=True, choices=sorted(PRESETS))
    args = p.parse_args(argv)

    spec, genes = assemble_for_preset(args.preset)
    run_name = spec.get("run_name", args.preset)
    outdir = ROOT / "data" / "output" / run_name / "structures"

    n_iso, n_can, n_missing, man_path = export_structures(genes, outdir)

    print(f"wrote {n_iso} isoform + {n_can} canonical .cif → {outdir}")
    print(f"  manifest: {man_path}")
    print(f"  {n_missing} isoforms had no cached structure (too long for Boltz, or not folded)")
    print("View in ChimeraX (colour by bfactor = pLDDT) or molstar.org/viewer")

    # PAE heatmaps for the folding panel — one JSON per structure from the
    # cached pae.npy (entries folded before PAE capture are skipped).
    n_pae, n_pae_missing, pae_dir = export_pae(genes, outdir)
    print(f"wrote {n_pae} PAE map(s) → {pae_dir} ({n_pae_missing} without cached PAE)")

    # Precompute the per-residue folding colour maps the site's 3Dmol viewer
    # applies (whole protein by pLDDT, differential region on a diverging ramp).
    parquet = ROOT / "data" / "output" / run_name / "all_paired.parquet"
    if parquet.is_file():
        print(f"building folding colour maps from {parquet.name}…")
        build_folding_colors(parquet, outdir)
    else:
        print(f"  (skipped colour maps: {parquet} not found — run scripts/run.py first)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
