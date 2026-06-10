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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))  # so `import run` (scripts/run.py) resolves

import run  # noqa: E402, I001
from swissisoform.export.folding_colors import build_folding_colors  # noqa: E402
from swissisoform.export.structures import export_structures  # noqa: E402


def assemble_for_preset(preset_name: str):
    """Reproduce run.py's load + assemble for *preset_name*; return (spec, genes)."""
    spec = run.PRESETS[preset_name]
    ref = run.UpstreamReference.load(
        gtf_path=run.GTF, genome_fasta=run.GENOME, protein_fasta=run.PROTEIN
    )
    if "isoforms" in spec:
        combined = run.load_combined(run.ALL_CELL_LINES, ref, run.build_config())
        final, gene_names = run.restrict_to_isoforms(
            combined, run.load_isoform_picks(spec["isoforms"])
        )
    else:
        final = run.load_single_sample(spec.get("cell_lines", ["HeLa"])[0], ref)
        gene_names = spec["genes"]
    genes = run.assemble_genes(
        final, gene_names=gene_names, genome_fasta=run.GENOME, exon_skeletons=ref.exon_skeletons
    )
    return spec, genes


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--preset", required=True, choices=sorted(run.PRESETS))
    args = p.parse_args(argv)

    spec, genes = assemble_for_preset(args.preset)
    run_name = spec.get("run_name", args.preset)
    outdir = ROOT / "data" / "output" / run_name / "structures"

    n_iso, n_can, n_missing, man_path = export_structures(genes, outdir)

    print(f"wrote {n_iso} isoform + {n_can} canonical .cif → {outdir}")
    print(f"  manifest: {man_path}")
    print(f"  {n_missing} isoforms had no cached structure (too long for Boltz, or not folded)")
    print("View in ChimeraX (colour by bfactor = pLDDT) or molstar.org/viewer")

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
