"""Thin CLI for the named-structure export.

Logic lives in ``swissisoform.export.structures``. Re-assembles a preset's
genes (via scripts/run.py), copies each isoform/canonical's cached Boltz
model.cif into ``data/output/<run_name>/structures/`` under human-readable
names, then precomputes the per-residue folding colour maps.

A run built from external start positions (``scripts/run_external_starts.py``) has
no preset; ``--run`` re-assembles its genes from the rows that run wrote
(``data/output/<run>/external_tis_rows.csv``) instead.

Usage:
    python scripts/export/export_structures.py --preset cheeseman13
    python scripts/export/export_structures.py --run BE_nc_hits
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

from swissisoform.export.folding_colors import build_folding_colors  # noqa: E402
from swissisoform.export.pae import export_pae  # noqa: E402
from swissisoform.export.presets import assemble_for_preset  # noqa: E402
from swissisoform.export.structures import export_structures  # noqa: E402
from swissisoform.references import (  # noqa: E402
    PRESETS,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--preset", choices=sorted(PRESETS))
    src.add_argument("--run", help="an external-starts run (reads its external_tis_rows.csv)")
    args = p.parse_args(argv)

    if args.preset:
        spec, genes = assemble_for_preset(args.preset)
        run_name = spec.get("run_name", args.preset)
    else:
        import pandas as pd

        from swissisoform.assembly import assemble_genes

        run_name = args.run
        rows_csv = ROOT / "data" / "output" / run_name / "external_tis_rows.csv"
        if not rows_csv.is_file():
            p.error(f"{rows_csv} not found — --run is for runs made by run_external_starts.py")
        genes = assemble_genes(pd.read_csv(rows_csv))
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
