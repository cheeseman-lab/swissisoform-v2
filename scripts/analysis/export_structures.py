"""Export named, viewable Boltz structures (.cif) for a run's isoforms.

Boltz caches structures by sequence sha1 (data/cache/structure/boltz/<hash>/
model.cif), which is unusable for a human. This re-assembles a preset's genes,
maps each isoform (and its canonical) protein to its cache entry by hash, and
copies the model.cif into data/output/<run_name>/structures/ with names like
``CBX1__extended__953aa__chr17-48071434--CTG.cif`` — plus a manifest.tsv.

Open the resulting .cif in ChimeraX (`open file.cif`, colour by B-factor =
pLDDT), PyMOL, or the Mol* web viewer (molstar.org/viewer, drag-and-drop).

Usage:
    python scripts/analysis/export_structures.py --preset cheeseman12
    python scripts/analysis/export_structures.py --preset 5gene
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))  # so `import run` (scripts/run.py) resolves

import build_folding_colors  # noqa: E402, I001
import run  # noqa: E402, I001
from swissisoform.plm.embed import protein_hash  # noqa: E402

BOLTZ = ROOT / "data" / "cache" / "structure" / "boltz"


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "-", s)


def _cif_for(seq: str | None) -> Path | None:
    """Locate the cached Boltz model.cif for a protein sequence, or None."""
    if not seq:
        return None
    cif = BOLTZ / protein_hash(seq.rstrip("*").upper()) / "model.cif"
    return cif if cif.exists() else None


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
    outdir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    n_iso = n_can = n_missing = 0
    seen_canonical: set[str] = set()

    for gene in sorted(genes, key=lambda g: g.gene_name):
        if gene.canonical_protein and gene.gene_name not in seen_canonical:
            seen_canonical.add(gene.gene_name)
            cif = _cif_for(gene.canonical_protein)
            if cif:
                aa = len(gene.canonical_protein.rstrip("*"))
                name = f"{gene.gene_name}__canonical__{aa}aa.cif"
                shutil.copyfile(cif, outdir / name)
                n_can += 1
                manifest.append({"file": name, "gene": gene.gene_name, "kind": "canonical",
                                 "orf_type": "", "aa_len": aa, "tis_id": ""})
        for site in gene.tis_sites:
            if not site.isoform_protein:
                continue
            cif = _cif_for(site.isoform_protein)
            aa = len(site.isoform_protein.rstrip("*"))
            ot = site.orf_type.value
            name = f"{gene.gene_name}__{ot}__{aa}aa__{_safe(site.tis_id)}.cif"
            if cif:
                shutil.copyfile(cif, outdir / name)
                n_iso += 1
                manifest.append({"file": name, "gene": gene.gene_name, "kind": "isoform",
                                 "orf_type": ot, "aa_len": aa, "tis_id": site.tis_id})
            else:
                n_missing += 1

    man_path = outdir / "manifest.tsv"
    with open(man_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "gene", "kind", "orf_type", "aa_len", "tis_id"],
                           delimiter="\t")
        w.writeheader()
        w.writerows(manifest)

    print(f"wrote {n_iso} isoform + {n_can} canonical .cif → {outdir}")
    print(f"  manifest: {man_path}")
    print(f"  {n_missing} isoforms had no cached structure (too long for Boltz, or not folded)")
    print("View in ChimeraX (colour by bfactor = pLDDT) or molstar.org/viewer")

    # Precompute the per-residue folding colour maps the site's 3Dmol viewer
    # applies (whole protein by pLDDT, differential region on a diverging ramp).
    parquet = ROOT / "data" / "output" / run_name / "all_paired.parquet"
    if parquet.is_file():
        print(f"building folding colour maps from {parquet.name}…")
        build_folding_colors.main(["--parquet", str(parquet), "--structures", str(outdir)])
    else:
        print(f"  (skipped colour maps: {parquet} not found — run scripts/run.py first)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
