"""Export named, viewable Boltz structures (.cif) for a run's isoforms.

Folding backends cache structures by sequence sha1
(data/cache/structure/<backend>/<hash>/model.cif, default backend ``esmfold2``),
which is unusable for a human. This maps each isoform (and its
canonical) protein to its cache entry by hash and copies the model.cif into a
run's ``structures/`` directory with names like
``CBX1__extended__953aa__chr17-48071434--CTG.cif`` — plus a manifest.tsv.

Open the resulting .cif in ChimeraX (`open file.cif`, colour by B-factor =
pLDDT), PyMOL, or the Mol* web viewer (molstar.org/viewer, drag-and-drop).
"""

from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path

from swissisoform.plm.embed import protein_hash
from swissisoform.structure.fold import DEFAULT_BACKEND, DEFAULT_CACHE_DIR, cache_path

ROOT = Path(__file__).resolve().parents[3]


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "-", s)


def _cif_for(seq: str | None, backend: str = DEFAULT_BACKEND) -> Path | None:
    """Locate the cached model.cif for a protein sequence, or None."""
    if not seq:
        return None
    cif = cache_path(DEFAULT_CACHE_DIR, backend, protein_hash(seq.rstrip("*").upper())) / "model.cif"
    return cif if cif.exists() else None


def export_structures(
    genes, outdir: Path, backend: str = DEFAULT_BACKEND
) -> tuple[int, int, int, Path]:
    """Copy cached folded .cif files for *genes* into *outdir* + write a manifest.

    Reads the ``data/cache/structure/<backend>/<hash>/model.cif`` cache (default
    backend ``esmfold2``). Returns ``(n_iso, n_can, n_missing, manifest_path)``.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    n_iso = n_can = n_missing = 0
    seen_canonical: set[str] = set()

    for gene in sorted(genes, key=lambda g: g.gene_name):
        if gene.canonical_protein and gene.gene_name not in seen_canonical:
            seen_canonical.add(gene.gene_name)
            cif = _cif_for(gene.canonical_protein, backend)
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
            cif = _cif_for(site.isoform_protein, backend)
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

    return n_iso, n_can, n_missing, man_path
