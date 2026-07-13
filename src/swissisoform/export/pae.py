"""Export per-protein PAE (predicted aligned error) maps for the website.

ESMFold2 computes the full L×L PAE every fold; the fold cache now persists it as
``<hash>/pae.npy`` (float16). This maps each isoform (and its canonical) protein
to its cache entry by sequence hash and writes a compact per-structure JSON into
a run's ``structures/pae/`` directory, named to match the folding-colour maps:

    <gene>__<side>__<segment>.pae.json   # side ∈ {canonical, isoform}

where ``<segment>`` is the sanitized tis_id (same convention as the CIF /
colour-map filenames). The payload is ``{"L": n, "pae": [row-major floats]}``
rounded to 1 decimal (PAE is 0..~32 Å) to keep the file small; the website's
canvas renderer reshapes it by ``L``.

Pure Python + numpy — it just reshapes the cached array. Entries folded before
PAE capture (no ``pae.npy``) are skipped and counted as missing.
"""

from __future__ import annotations

import json
from pathlib import Path

from swissisoform.export.folding_colors import _tis_id_to_struct_segment
from swissisoform.plm.embed import protein_hash
from swissisoform.structure.fold import DEFAULT_BACKEND, DEFAULT_CACHE_DIR, cache_path


def _pae_for(seq: str | None, backend: str = DEFAULT_BACKEND) -> Path | None:
    """Locate the cached pae.npy for a protein sequence, or None."""
    if not seq:
        return None
    pae = cache_path(DEFAULT_CACHE_DIR, backend, protein_hash(seq.rstrip("*").upper())) / "pae.npy"
    return pae if pae.exists() else None


def _write_pae_json(pae_path: Path, out_path: Path) -> bool:
    """Reshape one cached pae.npy into the website JSON. Returns True on success."""
    import numpy as np

    arr = np.asarray(np.load(pae_path))
    arr = np.squeeze(arr)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return False
    payload = {
        "L": int(arr.shape[0]),
        # row-major, rounded to 0.1 Å — the renderer only needs colour resolution.
        "pae": [round(float(v), 1) for v in arr.ravel()],
    }
    out_path.write_text(json.dumps(payload, separators=(",", ":")))
    return True


def export_pae(
    genes, outdir: Path, backend: str = DEFAULT_BACKEND
) -> tuple[int, int, Path]:
    """Write per-structure PAE JSON for *genes* into ``<outdir>/pae/``.

    Reads the ``data/cache/structure/<backend>/<hash>/pae.npy`` cache. One JSON
    per (gene, side, isoform-segment); canonical maps are written once per
    isoform segment so the website lookup mirrors the colour-map convention.

    Returns ``(n_written, n_missing, pae_dir)`` where ``n_missing`` counts
    (gene, side, segment) targets whose structure had no cached PAE.
    """
    pae_dir = Path(outdir) / "pae"
    pae_dir.mkdir(parents=True, exist_ok=True)

    n_written = n_missing = 0
    seen: set[str] = set()

    for gene in sorted(genes, key=lambda g: g.gene_name):
        for site in gene.tis_sites:
            if not site.isoform_protein:
                continue
            seg = _tis_id_to_struct_segment(str(site.tis_id))
            targets = (
                ("isoform", site.isoform_protein),
                ("canonical", gene.canonical_protein),
            )
            for side, seq in targets:
                if not seq:
                    continue
                out_name = f"{gene.gene_name}__{side}__{seg}.pae.json"
                if out_name in seen:
                    continue
                seen.add(out_name)
                pae_path = _pae_for(seq, backend)
                if pae_path is None or not _write_pae_json(pae_path, pae_dir / out_name):
                    n_missing += 1
                    continue
                n_written += 1

    return n_written, n_missing, pae_dir
