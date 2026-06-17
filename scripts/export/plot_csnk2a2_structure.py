"""Boltz-vs-ESMFold2 structure comparison for the CSNK2A2 N-terminal extension
(isoform chr16:58197883:-:CTG:ENST00000262506.8, 399 aa, 49-aa extension).

Loads both backends' CIFs for the isoform, superposes them (tmtools), and draws
side-by-side Cα backbones in a common orientation with the 49-aa extension
highlighted; annotates TM-score + RMSD between the two folds. Self-contained PNG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from swissisoform.structure.compare import (  # noqa: E402
    _ca_coords_and_seq,
    _try_load_structure,
    compare_structures,
)

ROOT = Path(__file__).resolve().parents[2]
H = "5545f11f038d446a2466d31dafef89c3bfcf15af"  # CSNK2A2 isoform protein hash
EXT = 49  # N-terminal extension length (residues)
OUT = ROOT / "data" / "output" / "cheeseman_13gene" / "structure_confidence_plots" / "structure_csnk2a2_extension.png"


def _ca(backend: str):
    cif = ROOT / "data" / "cache" / "structure" / backend / H / "model.cif"
    xyz, seq = _ca_coords_and_seq(_try_load_structure(cif))
    return np.asarray(xyz, float), seq


def main() -> int:
    """Render the side-by-side superposed backbones + TM/RMSD."""
    boltz_xyz, boltz_seq = _ca("boltz")
    esm_xyz, esm_seq = _ca("esmfold2")

    cmp = compare_structures(
        ROOT / "data" / "cache" / "structure" / "boltz" / H / "model.cif",
        ROOT / "data" / "cache" / "structure" / "esmfold2" / H / "model.cif",
        diff_isoform_start=0, diff_isoform_end=EXT,
    )
    tm, rmsd = cmp["tm_score"], cmp["rmsd_global"]
    print(f"CSNK2A2 isoform boltz-vs-esmfold2: TM={tm:.3f} RMSD={rmsd:.2f} Å  "
          f"(boltz {len(boltz_xyz)} CA, esmfold2 {len(esm_xyz)} CA)")

    # Superpose boltz onto esmfold2 (tmtools rotation) so both share an orientation.
    from tmtools import tm_align
    ali = tm_align(boltz_xyz, esm_xyz, boltz_seq, esm_seq)
    boltz_agn = boltz_xyz @ ali.u.T + ali.t

    fig = plt.figure(figsize=(11, 5.2))
    for k, (name, xyz) in enumerate([("Boltz-2", boltz_agn), ("ESMFold2", esm_xyz)]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        n = len(xyz)
        ax.plot(*xyz[EXT:].T, color="#9aa7b4", lw=1.6, label=f"shared ({n - EXT} aa)")
        ax.plot(*xyz[:EXT].T, color="#b5532a", lw=3.0, label=f"N-ext ({EXT} aa)")
        ax.scatter(*xyz[0].T, color="#b5532a", s=30)  # N-terminus
        ax.set_title(f"{name}", fontsize=13)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.legend(loc="upper left", fontsize=8)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
    fig.suptitle(
        f"CSNK2A2 extension (399 aa) — Boltz-2 vs ESMFold2   "
        f"TM-score = {tm:.3f},  RMSD = {rmsd:.2f} Å",
        fontsize=13)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
