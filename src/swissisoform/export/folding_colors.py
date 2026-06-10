"""Per-residue folding colours for the interactive 3Dmol viewer.

Flag G. The differential-region recolouring is decided **offline, on disk** here;
the website's 3Dmol.js viewer is a dumb display that loads the CIF and applies
this precomputed palette (no fragile in-browser recolouring).

Per structure we write ``<gene>__<side>__<segment>.colors.json`` =
``{"<resi>": "#rrggbb", ...}`` into ``<structures>/colors/``. Every residue is
coloured by its pLDDT (CA b-factor) on the standard ramp; residues in the
differential region (on whichever structure carries it — canonical for
truncations, isoform for extensions) use a distinct diverging ramp so the region
stands out while staying confidence-readable.

Pure Python (no PyMOL) — it's just ramp arithmetic over the CIF b-factor column.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# pLDDT band (b-factor); clamp to the informative range so mid-confidence
# variation is visible rather than washed out. Mirrors the values used in the
# PyMOL prototype Matteo signed off on.
PLDDT_MIN, PLDDT_MAX = 50.0, 90.0
# Three-stop ramps as (R, G, B). Core = standard pLDDT (orange→white→blue);
# diff = diverging (yellow→white→purple).
CORE_RAMP = ((230, 126, 34), (245, 245, 245), (30, 80, 200))
DIFF_RAMP = ((245, 220, 40), (245, 245, 245), (140, 40, 160))


def _ramp(b: float, ramp: tuple) -> str:
    t = max(0.0, min(1.0, (b - PLDDT_MIN) / (PLDDT_MAX - PLDDT_MIN)))
    c0, c1, c2 = ramp
    if t < 0.5:
        u, a, c = t * 2, c0, c1
    else:
        u, a, c = (t - 0.5) * 2, c1, c2
    rgb = tuple(round(a[i] + (c[i] - a[i]) * u) for i in range(3))
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _ca_bfactors(cif_path: Path) -> dict[int, float]:
    """{auth_seq_id: pLDDT} from a Boltz/AlphaFold CIF's CA B-factor column."""
    lines = cif_path.read_text().splitlines()
    headers = [ln.strip().split(".", 1)[1] for ln in lines if ln.startswith("_atom_site.")]
    try:
        ai = headers.index("label_atom_id")
        si = headers.index("auth_seq_id")
        bi = headers.index("B_iso_or_equiv")
    except ValueError:
        return {}
    out: dict[int, float] = {}
    for ln in lines:
        if not ln.startswith("ATOM"):
            continue
        f = ln.split()
        if len(f) <= max(ai, si, bi) or f[ai] != "CA":
            continue
        try:
            out[int(f[si])] = float(f[bi])
        except ValueError:
            continue
    return out


def _colors_for(cif_path: Path, diff_lo: int | None, diff_hi: int | None) -> dict[str, str]:
    cas = _ca_bfactors(cif_path)
    colors: dict[str, str] = {}
    for resi, b in cas.items():
        in_diff = diff_lo is not None and diff_hi is not None and diff_lo <= resi <= diff_hi
        colors[str(resi)] = _ramp(b, DIFF_RAMP if in_diff else CORE_RAMP)
    return colors


def _tis_id_to_struct_segment(tis_id: str) -> str:
    parts = tis_id.split(":")
    if len(parts) >= 5:
        chrom, pos, codon, tid = parts[0], parts[1], parts[3], ":".join(parts[4:])
        return f"{chrom}-{pos}---{codon}-{tid}"
    return tis_id.replace(":", "-")


def _index_structures(structures_dir: Path) -> dict[tuple[str, str], str]:
    index: dict[tuple[str, str], str] = {}
    for cif in structures_dir.glob("*.cif"):
        parts = cif.name[:-4].split("__")
        if len(parts) < 3:
            continue
        gene, kind = parts[0], parts[1]
        if kind == "canonical":
            index[(gene, "canonical")] = cif.name
        elif len(parts) >= 4:
            index[(gene, parts[3])] = cif.name
    return index


def build_folding_colors(
    parquet: Path, structures: Path, colors: Path | None = None
) -> int:
    """Write per-residue colour maps for the structures of a paired parquet.

    Returns the number of colour maps written (0 if none were written).
    """
    structures_dir = structures.resolve()
    colors_dir = (colors or structures_dir / "colors").resolve()
    colors_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet)
    index = _index_structures(structures_dir)
    written = 0
    seen: set[str] = set()

    for _, row in df.iterrows():
        gene = str(row["gene_name"])
        seg = _tis_id_to_struct_segment(str(row["tis_id"]))
        diff_space = str(row.get("diff_space") or "")
        try:
            lo, hi = int(row.get("diff_start")) + 1, int(row.get("diff_end"))
        except (TypeError, ValueError):
            lo = hi = None
        has_diff = lo is not None and hi is not None and hi >= lo

        for side, key in (("canonical", (gene, "canonical")), ("isoform", (gene, seg))):
            cif = index.get(key)
            if not cif:
                continue
            out_name = f"{gene}__{side}__{seg}.colors.json"
            if out_name in seen:
                continue
            seen.add(out_name)
            carries = has_diff and (
                (side == "canonical" and diff_space == "canonical")
                or (side == "isoform" and diff_space != "canonical")
            )
            colors_map = _colors_for(
                structures_dir / cif,
                lo if carries else None,
                hi if carries else None,
            )
            if not colors_map:
                print(f"WARN no CA b-factors in {cif}", file=sys.stderr)
                continue
            (colors_dir / out_name).write_text(json.dumps(colors_map, separators=(",", ":")))
            written += 1

    print(f"wrote {written} colour maps -> {colors_dir}")
    return written
