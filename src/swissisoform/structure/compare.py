"""Structure comparison metrics — pLDDT-derived (numpy) + structural (lazy).

Two layers:

- :func:`compare_confidence` — pure pLDDT arithmetic. Diff-region mean,
  shared-region delta, per-region std. Pure-Python / numpy, no heavy
  deps. Always available, always testable.
- :func:`compare_structures` — TM-score, backbone RMSD over shared
  residues, extension↔canonical contact count. Requires ``tmtools`` +
  ``biotite``; imports lazy so the main pipeline env can stay light.
  Each metric returns ``None`` when the dep is missing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _slice_indices(
    n: int, lo: int | None, hi: int | None
) -> list[int]:
    """0-based half-open slice clamped to [0, n)."""
    if lo is None or hi is None:
        return []
    a = max(0, int(lo))
    b = min(n, int(hi))
    if a >= b:
        return []
    return list(range(a, b))


def _mean(xs: list[float]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


def _std(xs: list[float]) -> float | None:
    if not xs:
        return None
    import math

    m = sum(xs) / len(xs)
    return math.sqrt(sum((v - m) ** 2 for v in xs) / len(xs))


def compare_confidence(
    canonical_plddt: list[float] | None,
    isoform_plddt: list[float] | None,
    *,
    diff_isoform_start: int | None,
    diff_isoform_end: int | None,
    diff_canonical_start: int | None,
    diff_canonical_end: int | None,
    orf_type: str | None = None,
) -> dict[str, Any]:
    """Compute pLDDT-derived comparison metrics.

    Args:
        canonical_plddt: Per-residue pLDDT (0-100) over the canonical protein.
        isoform_plddt: Per-residue pLDDT (0-100) over the isoform protein.
        diff_isoform_start: 0-based half-open diff-region start in isoform coords.
        diff_isoform_end: 0-based half-open diff-region end in isoform coords.
        diff_canonical_start: 0-based half-open diff-region start in canonical coords.
        diff_canonical_end: 0-based half-open diff-region end in canonical coords.
        orf_type: ORFType.value (used to pick which protein space carries
            the unique region for truncations).

    Returns:
        Dict with ``plddt_diffregion_mean``, ``plddt_diffregion_std``,
        ``plddt_canonical_mean``, ``plddt_isoform_mean``,
        ``plddt_delta_shared``. Any field is ``None`` when inputs don't
        support it.
    """
    out: dict[str, Any] = {
        "plddt_canonical_mean": _mean(canonical_plddt or []),
        "plddt_isoform_mean": _mean(isoform_plddt or []),
        "plddt_diffregion_mean": None,
        "plddt_diffregion_std": None,
        "plddt_delta_shared": None,
    }

    truncated = (orf_type == "truncated") or (
        diff_isoform_start is None and diff_canonical_start is not None
    )

    if truncated and canonical_plddt:
        idx = _slice_indices(
            len(canonical_plddt), diff_canonical_start, diff_canonical_end
        )
        diff_vals = [canonical_plddt[i] for i in idx]
        out["plddt_diffregion_mean"] = _mean(diff_vals)
        out["plddt_diffregion_std"] = _std(diff_vals)
    elif (not truncated) and isoform_plddt:
        idx = _slice_indices(len(isoform_plddt), diff_isoform_start, diff_isoform_end)
        diff_vals = [isoform_plddt[i] for i in idx]
        out["plddt_diffregion_mean"] = _mean(diff_vals)
        out["plddt_diffregion_std"] = _std(diff_vals)

    # Shared-region delta — extension and truncation differ in which
    # protein space is "complete". Using the isoform protein as anchor is
    # natural for extensions; for truncations it is the canonical.
    if isoform_plddt and canonical_plddt:
        if not truncated and diff_isoform_end is not None:
            iso_shared = isoform_plddt[diff_isoform_end:]
            # Pair against canonical shared region (entirety, since
            # extensions don't lose any canonical residues).
            can_shared = canonical_plddt
            n = min(len(iso_shared), len(can_shared))
            if n > 0:
                iso_m = sum(iso_shared[:n]) / n
                can_m = sum(can_shared[:n]) / n
                out["plddt_delta_shared"] = iso_m - can_m
        elif truncated and diff_canonical_end is not None:
            iso_shared = isoform_plddt
            can_shared = canonical_plddt[diff_canonical_end:]
            n = min(len(iso_shared), len(can_shared))
            if n > 0:
                iso_m = sum(iso_shared[:n]) / n
                can_m = sum(can_shared[:n]) / n
                out["plddt_delta_shared"] = iso_m - can_m

    return out


# ---------------------------------------------------------------------------
# Structural metrics (lazy: tmtools, biotite)
# ---------------------------------------------------------------------------


def _try_load_structure(path: Path) -> Any | None:
    """Load a CIF/PDB into a biotite AtomArray; ``None`` if biotite missing."""
    try:
        from biotite.structure.io.pdbx import CIFFile, get_structure
    except Exception:
        return None

    try:
        cif = CIFFile.read(str(path))
        # Boltz / Chai both write a single model; take model 1.
        struct = get_structure(cif, model=1)
        return struct
    except Exception as exc:  # noqa: BLE001
        logger.debug("biotite failed to load %s: %s", path, exc)
        return None


def _ca_coords_and_seq(struct: Any) -> tuple[Any, str] | None:
    """Return (CA xyz array, single-letter sequence) for a chain."""
    try:
        import numpy as np

        ca_mask = (struct.atom_name == "CA") & (struct.element == "C")
        ca = struct[ca_mask]
        if len(ca) == 0:
            return None
        # Single-letter sequence from 3-letter residue names.
        from biotite.sequence import ProteinSequence

        try:
            seq = ProteinSequence(
                ProteinSequence.convert_letter_3to1(r) for r in ca.res_name
            )
            seq_str = str(seq)
        except Exception:
            # Fallback: unknown residues get X.
            seq_str = "".join(
                ProteinSequence.convert_letter_3to1(r) if r in {
                    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
                    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
                } else "X"
                for r in ca.res_name
            )
        return np.asarray(ca.coord), seq_str
    except Exception as exc:  # noqa: BLE001
        logger.debug("CA extraction failed: %s", exc)
        return None


def compare_structures(
    canonical_cif: Path | None,
    isoform_cif: Path | None,
    *,
    diff_isoform_start: int | None = None,
    diff_isoform_end: int | None = None,
    contact_distance: float = 8.0,
) -> dict[str, Any]:
    """Compute TM-score, shared-region RMSD, and extension contacts.

    Pure scalars. Returns all-``None`` payload when files missing or deps
    unavailable — caller decides whether that's a soft failure.

    Args:
        canonical_cif: Path to the canonical protein's predicted CIF.
        isoform_cif: Path to the isoform protein's predicted CIF.
        diff_isoform_start: Diff-region start in isoform coords (extension contacts).
        diff_isoform_end: Diff-region end in isoform coords.
        contact_distance: Cα-Cα distance threshold (Å) for contacts.

    Returns:
        Dict with ``tm_score``, ``rmsd_shared``, ``extension_contacts``.
    """
    out: dict[str, Any] = {
        "tm_score": None,
        "rmsd_shared": None,
        "extension_contacts": None,
    }
    if canonical_cif is None or isoform_cif is None:
        return out
    if not Path(canonical_cif).exists() or not Path(isoform_cif).exists():
        return out

    can_struct = _try_load_structure(Path(canonical_cif))
    iso_struct = _try_load_structure(Path(isoform_cif))
    if can_struct is None or iso_struct is None:
        return out

    can_pair = _ca_coords_and_seq(can_struct)
    iso_pair = _ca_coords_and_seq(iso_struct)
    if can_pair is None or iso_pair is None:
        return out
    can_xyz, can_seq = can_pair
    iso_xyz, iso_seq = iso_pair

    # TM-score via tmtools (if available).
    try:
        from tmtools import tm_align

        ali = tm_align(can_xyz, iso_xyz, can_seq, iso_seq)
        # tmtools returns tm_norm_chain1, tm_norm_chain2 — average per Zhang.
        tm = (float(ali.tm_norm_chain1) + float(ali.tm_norm_chain2)) / 2.0
        out["tm_score"] = tm
        out["rmsd_shared"] = float(ali.rmsd)
    except Exception as exc:  # noqa: BLE001
        logger.debug("tm_align failed: %s", exc)

    # Extension contacts: residues in [diff_isoform_start, diff_isoform_end)
    # whose Cα is within `contact_distance` of any canonical-body Cα in
    # the isoform structure.
    if (
        diff_isoform_start is not None
        and diff_isoform_end is not None
        and iso_xyz is not None
        and len(iso_xyz) >= diff_isoform_end
    ):
        try:
            import numpy as np

            ext_idx = list(range(int(diff_isoform_start), int(diff_isoform_end)))
            body_idx = [i for i in range(len(iso_xyz)) if i not in set(ext_idx)]
            if ext_idx and body_idx:
                ext = iso_xyz[ext_idx]
                body = iso_xyz[body_idx]
                # Pairwise distances, count any within threshold.
                d2 = np.sum((ext[:, None, :] - body[None, :, :]) ** 2, axis=-1)
                n_contacts = int(np.sum(d2 <= contact_distance ** 2))
                out["extension_contacts"] = n_contacts
        except Exception as exc:  # noqa: BLE001
            logger.debug("contact count failed: %s", exc)

    return out
