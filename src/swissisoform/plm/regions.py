"""Diff-region → protein-coordinate index helper, shared across PLM modules.

The isoform-unique region of a TIS lives in *protein* coordinates that differ by
ORF type: truncations describe the lost stretch in canonical-protein coords, while
extensions / uORFs / altORFs / 3'UTR ORFs describe the gained stretch in
isoform-protein coords. Both :class:`~swissisoform.plm.module.PLMVEPModule` (slicing
per-residue LLR) and :class:`~swissisoform.plm.sae_module.SAEFeatureModule` (slicing
the per-residue SAE activation matrix) need the same index logic, so it lives here
as the single source of truth.
"""

from __future__ import annotations

from swissisoform.models import ORFType, TranslationInitiationSite


def _indices(
    protein: str, start: int | None, end: int | None
) -> tuple[list[int], list[int]] | None:
    """Return (unique, shared) 0-based indices into ``protein`` for ``[start, end)``.

    ``None`` when the region is uninterpretable (no coords, empty protein).
    """
    if start is None or end is None:
        return None
    length = len(protein.rstrip("*"))
    if length <= 0:
        return None
    u_lo = max(0, int(start))
    u_hi = min(length, int(end))
    unique = list(range(u_lo, u_hi))
    unique_set = set(unique)
    shared = [i for i in range(length) if i not in unique_set]
    return unique, shared


def diff_region_indices(
    site: TranslationInitiationSite,
) -> tuple[str, list[int], list[int]]:
    """Return ``(space, unique_idx, shared_idx)`` for a TIS's diff region.

    ``space`` is ``"canonical"`` for truncations (region resolved in canonical
    protein coords), ``"isoform"`` for extensions / uORFs / altORFs / 3'UTR ORFs
    (isoform protein coords), or ``"none"`` when the diff region has no coordinate
    interpretation (e.g. Annotated sites). The index lists are empty when
    ``space == "none"``.
    """
    dr = site.diff_region
    if dr is None:
        return "none", [], []

    if site.orf_type == ORFType.TRUNCATED:
        res = _indices(site.canonical_protein, dr.canonical_start, dr.canonical_end)
        if res is not None:
            return "canonical", res[0], res[1]
        return "none", [], []

    res = _indices(site.isoform_protein, dr.isoform_start, dr.isoform_end)
    if res is not None:
        return "isoform", res[0], res[1]
    return "none", [], []
