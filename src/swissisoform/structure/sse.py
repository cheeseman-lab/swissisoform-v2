"""Secondary-structure assignment from predicted coordinates.

ESMFold emits no secondary-structure annotation, so this derives it from the
coordinates it produced, via ``biotite.structure.annotate_sse`` (P-SEA,
Labesse 1997). Cα-only — distances and dihedral angles, no backbone hydrogen
bonds — and biotite warns it deviates from the reference implementation in some
cases. Good enough for "is there a helix here", not for publication-grade
assignment.

Because it is derived from a prediction it inherits that prediction's
uncertainty, which is why :func:`sse_elements` attaches a per-element mean pLDDT:
a geometrically clean ``aaaaaaa`` run through a pLDDT-0.5 stretch is geometry
fitted to a guess, and every consumer needs to be able to see that.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# biotite's P-SEA alphabet.
HELIX = "a"
STRAND = "b"
COIL = "c"
_STRUCTURED = (HELIX, STRAND)
_LABEL = {HELIX: "helix", STRAND: "strand"}

# Biophysical floor: below this a run is not an element, it is a labelling blip.
# An alpha-helix is 3.6 residues per turn, so 4 is a single turn (DSSP's own
# minimum) and 5 is a safe floor for "a real helix" — P-SEA emitted a 2-residue
# and a 4-residue "helix" on the cheeseman set, against a clean gap up to 13.
# Beta-strands are legitimately shorter: 5-residue strands are common and a
# 3-residue strand in a sheet is real, so they need their own, lower floor. A
# single shared cutoff of 5-6 would be right for helices and would discard 7 of
# 19 genuine strands.
MIN_LENGTH = {HELIX: 5, STRAND: 3}


def annotate_sse(cif_path: Path | str | None) -> str | None:
    """Per-residue secondary structure for one structure, or ``None``.

    Returns a string over the P-SEA alphabet — ``a`` helix, ``b`` strand,
    ``c`` coil, ``''`` for residues with no Cα — one character per residue.
    ``None`` when the file is missing or biotite/the parse is unavailable, so
    callers degrade rather than raise.
    """
    if not cif_path or not Path(cif_path).exists():
        return None
    try:
        import biotite.structure as struc
        from biotite.structure.io.pdbx import CIFFile, get_structure
    except Exception:  # pragma: no cover - optional dep
        return None
    try:
        structure = get_structure(CIFFile.read(str(cif_path)), model=1)
        return "".join(str(x) for x in struc.annotate_sse(structure))
    except Exception as exc:  # noqa: BLE001
        logger.debug("annotate_sse failed for %s: %s", cif_path, exc)
        return None


def sse_elements(
    sse: str | None,
    plddt: list[float] | None = None,
    *,
    start: int = 1,
    end: int | None = None,
    min_length: dict[str, int] | int | None = None,
) -> list[dict[str, Any]]:
    """Collapse a per-residue SSE string into helix/strand elements.

    Coil is not an element and is never returned. Coordinates are 1-based
    inclusive, matching the P readers, so an element's ``start``/``end`` can be
    handed straight back to ``plddt_profile`` / ``contacts`` / ``pae_block``.

    Args:
        sse: Per-residue assignment from :func:`annotate_sse`.
        plddt: Per-residue confidence, same indexing as ``sse``. When given,
            each element carries ``plddt_mean`` — the temper that keeps a
            geometrically clean element in a disordered region from being read
            as a real one.
        start: First residue of the window to scan, 1-based inclusive.
        end: Last residue, 1-based inclusive; defaults to the end of ``sse``.
        min_length: Per-type floor, defaulting to :data:`MIN_LENGTH`. An int
            applies one floor to both types; ``0`` disables filtering.
    """
    if min_length is None:
        floors = MIN_LENGTH
    elif isinstance(min_length, dict):
        floors = min_length
    else:
        floors = dict.fromkeys(_STRUCTURED, min_length)
    if not sse:
        return []
    n = len(sse)
    lo = max(1, int(start))
    hi = n if end is None else min(n, int(end))
    if lo > hi:
        return []

    out: list[dict[str, Any]] = []
    run_type: str | None = None
    run_start = lo
    for i in range(lo, hi + 2):  # one past the end to flush the final run
        ch = sse[i - 1] if i <= hi else None
        if ch != run_type:
            if run_type in _STRUCTURED:
                length = i - run_start
                if length >= floors.get(run_type, 0):
                    element: dict[str, Any] = {
                        "type": _LABEL[run_type],
                        "start": run_start,
                        "end": i - 1,
                        "length": length,
                    }
                    if plddt:
                        seg = [float(x) for x in plddt[run_start - 1 : i - 1]]
                        if seg:
                            element["plddt_mean"] = round(sum(seg) / len(seg), 3)
                    out.append(element)
            run_type = ch
            run_start = i
    return out


UNIQUE = "unique"
SHARED = "shared"
SPANS = "spans"


def classify_elements(
    elements: list[dict[str, Any]], *, diff_start: int, diff_end: int
) -> list[dict[str, Any]]:
    """Tag each element by where it sits relative to the differential region.

    Adds a ``region`` field in place and returns the same list: ``unique`` when
    the element lies wholly inside the differential region, ``shared`` when
    wholly outside it, ``spans`` when it crosses the boundary.

    Classifying a single whole-protein scan is deliberate. Scanning the two
    regions separately would clip a boundary-crossing element at each window
    edge, so it would appear twice with two wrong lengths; here it appears once,
    at its true extent, flagged. That case is worth seeing rather than hiding —
    a helix that begins in an extension and continues into the core is a
    structural-continuity claim, not a bookkeeping nuisance.

    Args:
        elements: From :func:`sse_elements`, 1-based inclusive.
        diff_start: First residue of the differential region, 1-based inclusive.
        diff_end: Last residue of the differential region, 1-based inclusive.
    """
    for element in elements:
        start, end = element["start"], element["end"]
        if start >= diff_start and end <= diff_end:
            element["region"] = UNIQUE
        elif end < diff_start or start > diff_end:
            element["region"] = SHARED
        else:
            element["region"] = SPANS
    return elements


def summarise_elements(
    elements: list[dict[str, Any]], *, min_plddt: float | None = None
) -> dict[str, Any]:
    """Counts and longest-run summary over a list of elements.

    ``longest_confident`` is the longest element whose own ``plddt_mean`` clears
    ``min_plddt`` — the value P3 scores on. It is deliberately distinct from
    ``longest_helix``/``longest_strand``, which ignore confidence: a caller must
    be able to see both "there is a long helix" and "but it is not confident".
    """
    helices = [e for e in elements if e["type"] == "helix"]
    strands = [e for e in elements if e["type"] == "strand"]
    confident = [
        e
        for e in elements
        if min_plddt is None or (e.get("plddt_mean") is not None and e["plddt_mean"] >= min_plddt)
    ]
    return {
        "n_helix": len(helices),
        "n_strand": len(strands),
        "longest_helix": max((e["length"] for e in helices), default=0),
        "longest_strand": max((e["length"] for e in strands), default=0),
        "longest_confident": max((e["length"] for e in confident), default=0),
    }
