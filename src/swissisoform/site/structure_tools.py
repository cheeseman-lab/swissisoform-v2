r"""Query-time readers over the fold cache for the P-category LLM pass.

The Predicted-Structure read gets 26 scalars aggregating a per-residue confidence
array and an L×L predicted-aligned-error matrix, so it can quote a mean pLDDT but
cannot ask whether the region is uniformly floppy or a confident element beside a
disordered linker. These readers query those arrays directly: exposed as
:data:`P_TOOLS`, executed by :func:`make_p_dispatch`, driven by
``swissisoform.site.llm.run_tool_loop``.

Unlike M, the P slice's hit rows are KEPT in the opening context — see
``STRIP_HITS_FOR_TOOLS`` in ``site.llm``. The bulk therefore sits behind the
readers, which makes what a reader RETURNS the constraint.
Complexity in L decides: O(L) may return raw values (:func:`plddt_profile`);
O(L²) must aggregate (:func:`pae_block` returns mean/min/max, never the
sub-matrix — a full PAE is ~332k JSON chars at L=239, re-sent every turn);
bounded output still caps (:func:`contacts`).

Cache addressing goes through ``StructureModule``'s
``structure_{canonical,isoform}_hash`` columns, which reach here via
``build_gene_record``'s ``_raw`` mirror. Pure reads throughout. See
``docs/architecture/structure_subsystem.md``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from swissisoform.structure.compare import (
    _ca_coords_and_seq,
    _try_load_structure,
    load_pae,
)
from swissisoform.structure.fold import (
    DEFAULT_BACKEND,
    DEFAULT_CACHE_DIR,
    load_cache,
)

__all__ = [
    "DATA_TOOL_NAMES",
    "EMIT_VERDICT",
    "P_TOOLS",
    "StructureHashesMissing",
    "contacts",
    "make_p_dispatch",
    "pae_block",
    "plddt_profile",
    "require_structure_hashes",
    "secondary_structure",
]


# ── Limits ────────────────────────────────────────────────────────────────

# A whole per-residue profile is cheap (~1.4k chars at L=239) but a 1000-residue
# protein is not, so the raw array is capped and the summary statistics stay
# exact regardless.
MAX_PROFILE_RESIDUES = 400
# Contact partners are naturally bounded by the fold, but cap anyway — the same
# discipline as query_variants' row cap.
MAX_CONTACT_PARTNERS = 40
DEFAULT_CONTACT_CUTOFF = 8.0  # Å, matching compare_structures' extension_contacts

_SIDES = ("isoform", "canonical")

# pLDDT scale of the cached confidence.json. Stated in every result because the
# aggregate columns the model also sees are on this scale, as is
# ScoringConfig.p1_plddt_threshold (0.70) — while the exported CIF B-factors are
# 0-100. A result that did not say which would invite the wrong comparison.
PLDDT_SCALE = "0-1"

# Columns read out of the evidence record's ``_raw`` mirror.
_HASH_COL = {"canonical": "isoform_structure_canonical_hash",
             "isoform": "isoform_structure_isoform_hash"}
_BACKEND_COL = "isoform_structure_backend"


class StructureHashesMissing(KeyError):
    """Raised when the evidence record predates the fold-cache hash columns."""


def require_structure_hashes(raw: dict[str, Any]) -> None:
    """Validate that a record can address the fold cache at all.

    Called once at wiring time so a run produced before ``StructureModule``
    emitted the hash columns fails immediately with an actionable message,
    rather than every reader returning ``no_hash`` and the model quietly
    concluding "no structural evidence". Deliberately not a soft fallback: a P
    verdict with the readers and one without are not comparable.
    """
    missing = [c for c in _HASH_COL.values() if c not in (raw or {})]
    if missing:
        raise StructureHashesMissing(
            f"evidence record lacks {missing}. These columns come from "
            "StructureModule (src/swissisoform/structure/module.py) and land in "
            "all_paired.parquet; a run produced before they existed cannot "
            "address the fold cache. Re-run the pipeline for this preset, then "
            "rebuild the evidence records — or pass --no-tools to run the P "
            "category through the single-shot path."
        )


# ── Resolution ────────────────────────────────────────────────────────────


def _diff_bounds(raw: dict[str, Any]) -> tuple[int | None, int | None, str | None]:
    """Differential-region bounds and the space they are expressed in.

    ``diff_space`` is the authoritative statement of WHICH structure contains the
    differential region — ``isoform`` for extensions (the added N-terminal
    segment exists only in the isoform), ``canonical`` for truncations (the
    removed segment exists only in the canonical protein). Re-deriving it from
    ``orf_type`` duplicates a fact the pipeline already recorded.
    """
    def _int(key: str) -> int | None:
        v = (raw or {}).get(key)
        if v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return None if f != f else int(f)  # NaN check

    space = (raw or {}).get("diff_space")
    return _int("diff_start"), _int("diff_end"), (str(space) if space else None)


def _default_side(raw: dict[str, Any]) -> str:
    """The side that CONTAINS the differential region, per ``diff_space``.

    A caller who names no side almost always wants the structure the category is
    about: the isoform for an extension (the added segment exists only there) and
    the canonical for a truncation (the removed segment exists only there).
    Hardcoding ``"isoform"`` silently profiles the whole isoform on every
    truncation — the differential window cannot be applied across spaces, so the
    reader falls back to the entire protein and answers a different question than
    the one asked. Falls back to ``isoform`` when ``diff_space`` is absent or is
    something else (separate ORFs, where the whole isoform IS the region).
    """
    space = (raw or {}).get("diff_space")
    return str(space) if space in _SIDES else "isoform"


def _resolve(
    raw: dict[str, Any], side: str, cache_dir: Path | str, backend: str | None
) -> tuple[dict[str, Any] | None, str]:
    """Load one side's cache entry. Returns ``(entry, status)``.

    Statuses: ``ok`` (folded, confidence present), ``not_folded`` (a cache
    directory exists but nothing was folded — ``status="too_long"``, >1024 aa),
    ``no_cache`` (nothing at that hash), ``no_hash`` (the record has no hash for
    this side).
    """
    if side not in _SIDES:
        raise ValueError(f"side must be one of {list(_SIDES)}; got {side!r}")
    h = (raw or {}).get(_HASH_COL[side])
    if not h:
        return None, "no_hash"
    entry = load_cache(str(h), cache_dir, backend or DEFAULT_BACKEND)
    if entry is None:
        return None, "no_cache"
    metrics = entry.get("metrics") or {}
    if not (entry.get("confidence") or {}).get("plddt"):
        return entry, "not_folded" if metrics.get("status") == "too_long" else "no_cache"
    return entry, "ok"


def _window(
    start: Any, end: Any, length: int, raw: dict[str, Any], side: str
) -> tuple[int, int]:
    """Resolve a residue window, defaulting to the differential region.

    Bounds are 1-based inclusive on the wire (residue numbering as a biologist
    reads it) and clamped to the structure. The differential-region default only
    applies on the side that actually contains it, per ``diff_space``.

    A window overlapping the protein is clamped to it; one entirely outside
    raises. ``ValueError`` reaches the model as ``{"error": ...}`` through
    ``make_p_dispatch`` — the channel a bad ``side`` already uses — so it can
    correct the numbers, rather than reading an empty ``status="ok"`` as a
    confident negative.
    """
    if start is None and end is None:
        d_start, d_end, space = _diff_bounds(raw)
        if space == side and d_start is not None and d_end is not None:
            start, end = d_start + 1, d_end  # 0-based half-open -> 1-based inclusive
        else:
            start, end = 1, length
    if start is not None and int(start) > length:
        raise ValueError(
            f"start={int(start)} is past the end of the {side} structure "
            f"({length} residues). Residue numbering is 1-based and each side has "
            f"its own length."
        )
    if end is not None and int(end) < 1:
        raise ValueError(
            f"end={int(end)} is before the start of the {side} structure. "
            f"Residue numbering is 1-based."
        )
    s = 1 if start is None else min(length, max(1, int(start)))
    e = length if end is None else min(length, int(end))
    return s, max(s, e)


# ── Readers ───────────────────────────────────────────────────────────────


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": round(sum(values) / len(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def plddt_profile(
    raw: dict[str, Any],
    *,
    side: str | None = None,
    start: Any = None,
    end: Any = None,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    backend: str | None = None,
) -> dict[str, Any]:
    """Per-residue fold confidence over a residue window.

    Answers what a single mean cannot: is low confidence spread evenly across the
    segment, or is it one disordered stretch beside a well-formed element? Values
    are on the 0-1 scale of the cached ``confidence.json`` — the same scale as
    the aggregate columns and the P1 threshold.
    """
    side = side or _default_side(raw)
    entry, status = _resolve(raw, side, cache_dir, backend)
    if status != "ok":
        return {"side": side, "status": status, "scale": PLDDT_SCALE,
                "region": None, "n_residues": 0, "per_residue": [],
                "mean": None, "min": None, "max": None}

    plddt = list(entry["confidence"]["plddt"])
    s, e = _window(start, end, len(plddt), raw, side)
    seg = [float(x) for x in plddt[s - 1:e]]
    truncated = len(seg) > MAX_PROFILE_RESIDUES
    return {
        "side": side,
        "status": "ok",
        "scale": PLDDT_SCALE,
        "region": [s, e],
        "protein_length": len(plddt),
        "n_residues": len(seg),
        **_stats(seg),
        "per_residue": [round(x, 3) for x in seg[:MAX_PROFILE_RESIDUES]],
        "per_residue_truncated": truncated,
    }


def pae_block(
    raw: dict[str, Any],
    *,
    rows: Any = None,
    cols: Any = None,
    side: str | None = None,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    backend: str | None = None,
) -> dict[str, Any]:
    """Mean predicted aligned error between two residue ranges.

    PAE[i, j] is the expected positional error at residue ``j`` when the
    structure is superposed on residue ``i``. Low between two regions means they
    are confidently placed RELATIVE to each other; high means their relative
    orientation is unresolved even if each is individually well-folded.

    Returns aggregates only. The full matrix is O(L²) — ~332k JSON chars at
    L=239 — and would dominate the conversation; the model gets its resolution by
    choosing ``rows`` and ``cols``.

    PAE is asymmetric, so an off-diagonal block averages BOTH rectangles
    (rows→cols and cols→rows), matching ``compare.pae_region_blocks``.
    """
    side = side or _default_side(raw)
    entry, status = _resolve(raw, side, cache_dir, backend)
    empty = {"side": side, "rows": None, "cols": None, "mean": None, "min": None,
             "max": None, "n_pairs": 0, "status": status}
    if status != "ok":
        return empty

    pae = load_pae(entry.get("pae_path"))
    if pae is None:
        return {**empty, "status": "no_pae"}

    import numpy as np

    m = np.asarray(pae, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] == 0:
        return {**empty, "status": "no_pae"}

    length = m.shape[0]
    r0, r1 = _window(*(rows or (None, None)), length, raw, side)
    c0, c1 = _window(*(cols or (None, None)), length, raw, side)
    r = np.arange(r0 - 1, r1)
    c = np.arange(c0 - 1, c1)
    if r.size == 0 or c.size == 0:
        return {**empty, "status": "no_diff_region"}

    block = m[np.ix_(r, c)]
    if r0 == c0 and r1 == c1:
        vals = block  # diagonal block: symmetric by construction
    else:
        vals = np.concatenate([block.ravel(), m[np.ix_(c, r)].ravel()])
    return {
        "side": side,
        "rows": [r0, r1],
        "cols": [c0, c1],
        "mean": round(float(vals.mean()), 2),
        "min": round(float(vals.min()), 2),
        "max": round(float(vals.max()), 2),
        "n_pairs": int(vals.size),
        "units": "angstrom",
        "status": "ok",
    }


_SSE_COL = {"canonical": "isoform_structure_sse_canonical",
            "isoform": "isoform_structure_sse_isoform"}


@lru_cache(maxsize=32)
def _sse_from_cif(cif_path: str) -> str | None:
    """Per-residue SSE for one structure, parsed once per process."""
    from swissisoform.structure.sse import annotate_sse as _annotate

    return _annotate(cif_path)


def secondary_structure(
    raw: dict[str, Any],
    *,
    side: str | None = None,
    start: Any = None,
    end: Any = None,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    backend: str | None = None,
) -> dict[str, Any]:
    """Helices and strands over a residue window, with per-element confidence.

    Answers "where IS the structure" — which pLDDT does not. Measured on UBE2M,
    the highest-confidence stretch of its extension (pLDDT 0.91) is coil, while
    the real 17-residue helix scores 0.75; selecting a window by confidence finds
    the wrong region. Every element carries its own ``plddt_mean`` so a
    geometrically clean helix through a disordered stretch stays distinguishable
    from a genuine one.

    Prefers the per-residue assignment already on the record and falls back to
    computing it from the CIF, so it works on runs predating those columns. This
    is not the pre-computed-answer anti-pattern that motivated stripping the PAE
    block means: the stored value is the FULL per-residue assignment, not a lossy
    summary, so reading it loses nothing.

    Scanned ONCE, whole; the window SELECTS which elements are returned rather
    than clipping them, so each is reported at its true extent and may run past
    ``region``. Clipping instead is the defect ``StructureModule._sse`` documents.

    ``region`` is membership in the DIFFERENTIAL region, never in the queried
    window, so it matches ``isoform_structure_sse_all_elements``, the P3 card and
    the P3 scorer. It is absent when the queried side does not hold that region
    (``side="isoform"`` on a truncation), since the bounds are in the other
    protein's numbering; ``region_is_differential`` says which you got.
    """
    from swissisoform.config import ScoringConfig
    from swissisoform.structure.sse import classify_elements, sse_elements, summarise_elements

    min_plddt = ScoringConfig().p3_min_sse_plddt
    side = side or _default_side(raw)
    entry, status = _resolve(raw, side, cache_dir, backend)
    empty = {"side": side, "region": None, "region_is_differential": False,
             "sse_string": None, "elements": [], "n_helix": 0, "n_strand": 0,
             "longest_helix": 0, "longest_strand": 0, "longest_confident": 0,
             "confident_min_plddt": min_plddt, "status": status}
    if status != "ok":
        return empty

    sse = (raw or {}).get(_SSE_COL[side]) or _sse_from_cif(str(entry.get("cif_path") or ""))
    if not sse:
        return {**empty, "status": "no_structure"}

    plddt = list(entry["confidence"]["plddt"])
    s, e = _window(start, end, len(sse), raw, side)

    all_elements = sse_elements(sse, plddt)
    d_start, d_end, space = _diff_bounds(raw)
    tagged = space == side and d_start is not None and d_end is not None
    if tagged:
        # diff bounds are 0-based half-open; classify_elements is 1-based inclusive.
        classify_elements(all_elements, diff_start=d_start + 1, diff_end=d_end)
    elements = [el for el in all_elements if el["end"] >= s and el["start"] <= e]

    return {
        "side": side,
        "region": [s, e],
        "region_is_differential": tagged,
        "protein_length": len(sse),
        "sse_string": sse[s - 1 : e],
        "elements": elements,
        # min_plddt is the P3 threshold, so longest_confident is the length P3
        # scores on. Without it the field silently reported the longest element of
        # ANY confidence under a name asserting the opposite.
        **summarise_elements(elements, min_plddt=min_plddt),
        "confident_min_plddt": min_plddt,
        "status": "ok",
    }


@lru_cache(maxsize=32)
def _ca_coords(cif_path: str):
    """Cα coordinates for one structure, parsed once per process."""
    struct = _try_load_structure(Path(cif_path))
    if struct is None:
        return None
    pair = _ca_coords_and_seq(struct)
    return None if pair is None else pair[0]


def contacts(
    raw: dict[str, Any],
    *,
    start: Any = None,
    end: Any = None,
    side: str | None = None,
    cutoff: float = DEFAULT_CONTACT_CUTOFF,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    backend: str | None = None,
) -> dict[str, Any]:
    """Cα-Cα contacts between a residue window and the rest of the structure.

    ``compare_structures`` emits a single ``extension_contacts`` count; this
    exposes WHICH residues are contacted, which is the difference between twelve
    contacts scattered over the surface and twelve concentrated on one patch —
    only the second suggests an interface.

    The CIF is never returned, only derived counts.
    """
    side = side or _default_side(raw)
    entry, status = _resolve(raw, side, cache_dir, backend)
    empty = {"side": side, "region": None, "cutoff_angstrom": float(cutoff),
             "n_contacts": 0, "n_residues_in_contact": 0, "contact_partners": [],
             "status": status}
    if status != "ok":
        return empty
    cif = entry.get("cif_path")
    if not cif:
        return {**empty, "status": "no_structure"}

    xyz = _ca_coords(str(cif))
    if xyz is None:
        return {**empty, "status": "no_structure"}

    import numpy as np

    length = len(xyz)
    s, e = _window(start, end, length, raw, side)
    region = np.arange(s - 1, e)
    body = np.array([i for i in range(length) if not (s - 1 <= i < e)], dtype=int)
    if region.size == 0 or body.size == 0:
        return {**empty, "region": [s, e], "status": "no_body"}

    d2 = np.sum((xyz[region][:, None, :] - xyz[body][None, :, :]) ** 2, axis=-1)
    within = d2 <= float(cutoff) ** 2
    per_partner = within.sum(axis=0)
    order = np.argsort(-per_partner)
    partners = [
        {"residue": int(body[i]) + 1, "n_contacts": int(per_partner[i])}
        for i in order
        if per_partner[i] > 0
    ]
    return {
        "side": side,
        "region": [s, e],
        "cutoff_angstrom": float(cutoff),
        "n_contacts": int(within.sum()),
        "n_residues_in_contact": len(partners),
        "contact_partners": partners[:MAX_CONTACT_PARTNERS],
        "contact_partners_truncated": len(partners) > MAX_CONTACT_PARTNERS,
        "status": "ok",
    }


# ── Tool schemas ──────────────────────────────────────────────────────────

EMIT_VERDICT = "emit_verdict"

_SIDE_PROP = {
    "type": "string",
    "enum": list(_SIDES),
    "description": (
        "Which predicted structure to read. 'isoform' is the alternative protein, "
        "'canonical' the reference. The differential region exists in only one of "
        "them — the isoform for extensions, the canonical for truncations — and "
        "every result reports which side it came from. Omit this to read the side "
        "that CONTAINS the differential region, which is what you usually want; "
        "name a side explicitly only to compare the two structures."
    ),
}
_RANGE_ITEMS = {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}

P_TOOLS: list[dict[str, Any]] = [
    {
        "name": "plddt_profile",
        "description": (
            "Per-residue fold confidence (pLDDT, 0-1) over a residue window, with "
            "mean/min/max. Omit start and end to profile the differential region. "
            "Use it to tell a uniformly disordered segment from a confident element "
            "beside a floppy linker — a single mean cannot express the difference."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "side": _SIDE_PROP,
                "start": {"type": "integer", "description": "First residue, 1-based inclusive."},
                "end": {"type": "integer", "description": "Last residue, 1-based inclusive."},
            },
            "required": [],
        },
    },
    {
        "name": "pae_block",
        "description": (
            "Mean predicted aligned error (angstroms) between two residue ranges: "
            "how confidently one region is positioned RELATIVE to another. Low = "
            "confidently docked, high = orientation unresolved even if each region "
            "folds well. Returns aggregates, never the matrix. Use it to test "
            "whether an extension is placed against the core, and to check the "
            "confidence qualifiers before treating a shared-region RMSD as a real "
            "conformational change."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "side": _SIDE_PROP,
                "rows": {**_RANGE_ITEMS, "description": "[start, end] 1-based inclusive; "
                         "defaults to the differential region."},
                "cols": {**_RANGE_ITEMS, "description": "[start, end] 1-based inclusive; "
                         "defaults to the differential region."},
            },
            "required": [],
        },
    },
    {
        "name": "secondary_structure",
        "description": (
            "Where the helices and strands actually are, as a list of elements with "
            "their residue ranges, lengths and per-element mean pLDDT. Use this to "
            "LOCATE structure — pLDDT tells you how confident a region is, NOT "
            "whether it is a helix, and the two frequently disagree. Omit start/end "
            "to cover the differential region. Elements are returned at their FULL "
            "extent, so one that begins inside your window and continues past it is "
            "reported at its true length and its end may exceed the window — that "
            "is a real element crossing the boundary, not an error. Each carries "
            "region=unique/shared/spans, meaning membership in the DIFFERENTIAL "
            "region (not in your window); unique+spans is the set the secondary-"
            "structure criterion scores. When region_is_differential is false you "
            "asked for the side that does not contain the differential region, so "
            "no membership can be computed and the tags are absent."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "side": _SIDE_PROP,
                "start": {"type": "integer", "description": "First residue, 1-based inclusive."},
                "end": {"type": "integer", "description": "Last residue, 1-based inclusive."},
            },
            "required": [],
        },
    },
    {
        "name": "contacts",
        "description": (
            "Ca-Ca contacts between a residue window and the rest of the structure, "
            "listing which residues are contacted and how often. Distinguishes "
            "contacts scattered over the surface from contacts concentrated on one "
            "patch; only the second suggests a genuine interface. Note these are "
            "distances within a PREDICTED structure, so weigh them against the "
            "pLDDT and PAE of the regions involved."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "side": _SIDE_PROP,
                "start": {"type": "integer", "description": "First residue, 1-based inclusive."},
                "end": {"type": "integer", "description": "Last residue, 1-based inclusive."},
                "cutoff": {
                    "type": "number",
                    "description": f"Ca-Ca distance threshold in angstroms "
                                   f"(default {DEFAULT_CONTACT_CUTOFF}).",
                },
            },
            "required": [],
        },
    },
    {
        "name": EMIT_VERDICT,
        "description": (
            "Terminal call: record the category verdict and stop. Call it exactly "
            "once, after gathering enough data with the reader tools to justify it."
        ),
        # See tools.py's EMIT_VERDICT for why this one tool is strict.
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["interesting", "neutral", "not_interesting"],
                },
                "reasoning": {
                    "type": "string",
                    "description": (
                        "A few tight sentences, bottom line first, citing the specific "
                        "numbers that justify the verdict. Never write the submodule "
                        "ID codes (P1, P2)."
                    ),
                },
                "evidence_used": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Short notes on which tool findings drove the verdict.",
                },
            },
            "required": ["verdict", "reasoning", "evidence_used"],
            "additionalProperties": False,
        },
    },
]

DATA_TOOL_NAMES = frozenset(t["name"] for t in P_TOOLS) - {EMIT_VERDICT}


# ── Dispatch ──────────────────────────────────────────────────────────────


def make_p_dispatch(
    raw: dict[str, Any],
    *,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
) -> Callable[[str, dict[str, Any]], dict[str, Any]]:
    """Build the local executor for the P reader tools, bound to one isoform.

    ``raw`` is the evidence record's ``_raw`` mirror, carrying the fold-cache
    hashes, the backend, and the differential-region coordinates. The returned
    callable never raises: a bad tool name or argument comes back as
    ``{"error": ...}`` so the model can correct itself next turn.
    """
    backend = (raw or {}).get(_BACKEND_COL) or DEFAULT_BACKEND
    readers: dict[str, Callable[..., dict[str, Any]]] = {
        "plddt_profile": plddt_profile,
        "secondary_structure": secondary_structure,
        "pae_block": pae_block,
        "contacts": contacts,
    }

    def dispatch(name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        reader = readers.get(name)
        if reader is None:
            return {
                "error": f"Unknown tool {name!r}. Available: {sorted(readers)} "
                f"plus {EMIT_VERDICT}."
            }
        kwargs = dict(tool_input or {})
        # Bound here, never model-supplied: these describe the run, not a choice.
        for reserved in ("cache_dir", "backend"):
            kwargs.pop(reserved, None)
        try:
            return reader(raw, cache_dir=cache_dir, backend=backend, **kwargs)
        except ValueError as e:
            return {"error": str(e)}
        except TypeError as e:
            return {"error": f"Bad arguments for {name}: {e}"}
        except Exception as e:  # pragma: no cover - defensive
            return {"error": f"{name} failed: {type(e).__name__}: {e}"}

    return dispatch
