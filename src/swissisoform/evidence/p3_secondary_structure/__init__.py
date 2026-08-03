"""P3 — secondary structure in the differential region. Plumbing: swissisoform.structure.sse."""

from __future__ import annotations

from swissisoform.config import ScoringConfig
from swissisoform.evidence.common import CriterionResult, _annotation
from swissisoform.models import TranslationInitiationSite

NAME = "P3_secondary_structure"


def score(site: TranslationInitiationSite, cfg: ScoringConfig) -> CriterionResult:
    """P3: the differential region contains a confident secondary-structure element.

    Fires when the region holds a helix or strand of at least
    ``cfg.p3_min_sse_length`` residues whose own mean pLDDT clears
    ``cfg.p3_min_sse_plddt``. **Both conditions are required.** Secondary
    structure here is derived from predicted coordinates via P-SEA, not predicted
    directly, so a geometrically clean helix running through a disordered stretch
    is geometry fitted to a guess — length alone would score it identically to a
    real element.

    Symmetric across ORF types, like P1, with the direction living in the
    narrative rather than the measurement:

    * extension / uORF / altORF — the isoform GAINS a structured element; a helix
      or strand in newly added sequence is a candidate functional addition.
    * truncation — the isoform LOSES one. The element is read off the CANONICAL
      structure (the removed segment exists only there), so a hit means the
      truncation deletes real structure rather than a disordered tail. On
      cheeseman_test this separates four truncations that remove a helix or
      strand from three that remove coil only.

    P3 says nothing about whether the element is *integrated* with the rest of
    the fold — whether a gained helix packs against the core, or a lost one was
    load-bearing. That is what the contact and PAE evidence answers, and
    duplicating it here would double-count.

    Reads ``site.isoform_annotations['structure']`` written by
    ``StructureModule``. Returns ``None`` when the fold is unusable — the same
    guards as P1, plus ``uniform_plddt``, where the per-element confidence that
    does the tempering would be a planted constant.
    """
    ann = _annotation(site, "structure")
    if ann is None:
        return CriterionResult(NAME, None, "structure annotation missing")

    status = ann.get("status")
    if status in ("no_cache", "too_long", "failed", "uniform_plddt"):
        return CriterionResult(NAME, None, f"structure status={status}")

    sse_status = ann.get("sse_status")
    if sse_status != "ok":
        return CriterionResult(NAME, None, f"sse status={sse_status or 'missing'}")

    min_len = cfg.p3_min_sse_length
    min_plddt = cfg.p3_min_sse_plddt
    elements = ann.get("sse_diff_elements") or []
    qualifying = [
        e
        for e in elements
        if isinstance(e, dict)
        and (e.get("length") or 0) >= min_len
        and e.get("plddt_mean") is not None
        and e["plddt_mean"] >= min_plddt
    ]

    if not qualifying:
        longest = max((e.get("length") or 0 for e in elements), default=0)
        detail = (
            f"longest element {longest} aa (< {min_len} aa or below pLDDT {min_plddt})"
            if elements
            else "no helix or strand in the differential region"
        )
        return CriterionResult(NAME, False, detail)

    best = max(qualifying, key=lambda e: e["length"])
    return CriterionResult(
        NAME,
        True,
        f"{best['length']} aa {best['type']} at {best['start']}-{best['end']} "
        f"(pLDDT {best['plddt_mean']:.2f}, thresholds {min_len} aa / {min_plddt})",
    )
