"""F7 — shared-region structural change. Plumbing: swissisoform.structure.

The shared region is the amino-acid sequence retained in both the isoform and
its canonical protein (canonical body for an extension; post-truncation body for
a truncation). Folded in both contexts it normally comes out nearly identical, so
its Cα backbone RMSD ≈ 0. The rare case where the extension/truncation
*reorganizes* how the retained region folds — an elevated shared-region RMSD — is
the functional signal this criterion scores.

Metric produced by ``StructureModule`` via ``compare_structures`` (strict Kabsch
superposition on the shared residues only). Descriptive companion ``tm_score_shared``
(length-normalized) rides along in the reason string.
"""

from __future__ import annotations

from swissisoform.config import ScoringConfig
from swissisoform.evidence.common import CriterionResult, _annotation
from swissisoform.models import TranslationInitiationSite

_NAME = "F7_shared_structural_change"


def score(site: TranslationInitiationSite, cfg: ScoringConfig) -> CriterionResult:
    """F7: the retained (shared) region folds significantly differently.

    ``True`` requires the shared region to be confidently folded in BOTH
    structures (min per-side mean pLDDT >= ``cfg.f7_plddt_min``) AND a shared-
    region Cα RMSD >= ``cfg.f7_rmsd_shared_min`` (Å). ``False`` when confidently
    folded but below threshold. Returns ``None`` (not evaluable) when:

    - the structure module didn't run / cache is cold (``status`` unusable),
    - the backend produced only a uniform pLDDT fill (``uniform_plddt``),
    - there is no shared region (uORF/altORF/internal) or the diff-region
      alignment is unverified (``rmsd_shared_status`` != ``ok``),
    - the shared region is shorter than ``cfg.f7_min_shared_len`` (too noisy),
    - the shared region is not confidently folded (low pLDDT).

    A low-confidence floppy loop must not masquerade as a structural change, so
    those cases are ``None`` rather than a misleading ``True``/``False``.
    """
    ann = _annotation(site, "structure")
    if ann is None:
        return CriterionResult(_NAME, None, "structure annotation missing")

    # Overall structure status gates the same failure modes as F1 (uniform_plddt
    # gives a planted scalar pLDDT that would defeat the confidence gate).
    status = ann.get("status")
    if status in ("no_cache", "too_long", "failed", "uniform_plddt", "partial"):
        return CriterionResult(_NAME, None, f"structure status={status}")

    rmsd_status = ann.get("rmsd_shared_status")
    rmsd = ann.get("rmsd_shared")
    if rmsd_status != "ok" or rmsd is None:
        return CriterionResult(
            _NAME, None, f"shared-region RMSD not computed (status={rmsd_status})"
        )

    n = ann.get("shared_region_len")
    if n is None or n < cfg.f7_min_shared_len:
        return CriterionResult(
            _NAME, None, f"shared region too short ({n} < {cfg.f7_min_shared_len} aa)"
        )

    plddt_iso = ann.get("plddt_shared_mean_isoform")
    plddt_can = ann.get("plddt_shared_mean_canonical")
    if plddt_iso is None or plddt_can is None:
        return CriterionResult(_NAME, None, "shared-region pLDDT unavailable")
    plddt = min(plddt_iso, plddt_can)
    if plddt < cfg.f7_plddt_min:
        return CriterionResult(
            _NAME,
            None,
            f"shared region not confidently folded "
            f"(min pLDDT {plddt:.2f} < {cfg.f7_plddt_min})",
        )

    tm = ann.get("tm_score_shared")
    tm_str = f"{tm:.2f}" if isinstance(tm, (int, float)) else "n/a"
    passed = rmsd >= cfg.f7_rmsd_shared_min
    rel = "≥" if passed else "<"
    return CriterionResult(
        _NAME,
        passed,
        f"shared-region Cα RMSD {rmsd:.2f} Å ({rel} {cfg.f7_rmsd_shared_min} Å "
        f"threshold), TM-score {tm_str}, {n} aa, min shared pLDDT {plddt:.2f}",
    )
