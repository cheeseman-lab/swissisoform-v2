"""S3 — magnitude of differential SAE features (isoform vs canonical). Plumbing: plm.sae_module."""

from __future__ import annotations

from swissisoform.config import ScoringConfig
from swissisoform.evidence.common import CriterionResult, _annotation
from swissisoform.models import TranslationInitiationSite

__all__ = ["score"]


def score(site: TranslationInitiationSite, cfg: ScoringConfig) -> CriterionResult:
    """S3: is the differential interpretable-SAE signal substantial?

    Reads ``site.isoform_annotations['sae']`` written by ``SAEFeatureModule``.

    The scored claim is a **magnitude** check on the strongest shared-feature
    activation shift, ``max(|top_gained_delta_max|, |top_lost_delta_max|)``,
    against ``cfg.s3_top_delta_min``.

    This replaced a categorical presence check
    (``n_isoform_only + n_canonical_only > 0``) that was **True for 100.0% of the
    6,462 isoforms** in the genome-wide ``full_catalog`` run, and so carried no
    information: two proteins of different length always differ in hundreds of
    features (median 197 gained+lost). The gained/lost counts remain in the reason
    string as context, but they cannot express *how much* changed — only the
    per-feature activation deltas can.

    * ``True``  — the top shared-feature |delta| meets the threshold.
    * ``False`` — a differential exists but no feature shifted that strongly.
    * ``None``  — the SAE step did not run (no cache / ``status != 'ok'``), or no
      shared-feature deltas are available (nothing to measure magnitude on) — so an
      unrun stage never reads as evidence-absent.
    """
    ann = _annotation(site, "sae")
    if ann is None:
        return CriterionResult("S3_sae", None, "sae annotation missing")
    status = ann.get("status")
    if status != "ok":
        return CriterionResult("S3_sae", None, f"sae status={status}")

    deltas = [
        abs(float(v))
        for v in (ann.get("top_gained_delta_max"), ann.get("top_lost_delta_max"))
        if isinstance(v, (int, float))
    ]
    if not deltas:
        return CriterionResult("S3_sae", None, "sae shared-feature deltas unavailable")
    top_delta = max(deltas)

    iso_only = ann.get("n_isoform_only")
    can_only = ann.get("n_canonical_only")
    ctx = ""
    if isinstance(iso_only, (int, float)) and isinstance(can_only, (int, float)):
        ctx = f"; {int(iso_only)} gained + {int(can_only)} lost features"

    return CriterionResult(
        "S3_sae",
        top_delta >= cfg.s3_top_delta_min,
        f"top shared-feature |delta| {top_delta:.2f} vs threshold "
        f"{cfg.s3_top_delta_min}{ctx}",
    )
