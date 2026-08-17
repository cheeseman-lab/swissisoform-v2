"""Module: PLM VEP — variant-effect scores from ESM-C masked-marginal LLR.

Consumes per-residue log-likelihood scores (computed offline by
``swissisoform.plm.embed.precompute_plm``) and emits TIS-level constraint
metrics.

The cached per-residue value is ``logP(wt)``, the model's log-probability of the
residue actually present — no alternate allele. **Higher (nearer zero) means more
conserved**: the residue is strongly determined by its context. Verified against
Zoonomia 241-mammal PhyloP over the same codons — pearson +0.40, spearman +0.48
over 6,842 residues, positive in 27/27 proteins (``figures/plm_direction/``).

- mean logP(wt) over the isoform-unique region,
- mean logP(wt) over the shared region (canonical-frame body),
- constraint delta (unique − shared; positive = unique region more conserved),
- count of strongly-conserved positions (logP(wt) >= threshold) in each region.

Why a SiteModule, not a ProteinModule: LLR is context-dependent — running
ESM-C on ``diff_region.sequence`` alone (Scope-A re-run) gives different
scores than slicing the same positions out of the full-protein forward
pass. So we compute LLR on the full canonical and isoform proteins, then
slice to unique/shared regions per the diff_region coordinates.

M1 (germline tolerance / constraint) reads ``constraint_delta`` from here as one
of its two either-or inputs; it abstains entirely on ORF types whose unique
region was never canonical coding sequence, where this contrast has no baseline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from swissisoform.config import PipelineConfig
from swissisoform.models import TranslationInitiationSite
from swissisoform.plm.embed import DEFAULT_CACHE_DIR, load_cache, protein_hash
from swissisoform.plm.regions import diff_region_indices

logger = logging.getLogger(__name__)

# logP(wt) AT OR ABOVE which a position counts as strongly-conserved.
#
# Direction matters and was previously inverted here. The cached array is
# logP(wt) — the model's confidence in the residue actually present — so HIGH
# (near 0) means the residue is strongly determined by its context. Measured
# against Zoonomia 241-mammal PhyloP over the same codons (6,842 residues, 27
# proteins): pearson +0.40 / spearman +0.48, positive in 27/27 proteins, decile
# means rising monotonically from PhyloP 2.0 to 5.4. The old threshold counted
# logP(wt) < -5.0 — the ~1% of residues the model finds LEAST expected — as
# "constrained", inherited from an ESM-2 650M top-decile cutoff on ΔLLR, a
# different population entirely.
#
# -0.5 is PROVISIONAL: it is the median of the observed logP(wt) distribution
# (49% of positions sit above it), not a calibrated cutoff. Recalibrate on the
# genome-wide run against PhyloP, not against a ΔLLR-derived number.
DEFAULT_CONSERVED_THRESHOLD = -0.5


def _safe_mean(arr: Any) -> float | None:
    if arr is None:
        return None
    if hasattr(arr, "size") and arr.size == 0:
        return None
    if len(arr) == 0:
        return None
    return float(sum(arr) / len(arr))


class PLMVEPModule:
    """ESM-C masked-marginal variant-effect predictor (SiteModule)."""

    MODULE_NAME: str = "plm_vep"
    OUTPUT_COLUMNS: list[str] = [
        "plm_vep_status",
        "plm_vep_mean_llr_isoform",
        "plm_vep_mean_llr_canonical",
        "plm_vep_mean_llr_unique_region",
        "plm_vep_mean_llr_shared_region",
        "plm_vep_constraint_delta",
        "plm_vep_n_constrained_positions_unique",
        "plm_vep_n_constrained_positions_shared",
    ]
    SCOPE: str = "C"

    def __init__(
        self,
        config: PipelineConfig,
        *,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        conserved_threshold: float = DEFAULT_CONSERVED_THRESHOLD,
    ) -> None:
        """Initialize the module.

        Args:
            config: Pipeline configuration.
            cache_dir: Directory holding the ``<hash>.npz`` LLR cache files.
            conserved_threshold: logP(wt) at or ABOVE which a position counts as
                strongly-conserved. See :data:`DEFAULT_CONSERVED_THRESHOLD`.
        """
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.conserved_threshold = conserved_threshold

    def _load_llr(self, protein: str) -> Any | None:
        if not protein:
            return None
        cached = load_cache(protein_hash(protein), self.cache_dir)
        if cached is None:
            return None
        return cached.get("llr")

    def annotate_site(self, site: TranslationInitiationSite) -> dict[str, Any]:
        """Compute LLR-derived constraint metrics for a single TIS."""
        empty = {
            "status": "not_run",
            "mean_llr_isoform": None,
            "mean_llr_canonical": None,
            "mean_llr_unique_region": None,
            "mean_llr_shared_region": None,
            "constraint_delta": None,
            "n_constrained_positions_unique": None,
            "n_constrained_positions_shared": None,
        }

        iso_llr = self._load_llr(site.isoform_protein)
        can_llr = self._load_llr(site.canonical_protein)

        if iso_llr is None and can_llr is None:
            empty["status"] = "no_cache"
            return empty

        out = dict(empty)
        out["status"] = "ok"
        out["mean_llr_isoform"] = _safe_mean(iso_llr) if iso_llr is not None else None
        out["mean_llr_canonical"] = _safe_mean(can_llr) if can_llr is not None else None

        # Pick the protein space matching the diff region.
        space, unique_idx, shared_idx = diff_region_indices(site)
        scores = can_llr if space == "canonical" else iso_llr if space == "isoform" else None

        if scores is None or len(unique_idx) == 0:
            out["status"] = "no_diff_region"
            return out

        # Guard length mismatch (shouldn't happen if cache built from same
        # canonical/isoform proteins, but be defensive).
        if len(scores) < max(unique_idx + shared_idx + [0]) + 1:
            logger.warning(
                "plm_vep: cached LLR length %d shorter than %s protein indices for %s",
                len(scores),
                space,
                site.tis_id,
            )
            out["status"] = "length_mismatch"
            return out

        unique_vals = [float(scores[i]) for i in unique_idx]
        shared_vals = [float(scores[i]) for i in shared_idx]

        unique_mean = (sum(unique_vals) / len(unique_vals)) if unique_vals else None
        shared_mean = (sum(shared_vals) / len(shared_vals)) if shared_vals else None
        out["mean_llr_unique_region"] = unique_mean
        out["mean_llr_shared_region"] = shared_mean

        # A DIFFERENCE, not a ratio. Both means are negative log-probabilities, so
        # a ratio inverts (a more-negative unique region gives a value above 1) and
        # explodes as the denominator approaches zero on a well-predicted core —
        # cheeseman_test produced 412x and 257x that way. The difference is signed
        # in the direction it reads: POSITIVE means the unique region is better
        # predicted, i.e. more conserved, than the shared core.
        if unique_mean is not None and shared_mean is not None:
            out["constraint_delta"] = unique_mean - shared_mean

        # >= thr, not < thr: a residue the model predicts confidently is the
        # conserved one (see DEFAULT_CONSERVED_THRESHOLD for the measurement).
        thr = self.conserved_threshold
        out["n_constrained_positions_unique"] = sum(1 for v in unique_vals if v >= thr)
        out["n_constrained_positions_shared"] = sum(1 for v in shared_vals if v >= thr)
        return out

    def run(self, tis_sites: list[TranslationInitiationSite]) -> list[TranslationInitiationSite]:
        """Attach LLR-derived constraint annotations to each TIS."""
        for site in tis_sites:
            site.isoform_annotations[self.MODULE_NAME] = self.annotate_site(site)
        return tis_sites
