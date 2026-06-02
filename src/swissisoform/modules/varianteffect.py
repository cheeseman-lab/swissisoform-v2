"""Module: Variant Effect — per-variant effect scores on the identified mutants.

Combines two complementary per-variant predictors onto every clinical hit
already attached to a TIS, then aggregates over the isoform-unique region:

1. **ESM-2 masked-marginal ΔLLR** — ``logP(alt) − logP(wt)`` at the variant's
   residue, read from the per-position distribution cached by
   ``swissisoform.plm.embed`` (``aa_logprobs``). More negative ⇒ the
   substitution is less tolerated by the language model. Looked up in the
   **canonical** protein space, since the clinical module's ``protein_pos``
   is canonical-transcript-frame (set by ``ConsequenceValidator`` against the
   canonical CDS). Valid for shared-region variants and truncation-unique
   variants (both live in the canonical protein); extension-unique variants
   fall outside the canonical CDS and never reach this module with a
   ``protein_pos``.
2. **AlphaMissense** — DeepMind's calibrated missense pathogenicity
   (0-1 score + class) by genomic ``(chrom, pos, ref, alt)``.

Runs as a SiteModule **after** ``ClinicalModule`` and
``VariantIntersectionModule`` so it can read the genomic-membership flags
(``in_isoform_unique``) the latter writes. Emits a per-variant table plus
unique-region aggregates that feed evidence-scoring criterion F5.

Two independent damaging branches (mirrors v1):
1. **Loss-of-function** — a frameshift / stop-gained / splice / start-lost
   consequence is damaging on its own; neither AlphaMissense nor ESM-2 (both
   missense-only) can see these, so they are flagged from the consequence term.
2. **Missense** — AlphaMissense (canonical frame only) or ESM-2 ΔLLR.

Single-residue missense variants get ΔLLR / AlphaMissense scores; LoF variants
get ``effect_lof=True`` with ``None`` numeric scores but still count as damaging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from swissisoform.clinical.alphamissense import PATHOGENIC_CLASS, AlphaMissenseLookup
from swissisoform.config import PipelineConfig, ScoringConfig
from swissisoform.models import TranslationInitiationSite
from swissisoform.plm.embed import DEFAULT_CACHE_DIR, aa_column, load_cache, protein_hash

logger = logging.getLogger(__name__)

# ESM-2 ΔLLR cutoff below which a substitution is treated as damaging. The
# natural-log masked-marginal margin is comparable to ESM-1v variant scores;
# −7.5 is the threshold Brandes et al. (2023) use for the analogous LLR.
DEFAULT_LLR_DAMAGING_THRESHOLD = -7.5

# Loss-of-function (high-impact) consequence terms. A variant with any of these
# is functionally damaging independent of any missense pathogenicity score —
# AlphaMissense and ESM-2 ΔLLR are missense-only and never see these. Restores
# v1's first clinical branch (is-there-a-LoF-variant) alongside the missense
# branch. SequenceOntology high-impact terms plus the short forms our annotators
# emit.
LOF_CONSEQUENCES = frozenset(
    {
        "frameshift_variant",
        "frameshift",
        "stop_gained",
        "stop_lost",
        "start_lost",
        "start_loss",
        "splice_acceptor_variant",
        "splice_donor_variant",
        "transcript_ablation",
        "feature_truncation",
    }
)


class VariantEffectModule:
    """Per-variant ESM-2 ΔLLR + AlphaMissense effect scoring (SiteModule).

    Attributes:
        MODULE_NAME: ``"varianteffect"``.
        OUTPUT_COLUMNS: Columns produced per TIS.
        SCOPE: ``"C"`` (per-candidate).
    """

    MODULE_NAME: str = "varianteffect"
    OUTPUT_COLUMNS: list[str] = [
        "varianteffect_hits",
        "varianteffect_n_scored_plm",
        "varianteffect_n_scored_am",
        "varianteffect_n_scorable_in_unique",
        "varianteffect_n_damaging_in_unique",
        "varianteffect_n_lof_in_unique",
        "varianteffect_mean_delta_llr_unique",
        "varianteffect_min_delta_llr_unique",
        "varianteffect_mean_am_pathogenicity_unique",
        "varianteffect_max_am_pathogenicity_unique",
        "varianteffect_n_am_pathogenic_in_unique",
        "varianteffect_summary",
    ]
    SCOPE: str = "C"

    def __init__(
        self,
        config: PipelineConfig,
        *,
        plm_cache_dir: Path | str = DEFAULT_CACHE_DIR,
        alphamissense: AlphaMissenseLookup | None = None,
        alphamissense_db: Path | str | None = None,
        llr_damaging_threshold: float | None = None,
    ) -> None:
        """Initialize the module.

        Args:
            config: Pipeline configuration (reads ``config.scoring`` for the
                ΔLLR damaging threshold when not given explicitly).
            plm_cache_dir: Directory holding the ``<hash>.npz`` PLM caches with
                the per-position ``aa_logprobs`` distribution.
            alphamissense: A pre-built lookup; takes precedence over
                ``alphamissense_db``.
            alphamissense_db: Path to the tabix-indexed AlphaMissense hg38
                table; used to build a lookup when one isn't passed.
            llr_damaging_threshold: ESM-2 ΔLLR cutoff for the damaging flag.
        """
        self.config = config
        self.cache_dir = Path(plm_cache_dir)
        if alphamissense is not None:
            self._am: AlphaMissenseLookup | None = alphamissense
        elif alphamissense_db is not None:
            self._am = AlphaMissenseLookup(alphamissense_db)
        else:
            self._am = None
        scoring: ScoringConfig | None = config.scoring
        if llr_damaging_threshold is not None:
            self.llr_damaging_threshold = llr_damaging_threshold
        elif scoring is not None:
            self.llr_damaging_threshold = scoring.f5_llr_damaging_threshold
        else:
            self.llr_damaging_threshold = DEFAULT_LLR_DAMAGING_THRESHOLD
        # Per-protein-hash aa_logprobs cache, populated lazily within a run.
        self._llr_cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # ESM-2 ΔLLR
    # ------------------------------------------------------------------

    def _aa_logprobs(self, protein: str) -> Any | None:
        """Load (and memoize) the per-position aa_logprobs matrix for *protein*."""
        if not protein:
            return None
        h = protein_hash(protein)
        if h in self._llr_cache:
            return self._llr_cache[h]
        cached = load_cache(h, self.cache_dir)
        mat = cached.get("aa_logprobs") if cached else None
        self._llr_cache[h] = mat
        return mat

    def _delta_llr_in_frame(
        self,
        protein_seq: str,
        aa_logprobs: Any | None,
        pos: Any,
        aa_ref: Any,
        aa_alt: Any,
        frame: str,
    ) -> dict[str, Any]:
        """Compute ESM-2 ΔLLR for one (pos, ref→alt) in a given protein frame.

        Returns ``plm_llr_wt`` / ``plm_llr_alt`` / ``plm_delta_llr`` + a
        ``plm_status`` reason and the ``plm_frame`` actually scored
        (``"canonical"`` or ``"isoform"``). All four numeric fields are
        ``None`` when not scorable.
        """
        out = {
            "plm_llr_wt": None,
            "plm_llr_alt": None,
            "plm_delta_llr": None,
            "plm_frame": frame,
        }
        if aa_logprobs is None:
            return {**out, "plm_status": "no_aa_logprobs"}
        col_ref = aa_column(aa_ref)
        col_alt = aa_column(aa_alt)
        if not isinstance(pos, int) or col_ref is None or col_alt is None:
            return {**out, "plm_status": "not_missense"}
        if pos < 0 or pos >= len(aa_logprobs):
            return {**out, "plm_status": "pos_out_of_range"}
        # Frame/space sanity guard: cached residue at pos must equal the
        # variant's ref AA in this frame. Catches off-by-one or wrong-frame
        # lookups loudly instead of silently mis-scoring.
        if pos < len(protein_seq) and protein_seq[pos].upper() != str(aa_ref).upper():
            return {**out, "plm_status": "aa_ref_mismatch"}
        wt = float(aa_logprobs[pos][col_ref])
        alt = float(aa_logprobs[pos][col_alt])
        return {
            "plm_llr_wt": wt,
            "plm_llr_alt": alt,
            "plm_delta_llr": alt - wt,
            "plm_status": "ok",
            "plm_frame": frame,
        }

    def _score_hit_plm(
        self,
        hit: dict[str, Any],
        canonical_seq: str,
        canonical_lp: Any | None,
        isoform_seq: str,
        isoform_lp: Any | None,
    ) -> dict[str, Any]:
        """Pick the right protein frame for a hit and compute ESM-2 ΔLLR.

        Rule: variants in the isoform-unique region for an extension/uORF/altORF
        live in *isoform* coordinates, so score against the isoform cache when
        ``isoform_protein_pos`` is set (the validator wrote it during
        variant_intersection's per-TIS pass). Every other case (shared region,
        truncation-unique lost residues, or no isoform mapping) defaults to
        the canonical frame using ``protein_pos`` / ``aa_ref`` / ``aa_alt``.
        """
        iso_pos = hit.get("isoform_protein_pos")
        in_unique = hit.get("in_isoform_unique")
        if iso_pos is not None and in_unique is True:
            return self._delta_llr_in_frame(
                isoform_seq,
                isoform_lp,
                iso_pos,
                hit.get("isoform_aa_ref"),
                hit.get("isoform_aa_alt"),
                frame="isoform",
            )
        return self._delta_llr_in_frame(
            canonical_seq,
            canonical_lp,
            hit.get("protein_pos"),
            hit.get("aa_ref"),
            hit.get("aa_alt"),
            frame="canonical",
        )

    # ------------------------------------------------------------------
    # SiteModule protocol
    # ------------------------------------------------------------------

    def annotate_site(self, site: TranslationInitiationSite) -> dict[str, Any]:
        """Attach per-variant effect scores and aggregate over the unique region."""
        # Prefer the intersection-tagged hits (carry in_isoform_unique).
        vi = site.isoform_annotations.get("variant_intersection")
        raw_hits = vi.get("hits") if isinstance(vi, dict) else None
        tagged = True
        if not isinstance(raw_hits, list):
            clinical = site.isoform_annotations.get("clinical") or {}
            raw_hits = clinical.get("hits") if isinstance(clinical, dict) else None
            tagged = False
        if not isinstance(raw_hits, list):
            raw_hits = []

        canonical_seq = (site.canonical_protein or "").rstrip("*")
        canonical_lp = self._aa_logprobs(site.canonical_protein) if canonical_seq else None
        isoform_seq = (site.isoform_protein or "").rstrip("*")
        isoform_lp = self._aa_logprobs(site.isoform_protein) if isoform_seq else None
        am = self._am

        hits_out: list[dict[str, Any]] = []
        n_scored_plm = 0
        n_scored_am = 0
        # Unique-region aggregates.
        deltas_unique: list[float] = []
        am_path_unique: list[float] = []
        n_am_pathogenic_unique = 0
        n_lof_unique = 0
        scorable_unique_ids: set[int] = set()
        damaging_unique_ids: set[int] = set()

        for idx, hit in enumerate(raw_hits):
            tagged_hit = dict(hit)
            in_unique = hit.get("in_isoform_unique") if tagged else None

            plm = self._score_hit_plm(
                hit, canonical_seq, canonical_lp, isoform_seq, isoform_lp,
            )
            tagged_hit.update(plm)
            if plm["plm_delta_llr"] is not None:
                n_scored_plm += 1

            am_rec = None
            if am is not None:
                am_rec = am.lookup(
                    hit.get("chrom", ""),
                    hit.get("genomic_pos") if isinstance(hit.get("genomic_pos"), int) else 0,
                    hit.get("ref") or "",
                    hit.get("alt") or "",
                    transcript_id=getattr(site, "transcript_id", None),
                )
            if am_rec is not None:
                tagged_hit["am_pathogenicity"] = am_rec["am_pathogenicity"]
                tagged_hit["am_class"] = am_rec["am_class"]
                n_scored_am += 1
            else:
                tagged_hit["am_pathogenicity"] = None
                tagged_hit["am_class"] = None

            # Frame for this hit. An extension/uORF/altORF variant in the
            # isoform-unique region acts in *isoform* coordinates (it maps to
            # isoform_protein_pos); everything else — shared region, the lost
            # N-terminus of a truncation, or any unmapped variant — acts in the
            # canonical frame. Mirrors _score_hit_plm's frame choice (A2).
            use_isoform_frame = (
                hit.get("isoform_protein_pos") is not None and in_unique is True
            )
            consequence = (
                hit.get("isoform_consequence") if use_isoform_frame else hit.get("consequence")
            )
            consequence = consequence or hit.get("consequence") or hit.get("isoform_consequence")
            is_lof = bool(consequence) and str(consequence) in LOF_CONSEQUENCES

            # AlphaMissense is canonical-frame, missense-only — it is meaningless
            # on an extension-unique (isoform-frame) variant, so it must not drive
            # the damaging flag there (A3). PLM ΔLLR is already frame-aware.
            am_damaging = (
                am_rec is not None
                and am_rec["am_class"] == PATHOGENIC_CLASS
                and not use_isoform_frame
            )
            plm_damaging = (
                plm["plm_delta_llr"] is not None
                and plm["plm_delta_llr"] <= self.llr_damaging_threshold
            )
            # Two independent branches (A1/A4): a loss-of-function consequence,
            # OR a missense call from AlphaMissense / ESM-2.
            is_damaging = is_lof or am_damaging or plm_damaging
            tagged_hit["effect_lof"] = is_lof
            tagged_hit["effect_consequence"] = consequence
            tagged_hit["effect_damaging"] = is_damaging

            if in_unique is True:
                # A LoF variant is assessable even though no missense predictor
                # scores it — count it as scorable so F5 isn't blocked when the
                # only unique-region variants are frameshifts / stop-gains.
                scorable = (
                    plm["plm_delta_llr"] is not None or am_rec is not None or is_lof
                )
                if scorable:
                    scorable_unique_ids.add(idx)
                if is_lof:
                    n_lof_unique += 1
                if plm["plm_delta_llr"] is not None:
                    deltas_unique.append(plm["plm_delta_llr"])
                if am_rec is not None and not use_isoform_frame:
                    am_path_unique.append(am_rec["am_pathogenicity"])
                    if am_rec["am_class"] == PATHOGENIC_CLASS:
                        n_am_pathogenic_unique += 1
                if is_damaging:
                    damaging_unique_ids.add(idx)

            hits_out.append(tagged_hit)

        plm_available = canonical_lp is not None or isoform_lp is not None
        status = "ok"
        if not tagged:
            status = "no_intersection"
        elif am is None and not plm_available:
            status = "no_predictors"

        return {
            "hits": hits_out,
            "n_scored_plm": n_scored_plm,
            "n_scored_am": n_scored_am,
            "n_scorable_in_unique": len(scorable_unique_ids),
            "n_damaging_in_unique": len(damaging_unique_ids),
            "n_lof_in_unique": n_lof_unique,
            "mean_delta_llr_unique": (
                sum(deltas_unique) / len(deltas_unique) if deltas_unique else None
            ),
            "min_delta_llr_unique": min(deltas_unique) if deltas_unique else None,
            "mean_am_pathogenicity_unique": (
                sum(am_path_unique) / len(am_path_unique) if am_path_unique else None
            ),
            "max_am_pathogenicity_unique": max(am_path_unique) if am_path_unique else None,
            "n_am_pathogenic_in_unique": n_am_pathogenic_unique,
            "summary": {
                "status": status,
                "n_total": len(raw_hits),
                "plm_available": plm_available,
                "alphamissense_available": am is not None and am.available,
                "llr_damaging_threshold": self.llr_damaging_threshold,
            },
        }

    def run(self, tis_sites: list[TranslationInitiationSite]) -> list[TranslationInitiationSite]:
        """Annotate every TIS and attach results to ``isoform_annotations``."""
        for site in tis_sites:
            site.isoform_annotations[self.MODULE_NAME] = self.annotate_site(site)
        return tis_sites
