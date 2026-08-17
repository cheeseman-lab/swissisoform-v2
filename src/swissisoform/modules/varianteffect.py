"""Module: Variant Effect — per-variant effect scores on the identified mutants.

Combines two complementary per-variant predictors onto every clinical hit
already attached to a TIS, then aggregates over the isoform-unique region:

1. **ESM-C masked-marginal ΔLLR** — ``logP(alt) − logP(wt)`` at the variant's
   residue, read from the per-position distribution cached by
   ``swissisoform.plm.embed`` (``aa_logprobs``). More negative ⇒ the
   substitution is less tolerated by the language model. **Frame-aware**
   (``_score_hit_plm``): an ``in_isoform_unique`` hit carrying an
   ``isoform_protein_pos`` — an extension's or separate ORF's own residues —
   is scored against the ISOFORM protein; everything else (shared region, a
   truncation's lost N-terminus, any unmapped variant) against the canonical
   protein, whose ``protein_pos`` ``ConsequenceValidator`` set. Both frames
   are real sequence, so both are scoreable; what an extension's unique region
   lacks is not a frame but an evolutionary history, which makes the score a
   statement about model-perceived disruptiveness rather than about selection.
2. **AlphaMissense** — DeepMind's calibrated missense pathogenicity
   (0-1 score + class) by genomic ``(chrom, pos, ref, alt)``. Canonical-frame
   only: a precomputed canonical-transcript table has no entry for a position
   outside the canonical CDS, so it is absent by construction over an
   extension's or separate ORF's unique region, and is excluded from the
   damaging flag there.

Runs as a SiteModule **after** ``ClinicalModule`` and
``VariantIntersectionModule`` so it can read the genomic-membership flags
(``in_isoform_unique``) the latter writes. Emits a per-variant table plus
unique-region aggregates that feed evidence-scoring criterion M1.

Two independent damaging branches (mirrors v1):
1. **Loss-of-function** — a frameshift / stop-gained / splice / start-lost
   consequence is damaging on its own; neither AlphaMissense nor ESM-C (both
   missense-only) can see these, so they are flagged from the consequence term.
2. **Missense** — AlphaMissense (canonical frame only) or ESM-C ΔLLR.

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

# ESM-C ΔLLR cutoff below which a substitution is treated as damaging. The
# natural-log masked-marginal margin is comparable to ESM-1v variant scores;
# −7.5 is the threshold Brandes et al. (2023) use for the analogous LLR.
DEFAULT_LLR_DAMAGING_THRESHOLD = -7.5

# gnomAD is a *tolerance* catalogue, not a disease one: a predicted-damaging
# variant seen at appreciable frequency in healthy humans is tolerated, so it
# must not count as damaging (ACMG allele-frequency benign evidence). Variants
# from gnomAD with allele_frequency at or above this are gated out of the
# damaging flag. ClinVar / COSMIC (disease) variants are never gated.
DEFAULT_GNOMAD_TOLERATED_AF = 1e-3

# Loss-of-function (high-impact) consequence terms. A variant with any of these
# is functionally damaging independent of any missense pathogenicity score —
# AlphaMissense and ESM-C ΔLLR are missense-only and never see these. Restores
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


def _ve_metric_cols(prefix: str) -> list[str]:
    """The seven varianteffect aggregate columns for a region[,source] prefix."""
    return [
        f"varianteffect_n_scorable_in_{prefix}",
        f"varianteffect_n_damaging_in_{prefix}",
        f"varianteffect_n_lof_in_{prefix}",
        f"varianteffect_mean_delta_llr_{prefix}",
        f"varianteffect_min_delta_llr_{prefix}",
        f"varianteffect_mean_am_pathogenicity_{prefix}",
        f"varianteffect_n_am_pathogenic_in_{prefix}",
    ]


class VariantEffectModule:
    """Per-variant ESM-C ΔLLR + AlphaMissense effect scoring (SiteModule).

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
        # Blended (all sources) per region. The shared twins back M1's
        # differential-vs-shared enrichment (§2).
        *_ve_metric_cols("unique"),
        "varianteffect_max_am_pathogenicity_unique",
        *_ve_metric_cols("shared"),
        "varianteffect_max_am_pathogenicity_shared",
        # Source-separated (§4): the predictors are source-independent, so each
        # region splits into gnomad (germline → M1) and disease (ClinVar+COSMIC
        # → M2). Blended = gnomad + disease.
        *_ve_metric_cols("unique_gnomad"),
        *_ve_metric_cols("unique_disease"),
        *_ve_metric_cols("shared_gnomad"),
        *_ve_metric_cols("shared_disease"),
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
            llr_damaging_threshold: ESM-C ΔLLR cutoff for the damaging flag.
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
            self.llr_damaging_threshold = scoring.m1_llr_damaging_threshold
        else:
            self.llr_damaging_threshold = DEFAULT_LLR_DAMAGING_THRESHOLD
        self.gnomad_tolerated_af = DEFAULT_GNOMAD_TOLERATED_AF
        # Per-protein-hash aa_logprobs cache, populated lazily within a run.
        self._llr_cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # ESM-C ΔLLR
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
        """Compute ESM-C ΔLLR for one (pos, ref→alt) in a given protein frame.

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
        """Pick the right protein frame for a hit and compute ESM-C ΔLLR.

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
        # Predictor aggregates keyed by (region, source): region ∈ {unique,
        # shared}, source ∈ {gnomad, disease}. The predictors are
        # source-independent (they score a substitution's effect), so we apply
        # them to both pools and surface the gnomad slice in M1 (germline
        # tolerance) and the disease slice in M2 (clinical). The blended
        # region-only columns are derived as gnomad+disease (§4).
        def _bucket() -> dict[str, Any]:
            return {
                "scorable": set(), "damaging": set(), "lof": 0,
                "deltas": [], "am_path": [], "n_am_path": 0,
            }

        agg: dict[tuple[str, str], dict[str, Any]] = {
            (rg, sr): _bucket() for rg in ("unique", "shared") for sr in ("gnomad", "disease")
        }

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
            # OR a missense call from AlphaMissense / ESM-C.
            is_damaging = is_lof or am_damaging or plm_damaging

            # gnomAD tolerance gate (§3): a predicted-damaging *gnomAD* variant
            # common in healthy humans is tolerated → not damaging. LoF stands
            # (it is functionally definitive); disease-DB variants are untouched.
            af = hit.get("allele_frequency")
            tolerated = (
                str(hit.get("source")).lower() == "gnomad"
                and isinstance(af, (int, float))
                and af >= self.gnomad_tolerated_af
            )
            if tolerated and not is_lof:
                is_damaging = False
            tagged_hit["effect_tolerated_in_gnomad"] = bool(tolerated)
            tagged_hit["effect_lof"] = is_lof
            tagged_hit["effect_consequence"] = consequence
            tagged_hit["effect_damaging"] = is_damaging

            in_shared = hit.get("in_isoform_shared") if tagged else None
            region_key = (
                "unique" if in_unique is True else ("shared" if in_shared is True else None)
            )
            src = str(hit.get("source")).lower()
            source_key = "gnomad" if src == "gnomad" else "disease"  # clinvar/cosmic → disease
            if region_key is not None:
                b = agg[(region_key, source_key)]
                # A LoF variant is assessable even when no missense predictor
                # scores it (frameshift / stop-gain) — count it as scorable.
                if plm["plm_delta_llr"] is not None or am_rec is not None or is_lof:
                    b["scorable"].add(idx)
                if is_lof:
                    b["lof"] += 1
                if plm["plm_delta_llr"] is not None:
                    b["deltas"].append(plm["plm_delta_llr"])
                if am_rec is not None and not use_isoform_frame:
                    b["am_path"].append(am_rec["am_pathogenicity"])
                    if am_rec["am_class"] == PATHOGENIC_CLASS:
                        b["n_am_path"] += 1
                if is_damaging:
                    b["damaging"].add(idx)

            hits_out.append(tagged_hit)

        plm_available = canonical_lp is not None or isoform_lp is not None
        status = "ok"
        if not tagged:
            status = "no_intersection"
        elif am is None and not plm_available:
            status = "no_predictors"

        def _merge(b1: dict[str, Any], b2: dict[str, Any]) -> dict[str, Any]:
            # Variants have exactly one source, so the index sets are disjoint.
            return {
                "scorable": b1["scorable"] | b2["scorable"],
                "damaging": b1["damaging"] | b2["damaging"],
                "lof": b1["lof"] + b2["lof"],
                "deltas": b1["deltas"] + b2["deltas"],
                "am_path": b1["am_path"] + b2["am_path"],
                "n_am_path": b1["n_am_path"] + b2["n_am_path"],
            }

        def _metrics(prefix: str, b: dict[str, Any]) -> dict[str, Any]:
            deltas, am_path = b["deltas"], b["am_path"]
            # ``no_intersection`` means the region had no variants to evaluate;
            # report ``None`` so a true zero (ran, found nothing) stays distinct.
            # ``no_predictors`` still counts consequence-based LoF/damaging, so
            # those counts stay real.
            counted = status != "no_intersection"
            return {
                f"n_scorable_in_{prefix}": len(b["scorable"]) if counted else None,
                f"n_damaging_in_{prefix}": len(b["damaging"]) if counted else None,
                f"n_lof_in_{prefix}": b["lof"] if counted else None,
                f"mean_delta_llr_{prefix}": (sum(deltas) / len(deltas) if deltas else None),
                f"min_delta_llr_{prefix}": (min(deltas) if deltas else None),
                f"mean_am_pathogenicity_{prefix}": (
                    sum(am_path) / len(am_path) if am_path else None
                ),
                f"n_am_pathogenic_in_{prefix}": b["n_am_path"],
            }

        result: dict[str, Any] = {
            "hits": hits_out,
            "n_scored_plm": n_scored_plm,
            "n_scored_am": n_scored_am,
        }
        for rg in ("unique", "shared"):
            blended = _merge(agg[(rg, "gnomad")], agg[(rg, "disease")])
            result.update(_metrics(rg, blended))  # blended (backward-compat)
            result[f"max_am_pathogenicity_{rg}"] = (
                max(blended["am_path"]) if blended["am_path"] else None
            )
            for sr in ("gnomad", "disease"):  # source-separated (§4)
                result.update(_metrics(f"{rg}_{sr}", agg[(rg, sr)]))
        result["summary"] = {
            "status": status,
            "n_total": len(raw_hits),
            "plm_available": plm_available,
            "alphamissense_available": am is not None and am.available,
            "llr_damaging_threshold": self.llr_damaging_threshold,
        }
        return result

    def run(self, tis_sites: list[TranslationInitiationSite]) -> list[TranslationInitiationSite]:
        """Annotate every TIS and attach results to ``isoform_annotations``."""
        for site in tis_sites:
            site.isoform_annotations[self.MODULE_NAME] = self.annotate_site(site)
        return tis_sites
