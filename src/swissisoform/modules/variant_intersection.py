r"""Module: Variant Intersection — per-TIS isoform-frame variant calling.

Post-processes the raw variant list attached to a TIS by
:class:`swissisoform.modules.clinical.ClinicalModule` and:

1. Re-validates each variant in the *isoform*'s reading frame using the
   TIS's ``orf_exons`` (via
   :meth:`ConsequenceValidator.validate_variants_against_orf`). Writes
   ``isoform_protein_pos`` / ``isoform_aa_ref`` / ``isoform_aa_alt`` /
   ``isoform_consequence`` onto each hit in place. This is what makes
   extension-unique-region variants (canonical 5'UTR / intron) actually
   callable as isoform missense / nonsense / frameshift — they are dropped
   by the canonical-CDS-only pass.
2. Tags each hit by genomic membership in the **unique** vs **shared**
   region. Unique is defined per ORF type:
   - extension / uORF / altORF: ``isoform_orf \\ canonical_orf`` (new
     sequence introduced by the isoform).
   - truncation: ``canonical_orf \\ isoform_orf`` (sequence lost from
     canonical). The truncation's unique residues live in canonical-protein
     space; ``isoform_protein_pos`` will be ``None`` for them.
3. Filters the hit list to variants that fall inside *some* coding region
   we care about (canonical ORF ∪ isoform ORF), dropping pure-intronic /
   pure-UTR variants — those would just bloat the parquet without carrying
   isoform-relevant signal.

Aggregates: total / unique / shared / pathogenic-in-each.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from swissisoform.clinical.validate import ConsequenceValidator
from swissisoform.coords import interval_difference, interval_intersection
from swissisoform.models import ORFType, TranslationInitiationSite

logger = logging.getLogger(__name__)


def _is_pathogenic(hit: dict[str, Any]) -> bool:
    """Return True when the hit carries a clinical_significance containing 'pathogenic'."""
    sig = hit.get("clinical_significance")
    return bool(sig) and "pathogenic" in str(sig).lower()


def _point_in_intervals(pos: int, intervals: list[tuple[int, int]]) -> bool:
    """True when *pos* (1-based genomic) lies in any half-open ``[start, end)`` interval.

    The intervals are 0-based half-open plus-strand coordinates, so a 1-based
    ``pos`` matches when ``start < pos <= end``.
    """
    for start, end in intervals:
        if start < pos <= end:
            return True
    return False


class VariantIntersectionModule:
    r"""Per-TIS isoform-frame variant calling + genomic membership tagging.

    Construct with an optional :class:`ConsequenceValidator` so the module
    can re-validate each variant against the TIS's ``orf_exons``. Without a
    validator, the module skips the isoform-frame pass and only does
    membership tagging on whatever canonical-frame fields are already on the
    hits (back-compat path used by unit tests).

    Output dict per TIS:

    - ``hits`` — augmented copies, each with both canonical-frame
      (``protein_pos`` / ``aa_ref`` / ``aa_alt``) and isoform-frame
      (``isoform_protein_pos`` / ``isoform_aa_ref`` / ``isoform_aa_alt``)
      fields, plus ``in_isoform_unique`` / ``in_isoform_shared`` / ``in_isoform``.
    - aggregate counts: ``n_total`` / ``n_in_unique_region`` /
      ``n_in_shared_region`` / ``n_pathogenic_in_unique_region`` /
      ``n_pathogenic_in_shared_region`` / ``n_dropped_outside_coding``.
    - ``summary.unique_space`` — ``"isoform"`` for extensions/uORFs,
      ``"canonical"`` for truncations (which protein space the unique region
      lives in).

    Attributes:
        MODULE_NAME: ``"variant_intersection"``.
        OUTPUT_COLUMNS: Columns produced per TIS.
        SCOPE: ``"C"`` (per-candidate).
    """

    MODULE_NAME: str = "variant_intersection"
    OUTPUT_COLUMNS: list[str] = [
        "variant_intersection_hits",
        "variant_intersection_n_total",
        "variant_intersection_n_in_unique_region",
        "variant_intersection_n_in_shared_region",
        "variant_intersection_n_pathogenic_in_unique_region",
        "variant_intersection_n_pathogenic_in_shared_region",
        "variant_intersection_n_disease_in_unique_region",
        "variant_intersection_n_disease_in_shared_region",
        "variant_intersection_n_gnomad_in_unique_region",
        "variant_intersection_n_gnomad_in_shared_region",
        "variant_intersection_n_dropped_outside_coding",
        "variant_intersection_summary",
    ]
    SCOPE: str = "C"

    def __init__(self, validator: ConsequenceValidator | None = None) -> None:
        """Initialize with an optional ConsequenceValidator for isoform-frame calls.

        Args:
            validator: Reused across TIS sites; its position-map and
                coding-sequence caches dedupe per-ORF work. When ``None``,
                the module skips isoform-frame validation and only does
                genomic-membership tagging.
        """
        self._validator = validator

    def annotate_site(self, site: TranslationInitiationSite) -> dict[str, Any]:
        """Augment + filter + tag clinical hits for *site*.

        Args:
            site: A TIS whose ``isoform_annotations['clinical']`` has already
                been populated by :class:`ClinicalModule`. The TIS must also
                carry ``orf_exons`` and ``canonical_orf_exons``; missing
                skeleton → ``status='no_skeleton'`` passthrough.

        Returns:
            Dict described in the class docstring.
        """
        clinical = site.isoform_annotations.get("clinical") or {}
        raw_hits = clinical.get("hits") if isinstance(clinical, dict) else None
        if not isinstance(raw_hits, list):
            raw_hits = []

        if not site.orf_exons:
            return self._fallback(raw_hits, status="no_skeleton")

        # ORF-type-aware unique region. For truncations the "unique" coding
        # nucleotides live in canonical (the lost N-term); for everything else
        # they live in the isoform.
        if site.orf_type == ORFType.TRUNCATED:
            unique = interval_difference(site.canonical_orf_exons, site.orf_exons)
            unique_space = "canonical"
        else:
            unique = interval_difference(site.orf_exons, site.canonical_orf_exons)
            unique_space = "isoform"
        shared = interval_intersection(site.orf_exons, site.canonical_orf_exons)

        # Re-validate every variant in the isoform's reading frame, writing
        # isoform_* fields onto each dict in place. Caches by tis_id so calls
        # for multiple variants on the same TIS reuse the position map and
        # coding sequence.
        if self._validator is not None:
            self._validator.validate_variants_against_orf(
                raw_hits,
                orf_exons=site.orf_exons,
                strand=site.strand,
                chrom=site.chrom,
                orf_key=site.tis_id,
                context=site.tis_id,
                field_prefix="isoform",
            )
            # Per-Tid canonical-frame revalidation: rewrites protein_pos /
            # aa_ref / aa_alt to the per-Tid frame (matching site.canonical_protein)
            # whenever this TIS's canonical ORF placed the variant. Clinical's
            # initial validation used the gene-level canonical transcript, which
            # diverges from per-Tid for FZR1-style cases — that mismatch causes
            # varianteffect's aa-ref guard to trip on shared-body hits because
            # the cached protein is per-Tid while protein_pos is gene-level.
            # No-op when per-Tid canonical == gene-level (most genes).
            if site.canonical_orf_exons:
                self._validator.validate_variants_against_orf(
                    raw_hits,
                    orf_exons=site.canonical_orf_exons,
                    strand=site.strand,
                    chrom=site.chrom,
                    orf_key=(site.tis_id, "canonical"),
                    context=f"{site.tis_id}:canonical",
                    field_prefix=None,
                )

        canonical_orf = site.canonical_orf_exons or []

        hits_out: list[dict[str, Any]] = []
        n_unique = 0
        n_shared = 0
        n_path_unique = 0
        n_path_shared = 0
        # Source-separated (§3): gnomAD answers "tolerated in healthy humans?",
        # ClinVar/COSMIC answer "disease-associated?" — opposite meanings, so F6
        # (disease burden) should read the disease counts, not the lumped total.
        n_disease_unique = 0
        n_disease_shared = 0
        n_gnomad_unique = 0
        n_gnomad_shared = 0
        n_scored = 0
        n_dropped_outside = 0
        n_unscored = 0

        for hit in raw_hits:
            tagged = deepcopy(hit)
            pos = hit.get("genomic_pos")
            if not isinstance(pos, int):
                tagged["in_isoform_unique"] = None
                tagged["in_isoform_shared"] = None
                tagged["in_isoform"] = None
                hits_out.append(tagged)
                n_unscored += 1
                continue

            in_unique = _point_in_intervals(pos, unique)
            in_shared = _point_in_intervals(pos, shared)
            in_isoform_orf = _point_in_intervals(pos, site.orf_exons)
            in_canonical_orf = _point_in_intervals(pos, canonical_orf)

            # Drop pure-intronic / pure-UTR variants that don't touch either
            # coding region. Keeps the hit list focused without losing any
            # variant we could meaningfully call.
            if not (in_isoform_orf or in_canonical_orf or in_unique):
                n_dropped_outside += 1
                continue

            tagged["in_isoform_unique"] = in_unique
            tagged["in_isoform_shared"] = in_shared
            tagged["in_isoform"] = in_isoform_orf

            n_scored += 1
            src = str(hit.get("source")).lower()
            is_disease = src in ("clinvar", "cosmic")
            is_gnomad = src == "gnomad"
            if in_unique:
                n_unique += 1
                if _is_pathogenic(hit):
                    n_path_unique += 1
                if is_disease:
                    n_disease_unique += 1
                if is_gnomad:
                    n_gnomad_unique += 1
            if in_shared:
                n_shared += 1
                if _is_pathogenic(hit):
                    n_path_shared += 1
                if is_disease:
                    n_disease_shared += 1
                if is_gnomad:
                    n_gnomad_shared += 1
            hits_out.append(tagged)

        # Share the augmented + filtered hits back to clinical's annotation so
        # downstream consumers (notably the comparator's positional subset)
        # see the isoform-frame fields without having to know about
        # variant_intersection. Per-TIS lists are independent (deepcopy above),
        # so this doesn't introduce cross-TIS shared state.
        if isinstance(clinical, dict):
            clinical["hits"] = hits_out

        return {
            "hits": hits_out,
            "n_total": len(hits_out),
            "n_in_unique_region": n_unique,
            "n_in_shared_region": n_shared,
            "n_pathogenic_in_unique_region": n_path_unique,
            "n_pathogenic_in_shared_region": n_path_shared,
            # Disease (ClinVar+COSMIC) vs tolerance (gnomAD), per region (§3).
            "n_disease_in_unique_region": n_disease_unique,
            "n_disease_in_shared_region": n_disease_shared,
            "n_gnomad_in_unique_region": n_gnomad_unique,
            "n_gnomad_in_shared_region": n_gnomad_shared,
            "n_dropped_outside_coding": n_dropped_outside,
            "summary": {
                "status": "ok",
                "n_scored": n_scored,
                "n_unscored": n_unscored,
                "unique_region_nt": sum(e - s for s, e in unique),
                "shared_region_nt": sum(e - s for s, e in shared),
                "unique_space": unique_space,
            },
        }

    def run(
        self,
        tis_sites: list[TranslationInitiationSite],
    ) -> list[TranslationInitiationSite]:
        """Annotate every TIS and attach results to ``isoform_annotations``."""
        for site in tis_sites:
            site.isoform_annotations[self.MODULE_NAME] = self.annotate_site(site)
        return tis_sites

    @staticmethod
    def _fallback(
        raw_hits: list[dict[str, Any]], *, status: str
    ) -> dict[str, Any]:
        """Return a "no-intersection-computed" result with flags set to None."""
        hits_out: list[dict[str, Any]] = []
        for hit in raw_hits:
            tagged = deepcopy(hit)
            tagged["in_isoform_unique"] = None
            tagged["in_isoform_shared"] = None
            tagged["in_isoform"] = None
            hits_out.append(tagged)
        return {
            "hits": hits_out,
            "n_total": len(raw_hits),
            "n_in_unique_region": None,
            "n_in_shared_region": None,
            "n_pathogenic_in_unique_region": None,
            "n_pathogenic_in_shared_region": None,
            "n_dropped_outside_coding": None,
            "summary": {
                "status": status,
                "n_scored": 0,
                "n_unscored": len(raw_hits),
            },
        }
