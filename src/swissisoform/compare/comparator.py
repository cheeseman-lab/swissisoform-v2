"""Paired canonical-vs-isoform comparator.

Operates on annotated Gene objects after ``AnnotationPipeline.run`` has
populated ``gene.canonical_annotations`` and per-TIS
``site.isoform_annotations``. Produces three kinds of comparison results:

1. **Scalar deltas** — ``isoform - canonical`` for every numeric field
   produced by a ProteinModule (pI, GRAVY, length, conservation score,
   etc.).
2. **Positional subsetting** — for every module whose output carries a
   ``hits`` list (motifs, clinical, conservation, massspec), filter the
   isoform (or canonical, for truncations) hits to those that overlap the
   differential-region coordinates. Counts and per-source breakdowns are
   derived from the filtered list.
3. **Scope-A enrichment** — re-annotate ``diff_region.sequence`` with a
   small set of ProteinModules (typically biophysics) to obtain the
   *unique region* value, and pair it with the *shared region* value.
   For ``tail_verified`` extensions, the shared region has the same
   sequence as the canonical protein, so its biophysical descriptors
   equal the canonical annotations and no separate annotation run is
   needed; symmetrically for tail-verified truncations, the shared
   region equals the isoform. The enrichment ratio (unique / shared) is
   reported only for ``tail_verified`` in-frame events.

The comparator writes into ``site.comparison[module_name]`` and
``site.diff_annotations[module_name]`` in place and returns the same
list of Gene objects.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from swissisoform.compare.paired import PairedComparison
from swissisoform.models import Gene, ORFType, TranslationInitiationSite

logger = logging.getLogger(__name__)

# ORF types for which the shared-region sequence equals a whole pane
# (canonical for extensions, isoform for truncations), allowing Scope-A
# enrichment ratios to reuse the existing annotations without extra compute.
_IN_FRAME_ORF_TYPES = (ORFType.EXTENDED, ORFType.TRUNCATED, ORFType.UOORF)

# Fields that are definitionally positional (hit lists) and should never be
# compared as scalars.
_POSITIONAL_KEYS = {"hits", "summary"}


def _hits_overlapping(
    hits: list[dict[str, Any]], start: int | None, end: int | None
) -> list[dict[str, Any]]:
    """Return hits whose [pos, end) interval overlaps [start, end).

    A hit is kept when ``pos < end`` AND ``hit_end > start``.  Hits
    missing a positional field (e.g. gene-level summary rows) are
    conservatively dropped from the subset.
    """
    if start is None or end is None or start >= end:
        return []
    subset: list[dict[str, Any]] = []
    for hit in hits:
        pos = hit.get("pos")
        hit_end = hit.get("end", pos + 1 if pos is not None else None)
        if pos is None or hit_end is None:
            continue
        if pos < end and hit_end > start:
            subset.append(hit)
    return subset


def _is_numeric(value: Any) -> bool:
    """True for finite real numbers (excludes bools, NaN, inf)."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return not (math.isnan(value) or math.isinf(value))


def _scalar_deltas(canonical: dict[str, Any], isoform: dict[str, Any]) -> dict[str, float]:
    """Compute ``isoform - canonical`` for every shared numeric scalar field."""
    deltas: dict[str, float] = {}
    for key, iso_val in isoform.items():
        if key in _POSITIONAL_KEYS:
            continue
        can_val = canonical.get(key)
        if _is_numeric(iso_val) and _is_numeric(can_val):
            deltas[f"{key}_delta"] = float(iso_val) - float(can_val)
    return deltas


def _categorical_changes(canonical: dict[str, Any], isoform: dict[str, Any]) -> dict[str, Any]:
    """Flag categorical (non-numeric, non-list) fields whose values differ."""
    changes: dict[str, Any] = {}
    for key, iso_val in isoform.items():
        if key in _POSITIONAL_KEYS or _is_numeric(iso_val):
            continue
        if isinstance(iso_val, (list, dict)):
            continue
        can_val = canonical.get(key)
        if iso_val != can_val:
            changes[f"{key}_changed"] = True
            changes[f"{key}_canonical"] = can_val
            changes[f"{key}_isoform"] = iso_val
    return changes


def _scope_a_enrichment(diff_ann: dict[str, Any], shared_ann: dict[str, Any]) -> dict[str, Any]:
    """Enrichment ratios for every shared numeric field (``unique / shared``).

    Delegates to ``PairedComparison.compare_scalar_regions`` for the
    per-field dict. Returns a flat mapping keyed by
    ``{field}_{unique,shared,ratio,enriched}``.
    """
    out: dict[str, Any] = {}
    for key, diff_val in diff_ann.items():
        if key in _POSITIONAL_KEYS or not _is_numeric(diff_val):
            continue
        shared_val = shared_ann.get(key)
        if not _is_numeric(shared_val):
            continue
        result = PairedComparison.compare_scalar_regions(float(diff_val), float(shared_val))
        for sub_key, sub_val in result.items():
            out[f"{key}_{sub_key}"] = sub_val
    return out


class Comparator:
    """Derive canonical-vs-isoform comparisons from annotated Gene objects.

    The comparator assumes ``AnnotationPipeline.run`` has already populated
    ``gene.canonical_annotations`` and per-site ``isoform_annotations``.
    For each TIS it writes:

    - ``site.comparison[module_name]`` — scalar deltas, categorical change
      flags, positional hit subsets, and Scope-A enrichment ratios.
    - ``site.diff_annotations[module_name]`` — optional re-annotation of
      ``diff_region.sequence`` with ``scope_a_modules`` (for Scope-A
      enrichment denominators that cannot be reused from the
      canonical / isoform panes).

    Attributes:
        scope_a_modules: ProteinModule instances to re-run on
            ``diff_region.sequence``. Typically ``[BiophysicsModule(cfg)]``
            — motif / clinical / conservation modules don't need Scope-A
            re-annotation because their positional hits already carry the
            per-residue information needed to restrict to the diff region.
    """

    def __init__(self, scope_a_modules: list[Any] | None = None) -> None:
        """Store the Scope-A modules used for diff-region re-annotation."""
        self.scope_a_modules = scope_a_modules or []

    def compare(self, genes: list[Gene]) -> list[Gene]:
        """Derive comparison fields for every TIS of every gene.

        Args:
            genes: Gene objects with ``canonical_annotations`` and
                ``tis_sites[*].isoform_annotations`` populated.

        Returns:
            The same list of Gene objects with
            ``site.comparison`` and ``site.diff_annotations`` filled in.
        """
        for gene in genes:
            for site in gene.tis_sites:
                self._compare_site(gene, site)
        return genes

    def _compare_site(self, gene: Gene, site: TranslationInitiationSite) -> None:
        """Fill ``site.comparison`` and ``site.diff_annotations`` for one TIS."""
        site.comparison = {}
        site.diff_annotations = {}
        diff_region = site.diff_region

        # ── Scope-A re-annotation of the diff region ─────────────────────
        if diff_region is not None and diff_region.sequence and self.scope_a_modules:
            for mod in self.scope_a_modules:
                site.diff_annotations[mod.MODULE_NAME] = mod.annotate(diff_region.sequence)

        # ── Per-module comparison ───────────────────────────────────────
        modules = set(site.isoform_annotations) | set(gene.canonical_annotations)
        for module_name in modules:
            canonical_ann = gene.canonical_annotations.get(module_name, {})
            isoform_ann = site.isoform_annotations.get(module_name, {})
            diff_ann = site.diff_annotations.get(module_name, {})
            site.comparison[module_name] = self._compare_module(
                canonical_ann, isoform_ann, diff_ann, site
            )

    def _compare_module(
        self,
        canonical: dict[str, Any],
        isoform: dict[str, Any],
        diff: dict[str, Any],
        site: TranslationInitiationSite,
    ) -> dict[str, Any]:
        """Build the comparison dict for a single module's outputs."""
        result: dict[str, Any] = {}

        result.update(_scalar_deltas(canonical, isoform))
        result.update(_categorical_changes(canonical, isoform))

        # Positional subsetting: filter to diff region coords.
        hits = isoform.get("hits")
        if isinstance(hits, list):
            subset, source_pane = self._subset_hits(hits, canonical, site)
            result["hits_in_diff_region"] = subset
            result["n_hits_in_diff_region"] = len(subset)
            if source_pane is not None:
                result["hits_source_pane"] = source_pane

        # Scope-A enrichment: diff-region annotations vs. shared-region.
        if diff:
            shared = self._shared_annotations(canonical, isoform, site)
            if shared:
                result.update(_scope_a_enrichment(diff, shared))

        return result

    def _subset_hits(
        self,
        isoform_hits: list[dict[str, Any]],
        canonical_ann: dict[str, Any],
        site: TranslationInitiationSite,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Return positional hits overlapping the diff region.

        For extensions / uORFs / altORFs (diff in isoform frame), filter
        the isoform's hits. For truncations (diff in canonical frame), the
        lost region's hits come from the canonical's hit list, so filter
        those instead and return source_pane='canonical'.
        """
        diff_region = site.diff_region
        if diff_region is None:
            return [], None

        # Truncation: diff lives in canonical coordinates → filter canonical hits
        if diff_region.canonical_start is not None and diff_region.canonical_end is not None:
            canonical_hits = canonical_ann.get("hits", [])
            if isinstance(canonical_hits, list):
                subset = _hits_overlapping(
                    canonical_hits,
                    diff_region.canonical_start,
                    diff_region.canonical_end,
                )
                return subset, "canonical"
            return [], "canonical"

        # Extension / uORF / altORF: diff in isoform coords → filter isoform hits
        if diff_region.isoform_start is not None and diff_region.isoform_end is not None:
            subset = _hits_overlapping(
                isoform_hits, diff_region.isoform_start, diff_region.isoform_end
            )
            return subset, "isoform"

        return [], None

    def _shared_annotations(
        self,
        canonical: dict[str, Any],
        isoform: dict[str, Any],
        site: TranslationInitiationSite,
    ) -> dict[str, Any] | None:
        """Pick the annotation pane that represents the shared region.

        Returns ``None`` when no shared region exists (uORFs, altORFs) or
        when the diff region isn't trustworthy enough to assert that the
        shared sequence equals either pane (non tail-verified).
        """
        diff_region = site.diff_region
        if diff_region is None or diff_region.confidence != "tail_verified":
            return None
        if site.orf_type == ORFType.EXTENDED:
            # Shared region = canonical body (= canonical pane by sequence identity)
            return canonical
        if site.orf_type == ORFType.TRUNCATED:
            # Shared region = isoform body (= isoform pane by sequence identity)
            return isoform
        # UOORF, UORF, ALT_ORF, INTERNAL_OUT_OF_FRAME, THREE_UTR_ORF, ANNOTATED
        return None


def compare_genes(genes: list[Gene], scope_a_modules: list[Any] | None = None) -> list[Gene]:
    """Functional wrapper: instantiate a Comparator and run it in place."""
    return Comparator(scope_a_modules=scope_a_modules).compare(genes)
