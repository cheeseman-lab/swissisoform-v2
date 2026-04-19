"""Module: Mass Spectrometry Validation — in-silico tryptic digest and peptide validation.

Performs in-silico tryptic digestion of isoform proteins to identify unique peptides,
optionally cross-referencing with pre-computed PepQuery2 validation results.
"""

from __future__ import annotations

import logging
from typing import Any

from swissisoform.config import PipelineConfig
from swissisoform.models import TranslationInitiationSite

logger = logging.getLogger(__name__)


class MassSpecModule:
    """Mass spectrometry validation module (``ProteinModule`` protocol).

    Performs in-silico tryptic digestion to find peptides with their positions
    in the protein, marks peptides unique to the isoform (not found in canonical
    digest), and optionally validates against pre-computed PepQuery2 results.

    Attributes:
        MODULE_NAME: Unique module identifier.
        OUTPUT_COLUMNS: Column names produced (prefixed with MODULE_NAME_).
        SCOPE: Module scope ('C' for per-site classification).
    """

    MODULE_NAME: str = "massspec"
    OUTPUT_COLUMNS: list[str] = ["massspec_hits", "massspec_summary"]
    SCOPE: str = "C"

    def __init__(
        self,
        config: PipelineConfig,
        validated_peptides: dict[str, set[str]] | None = None,
    ) -> None:
        """Initialize with pipeline configuration.

        Args:
            config: Pipeline configuration.
            validated_peptides: Optional pre-computed PepQuery results.
                Dict mapping gene_name -> set of validated peptide sequences.
                If provided, peptides found in this set are marked as validated.
        """
        self.config = config
        self.validated_peptides = validated_peptides or {}

    def _tryptic_digest(
        self,
        protein: str,
        missed_cleavages: int = 1,
        min_length: int = 7,
        max_length: int = 30,
    ) -> list[dict[str, Any]]:
        """In-silico tryptic digestion returning peptides with positions.

        Trypsin cleaves after K or R, except when followed by P.

        Args:
            protein: Protein sequence, optionally ending with '*'.
            missed_cleavages: Maximum number of missed cleavages (0 and 1).
            min_length: Minimum peptide length to keep.
            max_length: Maximum peptide length to keep.

        Returns:
            List of dicts with keys: peptide, pos, end, length.
        """
        seq = protein.rstrip("*").upper()
        if not seq:
            return []

        # Find cleavage sites: positions where seq[i] in (K, R) and seq[i+1] != P
        cleavage_sites: list[int] = []
        for i in range(len(seq)):
            if seq[i] in ("K", "R"):
                if i + 1 < len(seq) and seq[i + 1] == "P":
                    continue  # KP/RP exception
                cleavage_sites.append(i)

        # Build site boundaries: start positions of each fragment
        sites = [0] + [s + 1 for s in cleavage_sites] + [len(seq)]

        # Generate peptides for each missed cleavage count
        seen: set[tuple[str, int]] = set()
        peptides: list[dict[str, Any]] = []

        for mc in range(missed_cleavages + 1):
            for i in range(len(sites) - 1 - mc):
                start = sites[i]
                end = sites[i + 1 + mc]
                pep = seq[start:end]
                pep_len = len(pep)
                if pep_len < min_length or pep_len > max_length:
                    continue
                key = (pep, start)
                if key in seen:
                    continue
                seen.add(key)
                peptides.append(
                    {
                        "peptide": pep,
                        "pos": start,
                        "end": end,
                        "length": pep_len,
                    }
                )

        return peptides

    def annotate(
        self,
        protein: str,
        canonical_protein: str | None = None,
        gene_name: str | None = None,
    ) -> dict[str, Any]:
        """Compute mass spectrometry annotations for a protein.

        When *canonical_protein* is ``None`` (unknown), ``unique_to_isoform``
        is set to ``None`` for every peptide — NOT ``False`` — because we
        cannot tell whether a peptide is unique without the canonical digest.
        Summary ``unique_peptides`` is also ``None`` in that case.

        Args:
            protein: Isoform protein sequence.
            canonical_protein: Canonical protein sequence for uniqueness
                comparison. ``None`` means "unknown" (output will not claim
                uniqueness). Empty string is treated as unknown.
            gene_name: Gene name for PepQuery result lookup. ``None``/empty
                means "unknown" (no validation performed).

        Returns:
            Dict with keys 'hits' (list of peptide dicts) and 'summary' (stats dict).
        """
        isoform_peptides = self._tryptic_digest(protein)

        # Normalize missing values — empty string is treated as "unknown"
        canonical_known = bool(canonical_protein)
        gene_known = bool(gene_name)

        if not canonical_known:
            logger.debug(
                "MassSpec.annotate called without canonical_protein — "
                "unique_to_isoform will be None (unknown) for every peptide"
            )

        if not isoform_peptides:
            return {
                "hits": [],
                "summary": {
                    "total_peptides": 0,
                    "unique_peptides": 0 if canonical_known else None,
                    "validated_peptides": 0 if gene_known else None,
                    "min_peptide_length": None,
                    "max_peptide_length": None,
                },
            }

        # Build canonical peptide set for uniqueness check
        canonical_pep_seqs: set[str] = set()
        if canonical_known:
            canonical_digested = self._tryptic_digest(canonical_protein)
            canonical_pep_seqs = {p["peptide"] for p in canonical_digested}

        # Get validated peptide set for this gene
        gene_validated: set[str] = (
            self.validated_peptides.get(gene_name, set()) if gene_known else set()
        )

        # Annotate each peptide
        hits: list[dict[str, Any]] = []
        unique_count = 0
        validated_count = 0

        for pep in isoform_peptides:
            # unique is None (unknown) when canonical is missing — NOT False
            if canonical_known:
                unique = pep["peptide"] not in canonical_pep_seqs
            else:
                unique = None

            # validated is None (unknown) when gene is missing — NOT False
            if gene_known:
                validated = pep["peptide"] in gene_validated
            else:
                validated = None

            if unique is True:
                unique_count += 1
            if validated is True:
                validated_count += 1

            hits.append(
                {
                    "peptide": pep["peptide"],
                    "pos": pep["pos"],
                    "end": pep["end"],
                    "length": pep["length"],
                    "unique_to_isoform": unique,
                    "validated": validated,
                }
            )

        lengths = [h["length"] for h in hits]
        summary = {
            "total_peptides": len(hits),
            "unique_peptides": unique_count if canonical_known else None,
            "validated_peptides": validated_count if gene_known else None,
            "min_peptide_length": min(lengths),
            "max_peptide_length": max(lengths),
        }

        return {"hits": hits, "summary": summary}

    def run(self, tis_sites: list[TranslationInitiationSite]) -> list[TranslationInitiationSite]:
        """Compute mass spec annotations for each TIS site.

        Args:
            tis_sites: Input TIS sites with proteins set.

        Returns:
            The same sites with isoform_annotations["massspec"] populated.
        """
        for site in tis_sites:
            site.isoform_annotations[self.MODULE_NAME] = self.annotate(
                site.isoform_protein,
                canonical_protein=site.canonical_protein,
                gene_name=site.gene_name,
            )
        return tis_sites
