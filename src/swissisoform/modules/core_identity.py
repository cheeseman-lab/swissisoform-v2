"""Module 1: Core Identity — ORF classification, lengths, and in-frame check.

Computes derived identity/classification annotations from pre-populated
canonical_protein, isoform_protein, differential_sequence, and orf_type fields.
"""

from __future__ import annotations

from swissisoform.config import PipelineConfig
from swissisoform.models import ORFType, TranslationInitiationSite

# ORF types that share the canonical reading frame
_IN_FRAME_TYPES: frozenset[ORFType] = frozenset(
    {ORFType.ANNOTATED, ORFType.EXTENDED, ORFType.TRUNCATED, ORFType.UOORF}
)


def _protein_length(seq: str) -> int:
    """Count residues in a protein sequence, excluding a trailing stop codon '*'.

    Args:
        seq: Protein sequence, optionally ending with '*'.

    Returns:
        Number of amino acid residues (excluding '*').
    """
    if seq.endswith("*"):
        return len(seq) - 1
    return len(seq)


def _is_in_frame(site: TranslationInitiationSite) -> bool:
    """Determine whether a TIS is in-frame with the canonical CDS.

    In-frame types: ANNOTATED, EXTENDED, TRUNCATED, UOORF.
    Out-of-frame types: UORF, INTERNAL_OUT_OF_FRAME, THREE_UTR_ORF, ALT_ORF.

    Args:
        site: A TranslationInitiationSite with orf_type set.

    Returns:
        True if the ORF type is in-frame with canonical CDS.
    """
    return site.orf_type in _IN_FRAME_TYPES


class CoreIdentityModule:
    """Module 1: Core Identity annotations.

    Computes variant IDs, ORF type labels, protein lengths, in-frame status,
    and large truncation warnings for each TIS site.

    Attributes:
        MODULE_NAME: Unique module identifier.
        OUTPUT_COLUMNS: Column names produced (prefixed with MODULE_NAME_).
        SCOPE: Module scope ('C' for per-site classification).
        truncation_max_aa: Maximum truncation length before warning.
    """

    MODULE_NAME: str = "core_identity"
    OUTPUT_COLUMNS: list[str] = [
        "core_identity_variant_id",
        "core_identity_orf_type",
        "core_identity_canonical_protein_length",
        "core_identity_isoform_protein_length",
        "core_identity_differential_length_aa",
        "core_identity_in_frame",
        "core_identity_large_truncation_warning",
    ]
    SCOPE: str = "C"

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize with pipeline configuration.

        Args:
            config: Pipeline configuration. Uses scoring.truncation_max_aa if
                scoring config is provided, otherwise defaults to 200.
        """
        self.truncation_max_aa = 200
        if config.scoring is not None:
            self.truncation_max_aa = config.scoring.truncation_max_aa

    def run(
        self, tis_sites: list[TranslationInitiationSite]
    ) -> list[TranslationInitiationSite]:
        """Compute core identity annotations for each TIS site.

        For each site, writes to site.annotations["core_identity"] with keys:
        variant_id, orf_type, canonical_protein_length, isoform_protein_length,
        differential_length_aa, in_frame, large_truncation_warning.

        Args:
            tis_sites: Input TIS sites with proteins and orf_type already set.

        Returns:
            The same sites with annotations["core_identity"] populated.
        """
        for site in tis_sites:
            diff_len = _protein_length(site.differential_sequence)
            in_frame = _is_in_frame(site)
            large_trunc = (
                site.orf_type == ORFType.TRUNCATED and diff_len > self.truncation_max_aa
            )

            site.annotations[self.MODULE_NAME] = {
                "variant_id": f"{site.gene_name}_{site.tis_id}",
                "orf_type": site.orf_type.value,
                "canonical_protein_length": _protein_length(site.canonical_protein),
                "isoform_protein_length": _protein_length(site.isoform_protein),
                "differential_length_aa": diff_len,
                "in_frame": in_frame,
                "large_truncation_warning": large_trunc,
            }

        return tis_sites
