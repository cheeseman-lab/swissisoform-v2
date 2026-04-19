"""Variant consequence validation via codon-level translation analysis.

Maps genomic variant positions to coding sequence positions using CDS exon
boundaries from GTF, then predicts consequences by translating reference
and mutant codons.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from Bio.Seq import Seq

logger = logging.getLogger(__name__)

COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}


class ConsequenceValidator:
    """Validates variant consequences by mapping genomic to coding positions.

    Uses CDS exon boundaries from a GTF-derived DataFrame to build a
    genomic_pos -> coding_pos map, then translates reference and mutant
    codons to classify consequences (missense, synonymous, nonsense, etc.).

    When no genome FASTA is provided, validation falls back to HGVSP
    parsing only (validated=False).
    """

    def __init__(
        self,
        cds_df: pd.DataFrame | None = None,
        genome_fasta: str | None = None,
    ) -> None:
        """Initialize the validator.

        Args:
            cds_df: DataFrame of CDS features from ``load_cds_features()``.
                Columns: chromosome, start, end, strand, gene_id,
                transcript_id, feature_type.
            genome_fasta: Path to genome FASTA for extracting reference
                sequence. If None, validation falls back to HGVSP parsing only.
        """
        self._cds_df = cds_df
        self._genome = None  # lazy-loaded pysam.FastaFile
        self._genome_path = genome_fasta
        self._position_map_cache: dict[str, dict[int, int]] = {}
        self._coding_seq_cache: dict[str, str] = {}
        self._strand_cache: dict[str, str] = {}

    def build_position_map(self, transcript_id: str) -> dict[int, int]:
        """Build genomic_pos -> coding_pos map for a transcript's CDS.

        Walks through CDS exon regions in strand-aware order, assigning
        each genomic position a linear 0-indexed coding position starting
        from the canonical start codon.

        Args:
            transcript_id: GENCODE transcript ID (e.g. ``"ENST00000269305.9"``).

        Returns:
            Dict mapping genomic positions (1-based, inclusive) to coding
            positions (0-based).
        """
        if transcript_id in self._position_map_cache:
            return self._position_map_cache[transcript_id]

        if self._cds_df is None or self._cds_df.empty:
            return {}

        # Filter to this transcript's CDS features
        tx_cds = self._cds_df[
            (self._cds_df["transcript_id"] == transcript_id)
            & (self._cds_df["feature_type"] == "CDS")
        ].copy()

        if tx_cds.empty:
            return {}

        strand = tx_cds.iloc[0]["strand"]
        self._strand_cache[transcript_id] = strand

        # Find canonical start from start_codon feature
        start_codons = self._cds_df[
            (self._cds_df["transcript_id"] == transcript_id)
            & (self._cds_df["feature_type"] == "start_codon")
        ]

        if not start_codons.empty:
            canonical_start = (
                start_codons.iloc[0]["start"] if strand == "+" else start_codons.iloc[0]["end"]
            )
        else:
            canonical_start = tx_cds["start"].min() if strand == "+" else tx_cds["end"].max()

        # Sort CDS regions
        if strand == "+":
            tx_cds = tx_cds.sort_values("start")
        else:
            tx_cds = tx_cds.sort_values("start", ascending=False)

        pos_map: dict[int, int] = {}
        coding_pos = 0

        for _, cds in tx_cds.iterrows():
            cds_start = int(cds["start"])
            cds_end = int(cds["end"])

            if strand == "+":
                effective_start = max(cds_start, canonical_start)
                if effective_start <= cds_end:
                    for gpos in range(effective_start, cds_end + 1):
                        pos_map[gpos] = coding_pos
                        coding_pos += 1
            else:
                effective_end = min(cds_end, canonical_start)
                if cds_start <= effective_end:
                    for gpos in range(effective_end, cds_start - 1, -1):
                        pos_map[gpos] = coding_pos
                        coding_pos += 1

        self._position_map_cache[transcript_id] = pos_map
        return pos_map

    def build_coding_sequence(self, transcript_id: str) -> str:
        """Extract the coding sequence from genome FASTA using the position map.

        Requires a genome FASTA path to have been set at init time. Bases are
        complemented for minus-strand transcripts.

        Args:
            transcript_id: GENCODE transcript ID.

        Returns:
            Coding sequence string, or empty string if no genome is available.
        """
        if transcript_id in self._coding_seq_cache:
            return self._coding_seq_cache[transcript_id]

        if self._genome_path is None:
            return ""

        # Lazy-load genome
        if self._genome is None:
            import pysam

            self._genome = pysam.FastaFile(self._genome_path)

        pos_map = self.build_position_map(transcript_id)
        if not pos_map:
            return ""

        strand = self._strand_cache.get(transcript_id, "+")

        # Get CDS features for chromosome lookup
        tx_cds = self._cds_df[
            (self._cds_df["transcript_id"] == transcript_id)
            & (self._cds_df["feature_type"] == "CDS")
        ]
        if tx_cds.empty:
            return ""
        chrom = tx_cds.iloc[0]["chromosome"]

        # Sort genomic positions by coding position
        sorted_positions = sorted(pos_map.items(), key=lambda x: x[1])

        # Extract bases
        coding_bases: list[str] = []
        for gpos, _cpos in sorted_positions:
            base = self._genome.fetch(chrom, gpos - 1, gpos)  # pysam is 0-based
            if strand == "-":
                base = COMPLEMENT.get(base.upper(), base)
            coding_bases.append(base.upper())

        seq = "".join(coding_bases)
        self._coding_seq_cache[transcript_id] = seq
        return seq

    def validate_variant(
        self,
        transcript_id: str,
        genomic_pos: int,
        ref: str,
        alt: str,
    ) -> dict[str, Any]:
        """Validate a single variant's consequence via codon-level translation.

        Maps the genomic position to a coding position, then classifies the
        consequence by translating reference and mutant codons.

        Args:
            transcript_id: GENCODE transcript ID.
            genomic_pos: 1-based genomic position of the variant.
            ref: Reference allele (e.g. ``"A"``).
            alt: Alternate allele (e.g. ``"G"``).

        Returns:
            Dict with keys: consequence, protein_pos, aa_ref, aa_alt,
            codon_ref, codon_alt, validated.
        """
        pos_map = self.build_position_map(transcript_id)

        if not pos_map:
            return {
                "consequence": None,
                "protein_pos": None,
                "aa_ref": None,
                "aa_alt": None,
                "codon_ref": None,
                "codon_alt": None,
                "validated": False,
            }

        # Check if position is in coding region
        if genomic_pos not in pos_map:
            return {
                "consequence": "intronic",
                "protein_pos": None,
                "aa_ref": None,
                "aa_alt": None,
                "codon_ref": None,
                "codon_alt": None,
                "validated": True,
            }

        coding_pos = pos_map[genomic_pos]

        # Handle indels
        if len(ref) != len(alt):
            length_diff = abs(len(alt) - len(ref))
            if length_diff % 3 == 0:
                consequence = "inframe_insertion" if len(alt) > len(ref) else "inframe_deletion"
            else:
                consequence = "frameshift_variant"
            return {
                "consequence": consequence,
                "protein_pos": coding_pos // 3,
                "aa_ref": None,
                "aa_alt": None,
                "codon_ref": None,
                "codon_alt": None,
                "validated": True,
            }

        # Multi-nucleotide variant (MNV): len(ref) == len(alt) > 1.
        # Codon-level translation would need to walk multiple codons; skip
        # for now and classify as mnv so it doesn't spam reference_mismatch.
        if len(ref) > 1:
            return {
                "consequence": "mnv",
                "protein_pos": coding_pos // 3,
                "aa_ref": None,
                "aa_alt": None,
                "codon_ref": None,
                "codon_alt": None,
                "validated": True,
            }

        # SNV: need coding sequence for codon analysis
        coding_seq = self._coding_seq_cache.get(transcript_id) or self.build_coding_sequence(
            transcript_id
        )

        if not coding_seq:
            return {
                "consequence": None,
                "protein_pos": coding_pos // 3,
                "aa_ref": None,
                "aa_alt": None,
                "codon_ref": None,
                "codon_alt": None,
                "validated": False,
            }

        # Single nucleotide variant
        codon_start = (coding_pos // 3) * 3
        offset = coding_pos % 3

        if codon_start + 3 > len(coding_seq):
            logger.warning(
                "Codon extends beyond coding sequence for %s at pos %d",
                transcript_id,
                genomic_pos,
            )
            return {
                "consequence": None,
                "protein_pos": coding_pos // 3,
                "aa_ref": None,
                "aa_alt": None,
                "codon_ref": None,
                "codon_alt": None,
                "validated": False,
            }

        codon_ref = coding_seq[codon_start : codon_start + 3]

        # Handle strand: complement ref/alt for minus strand
        strand = self._strand_cache.get(transcript_id, "+")
        if strand == "-":
            ref_base = COMPLEMENT.get(ref.upper(), ref.upper())
            alt_base = COMPLEMENT.get(alt.upper(), alt.upper())
        else:
            ref_base = ref.upper()
            alt_base = alt.upper()

        # Verify reference matches expectation
        if codon_ref[offset] != ref_base:
            logger.warning(
                "Reference mismatch at %s:%d — expected %s at offset %d of codon %s, got %s",
                transcript_id,
                genomic_pos,
                ref_base,
                offset,
                codon_ref,
                codon_ref[offset],
            )
            return {
                "consequence": "reference_mismatch",
                "protein_pos": coding_pos // 3,
                "aa_ref": None,
                "aa_alt": None,
                "codon_ref": codon_ref,
                "codon_alt": None,
                "validated": False,
            }

        # Apply mutation
        codon_alt = codon_ref[:offset] + alt_base + codon_ref[offset + 1 :]

        # Translate both codons
        aa_ref = str(Seq(codon_ref).translate())
        aa_alt = str(Seq(codon_alt).translate())

        # Classify consequence
        if aa_ref == aa_alt:
            consequence = "synonymous_variant"
        elif aa_alt == "*":
            consequence = "stop_gained"
        elif aa_ref == "*":
            consequence = "stop_lost"
        else:
            consequence = "missense_variant"

        return {
            "consequence": consequence,
            "protein_pos": coding_pos // 3,
            "aa_ref": aa_ref,
            "aa_alt": aa_alt,
            "codon_ref": codon_ref,
            "codon_alt": codon_alt,
            "validated": True,
        }

    def validate_variants(
        self,
        variants: list[dict[str, Any]],
        transcript_id: str,
    ) -> list[dict[str, Any]]:
        """Validate a batch of variants and update their consequence annotations.

        For each variant dict that contains genomic_pos, ref, and alt, runs
        codon-level validation and overwrites the consequence if validation
        succeeds.

        Args:
            variants: List of variant dicts (in the clinical module hit format).
            transcript_id: GENCODE transcript ID for position mapping.

        Returns:
            The same list of variant dicts, updated in place with validated
            consequences and protein positions.
        """
        for variant in variants:
            gpos = variant.get("genomic_pos")
            ref = variant.get("ref")
            alt = variant.get("alt")

            # Treat empty string / zero / None as "missing" — for e.g. indels
            # where one side is "" legitimately, we need a stricter check per
            # variant type, but at this level both must be present and truthy
            # for SNV validation to be meaningful.
            if not gpos or ref is None or alt is None or (ref == "" and alt == ""):
                continue

            result = self.validate_variant(transcript_id, gpos, ref, alt)

            if result["validated"] and result["consequence"]:
                variant["consequence"] = result["consequence"]
                if result["protein_pos"] is not None:
                    variant["protein_pos"] = result["protein_pos"]
                if result["aa_ref"] is not None:
                    variant["aa_ref"] = result["aa_ref"]
                if result["aa_alt"] is not None:
                    variant["aa_alt"] = result["aa_alt"]

        return variants
