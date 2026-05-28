"""Variant consequence validation via codon-level translation analysis.

Maps genomic variant positions to coding-sequence positions using exon
intervals — either the canonical CDS from a GTF (``validate_variant`` /
``build_position_map``) or an arbitrary ORF's ``orf_exons`` from the
assembly Layer-2 walker (``validate_variant_against_orf``) — then
predicts consequences by translating reference and mutant codons. The
ORF path is what lets the pipeline call mutations in the *isoform*
reading frame for extension/altORF unique regions that fall outside the
canonical CDS.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from Bio.Seq import Seq

logger = logging.getLogger(__name__)

COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}


def _revcomp(seq: str) -> str:
    """Reverse-complement an A/C/G/T/N sequence; non-ACGT bases pass through."""
    return "".join(COMPLEMENT.get(b, b) for b in reversed(seq))


class ConsequenceValidator:
    """Validates variant consequences by mapping genomic to coding positions.

    Two entry points:

    - :meth:`validate_variant` — canonical-CDS frame, takes a GENCODE
      ``transcript_id`` and uses the CDS / start_codon features from the
      GTF-derived ``cds_df``.
    - :meth:`validate_variant_against_orf` — arbitrary-ORF frame, takes
      ``orf_exons`` (0-based half-open plus-strand intervals in ascending
      genomic order, as produced by :func:`swissisoform.coords.orf_exons_from_skeleton`)
      plus strand + chromosome. Used to re-validate variants against an
      *isoform*'s reading frame so the pipeline can score mutations in the
      isoform-unique region (extensions, altORFs) that the canonical-CDS
      validator drops as 5'UTR/intronic.

    Both paths share the codon-level classification helper
    :meth:`_analyze_variant`, so consequence semantics (synonymous /
    missense / stop_gained / stop_lost / inframe_insertion /
    inframe_deletion / frameshift_variant / mnv / intronic /
    reference_mismatch) are identical regardless of which frame is used.

    When no genome FASTA is provided, validation falls back to position
    classification only (``validated=False`` on SNVs that require the
    reference codon).
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
        # Canonical-CDS caches keyed by transcript_id (str)
        self._position_map_cache: dict[str, dict[int, int]] = {}
        self._coding_seq_cache: dict[str, str] = {}
        self._strand_cache: dict[str, str] = {}
        # ORF-frame caches keyed by hashable orf_key (caller-supplied or
        # derived from the exons + strand). Separate to avoid str/tuple
        # key collisions with the canonical caches.
        self._orf_position_map_cache: dict[Any, dict[int, int]] = {}
        self._orf_coding_seq_cache: dict[Any, str] = {}

    # ------------------------------------------------------------------
    # Canonical-CDS path (transcript_id-based; original API)
    # ------------------------------------------------------------------

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

        # Sort CDS regions in mRNA order
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

        if self._genome is None:
            import pysam

            self._genome = pysam.FastaFile(self._genome_path)

        pos_map = self.build_position_map(transcript_id)
        if not pos_map:
            return ""

        strand = self._strand_cache.get(transcript_id, "+")

        tx_cds = self._cds_df[
            (self._cds_df["transcript_id"] == transcript_id)
            & (self._cds_df["feature_type"] == "CDS")
        ]
        if tx_cds.empty:
            return ""
        chrom = tx_cds.iloc[0]["chromosome"]

        # Sort genomic positions by coding position (i.e. mRNA order)
        sorted_positions = sorted(pos_map.items(), key=lambda x: x[1])

        coding_bases: list[str] = []
        for gpos, _cpos in sorted_positions:
            base = self._genome.fetch(chrom, gpos - 1, gpos)
            if strand == "-":
                base = COMPLEMENT.get(base.upper(), base)
            coding_bases.append(base.upper())

        seq = "".join(coding_bases)
        self._coding_seq_cache[transcript_id] = seq
        return seq

    # ------------------------------------------------------------------
    # ORF-frame path (orf_exons-based; new)
    # ------------------------------------------------------------------

    @staticmethod
    def _orf_cache_key(orf_exons: list[tuple[int, int]], strand: str, chrom: str = "") -> tuple:
        return (tuple((int(s), int(e)) for s, e in orf_exons), strand, chrom)

    def build_position_map_from_orf(
        self,
        orf_exons: list[tuple[int, int]],
        strand: str,
        *,
        orf_key: Any = None,
    ) -> dict[int, int]:
        """Build 1-based genomic_pos → 0-based coding_pos map from ORF exons.

        Args:
            orf_exons: 0-based half-open plus-strand intervals in **ascending**
                genomic order (the format produced by
                :func:`swissisoform.coords.orf_exons_from_skeleton`).
            strand: ``"+"`` or ``"-"``. Determines mRNA-order traversal —
                for minus strand the intervals are walked in descending order
                and within each from high to low genomic position.
            orf_key: Optional hashable cache key; defaults to a tuple derived
                from the exons + strand.

        Returns:
            Dict mapping 1-based genomic positions (inclusive) to 0-based
            coding positions starting from 0 at the ORF's first base.
        """
        if not orf_exons:
            return {}
        key = orf_key if orf_key is not None else self._orf_cache_key(orf_exons, strand)
        if key in self._orf_position_map_cache:
            return self._orf_position_map_cache[key]

        intervals = list(orf_exons)
        if strand == "-":
            intervals = list(reversed(intervals))

        pos_map: dict[int, int] = {}
        coding_pos = 0
        for start, end in intervals:
            if strand == "+":
                for gpos in range(start + 1, end + 1):
                    pos_map[gpos] = coding_pos
                    coding_pos += 1
            else:
                for gpos in range(end, start, -1):
                    pos_map[gpos] = coding_pos
                    coding_pos += 1

        self._orf_position_map_cache[key] = pos_map
        return pos_map

    def build_coding_sequence_from_orf(
        self,
        orf_exons: list[tuple[int, int]],
        strand: str,
        chrom: str,
        *,
        orf_key: Any = None,
    ) -> str:
        """Extract coding sequence in mRNA order from genome using ORF exons.

        Args:
            orf_exons: 0-based half-open plus-strand intervals in ascending
                genomic order.
            strand: ``"+"`` or ``"-"``.
            chrom: Chromosome (e.g. ``"chr17"``) for FASTA lookup.
            orf_key: Optional hashable cache key.

        Returns:
            Coding sequence (mRNA-order); empty string if the genome FASTA
            wasn't configured or the exons are empty.
        """
        if not orf_exons or self._genome_path is None:
            return ""
        key = orf_key if orf_key is not None else self._orf_cache_key(orf_exons, strand, chrom)
        if key in self._orf_coding_seq_cache:
            return self._orf_coding_seq_cache[key]

        if self._genome is None:
            import pysam

            self._genome = pysam.FastaFile(self._genome_path)

        # Plus-strand bases in ascending genomic order; rev-comp at the end
        # for minus-strand to recover mRNA order.
        bases: list[str] = []
        for start, end in orf_exons:
            bases.append(self._genome.fetch(chrom, start, end).upper())
        plus_seq = "".join(bases)
        seq = _revcomp(plus_seq) if strand == "-" else plus_seq
        self._orf_coding_seq_cache[key] = seq
        return seq

    def validate_variant_against_orf(
        self,
        *,
        orf_exons: list[tuple[int, int]],
        strand: str,
        chrom: str,
        genomic_pos: int,
        ref: str,
        alt: str,
        orf_key: Any = None,
        context: str = "",
    ) -> dict[str, Any]:
        """Validate a variant in the reading frame of an arbitrary ORF.

        Mirror of :meth:`validate_variant` but using ``orf_exons`` directly
        instead of a transcript_id + canonical CDS lookup. Used to re-call
        each variant in the *isoform* reading frame for the isoform-unique
        region.
        """
        pos_map = self.build_position_map_from_orf(orf_exons, strand, orf_key=orf_key)
        coding_seq = self.build_coding_sequence_from_orf(
            orf_exons, strand, chrom, orf_key=orf_key
        )
        return self._analyze_variant(
            pos_map=pos_map,
            coding_seq=coding_seq,
            strand=strand,
            genomic_pos=genomic_pos,
            ref=ref,
            alt=alt,
            context=context,
        )

    # ------------------------------------------------------------------
    # Shared codon analysis
    # ------------------------------------------------------------------

    def _analyze_variant(
        self,
        *,
        pos_map: dict[int, int],
        coding_seq: str,
        strand: str,
        genomic_pos: int,
        ref: str,
        alt: str,
        context: str = "",
    ) -> dict[str, Any]:
        """Classify a variant given a precomputed position map + coding seq.

        Shared by the canonical-CDS (:meth:`validate_variant`) and ORF-frame
        (:meth:`validate_variant_against_orf`) entry points so consequence
        semantics are identical across frames.
        """
        if not pos_map:
            return _null_result(validated=False)

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

        # Indels
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

        # MNV — skip codon walk to avoid reference_mismatch spam.
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

        # SNV
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

        codon_start = (coding_pos // 3) * 3
        offset = coding_pos % 3

        if codon_start + 3 > len(coding_seq):
            logger.warning(
                "Codon extends beyond coding sequence for %s at gpos %d",
                context or "<unknown-orf>",
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

        if strand == "-":
            ref_base = COMPLEMENT.get(ref.upper(), ref.upper())
            alt_base = COMPLEMENT.get(alt.upper(), alt.upper())
        else:
            ref_base = ref.upper()
            alt_base = alt.upper()

        if codon_ref[offset] != ref_base:
            logger.warning(
                "Reference mismatch at %s gpos=%d — expected %s at offset %d of codon %s, got %s",
                context or "<unknown-orf>",
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

        codon_alt = codon_ref[:offset] + alt_base + codon_ref[offset + 1 :]
        aa_ref = str(Seq(codon_ref).translate())
        aa_alt = str(Seq(codon_alt).translate())

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

    # ------------------------------------------------------------------
    # Thin canonical-CDS variant entry point (wraps _analyze_variant)
    # ------------------------------------------------------------------

    def validate_variant(
        self,
        transcript_id: str,
        genomic_pos: int,
        ref: str,
        alt: str,
    ) -> dict[str, Any]:
        """Validate a single variant's consequence in the canonical CDS frame."""
        pos_map = self.build_position_map(transcript_id)
        strand = self._strand_cache.get(transcript_id, "+")
        coding_seq = (
            self._coding_seq_cache.get(transcript_id)
            if transcript_id in self._coding_seq_cache
            else self.build_coding_sequence(transcript_id)
        )
        return self._analyze_variant(
            pos_map=pos_map,
            coding_seq=coding_seq or "",
            strand=strand,
            genomic_pos=genomic_pos,
            ref=ref,
            alt=alt,
            context=transcript_id,
        )

    def validate_variants(
        self,
        variants: list[dict[str, Any]],
        transcript_id: str,
    ) -> list[dict[str, Any]]:
        """Canonical-CDS batch validation; updates each variant dict in place.

        For each variant dict that contains genomic_pos, ref, and alt, runs
        codon-level validation and overwrites the consequence + canonical-frame
        ``protein_pos`` / ``aa_ref`` / ``aa_alt`` if validation succeeds.
        Variants whose position lies outside the canonical CDS keep their
        existing ``protein_pos`` (typically ``None``) — the isoform-frame
        re-pass (see :meth:`validate_variants_against_orf`) handles those.
        """
        for variant in variants:
            gpos = variant.get("genomic_pos")
            ref = variant.get("ref")
            alt = variant.get("alt")

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

    def validate_variants_against_orf(
        self,
        variants: list[dict[str, Any]],
        *,
        orf_exons: list[tuple[int, int]],
        strand: str,
        chrom: str,
        orf_key: Any = None,
        context: str = "",
        field_prefix: str | None = "isoform",
    ) -> list[dict[str, Any]]:
        """Re-call a batch of variants in an ORF reading frame.

        Two write modes:

        - ``field_prefix="isoform"`` (default) — writes
          ``isoform_protein_pos`` / ``isoform_aa_ref`` / ``isoform_aa_alt`` /
          ``isoform_consequence`` on each variant; the canonical-frame fields
          (``protein_pos`` / ``aa_ref`` / ``aa_alt`` / ``consequence``, set
          earlier by ``validate_variants`` against the canonical transcript)
          are untouched. Every variant is recorded — including
          ``intronic`` / ``reference_mismatch`` — so downstream code can
          distinguish "not in this ORF" from "ORF not run".

        - ``field_prefix=None`` — *overwrites* the canonical-frame fields
          (``protein_pos`` / ``aa_ref`` / ``aa_alt`` / ``consequence``) with
          the new values whenever validation against this ORF succeeds and
          produced a real position. Used by the per-TIS canonical-frame
          revalidation in :class:`VariantIntersectionModule` so that
          ``protein_pos`` reflects the **per-Tid** canonical CDS frame
          (matching the protein cached in ``site.canonical_protein``)
          instead of the gene-level transcript that clinical fetched
          against. For the FZR1 case (gene-level 496 aa vs per-Tid 493 aa)
          this is what eliminates the 3-aa frame offset that the
          ``varianteffect`` aa-ref guard catches as ``aa_ref_mismatch``.
          Variants that don't map into this ORF (intronic / outside its
          exons / reference_mismatch) leave the existing canonical fields
          alone.

        Any other string prefix writes ``<prefix>_*`` fields, mirroring the
        isoform mode.
        """
        for variant in variants:
            gpos = variant.get("genomic_pos")
            ref = variant.get("ref")
            alt = variant.get("alt")
            if not gpos or ref is None or alt is None or (ref == "" and alt == ""):
                continue
            result = self.validate_variant_against_orf(
                orf_exons=orf_exons,
                strand=strand,
                chrom=chrom,
                genomic_pos=gpos,
                ref=ref,
                alt=alt,
                orf_key=orf_key,
                context=context,
            )
            if field_prefix is None:
                # Overwrite canonical-frame fields when the per-Tid revalidation
                # actually placed the variant in this ORF.
                if result["validated"] and result["protein_pos"] is not None:
                    variant["consequence"] = result["consequence"]
                    variant["protein_pos"] = result["protein_pos"]
                    if result["aa_ref"] is not None:
                        variant["aa_ref"] = result["aa_ref"]
                    if result["aa_alt"] is not None:
                        variant["aa_alt"] = result["aa_alt"]
            else:
                variant[f"{field_prefix}_consequence"] = result["consequence"]
                variant[f"{field_prefix}_protein_pos"] = result["protein_pos"]
                variant[f"{field_prefix}_aa_ref"] = result["aa_ref"]
                variant[f"{field_prefix}_aa_alt"] = result["aa_alt"]
        return variants


def _null_result(validated: bool) -> dict[str, Any]:
    return {
        "consequence": None,
        "protein_pos": None,
        "aa_ref": None,
        "aa_alt": None,
        "codon_ref": None,
        "codon_alt": None,
        "validated": validated,
    }
