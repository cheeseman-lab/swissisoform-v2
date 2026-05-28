"""AlphaMissense hg38 lookup — per-variant calibrated missense pathogenicity.

Random-access lookup against the tabix-indexed AlphaMissense hg38 table
(``AlphaMissense_hg38.tsv.gz`` + ``.tbi``, built by
``scripts/setup/setup_databases.py alphamissense``). One bgzf block read
per query, so this is cheap enough to call per clinical variant inline.

Table columns (tab-separated, ``#``-prefixed header)::

    CHROM POS REF ALT genome uniprot_id transcript_id protein_variant \
        am_pathogenicity am_class

``am_pathogenicity`` is a 0-1 score; ``am_class`` ∈ {likely_benign,
ambiguous, likely_pathogenic}. AlphaMissense only scores missense SNVs, so
indels / non-SNV alleles return ``None`` by construction (no matching row).

Licence: CC BY-NC-SA 4.0 (DeepMind) — research use.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Pathogenicity class AlphaMissense assigns above its calibrated threshold.
PATHOGENIC_CLASS = "likely_pathogenic"


def _strip_version(transcript_id: str | None) -> str | None:
    """Drop the ``.N`` version suffix from an Ensembl transcript id."""
    if not transcript_id:
        return None
    return transcript_id.split(".", 1)[0]


class AlphaMissenseLookup:
    """Tabix-backed point lookup into the AlphaMissense hg38 table.

    The :class:`pysam.TabixFile` handle is opened lazily on first query and
    reused, so a single instance can be shared across a run. Construct with
    the path to ``AlphaMissense_hg38.tsv.gz`` (its ``.tbi`` must sit
    alongside).
    """

    def __init__(self, db_path: Path | str) -> None:
        """Store the table path; the tabix handle opens on first lookup.

        Args:
            db_path: Path to the bgzipped, tabix-indexed AlphaMissense hg38
                table.
        """
        self.db_path = Path(db_path)
        self._tbx: Any | None = None
        self._open_failed = False

    @property
    def available(self) -> bool:
        """True when the table file and its tabix index both exist."""
        return self.db_path.exists() and self.db_path.with_suffix(
            self.db_path.suffix + ".tbi"
        ).exists()

    def _handle(self) -> Any | None:
        if self._tbx is not None or self._open_failed:
            return self._tbx
        try:
            import pysam

            self._tbx = pysam.TabixFile(str(self.db_path))
        except Exception as exc:  # noqa: BLE001 — degrade gracefully if index/file absent
            logger.warning("AlphaMissenseLookup: could not open %s: %s", self.db_path, exc)
            self._open_failed = True
            self._tbx = None
        return self._tbx

    def lookup(
        self,
        chrom: str,
        pos: int,
        ref: str,
        alt: str,
        *,
        transcript_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Return the AlphaMissense record for a genomic SNV, or ``None``.

        Args:
            chrom: Chromosome in UCSC style (``"chr17"``).
            pos: 1-based genomic position.
            ref: Reference allele (single base for a scorable SNV).
            alt: Alternate allele (single base).
            transcript_id: Optional Ensembl transcript id. When the position
                has rows for several transcripts, the row whose (versionless)
                transcript matches is preferred; otherwise the first matching
                allele row is returned.

        Returns:
            ``{"am_pathogenicity": float, "am_class": str, "protein_variant":
            str, "transcript_id": str, "uniprot_id": str}`` or ``None`` when
            the table is unavailable or no row matches the allele.
        """
        if not chrom or not isinstance(pos, int) or pos <= 0:
            return None
        # AlphaMissense only scores single-base substitutions.
        if not ref or not alt or len(ref) != 1 or len(alt) != 1:
            return None

        tbx = self._handle()
        if tbx is None:
            return None

        ref_u = ref.upper()
        alt_u = alt.upper()
        want_tx = _strip_version(transcript_id)

        try:
            rows = list(tbx.fetch(chrom, pos - 1, pos))
        except (ValueError, OSError):
            # Chromosome absent from the index (e.g. alt contigs) → no score.
            return None

        first_match: dict[str, Any] | None = None
        for line in rows:
            fields = line.split("\t")
            if len(fields) < 10:
                continue
            r_pos, r_ref, r_alt = fields[1], fields[2], fields[3]
            if r_pos != str(pos) or r_ref.upper() != ref_u or r_alt.upper() != alt_u:
                continue
            try:
                am_path = float(fields[8])
            except ValueError:
                continue
            record = {
                "am_pathogenicity": am_path,
                "am_class": fields[9],
                "protein_variant": fields[7],
                "transcript_id": fields[6],
                "uniprot_id": fields[5],
            }
            if want_tx is not None and _strip_version(fields[6]) == want_tx:
                return record
            if first_match is None:
                first_match = record
        return first_match
