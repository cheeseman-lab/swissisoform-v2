"""Module: Conservation — evolutionary conservation via sequence similarity search.

Runs DIAMOND, blastp, or MMseqs2 against a protein database. Auto-detects
which tool is available. Returns alignment hits with positional data.

Implements the ``ProteinModule`` protocol: ``annotate(protein)`` does the work.
Can run inline searches or return pre-computed results for batch runs.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from swissisoform.config import PipelineConfig
from swissisoform.models import TranslationInitiationSite

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONSERVATION_LABELS: dict[int, str] = {
    0: "No homology",
    1: "Weak homology",
    2: "Partial homology",
    3: "Stop codon interrupting homology",
    4: "Moderate-low conservation",
    5: "Moderate conservation",
    6: "Good conservation",
    7: "Strong conservation",
    8: "Very strong conservation",
    9: "Near-perfect conservation",
}

BLAST_TABULAR_COLUMNS = [
    "qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
    "qstart", "qend", "sstart", "send", "evalue", "bitscore",
]

# Tool preference order
TOOL_PREFERENCE = ["diamond", "mmseqs", "blastp"]


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class ConservationModule:
    """Annotates proteins with evolutionary conservation via sequence search.

    Implements the ``ProteinModule`` protocol. ``annotate(protein)`` returns
    alignment hits and a conservation summary. Can operate in three modes:

    1. **Pre-computed**: If ``precomputed`` dict is provided, results are
       returned directly without running any search.
    2. **Inline search**: If a database path and search tool are available,
       runs DIAMOND/blastp/MMseqs2 as a subprocess.
    3. **No-op**: If no database or tool is configured, returns empty results
       with ``conservation_score=0``.

    Attributes:
        MODULE_NAME: ``"conservation"``
        OUTPUT_COLUMNS: Two prefixed column names.
        SCOPE: ``"C"`` (per-candidate).
    """

    MODULE_NAME: str = "conservation"
    OUTPUT_COLUMNS: list[str] = [
        "conservation_hits",
        "conservation_summary",
    ]
    SCOPE: str = "C"

    def __init__(
        self,
        config: PipelineConfig,
        precomputed: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize with pipeline configuration.

        Args:
            config: Pipeline config. Uses ``config.conservation`` for db path
                and tool settings.
            precomputed: Optional pre-loaded results. Dict mapping protein
                sequence (or hash/identifier) to annotation dict. If provided,
                ``annotate()`` returns from this cache.
        """
        self._config = config
        self._precomputed = precomputed or {}
        self._tool: str | None = None
        self._db_path: Path | None = None

        # Resolve db path and tool from config
        if config.conservation and config.conservation.diamond_db:
            self._db_path = config.conservation.diamond_db
        self._tool = self._detect_tool()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_tool() -> str | None:
        """Auto-detect the best available sequence search tool.

        Returns:
            Tool name (``"diamond"``, ``"mmseqs"``, or ``"blastp"``), or
            ``None`` if no tool is found.
        """
        for tool in TOOL_PREFERENCE:
            if shutil.which(tool):
                return tool
        return None

    @staticmethod
    def _score_pident(pident: float) -> int:
        """Map percent identity to a 0-9 conservation score.

        Args:
            pident: Best percent identity from search results.

        Returns:
            Integer score from 0 to 9.
        """
        if pident >= 95:
            return 9
        if pident >= 85:
            return 8
        if pident >= 75:
            return 7
        if pident >= 65:
            return 6
        if pident >= 55:
            return 5
        if pident >= 45:
            return 3
        if pident >= 40:
            return 2
        if pident >= 30:
            return 1
        return 0

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        """Return an empty annotation with score 0 and no hits.

        Returns:
            Dict with ``"hits"`` (empty list) and ``"summary"`` (score 0).
        """
        return {
            "hits": [],
            "summary": {
                "conservation_score": 0,
                "conservation_label": CONSERVATION_LABELS[0],
                "best_pident": None,
                "best_evalue": None,
                "n_hits": 0,
                "tool_used": None,
            },
        }

    @staticmethod
    def _parse_blast_tabular(output_file: str) -> list[dict[str, Any]]:
        """Parse BLAST tabular format 6 output into a list of hit dicts.

        Args:
            output_file: Path to the tab-separated output file.

        Returns:
            List of hit dicts with keys: ``subject_id``, ``pident``,
            ``length``, ``evalue``, ``bitscore``, ``qstart``, ``qend``,
            ``sstart``, ``send``.
        """
        hits: list[dict[str, Any]] = []
        if not os.path.exists(output_file):
            return hits
        with open(output_file) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 12:
                    continue
                try:
                    hits.append({
                        "subject_id": parts[1],
                        "pident": float(parts[2]),
                        "length": int(parts[3]),
                        "evalue": float(parts[10]),
                        "bitscore": float(parts[11]),
                        "qstart": int(parts[6]) - 1,  # convert to 0-indexed
                        "qend": int(parts[7]) - 1,
                        "sstart": int(parts[8]),
                        "send": int(parts[9]),
                    })
                except (ValueError, IndexError):
                    continue
        return hits

    def _build_search_cmd(
        self, query_fasta: str, db_path: str, output_file: str, tool: str,
    ) -> list[str]:
        """Build the subprocess command for the given tool.

        Args:
            query_fasta: Path to query FASTA file.
            db_path: Path to the reference database.
            output_file: Path for tabular output.
            tool: One of ``"diamond"``, ``"blastp"``, ``"mmseqs"``.

        Returns:
            Command list suitable for ``subprocess.run``.
        """
        if tool == "diamond":
            return [
                "diamond", "blastp",
                "-q", query_fasta,
                "-d", db_path,
                "-o", output_file,
                "--outfmt", "6", "qseqid", "sseqid", "pident", "length",
                "mismatch", "gapopen", "qstart", "qend", "sstart", "send",
                "evalue", "bitscore",
                "--sensitive",
            ]
        if tool == "blastp":
            return [
                "blastp",
                "-query", query_fasta,
                "-db", db_path,
                "-outfmt", "6 qseqid sseqid pident length mismatch gapopen"
                " qstart qend sstart send evalue bitscore",
                "-max_target_seqs", "10",
                "-out", output_file,
            ]
        # mmseqs
        tmp_dir = os.path.join(os.path.dirname(output_file), "mmseqs_tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        return [
            "mmseqs", "easy-search",
            query_fasta, db_path, output_file, tmp_dir,
            "--format-output", "query,target,pident,alnlen,mismatch,gapopen,"
            "qstart,qend,tstart,tend,evalue,bits",
            "-s", "7.5",
        ]

    def _run_search(
        self, protein: str, tool: str, db_path: str,
    ) -> list[dict[str, Any]]:
        """Run a sequence search subprocess and parse results.

        Creates a temporary directory (in the current working directory per HPC
        rules), writes the query FASTA, executes the search, parses output,
        and cleans up.

        Args:
            protein: Amino acid sequence to search.
            tool: Search tool name.
            db_path: Path to the reference database.

        Returns:
            List of hit dicts from ``_parse_blast_tabular``.
        """
        tmp_dir = tempfile.mkdtemp(dir=".")
        try:
            # Write query FASTA
            query_fasta = os.path.join(tmp_dir, "query.fasta")
            with open(query_fasta, "w") as fh:
                fh.write(">query\n")
                fh.write(protein.rstrip("*") + "\n")

            output_file = os.path.join(tmp_dir, "results.tsv")
            cmd = self._build_search_cmd(query_fasta, db_path, output_file, tool)

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=600,
                )
                if result.returncode != 0:
                    logger.warning(
                        "%s search failed (exit %d): %s",
                        tool, result.returncode, result.stderr[:200],
                    )
                    return []
            except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
                logger.warning("%s search error: %s", tool, exc)
                return []

            return self._parse_blast_tabular(output_file)
        finally:
            # Clean up temp directory
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Public API (ProteinModule protocol)
    # ------------------------------------------------------------------

    def annotate(self, protein: str) -> dict[str, Any]:
        """Annotate a single protein sequence with conservation data.

        Args:
            protein: Amino acid sequence (may include trailing ``'*'``).

        Returns:
            Dict with ``"hits"`` (list of alignment hit dicts) and
            ``"summary"`` (conservation score, label, best metrics).
        """
        # Empty protein
        if not protein or not protein.rstrip("*"):
            return self._empty_result()

        # Pre-computed results
        if protein in self._precomputed:
            return self._precomputed[protein]

        # No config or no db path
        if not self._config.conservation or not self._db_path:
            return self._empty_result()

        # No tool available
        if not self._tool:
            return self._empty_result()

        # Check db file exists
        if not self._db_path.exists():
            logger.warning("Database not found: %s", self._db_path)
            return self._empty_result()

        # Run search
        hits = self._run_search(protein, self._tool, str(self._db_path))

        if not hits:
            return self._empty_result()

        # Score from best hit (highest bitscore)
        best_hit = max(hits, key=lambda h: h["bitscore"])
        score = self._score_pident(best_hit["pident"])

        return {
            "hits": hits,
            "summary": {
                "conservation_score": score,
                "conservation_label": CONSERVATION_LABELS.get(score, "Unknown"),
                "best_pident": best_hit["pident"],
                "best_evalue": best_hit["evalue"],
                "n_hits": len(hits),
                "tool_used": self._tool,
            },
        }

    def run(
        self, tis_sites: list[TranslationInitiationSite],
    ) -> list[TranslationInitiationSite]:
        """Annotate all TIS sites with conservation data.

        Thin wrapper that calls ``annotate()`` for each site's isoform protein
        and stores the result in ``site.isoform_annotations[MODULE_NAME]``.

        Args:
            tis_sites: Input sites to annotate.

        Returns:
            The same sites with ``isoform_annotations["conservation"]``
            populated.
        """
        for site in tis_sites:
            site.isoform_annotations[self.MODULE_NAME] = self.annotate(
                site.isoform_protein,
            )
        return tis_sites
