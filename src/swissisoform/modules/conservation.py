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
    "qseqid",
    "sseqid",
    "pident",
    "length",
    "mismatch",
    "gapopen",
    "qstart",
    "qend",
    "sstart",
    "send",
    "evalue",
    "bitscore",
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
    def _empty_result(status: str = "no_hits", tool: str | None = None) -> dict[str, Any]:
        """Return an empty annotation distinguishing "didn't run" from "no hits".

        Args:
            status: ``"no_hits"`` (ran, found no homologs → score 0) or
                ``"not_run"`` (tool/db/config missing → score None).
            tool: Optional tool name used (for diagnostic purposes).

        Returns:
            Dict with ``"hits"`` (empty list) and ``"summary"`` with the
            appropriate semantic for the failure mode.
        """
        if status == "not_run":
            return {
                "hits": [],
                "summary": {
                    "conservation_score": None,  # couldn't compute
                    "conservation_label": None,
                    "best_pident": None,
                    "best_evalue": None,
                    "n_hits": None,
                    "tool_used": tool,
                    "status": "not_run",
                },
            }
        # Legit "ran but found nothing" → score 0
        return {
            "hits": [],
            "summary": {
                "conservation_score": 0,
                "conservation_label": CONSERVATION_LABELS[0],
                "best_pident": None,
                "best_evalue": None,
                "n_hits": 0,
                "tool_used": tool,
                "status": "no_hits",
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
                    hits.append(
                        {
                            "subject_id": parts[1],
                            "pident": float(parts[2]),
                            "length": int(parts[3]),
                            "evalue": float(parts[10]),
                            "bitscore": float(parts[11]),
                            "qstart": int(parts[6]) - 1,  # convert to 0-indexed
                            "qend": int(parts[7]) - 1,
                            "sstart": int(parts[8]),
                            "send": int(parts[9]),
                        }
                    )
                except (ValueError, IndexError):
                    continue
        return hits

    def _build_search_cmd(
        self,
        query_fasta: str,
        db_path: str,
        output_file: str,
        tool: str,
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
            # Default mode (not --sensitive) — ~10x faster and still
            # appropriate for per-protein conservation lookup against a
            # reviewed reference like SwissProt.
            return [
                "diamond",
                "blastp",
                "-q",
                query_fasta,
                "-d",
                db_path,
                "-o",
                output_file,
                "--outfmt",
                "6",
                "qseqid",
                "sseqid",
                "pident",
                "length",
                "mismatch",
                "gapopen",
                "qstart",
                "qend",
                "sstart",
                "send",
                "evalue",
                "bitscore",
                "--max-target-seqs",
                "10",
                "--quiet",
            ]
        if tool == "blastp":
            return [
                "blastp",
                "-query",
                query_fasta,
                "-db",
                db_path,
                "-outfmt",
                "6 qseqid sseqid pident length mismatch gapopen"
                " qstart qend sstart send evalue bitscore",
                "-max_target_seqs",
                "10",
                "-out",
                output_file,
            ]
        # mmseqs
        tmp_dir = os.path.join(os.path.dirname(output_file), "mmseqs_tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        return [
            "mmseqs",
            "easy-search",
            query_fasta,
            db_path,
            output_file,
            tmp_dir,
            "--format-output",
            "query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits",
            "-s",
            "7.5",
        ]

    def _run_search(
        self,
        protein: str,
        tool: str,
        db_path: str,
    ) -> list[dict[str, Any]] | None:
        """Run a sequence search subprocess and parse results.

        Creates a temporary directory (in the current working directory per HPC
        rules), writes the query FASTA, executes the search, parses output,
        and cleans up.

        Args:
            protein: Amino acid sequence to search.
            tool: Search tool name.
            db_path: Path to the reference database.

        Returns:
            List of hit dicts on successful run (may be empty).  Returns
            ``None`` if the subprocess failed — caller should treat this as
            "could not run" rather than "no homologs found".
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
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if result.returncode != 0:
                    logger.warning(
                        "%s search failed (exit %d): %s",
                        tool,
                        result.returncode,
                        result.stderr[:200],
                    )
                    return None  # couldn't run — don't claim "no hits"
            except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
                logger.warning("%s search error: %s", tool, exc)
                return None  # couldn't run — don't claim "no hits"

            return self._parse_blast_tabular(output_file)
        finally:
            # Clean up temp directory
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Public API (ProteinModule protocol)
    # ------------------------------------------------------------------

    def annotate(self, protein: str) -> dict[str, Any]:
        """Annotate a single protein sequence with conservation data.

        Distinguishes two "empty" outcomes:
        - ``status="no_hits"``: search ran successfully but found no homologs.
          ``conservation_score`` is ``0``.
        - ``status="not_run"``: could not run (no tool, no db, no config,
          subprocess failed). ``conservation_score`` is ``None``.

        Args:
            protein: Amino acid sequence (may include trailing ``'*'``).

        Returns:
            Dict with ``"hits"`` (list of alignment hit dicts) and
            ``"summary"`` (conservation score, label, best metrics, status).
        """
        # Empty protein — treat as "no_hits" (nothing to search, legit empty)
        if not protein or not protein.rstrip("*"):
            return self._empty_result(status="no_hits", tool=self._tool)

        # Pre-computed results — strip stop codon for stable lookup key
        stripped = protein.rstrip("*")
        if stripped in self._precomputed:
            return self._precomputed[stripped]
        if protein in self._precomputed:  # backward compat for raw keys
            return self._precomputed[protein]

        # No config or no db path → can't run
        if not self._config.conservation or not self._db_path:
            return self._empty_result(status="not_run")

        # No tool available → can't run
        if not self._tool:
            return self._empty_result(status="not_run")

        # Check db file exists → can't run
        if not self._db_path.exists():
            logger.warning("Database not found: %s", self._db_path)
            return self._empty_result(status="not_run", tool=self._tool)

        # Run search — returns None if subprocess failed, [] if ran with no hits
        hits = self._run_search(protein, self._tool, str(self._db_path))

        if hits is None:
            # subprocess error → can't claim "no conservation"
            return self._empty_result(status="not_run", tool=self._tool)

        if not hits:
            # ran successfully, no homologs → legit score 0
            return self._empty_result(status="no_hits", tool=self._tool)

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
                "status": "ok",
            },
        }

    def precompute_batch(self, proteins: list[str] | set[str]) -> int:
        """Run ONE DIAMOND call over a batch of proteins, populate cache.

        ``annotate(protein)`` spawns a fresh DIAMOND subprocess per call,
        which reloads the ~300 MB SwissProt DB every time — wildly slow
        for pipelines with dozens or hundreds of unique proteins.  This
        helper runs a single DIAMOND call on the whole batch, parses per-
        query hits, and populates ``self._precomputed`` keyed by the exact
        protein sequence.  Subsequent ``annotate()`` calls hit the cache.

        Args:
            proteins: Unique protein sequences to search.  Duplicates are
                deduped; stop codons stripped.

        Returns:
            Number of unique proteins inserted into the precomputed cache
            (both hits and no-hits).  Returns 0 if the module isn't
            configured to run (no tool / no DB).
        """
        if not self._config.conservation or not self._db_path or not self._tool:
            logger.warning(
                "ConservationModule.precompute_batch: not configured to run "
                "(tool=%r db=%r) — skipping",
                self._tool,
                self._db_path,
            )
            return 0
        if not self._db_path.exists():
            logger.warning("Database not found: %s", self._db_path)
            return 0

        # Dedup + strip stop codons
        unique_seqs: list[str] = []
        seen: set[str] = set()
        for raw in proteins:
            seq = (raw or "").rstrip("*")
            if seq and seq not in seen:
                seen.add(seq)
                unique_seqs.append(seq)
        if not unique_seqs:
            return 0

        qid_to_seq = {f"q{i}": s for i, s in enumerate(unique_seqs)}
        tmp_dir = tempfile.mkdtemp(dir=".")
        per_query: dict[str, list[dict[str, Any]]] = {qid: [] for qid in qid_to_seq}
        try:
            query_fasta = os.path.join(tmp_dir, "query.fasta")
            with open(query_fasta, "w") as fh:
                for qid, seq in qid_to_seq.items():
                    fh.write(f">{qid}\n{seq}\n")
            output_file = os.path.join(tmp_dir, "results.tsv")
            cmd = self._build_search_cmd(query_fasta, str(self._db_path), output_file, self._tool)
            logger.info(
                "ConservationModule.precompute_batch: %d unique proteins via %s",
                len(qid_to_seq),
                self._tool,
            )
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
            )
            if result.returncode != 0:
                logger.error(
                    "diamond batch failed (exit %d): %s",
                    result.returncode,
                    result.stderr[:400],
                )
                return 0

            # qseqid-aware parse (BLAST tabular format 6, first column = qseqid)
            if os.path.exists(output_file):
                with open(output_file) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = line.split("\t")
                        if len(parts) < 12:
                            continue
                        try:
                            qseqid = parts[0]
                            hit = {
                                "subject_id": parts[1],
                                "pident": float(parts[2]),
                                "length": int(parts[3]),
                                "evalue": float(parts[10]),
                                "bitscore": float(parts[11]),
                                "qstart": int(parts[6]) - 1,
                                "qend": int(parts[7]) - 1,
                                "sstart": int(parts[8]),
                                "send": int(parts[9]),
                            }
                        except (ValueError, IndexError):
                            continue
                        if qseqid in per_query:
                            per_query[qseqid].append(hit)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        # Populate self._precomputed, keyed on the protein sequence
        for qid, seq in qid_to_seq.items():
            hits = per_query.get(qid, [])
            if hits:
                best = max(hits, key=lambda h: h["bitscore"])
                score = self._score_pident(best["pident"])
                self._precomputed[seq] = {
                    "hits": hits,
                    "summary": {
                        "conservation_score": score,
                        "conservation_label": CONSERVATION_LABELS.get(score, "Unknown"),
                        "best_pident": best["pident"],
                        "best_evalue": best["evalue"],
                        "n_hits": len(hits),
                        "tool_used": self._tool,
                        "status": "ok",
                    },
                }
            else:
                # Ran successfully, no homologs → legit score 0
                self._precomputed[seq] = self._empty_result(status="no_hits", tool=self._tool)
        logger.info(
            "ConservationModule.precompute_batch: cached %d proteins",
            len(qid_to_seq),
        )
        return len(qid_to_seq)

    def run(
        self,
        tis_sites: list[TranslationInitiationSite],
    ) -> list[TranslationInitiationSite]:
        """Annotate all TIS sites with conservation data.

        Thin wrapper that calls ``annotate()`` for each site's isoform protein
        and stores the result in ``site.isoform_annotations[MODULE_NAME]``.
        """
        for site in tis_sites:
            site.isoform_annotations[self.MODULE_NAME] = self.annotate(
                site.isoform_protein,
            )
        return tis_sites
