"""Module: InterProScan — domain annotation consumer.

Attaches InterProScan domain hits to proteins. Predictions are
precomputed in batch via :func:`precompute_interproscan` (subprocess
into a Singularity image), then hash-keyed by protein sequence. The
module itself only performs lookups — identical pattern to
``modules/localization.py`` and ``modules/signalp.py``.

InterProScan aggregates hits from multiple member databases (Pfam,
SMART, PROSITE, CDD, ...). Each hit carries a signature accession
(e.g. ``PF12345``), the member-database name, genomic coordinates on
the protein, and optionally an InterPro cross-reference (``IPR001234``)
with description.

InterProScan 6 is a Nextflow pipeline that orchestrates member-DB
containers (HMMER, Pfam, SMART, ...) under a single entry point.
Databases auto-download on first run into a shared ``--datadir`` that
we pre-warm via ``scripts/setup_databases.py interproscan``.  The
module gracefully no-ops when Nextflow / datadir aren't set up.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from swissisoform.config import PipelineConfig
from swissisoform.models import TranslationInitiationSite

logger = logging.getLogger(__name__)


def _protein_hash(protein: str) -> str:
    """Stable hash of a protein sequence (stop codon stripped, uppercased)."""
    seq = protein.rstrip("*").upper()
    return hashlib.sha1(seq.encode("ascii"), usedforsecurity=False).hexdigest()


# Default install location used by scripts/setup_databases.py interproscan.
DEFAULT_INTERPROSCAN_DIR = (
    Path(__file__).resolve().parents[3] / "data" / "reference" / "interproscan"
)
# Comma-separated set of member-DB applications.  When ``None``, IPS6
# uses its default non-ML app set (Pfam, SMART, CDD, PANTHER, etc.).
# Pfam-only runs trip a COMBINE_MATCHES bug in IPS6 6.0.0 when any
# sequence has no hit, so the multi-app default is safer.
INTERPROSCAN_APPLICATIONS: str | None = None
INTERPROSCAN_VERSION = "6.0.0"
INTERPROSCAN_DATA_VERSION = "108.0"
INTERPROSCAN_NF_REPO = "ebi-pf-team/interproscan6"


def precompute_interproscan(
    proteins: dict[str, str] | list[str],
    *,
    applications: str | None = INTERPROSCAN_APPLICATIONS,
    install_dir: Path | None = None,
    nf_version: str = INTERPROSCAN_VERSION,
    data_version: str = INTERPROSCAN_DATA_VERSION,
    profile: str = "singularity",
) -> dict[str, dict[str, Any]]:
    """Run InterProScan 6 on a batch of unique proteins via Nextflow.

    Args:
        proteins: Either ``{label: sequence}`` or a list of sequences.
            Labels are discarded; output is keyed on the sha1 of the
            stop-stripped, uppercased sequence so duplicates dedupe.
        applications: Comma-separated InterProScan applications to run
            (e.g. ``"Pfam,SMART,CDD"``).  Default: ``"Pfam"``.
        install_dir: Directory that holds the shared Nextflow ``datadir``
            with downloaded DBs (default:
            ``data/reference/interproscan/datadir``).
        nf_version: Nextflow revision / release tag to pin (``-r``).
        data_version: InterPro DB version to pin (``--interpro``).
        profile: Nextflow profile (``singularity`` on HPC).

    Returns:
        ``{protein_hash: {"hits": [...], "summary": {...}}}``. Empty dict
        (with WARN) when Nextflow or the datadir aren't set up, or when
        the pipeline subprocess fails.
    """
    import shutil as _shutil
    import subprocess
    import tempfile
    from pathlib import Path as _P

    if isinstance(proteins, list):
        proteins = {f"seq_{i}": s for i, s in enumerate(proteins)}

    if _shutil.which("nextflow") is None:
        logger.warning(
            "precompute_interproscan: 'nextflow' not on PATH — returning empty."
        )
        return {}

    root = install_dir or DEFAULT_INTERPROSCAN_DIR
    datadir = root / "datadir"
    if not datadir.exists():
        logger.warning(
            "precompute_interproscan: datadir missing at %s — run "
            "`python scripts/setup_databases.py interproscan` first.  Returning empty.",
            datadir,
        )
        return {}

    # Dedup inputs by sequence hash
    hash_to_seq: dict[str, str] = {}
    for _label, seq in proteins.items():
        if not seq:
            continue
        h = _protein_hash(seq)
        hash_to_seq.setdefault(h, seq.rstrip("*").upper().replace("*", ""))

    if not hash_to_seq:
        return {}

    logger.info(
        "precompute_interproscan: %d inputs → %d unique sequences (applications=%s)",
        len(proteins),
        len(hash_to_seq),
        applications,
    )

    tmpdir = _P(tempfile.mkdtemp(prefix="interproscan_", dir=".")).resolve()
    fasta = tmpdir / "input.fa"
    outdir = tmpdir / "out"
    outdir.mkdir()
    with open(fasta, "w") as fh:
        for h, seq in hash_to_seq.items():
            fh.write(f">{h}\n{seq}\n")

    cmd = [
        "nextflow", "run", INTERPROSCAN_NF_REPO,
        "-r", nf_version,
        "-resume",  # reuse cached DB prep / lookup tasks across calls
        "-profile", profile,
        "--datadir", str(datadir),
        "--interpro", data_version,
        "--input", str(fasta),
        "--outdir", str(outdir),
        # Route through the container-based COMBINE_MATCHES path; the
        # LOCAL variant trips a Groovy classpath issue under Nextflow 25.x.
        "--batchSize", "50000",
    ]
    if applications:
        cmd.extend(["--applications", applications])
    logger.info("precompute_interproscan: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=tmpdir)
        if proc.stdout:
            logger.info("precompute_interproscan stdout (tail):\n%s", proc.stdout[-800:])
    except subprocess.CalledProcessError as exc:
        logger.error(
            "precompute_interproscan: nextflow run failed (exit %d). stderr=%s",
            exc.returncode,
            exc.stderr[-800:] if exc.stderr else "",
        )
        return {}

    # v6's Nextflow pipeline writes TSV under ``<outdir>/<basename>.tsv``.
    tsv_candidates = sorted(outdir.rglob("*.tsv"))
    if not tsv_candidates:
        logger.error("precompute_interproscan: no .tsv output under %s", outdir)
        return {}
    out_tsv = tsv_candidates[0]

    # TSV columns:
    # 0  protein_accession
    # 1  md5
    # 2  length
    # 3  analysis (member DB)
    # 4  signature_accession
    # 5  signature_description
    # 6  start  (1-based, inclusive)
    # 7  end    (1-based, inclusive)
    # 8  score (e-value; may be '-')
    # 9  status
    # 10 date
    # 11 interpro_accession  (may be '-')
    # 12 interpro_description (may be '-')
    per_protein: dict[str, list[dict[str, Any]]] = {}
    with open(out_tsv) as fh:
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 11:
                continue
            key = fields[0]
            try:
                start_1based = int(fields[6])
                end_1based = int(fields[7])
            except ValueError:
                continue
            score_raw = fields[8]
            try:
                score: float | None = float(score_raw)
            except ValueError:
                score = None
            interpro_id = fields[11] if len(fields) > 11 and fields[11] not in ("", "-") else None
            interpro_desc = (
                fields[12] if len(fields) > 12 and fields[12] not in ("", "-") else None
            )
            per_protein.setdefault(key, []).append(
                {
                    "name": fields[4],
                    "pos": start_1based - 1,  # convert to 0-based for comparator consistency
                    "end": end_1based,
                    "db": fields[3],
                    "description": fields[5] or None,
                    "score": score,
                    "interpro_id": interpro_id,
                    "interpro_description": interpro_desc,
                }
            )

    result: dict[str, dict[str, Any]] = {}
    for h in hash_to_seq:
        hits = per_protein.get(h, [])
        dbs = {hit["db"] for hit in hits}
        interpro_ids = {hit["interpro_id"] for hit in hits if hit["interpro_id"]}
        result[h] = {
            "hits": hits,
            "summary": {
                "n_hits": len(hits),
                "n_databases": len(dbs),
                "n_interpro": len(interpro_ids),
            },
        }

    try:
        _shutil.rmtree(tmpdir)
    except OSError:
        logger.warning("precompute_interproscan: failed to remove tempdir %s", tmpdir)

    return result


class InterProScanModule:
    """InterProScan domain annotation module.

    Consumes pre-computed InterProScan domain hits and attaches them to
    proteins. Hash-keyed lookup; no inline inference.

    Attributes:
        MODULE_NAME: ``"interproscan"``
        OUTPUT_COLUMNS: ``["interproscan_hits", "interproscan_summary"]``
        SCOPE: ``"C"`` (per-candidate site).
    """

    MODULE_NAME: str = "interproscan"
    OUTPUT_COLUMNS: list[str] = ["interproscan_hits", "interproscan_summary"]
    SCOPE: str = "C"

    def __init__(
        self,
        config: PipelineConfig,
        predictions: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize with pipeline config and optional pre-computed predictions."""
        self.config = config
        self.predictions = predictions or {}

    def annotate(self, protein: str) -> dict[str, Any]:
        """Look up InterProScan hits for *protein* by sequence hash.

        Returns an empty-hits structure when the hash isn't in the
        predictions dict — the pipeline still writes a well-formed
        column rather than ``None``.
        """
        h = _protein_hash(protein)
        pred = self.predictions.get(h)
        if pred is None:
            return {"hits": [], "summary": {"n_hits": 0, "n_databases": 0, "n_interpro": 0}}
        return {
            "hits": pred.get("hits", []),
            "summary": pred.get("summary", {"n_hits": 0, "n_databases": 0, "n_interpro": 0}),
        }

    def run(self, tis_sites: list[TranslationInitiationSite]) -> list[TranslationInitiationSite]:
        """Attach InterProScan annotations to each TIS's isoform protein."""
        for site in tis_sites:
            site.isoform_annotations[self.MODULE_NAME] = self.annotate(site.isoform_protein)
        return tis_sites
