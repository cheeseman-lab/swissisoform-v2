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
we pre-warm via ``scripts/setup/setup_databases.py interproscan``.  The
module gracefully no-ops when Nextflow / datadir aren't set up.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from swissisoform.config import PipelineConfig
from swissisoform.models import TranslationInitiationSite

logger = logging.getLogger(__name__)


def _protein_hash(protein: str) -> str:
    """Stable hash of a protein sequence (stop codon stripped, uppercased)."""
    seq = protein.rstrip("*").upper()
    return hashlib.sha1(seq.encode("ascii"), usedforsecurity=False).hexdigest()


# Member databases that report disorder / coiled-coil / signal-peptide /
# transmembrane / low-complexity regions rather than true functional
# domains. A "real functional domain" must additionally carry an InterPro
# cross-reference, so these are excluded from real-domain counting (F3).
# Matched case-insensitively against the hit ``db`` field.
DISORDER_STRUCTURAL_DBS = frozenset(
    {
        "mobidb-lite",
        "coils",
        "low_complexity",
        "signalp",
        "phobius",
        "tmhmm",
    }
)


def is_real_domain(hit: dict[str, Any]) -> bool:
    """Whether an InterProScan hit represents a real functional domain.

    A real functional domain has a real InterPro cross-reference
    (``interpro_id``) AND comes from a member DB that is not in the
    disorder / structural-only set (MobiDB-lite, Coils, low_complexity,
    SignalP, Phobius, TMHMM). The comparator uses this to count only
    genuine domain gain/loss for F3, ignoring disorder/coiled-coil noise.

    Args:
        hit: One entry from an interproscan ``hits`` list.

    Returns:
        ``True`` if the hit is a real functional domain, else ``False``.
    """
    interpro_id = hit.get("interpro_id")
    if not interpro_id or interpro_id in ("-", ""):
        return False
    db = str(hit.get("db", "")).strip().lower()
    return db not in DISORDER_STRUCTURAL_DBS


# Alias — the comparator imports this name for the F3 real-domain count.
is_real_functional_domain = is_real_domain


# Default install location used by scripts/setup/setup_databases.py interproscan.
DEFAULT_INTERPROSCAN_DIR = (
    Path(__file__).resolve().parents[4] / "data" / "reference" / "interproscan"
)
# Comma-separated set of member-DB applications.  When ``None``, IPS6
# uses its default non-ML app set (Pfam, SMART, CDD, PANTHER, etc.).
# Pfam-only runs trip a COMBINE_MATCHES bug in IPS6 6.0.0 when any
# sequence has no hit, so the multi-app default is safer.
INTERPROSCAN_APPLICATIONS: str | None = None
INTERPROSCAN_VERSION = "6.0.0"
INTERPROSCAN_DATA_VERSION = "108.0"
INTERPROSCAN_NF_REPO = "ebi-pf-team/interproscan6"


def _ips_fallback_from_workdir(
    tmpdir,
    hash_to_seq,
):
    """Recover IPS hits from ``calculatedMatches.json`` when combine step failed.

    The InterProScan 6 Nextflow pipeline writes a cross-analysis
    aggregated match dict at ``<workdir>/work/<task>/calculatedMatches.json``
    just before the final combine step. When combine fails (e.g. Groovy
    classpath issue), the per-analysis data is still on disk; we can read
    it directly.

    The JSON is keyed by MD5 (uppercase) of the input sequence; our cache
    is keyed by SHA1. We re-key using the input FASTA we wrote.

    Returns the same shape as a successful precompute:
    ``{sha1_hash: {hits: [...], summary: {n_hits, n_databases, n_interpro}}}``
    or ``{}`` if no fallback JSON is found.
    """
    workdir = Path(tmpdir) / "work"
    if not workdir.exists():
        logger.warning("ips fallback: no work/ dir under %s", tmpdir)
        return {}

    # Reject stale candidates whose mtime predates the run — Nextflow
    # work dirs are uniquely-named per run (random hex prefix) so this
    # double-checks against external reuse. We allow up to 1h backdate
    # to accommodate clock skew between worker nodes.
    try:
        tmpdir_mtime = Path(tmpdir).stat().st_mtime
        cutoff_mtime = tmpdir_mtime - 3600
    except OSError:
        cutoff_mtime = 0.0

    candidates = []
    for p in workdir.rglob("calculatedMatches.json"):
        try:
            stat = p.stat()
        except OSError:
            continue
        if stat.st_mtime < cutoff_mtime:
            continue
        candidates.append((stat.st_size, p))
    candidates.sort(reverse=True)
    if not candidates:
        logger.error(
            "ips fallback: no calculatedMatches.json found under %s", workdir
        )
        return {}

    # The aggregated JSON is the largest (per-task no_matches files are tiny).
    json_size, json_p = candidates[0]
    if json_size < 1024:
        logger.warning(
            "ips fallback: largest calculatedMatches.json is only %d bytes", json_size
        )
        return {}

    try:
        with open(json_p) as fh:
            matches = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("ips fallback: failed to read %s: %s", json_p, exc)
        return {}

    if not isinstance(matches, dict):
        return {}

    # Build md5 -> sha1 mapping for our input proteins
    md5_to_sha1 = {}
    for sha1, seq in hash_to_seq.items():
        md5 = hashlib.md5(seq.encode("ascii")).hexdigest().upper()
        md5_to_sha1[md5] = sha1

    result = {}
    for md5_key, sig_map in matches.items():
        sha1_key = md5_to_sha1.get(md5_key)
        if sha1_key is None:
            continue
        if not isinstance(sig_map, dict):
            continue
        hits = []
        dbs = set()
        interpro_ids = set()
        for sig_id, sig_data in sig_map.items():
            if not isinstance(sig_data, dict):
                continue
            sig = sig_data.get("signature") or {}
            sig_lib = sig.get("signatureLibraryRelease") or {}
            source = sig_data.get("source") or sig_lib.get("library", "")
            entry = sig.get("entry") or {}
            interpro_id = entry.get("accession") if entry else None
            interpro_desc = entry.get("description") if entry else None
            sig_name = sig.get("name") or sig.get("accession") or sig_id
            sig_desc = sig.get("description")
            locations = sig_data.get("locations") or []
            for loc in locations:
                if not isinstance(loc, dict):
                    continue
                try:
                    start = int(loc.get("start"))
                    end = int(loc.get("end"))
                except (TypeError, ValueError):
                    continue
                score = loc.get("evalue")
                if score is not None:
                    try:
                        score = float(score)
                    except (TypeError, ValueError):
                        score = None
                hits.append({
                    "name": sig.get("accession") or sig_id,
                    "pos": start - 1,  # 1-based inclusive -> 0-based for comparator
                    "end": end,
                    "db": source,
                    "description": sig_desc or sig_name,
                    "score": score,
                    "interpro_id": interpro_id,
                    "interpro_description": interpro_desc,
                })
                if source:
                    dbs.add(source)
                if interpro_id:
                    interpro_ids.add(interpro_id)
        result[sha1_key] = {
            "hits": hits,
            "summary": {
                "n_hits": len(hits),
                "n_databases": len(dbs),
                "n_interpro": len(interpro_ids),
            },
        }
    # Also fill in proteins from input that had no matches
    for sha1, _seq in hash_to_seq.items():
        result.setdefault(sha1, {
            "hits": [],
            "summary": {"n_hits": 0, "n_databases": 0, "n_interpro": 0},
        })
    return result


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
            "`python scripts/setup/setup_databases.py interproscan` first.  Returning empty.",
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

    # One cache root for all transient scratch (never the repo root or /tmp).
    _scratch = _P(__file__).resolve().parents[4] / "data" / "cache" / "tmp"
    _scratch.mkdir(parents=True, exist_ok=True)
    tmpdir = _P(tempfile.mkdtemp(prefix="interproscan_", dir=_scratch)).resolve()
    try:
        return _run_interproscan(tmpdir, hash_to_seq, applications, nf_version,
                                 profile, datadir, data_version)
    finally:
        # Always remove the scratch tree, on every return path and on exception.
        _shutil.rmtree(tmpdir, ignore_errors=True)


def _run_interproscan(
    tmpdir: Path,
    hash_to_seq: dict[str, str],
    applications: str,
    nf_version: str,
    profile: str,
    datadir: Path,
    data_version: str,
) -> dict[str, dict[str, Any]]:
    """Run the InterProScan Nextflow pipeline in ``tmpdir`` and parse its TSV.

    Cleanup of ``tmpdir`` is the caller's responsibility (a ``finally``).
    """
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
        # Fallback: try to recover from the per-task aggregated JSON.
        # The Nextflow pipeline runs all per-analysis steps successfully
        # but the final ``combine`` step (a Groovy script) can fail with
        # a classpath issue on some installs. The cross-analysis match
        # data lives in ``calculatedMatches.json`` in the work tree,
        # produced just before combine. If we find it, parse it.
        fallback = _ips_fallback_from_workdir(tmpdir, hash_to_seq)
        if fallback:
            logger.warning(
                "precompute_interproscan: recovered %d entries from "
                "calculatedMatches.json fallback (combine step skipped)",
                len(fallback),
            )
            return fallback
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

        Returns a well-formed dict with a ``summary.status`` field so
        downstream scoring can distinguish:

        - ``status='ok'`` — precompute ran and this protein was scanned
          (``n_hits`` may legitimately be 0 — a real measurement).
        - ``status='no_data'`` — precompute failed or this protein's
          hash isn't in the predictions dict. Scoring (F3) should
          return ``None`` rather than ``False``.
        """
        h = _protein_hash(protein)
        pred = self.predictions.get(h)
        if pred is None:
            return {
                "hits": [],
                "summary": {
                    "n_hits": 0,
                    "n_databases": 0,
                    "n_interpro": 0,
                    "status": "no_data",
                },
            }
        summary = dict(pred.get("summary", {"n_hits": 0, "n_databases": 0, "n_interpro": 0}))
        summary.setdefault("status", "ok")
        # Sort hits deterministically — the underlying Nextflow pipeline runs
        # member-DB scans in parallel and the resulting hit order isn't stable
        # across reruns, which makes back-to-back parquets diff on this one
        # column for no semantic reason. Sort by (pos, end, signature name)
        # so the *set* (already identical) becomes a deterministic *list*.
        raw_hits = pred.get("hits", [])
        hits = sorted(
            raw_hits,
            key=lambda h: (
                h.get("pos") if isinstance(h.get("pos"), int) else 10**9,
                h.get("end") if isinstance(h.get("end"), int) else 10**9,
                str(h.get("name", "")),
                str(h.get("db", "")),
            ),
        )
        return {
            "hits": hits,
            "summary": summary,
        }

    def run(self, tis_sites: list[TranslationInitiationSite]) -> list[TranslationInitiationSite]:
        """Attach InterProScan annotations to each TIS's isoform protein."""
        for site in tis_sites:
            site.isoform_annotations[self.MODULE_NAME] = self.annotate(site.isoform_protein)
        return tis_sites
