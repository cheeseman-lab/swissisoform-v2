"""Module: TargetP — subcellular targeting peptide prediction consumer.

Attaches TargetP 2.0 targeting-peptide predictions to proteins. TargetP
distinguishes three targeting classes (in addition to ``noTP``):

- ``SP`` — classical secretory signal peptide
- ``mTP`` — mitochondrial transit peptide
- ``cTP`` — chloroplastic transit peptide (plant-only; irrelevant for
  human isoforms but reported for completeness)

For human isoforms, the biologically informative calls are ``SP``
(overlapping with SignalP's call) and ``mTP`` (which SignalP cannot
predict). Using TargetP + SignalP together lets F4
(``targeting/signal change``) fire on either signal-peptide or
mitochondrial-presequence gain/loss in the extension region.

TargetP 2.0 is a **self-contained native Go binary** distributed as an
academic-license tarball from DTU.  It is *not* a Python package; no
conda env is needed.  ``scripts/setup_databases.py targetp`` extracts
the tarball to ``data/reference/targetp/targetp-2.0/`` and verifies the
binary runs.  This module subprocesses to that binary directly, setting
``LD_LIBRARY_PATH`` to the bundled TensorFlow libs.  If the install
directory is missing, the module degrades to all-``None`` annotations.
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


# Default install location used by scripts/setup_databases.py targetp.
# Callers can override via ``install_dir`` below.
DEFAULT_TARGETP_DIR = (
    Path(__file__).resolve().parents[3] / "data" / "reference" / "targetp" / "targetp-2.0"
)


def precompute_targetp(
    proteins: dict[str, str] | list[str],
    *,
    organism: str = "non-pl",
    install_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Run TargetP 2.0 on a batch of unique proteins via subprocess.

    Args:
        proteins: Either ``{label: sequence}`` or a list of sequences.
            Output is keyed on the sha1 of the stop-stripped, uppercased
            sequence so duplicates dedupe.
        organism: ``non-pl`` (default, humans) or ``pl`` (plant). Matches
            TargetP's ``-org`` flag.
        install_dir: Path to the extracted TargetP 2.0 directory
            (default: ``data/reference/targetp/targetp-2.0``).

    Returns:
        ``{protein_hash: {targetp_prediction, targetp_probability,
        targetp_sp_prob, targetp_mtp_prob, targetp_ctp_prob,
        targetp_cleavage_site}}``. Empty dict (with WARN) if the install
        is missing or the subprocess fails.
    """
    import shutil as _shutil
    import subprocess
    import tempfile
    from pathlib import Path as _P

    if isinstance(proteins, list):
        proteins = {f"seq_{i}": s for i, s in enumerate(proteins)}

    root = install_dir or DEFAULT_TARGETP_DIR
    targetp_bin = root / "bin" / "targetp"
    lib_dir = root / "lib"
    if not targetp_bin.exists():
        logger.warning(
            "precompute_targetp: binary not found at %s — run "
            "`python scripts/setup_databases.py targetp` first.  Returning empty.",
            targetp_bin,
        )
        return {}

    hash_to_seq: dict[str, str] = {}
    for _label, seq in proteins.items():
        h = _protein_hash(seq)
        hash_to_seq.setdefault(h, seq.rstrip("*").upper())

    logger.info(
        "precompute_targetp: %d inputs → %d unique sequences (organism=%s)",
        len(proteins),
        len(hash_to_seq),
        organism,
    )

    tmpdir = _P(tempfile.mkdtemp(prefix="targetp_", dir=".")).resolve()
    fasta = tmpdir / "input.fa"
    prefix = tmpdir / "result"
    with open(fasta, "w") as fh:
        for h, seq in hash_to_seq.items():
            fh.write(f">{h}\n{seq}\n")

    cmd = [
        str(targetp_bin),
        "-fasta", str(fasta),
        "-org", organism,
        "-format", "short",
        "-prefix", str(prefix),
    ]
    import os as _os

    env = dict(_os.environ)
    # TargetP's bundled TensorFlow .so lives alongside the binary — the
    # readme expects ``lib/`` sitting next to ``bin/`` to be found via
    # LD_LIBRARY_PATH.
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = (
        f"{lib_dir}:{existing_ld}" if existing_ld else str(lib_dir)
    )
    logger.info("precompute_targetp: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env, cwd=tmpdir)
        if proc.stdout:
            logger.info("precompute_targetp stdout (tail):\n%s", proc.stdout[-800:])
    except subprocess.CalledProcessError as exc:
        logger.error(
            "precompute_targetp: targetp failed (exit %d). stderr=%s",
            exc.returncode,
            exc.stderr[-800:] if exc.stderr else "",
        )
        return {}

    # TargetP writes ``<prefix>_summary.targetp2`` — tab-separated, commented
    # header. Non-plant columns: ID, Prediction, noTP, SP, mTP, CS Position
    summary_candidates = list(tmpdir.glob("result*summary*"))
    if not summary_candidates:
        logger.error("precompute_targetp: no summary file found in %s", tmpdir)
        return {}
    summary_path = summary_candidates[0]

    result: dict[str, dict[str, Any]] = {}
    with open(summary_path) as fh:
        header: list[str] | None = None
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                if "ID" in line and "Prediction" in line:
                    header = [c.strip() for c in line.lstrip("#").strip().split("\t")]
                continue
            if header is None:
                continue
            fields = line.split("\t")
            row = dict(zip(header, fields, strict=False))
            key = row.get("ID", "").strip()
            if not key:
                continue
            prediction = row.get("Prediction", "noTP").strip()

            def _maybe_float(name: str) -> float | None:
                raw = row.get(name)
                if raw is None or raw == "":
                    return None
                try:
                    return float(raw)
                except ValueError:
                    return None

            prob_field = prediction if prediction in row else "noTP"
            probability = _maybe_float(prob_field)
            cs = row.get("CS Position", "").strip() or None
            result[key] = {
                "targetp_prediction": prediction,
                "targetp_probability": probability,
                "targetp_sp_prob": _maybe_float("SP"),
                "targetp_mtp_prob": _maybe_float("mTP"),
                "targetp_ctp_prob": _maybe_float("cTP"),
                "targetp_cleavage_site": cs,
            }

    try:
        _shutil.rmtree(tmpdir)
    except OSError:
        logger.warning("precompute_targetp: failed to remove tempdir %s", tmpdir)
    return result


class TargetPModule:
    """TargetP 2.0 targeting-peptide annotation module.

    Consumes pre-computed TargetP predictions and attaches them to proteins.
    Hash-keyed lookup; no inline inference.
    """

    MODULE_NAME: str = "targetp"
    OUTPUT_COLUMNS: list[str] = [
        "targetp_prediction",
        "targetp_probability",
        "targetp_sp_prob",
        "targetp_mtp_prob",
        "targetp_ctp_prob",
        "targetp_cleavage_site",
    ]
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
        """Look up TargetP prediction for *protein* by sequence hash.

        Returns all-``None`` when the hash isn't in the predictions dict.
        """
        h = _protein_hash(protein)
        pred = self.predictions.get(h, {})
        return {
            "targetp_prediction": pred.get("targetp_prediction"),
            "targetp_probability": pred.get("targetp_probability"),
            "targetp_sp_prob": pred.get("targetp_sp_prob"),
            "targetp_mtp_prob": pred.get("targetp_mtp_prob"),
            "targetp_ctp_prob": pred.get("targetp_ctp_prob"),
            "targetp_cleavage_site": pred.get("targetp_cleavage_site"),
        }

    def run(self, tis_sites: list[TranslationInitiationSite]) -> list[TranslationInitiationSite]:
        """Attach TargetP annotations to each TIS's isoform protein."""
        for site in tis_sites:
            site.isoform_annotations[self.MODULE_NAME] = self.annotate(site.isoform_protein)
        return tis_sites
