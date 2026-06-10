"""Module: SignalP — signal peptide prediction consumer.

Attaches SignalP 6.0 signal peptide predictions to proteins. Predictions
are precomputed in batch via :func:`precompute_signalp` (subprocess into
an isolated conda env), then hash-keyed by protein sequence. The module
itself only performs lookups — identical pattern to
``modules/localization.py``.

Five signal peptide types are supported by SignalP 6.0:
``SP`` (classical), ``LIPO`` (Sec/SPII lipoprotein), ``TAT`` (Tat/SPI),
``TATLIPO`` (Tat/SPII lipoprotein), and ``PILIN`` (Sec/SPIII). For
isoform biology, the distinction between canonical and isoform SP call
is what scores F4 (targeting/signal change).

SignalP 6.0 is distributed as a pip-installable academic-license
tarball from DTU, installed into a dedicated ``swissisoform-v2-signalp``
conda env (see ``scripts/setup/setup_databases.py signalp``). This module
gracefully no-ops when the env is missing.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from swissisoform.config import PipelineConfig
from swissisoform.models import TranslationInitiationSite

logger = logging.getLogger(__name__)


def _protein_hash(protein: str) -> str:
    """Stable hash of a protein sequence (stop codon stripped, uppercased)."""
    seq = protein.rstrip("*").upper()
    return hashlib.sha1(seq.encode("ascii"), usedforsecurity=False).hexdigest()


SIGNALP_CONDA_ENV = "swissisoform-v2-signalp"


def precompute_signalp(
    proteins: dict[str, str] | list[str],
    *,
    organism: str = "eukarya",
    mode: str = "fast",
    conda_env: str = SIGNALP_CONDA_ENV,
) -> dict[str, dict[str, Any]]:
    """Run SignalP 6.0 on a batch of unique proteins via subprocess.

    SignalP 6.0 ships as a DTU academic-license tarball installed into
    ``swissisoform-v2-dtu`` (shared with TargetP 2.0).  This function
    writes a FASTA of deduplicated input sequences, invokes the env's
    ``signalp6`` CLI, and parses the resulting ``prediction_results.txt``.

    Args:
        proteins: Either ``{label: sequence}`` or a list of sequences.
            Labels are discarded — output is keyed on the sha1 of the
            stop-stripped, uppercased sequence so duplicates dedupe.
        organism: ``eukarya`` (default), ``other``, ``archaea``, or ``gram+/gram-``.
            For human isoforms, ``eukarya`` is correct.
        mode: ``fast`` (default, ProtBERT-based CPU-friendly) or ``slow``
            (Evolutionary-scale, better for edge cases). ``fast`` is
            sufficient for large batches.
        conda_env: Name of the shared DTU env.

    Returns:
        ``{protein_hash: {signalp_prediction, signalp_probability, signalp_cleavage_site}}``.
        Empty dict (with WARN) if the env is missing or subprocess fails.
    """
    import shutil as _shutil
    import subprocess
    import tempfile
    from pathlib import Path as _P

    if isinstance(proteins, list):
        proteins = {f"seq_{i}": s for i, s in enumerate(proteins)}

    if _shutil.which("conda") is None:
        logger.warning("precompute_signalp: 'conda' binary not on PATH — returning empty.")
        return {}

    env_list = subprocess.run(["conda", "env", "list"], capture_output=True, text=True, check=False)
    if conda_env not in env_list.stdout:
        logger.warning(
            "precompute_signalp: conda env %r not found — run "
            "`python scripts/setup/setup_databases.py signalp` first.  Returning empty.",
            conda_env,
        )
        return {}

    hash_to_seq: dict[str, str] = {}
    for _label, seq in proteins.items():
        h = _protein_hash(seq)
        hash_to_seq.setdefault(h, seq.rstrip("*").upper())

    logger.info(
        "precompute_signalp: %d inputs → %d unique sequences (organism=%s, mode=%s)",
        len(proteins),
        len(hash_to_seq),
        organism,
        mode,
    )

    # One cache root for all transient scratch (never the repo root or /tmp).
    _scratch = _P(__file__).resolve().parents[4] / "data" / "cache" / "tmp"
    _scratch.mkdir(parents=True, exist_ok=True)
    tmpdir = _P(tempfile.mkdtemp(prefix="signalp_", dir=_scratch))
    fasta = tmpdir / "input.fa"
    outdir = tmpdir / "out"
    outdir.mkdir()
    with open(fasta, "w") as fh:
        for h, seq in hash_to_seq.items():
            fh.write(f">{h}\n{seq}\n")

    # Resolve env python and signalp6 binary explicitly to dodge ~/.local shims.
    env_prefix = subprocess.run(
        ["conda", "run", "-n", conda_env, "python", "-c", "import sys; print(sys.prefix)"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    signalp_bin = _P(env_prefix) / "bin" / "signalp6"
    cmd = [
        str(signalp_bin),
        "--fastafile", str(fasta),
        "--organism", organism,
        "--mode", mode,
        "--format", "txt",
        "--output_dir", str(outdir),
    ]
    import os as _os

    env = dict(_os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    torch_cache = _P("./.torch_cache").resolve()
    torch_cache.mkdir(parents=True, exist_ok=True)
    env["TORCH_HOME"] = str(torch_cache)
    env["HF_HOME"] = str(torch_cache / "hf")
    env["XDG_CACHE_HOME"] = str(torch_cache / "xdg")
    logger.info("precompute_signalp: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        if proc.stdout:
            logger.info("precompute_signalp stdout (tail):\n%s", proc.stdout[-800:])
    except subprocess.CalledProcessError as exc:
        logger.error(
            "precompute_signalp: signalp6 failed (exit %d). stderr=%s",
            exc.returncode,
            exc.stderr[-800:] if exc.stderr else "",
        )
        return {}

    # SignalP 6 writes ``prediction_results.txt`` (tab-separated, commented header).
    # Columns: ID, Prediction, OTHER, SP(Sec/SPI), LIPO, TAT, TATLIPO, PILIN, CS Position
    results_path = outdir / "prediction_results.txt"
    if not results_path.exists():
        logger.error("precompute_signalp: no prediction_results.txt in %s", outdir)
        return {}

    result: dict[str, dict[str, Any]] = {}
    with open(results_path) as fh:
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
            prediction = row.get("Prediction", "OTHER").strip()
            # Store a fixed-axis probability ``1 − P(OTHER)`` so canonical and
            # isoform values are on the same scale (P(any signal peptide)); the
            # per-class call stays in ``signalp_prediction``.  Subtracting two
            # per-class probabilities (the old behaviour) gave a wrong-signed,
            # near-zero delta exactly when the class changed.
            try:
                probability: float | None = (
                    1.0 - float(row["OTHER"]) if "OTHER" in row else None
                )
            except ValueError:
                probability = None
            cs = row.get("CS Position", "").strip() or None
            result[key] = {
                "signalp_prediction": prediction,
                "signalp_probability": probability,
                "signalp_cleavage_site": cs,
            }

    try:
        _shutil.rmtree(tmpdir)
    except OSError:
        logger.warning("precompute_signalp: failed to remove tempdir %s", tmpdir)
    return result


class SignalPModule:
    """SignalP 6.0 signal peptide annotation module.

    Consumes pre-computed SignalP predictions and attaches them to proteins.
    The actual prediction runs separately via :func:`precompute_signalp`;
    this module only looks up by sequence hash.

    Attributes:
        MODULE_NAME: Unique module identifier.
        OUTPUT_COLUMNS: Column names produced (prefixed with MODULE_NAME_).
        SCOPE: Module scope ('C' for per-site classification; module is
            also used as a ProteinModule so it runs on canonical proteins).
    """

    MODULE_NAME: str = "signalp"
    OUTPUT_COLUMNS: list[str] = [
        "signalp_prediction",
        "signalp_probability",
        "signalp_cleavage_site",
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
        """Look up SignalP prediction for *protein* by sequence hash.

        Returns all-``None`` when the hash isn't in the predictions dict
        (module degrades gracefully without a precompute run).
        """
        h = _protein_hash(protein)
        pred = self.predictions.get(h, {})
        return {
            "signalp_prediction": pred.get("signalp_prediction"),
            "signalp_probability": pred.get("signalp_probability"),
            "signalp_cleavage_site": pred.get("signalp_cleavage_site"),
        }

    def run(self, tis_sites: list[TranslationInitiationSite]) -> list[TranslationInitiationSite]:
        """Attach SignalP annotations to each TIS's isoform protein."""
        for site in tis_sites:
            site.isoform_annotations[self.MODULE_NAME] = self.annotate(site.isoform_protein)
        return tis_sites
