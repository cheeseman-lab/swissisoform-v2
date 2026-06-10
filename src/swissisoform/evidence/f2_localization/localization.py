"""Module: Localization — subcellular localization prediction consumer.

Attaches DeepLoc subcellular localization predictions to proteins.
Predictions can be pre-computed (passed via ``predictions=...``) or
computed inline over a batch of unique protein sequences via
:func:`precompute_deeploc`.

Inline compute uses the ``deeploc-2`` Python package (CPU-capable; GPU
optional).  The module gracefully no-ops when the package isn't
importable — useful in test environments without the DL stack.
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


DEEPLOC_CONDA_ENV = "swissisoform-v2-deeploc"


def precompute_deeploc(
    proteins: dict[str, str] | list[str],
    *,
    model: str = "Fast",
    device: str = "cpu",
    conda_env: str = DEEPLOC_CONDA_ENV,
) -> dict[str, dict[str, Any]]:
    """Run DeepLoc on a batch of unique proteins via subprocess.

    DeepLoc 2.1 only ships as a Python 3.8-pinned tarball, so we run it
    in an isolated conda env (``swissisoform-v2-deeploc``) set up by
    ``scripts/setup/setup_databases.py deeploc``.  This function shells out to
    ``conda run -n <env> deeploc2 -f input.fa -m <model> -d <device>
    -o <outdir>`` and parses the resulting CSV.

    Args:
        proteins: Either ``{label: sequence}`` or a list of sequences.
            Labels are discarded — output is keyed on the sha1 of the
            stop-stripped, uppercased sequence so duplicates dedupe.
        model: "Fast" or "Accurate".  "Fast" is CPU-friendly.
        device: "cpu" or "cuda".
        conda_env: Name of the conda env hosting deeploc2.

    Returns:
        ``{protein_hash: {deeploc, deeploc_signals, deeploc_membrane}}``.
        Empty dict (with WARN) if the conda env is missing or subprocess
        fails — the module degrades gracefully so pipelines can still
        run without localization.
    """
    import shutil as _shutil
    import subprocess
    import tempfile
    from pathlib import Path as _P

    if isinstance(proteins, list):
        proteins = {f"seq_{i}": s for i, s in enumerate(proteins)}

    if _shutil.which("conda") is None:
        logger.warning("precompute_deeploc: 'conda' binary not on PATH — returning empty.")
        return {}

    # Check env existence via `conda env list`
    env_list = subprocess.run(["conda", "env", "list"], capture_output=True, text=True, check=False)
    if conda_env not in env_list.stdout:
        logger.warning(
            "precompute_deeploc: conda env %r not found — run "
            "`python scripts/setup/setup_databases.py deeploc` first.  Returning empty.",
            conda_env,
        )
        return {}

    # Dedup by hash
    hash_to_seq: dict[str, str] = {}
    for _label, seq in proteins.items():
        h = _protein_hash(seq)
        hash_to_seq.setdefault(h, seq.rstrip("*").upper())

    logger.info(
        "precompute_deeploc: %d inputs → %d unique sequences (model=%s, device=%s)",
        len(proteins),
        len(hash_to_seq),
        model,
        device,
    )

    # One cache root for all transient scratch (never the repo root or /tmp).
    _scratch = _P(__file__).resolve().parents[4] / "data" / "cache" / "tmp"
    _scratch.mkdir(parents=True, exist_ok=True)
    tmpdir = _P(tempfile.mkdtemp(prefix="deeploc_", dir=_scratch))
    fasta = tmpdir / "input.fa"
    outdir = tmpdir / "out"
    outdir.mkdir()
    with open(fasta, "w") as fh:
        for h, seq in hash_to_seq.items():
            fh.write(f">{h}\n{seq}\n")

    # Invoke deeploc2 via the env's python directly to dodge any stale
    # ~/.local/bin/deeploc2 shim, and set PYTHONNOUSERSITE=1 so the env
    # python can't pick up packages from ~/.local that shadow its own.
    # Resolve the env's deeploc2 binary explicitly so a stale
    # ~/.local/bin/deeploc2 shim can't shadow it.
    env_prefix = subprocess.run(
        ["conda", "run", "-n", conda_env, "python", "-c", "import sys; print(sys.prefix)"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    deeploc_bin = _P(env_prefix) / "bin" / "deeploc2"
    cmd = [
        str(deeploc_bin),
        "-f",
        str(fasta),
        "-m",
        model,
        "-d",
        device,
        "-o",
        str(outdir),
    ]
    import os as _os

    env = dict(_os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    # Redirect torch hub cache to cwd — home dir has limited quota on
    # shared HPC and ESM weights are ~1 GB.
    torch_cache = _P("./.torch_cache").resolve()
    torch_cache.mkdir(parents=True, exist_ok=True)
    env["TORCH_HOME"] = str(torch_cache)
    env["HF_HOME"] = str(torch_cache / "hf")
    env["XDG_CACHE_HOME"] = str(torch_cache / "xdg")
    logger.info("precompute_deeploc: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        if proc.stdout:
            logger.info("precompute_deeploc stdout (tail):\n%s", proc.stdout[-800:])
        if proc.stderr:
            logger.info("precompute_deeploc stderr (tail):\n%s", proc.stderr[-800:])
    except subprocess.CalledProcessError as exc:
        logger.error(
            "precompute_deeploc: deeploc2 failed (exit %d). stderr=%s",
            exc.returncode,
            exc.stderr[-800:] if exc.stderr else "",
        )
        return {}

    # Parse results CSV — schema: Protein_ID, Localizations, Signals,
    # Membrane types, <per-location probs>
    import pandas as pd

    results_csvs = list(outdir.glob("results_*.csv"))
    if not results_csvs:
        logger.error("precompute_deeploc: no results_*.csv in %s", outdir)
        return {}
    df = pd.read_csv(results_csvs[0])

    result: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        key = str(row["Protein_ID"])
        result[key] = {
            "deeploc": row.get("Localizations"),
            "deeploc_signals": row.get("Signals"),
            "deeploc_membrane": row.get("Membrane types"),
        }
    # Clean up the per-call tempdir (FASTA + DeepLoc output CSV) so repeat
    # runs don't accumulate ``deeploc_*`` directories in the repo root.
    try:
        _shutil.rmtree(tmpdir)
    except OSError:
        logger.warning("precompute_deeploc: failed to remove tempdir %s", tmpdir)
    return result


class LocalizationModule:
    """Localization annotation module.

    Consumes pre-computed subcellular localization predictions and attaches
    them to TIS sites. The actual prediction tool (DeepLoc) is run separately
    as a GPU job; this module only loads and merges results.

    Attributes:
        MODULE_NAME: Unique module identifier.
        OUTPUT_COLUMNS: Column names produced (prefixed with MODULE_NAME_).
        SCOPE: Module scope ('C' for per-site classification).
    """

    MODULE_NAME: str = "localization"
    OUTPUT_COLUMNS: list[str] = [
        "localization_deeploc_prediction",
        "localization_deeploc_signals",
        "localization_deeploc_membrane",
    ]
    SCOPE: str = "C"

    def __init__(
        self,
        config: PipelineConfig,
        predictions: dict[str, dict[str, str | None]] | None = None,
    ) -> None:
        """Initialize with pipeline configuration and pre-computed predictions.

        Args:
            config: Pipeline configuration.
            predictions: Mapping of key (e.g. tis_id) to prediction dict with keys:
                deeploc, deeploc_signals, deeploc_membrane.
        """
        self.config = config
        self.predictions = predictions or {}

    def annotate(self, protein: str) -> dict[str, Any]:
        """Look up localization prediction for *protein* by sequence hash.

        Implements the ``ProteinModule`` protocol.  The pipeline calls this
        on both canonical (per gene) and isoform (per TIS) proteins.  The
        internal ``predictions`` dict is hash-keyed, so the same sequence
        queried multiple times always returns the same answer without
        recomputation.

        Args:
            protein: Protein sequence (stop codon optional).

        Returns:
            Annotation dict with keys: ``deeploc_prediction``,
            ``deeploc_signals``, ``deeploc_membrane``.  All values are
            ``None`` if the sequence is not in the predictions dict.
        """
        h = _protein_hash(protein)
        pred = self.predictions.get(h, {})
        return {
            "deeploc_prediction": pred.get("deeploc"),
            "deeploc_signals": pred.get("deeploc_signals"),
            "deeploc_membrane": pred.get("deeploc_membrane"),
        }

    def annotate_by_key(self, key: str) -> dict[str, Any]:
        """Look up predictions by arbitrary key (legacy).

        Retained for callers that key on tis_id or gene_name instead of
        protein sequence.  Prefer :meth:`annotate` for new code.
        """
        pred = self.predictions.get(key, {})
        return {
            "deeploc_prediction": pred.get("deeploc"),
            "deeploc_signals": pred.get("deeploc_signals"),
            "deeploc_membrane": pred.get("deeploc_membrane"),
        }

    def run(self, tis_sites: list[TranslationInitiationSite]) -> list[TranslationInitiationSite]:
        """Attach localization annotations to each TIS site's isoform protein.

        Supports two keying strategies on ``self.predictions``:
        1) tis_id-keyed (legacy / test-fixture shape)
        2) protein-hash-keyed (what ``precompute_deeploc`` returns)

        Tries tis_id first, falls back to protein hash. Missing keys yield
        all-``None`` annotation fields so downstream code stays uniform.
        """
        for site in tis_sites:
            if site.tis_id in self.predictions:
                site.isoform_annotations[self.MODULE_NAME] = self.annotate_by_key(site.tis_id)
            else:
                site.isoform_annotations[self.MODULE_NAME] = self.annotate(site.isoform_protein)
        return tis_sites
