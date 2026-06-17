"""Cache I/O + backend dispatch for structure prediction.

Owns the on-disk cache schema (see :mod:`swissisoform.structure`) and
the :func:`precompute_fold` entrypoint that fans inputs out to the
backend chosen by ``backend=`` (ESMFold2 by default; Boltz-2 / Chai-1
selectable).

The backends themselves live in :mod:`swissisoform.structure.esmfold2`,
:mod:`swissisoform.structure.boltz` and :mod:`swissisoform.structure.chai`.
They are imported lazily so this module — and the SiteModule that consumes
it at pipeline runtime — has no hard dependency on torch / esm / boltz /
chai_lab.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache" / "structure"
DEFAULT_BACKEND = "esmfold2"
BACKENDS = ("esmfold2", "boltz", "chai")
DEFAULT_MAX_SEQ_LEN = 1024
FOLD_CONDA_ENV = "swissisoform-v2-fold"


def protein_hash(protein: str) -> str:
    """Stable hash of a protein sequence (stop codon stripped, uppercased)."""
    seq = protein.rstrip("*").upper()
    return hashlib.sha1(seq.encode("ascii"), usedforsecurity=False).hexdigest()


def cache_path(cache_dir: Path | str, backend: str, h: str) -> Path:
    """Return the per-protein cache directory for ``hash`` under ``backend``."""
    return Path(cache_dir) / backend / h


def load_cache(
    h: str,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    backend: str = DEFAULT_BACKEND,
) -> dict[str, Any] | None:
    """Load cached fold results by protein hash.

    Returns a dict with keys ``confidence``, ``metrics``, ``cif_path`` (or
    ``None`` if missing). Returns ``None`` if neither metrics.json nor
    confidence.json exists for ``hash`` under ``backend``.
    """
    base = cache_path(cache_dir, backend, h)
    metrics_p = base / "metrics.json"
    confidence_p = base / "confidence.json"
    cif_p = base / "model.cif"
    if not metrics_p.exists() and not confidence_p.exists():
        return None

    out: dict[str, Any] = {}
    if confidence_p.exists():
        with open(confidence_p) as fh:
            out["confidence"] = json.load(fh)
    if metrics_p.exists():
        with open(metrics_p) as fh:
            out["metrics"] = json.load(fh)
    out["cif_path"] = cif_p if cif_p.exists() else None
    out["base_dir"] = base
    return out


def write_cache(
    h: str,
    *,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    backend: str = DEFAULT_BACKEND,
    confidence: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    cif_text: str | None = None,
    cif_src: Path | None = None,
) -> Path:
    """Write a per-protein cache entry. Creates ``base_dir`` if missing.

    Either ``cif_text`` or ``cif_src`` populates ``model.cif`` (text takes
    priority). ``confidence`` and ``metrics`` are JSON-serialised to their
    respective files. Used by both backends + tests.
    """
    base = cache_path(cache_dir, backend, h)
    base.mkdir(parents=True, exist_ok=True)
    if confidence is not None:
        with open(base / "confidence.json", "w") as fh:
            json.dump(confidence, fh, indent=2)
    if metrics is not None:
        with open(base / "metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2)
    if cif_text is not None:
        (base / "model.cif").write_text(cif_text)
    elif cif_src is not None and Path(cif_src).exists():
        (base / "model.cif").write_bytes(Path(cif_src).read_bytes())
    return base


def derive_metrics(
    *,
    backend: str,
    plddt: list[float] | None,
    ptm: float | None,
    status: str,
) -> dict[str, Any]:
    """Compute the scalar ``metrics.json`` payload from per-residue pLDDT.

    Pure function — backends call this after parsing their native output.
    """
    import math

    if plddt and status == "ok":
        n = len(plddt)
        mean = sum(plddt) / n
        var = sum((v - mean) ** 2 for v in plddt) / n
        std = math.sqrt(var)
        return {
            "status": status,
            "backend": backend,
            "length": n,
            "plddt_mean": mean,
            "plddt_std": std,
            "ptm": ptm,
        }
    return {
        "status": status,
        "backend": backend,
        "length": len(plddt) if plddt else None,
        "plddt_mean": None,
        "plddt_std": None,
        "ptm": ptm,
    }


def precompute_fold(
    proteins: dict[str, str] | list[str],
    *,
    backend: str = DEFAULT_BACKEND,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    inline: bool = True,
    skip_missing: bool = True,
    **backend_kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Run (or load cached) structure predictions for a batch of proteins.

    Args:
        proteins: ``{label: seq}`` or list of seqs. Hash-keyed; duplicates dedupe.
        backend: ``"esmfold2"`` (default), ``"boltz"`` or ``"chai"``.
        cache_dir: Root cache directory (per-backend subdirs underneath).
        max_seq_len: Sequences longer than this skip with ``status="too_long"``.
        inline: When True, run uncached proteins inline (requires the GPU env).
            When False, only returns proteins already in cache — useful from
            CPU pipeline runs that delegate folding to a SLURM job.
        skip_missing: When True, only returns cached entries; missing entries
            are absent. When False, missing entries get ``None``-valued payloads.
        **backend_kwargs: Forwarded to the backend-specific ``fold_one``.

    Returns:
        ``{protein_hash: load_cache result}``.
    """
    cache_dir = Path(cache_dir)
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}")

    if isinstance(proteins, list):
        proteins = {f"seq_{i}": s for i, s in enumerate(proteins)}

    hash_to_seq: dict[str, str] = {}
    for _label, seq in proteins.items():
        if not seq:
            continue
        h = protein_hash(seq)
        hash_to_seq.setdefault(h, seq.rstrip("*").upper())

    logger.info(
        "precompute_fold[%s]: %d inputs → %d unique sequences (cache=%s, inline=%s)",
        backend,
        len(proteins),
        len(hash_to_seq),
        cache_dir,
        inline,
    )

    result: dict[str, dict[str, Any]] = {}
    missing: list[tuple[str, str]] = []
    for h, seq in hash_to_seq.items():
        cached = load_cache(h, cache_dir, backend)
        if cached is not None:
            result[h] = cached
        else:
            missing.append((h, seq))

    if missing and inline:
        too_long: list[tuple[str, str]] = []
        runnable: list[tuple[str, str]] = []
        for h, seq in missing:
            if len(seq) > max_seq_len:
                too_long.append((h, seq))
            else:
                runnable.append((h, seq))

        for h, seq in too_long:
            metrics = derive_metrics(
                backend=backend, plddt=None, ptm=None, status="too_long"
            )
            metrics["length"] = len(seq)
            write_cache(h, cache_dir=cache_dir, backend=backend, metrics=metrics)
            cached = load_cache(h, cache_dir, backend)
            if cached is not None:
                result[h] = cached

        if runnable:
            try:
                fold_one = _load_backend(backend)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "precompute_fold[%s]: backend import failed (%s); "
                    "skipping %d uncached proteins. Run scripts/slurm/run_fold.sbatch.",
                    backend,
                    exc,
                    len(runnable),
                )
                fold_one = None
            if fold_one is not None:
                for h, seq in runnable:
                    try:
                        fold_one(
                            h,
                            seq,
                            cache_dir=cache_dir,
                            **backend_kwargs,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.error(
                            "precompute_fold[%s]: failed for hash=%s len=%d: %s",
                            backend,
                            h,
                            len(seq),
                            exc,
                        )
                        metrics = derive_metrics(
                            backend=backend, plddt=None, ptm=None, status="failed"
                        )
                        metrics["length"] = len(seq)
                        metrics["error"] = str(exc)[:500]
                        write_cache(h, cache_dir=cache_dir, backend=backend, metrics=metrics)
                    cached = load_cache(h, cache_dir, backend)
                    if cached is not None:
                        result[h] = cached

    if not skip_missing:
        for h, _ in missing:
            result.setdefault(h, {"confidence": None, "metrics": None, "cif_path": None})

    return result


def _load_backend(backend: str):
    if backend == "esmfold2":
        from swissisoform.structure.esmfold2 import fold_one as fold_one_esmfold2
        return fold_one_esmfold2
    if backend == "boltz":
        from swissisoform.structure.boltz import fold_one as fold_one_boltz
        return fold_one_boltz
    if backend == "chai":
        from swissisoform.structure.chai import fold_one as fold_one_chai
        return fold_one_chai
    raise ValueError(f"unknown backend {backend!r}")
