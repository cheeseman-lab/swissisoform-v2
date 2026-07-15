"""ESMFold2 single-protein folding wrapper (default structure backend).

ESMFold2 (EvolutionaryScale) predicts all-atom structure directly from a
single sequence — no MSA, no subprocess. We load ``ESMFold2-Fast`` once per
worker (heavy weights), fold via ``ESMFold2InputBuilder().fold`` and normalise
the output into the per-protein cache schema owned by
:mod:`swissisoform.structure.fold`.

pLDDT is emitted on a **0-1** scale (verified on GPU), matching Boltz-2 and the
``f1_plddt_threshold`` default of 0.70 — no rescaling needed.

Requires ``esm`` (the EvolutionaryScale package) + the Biohub ``transformers``
fork in the active env (``.[fold]`` extra). See pyproject ``[fold]``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from swissisoform.structure.fold import (
    DEFAULT_CACHE_DIR,
    cache_path,
    derive_metrics,
    write_cache,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "biohub/ESMFold2-Fast"
# Quality/speed knobs. The library default (20/200) is slow; ESMFold2-Fast at
# 8/100 is a reasonable pipeline default (the 3/50 README quick-start already
# gives mean pLDDT ~0.82 / pTM ~0.77 on ubiquitin in ~22s on an A100).
DEFAULT_NUM_LOOPS = 8
DEFAULT_NUM_SAMPLING_STEPS = 100

# Loaded models, keyed by (model_name, device). The weights are large, so a
# batch of proteins reuses one resident model rather than reloading per call.
_MODELS: dict[tuple[str, str], Any] = {}


def _write_pae(base: Path, pae: Any, h: str) -> None:
    """Persist the L×L PAE map to ``base/pae.npy`` (float16), if present.

    ``pae`` is the raw ESMFold2 output tensor (torch), shape ``(L, L)`` or
    ``(1, L, L)`` for a single chain. Best-effort: any shape/backend surprise
    logs a warning and skips the file rather than failing the fold.
    """
    if pae is None:
        logger.info("esmfold2: no PAE on result for hash=%s (skipping pae.npy)", h)
        return
    try:
        import numpy as np

        arr = np.asarray(pae.detach().float().cpu().numpy())
        arr = np.squeeze(arr)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            logger.warning("esmfold2: unexpected PAE shape %s for hash=%s", arr.shape, h)
            return
        np.save(base / "pae.npy", arr.astype(np.float16))
    except Exception as exc:  # noqa: BLE001
        logger.warning("esmfold2: failed to write PAE for hash=%s: %s", h, exc)


def _load_model(model_name: str, device: str):
    """Load (and cache) an ESMFold2 model on ``device``."""
    key = (model_name, device)
    if key not in _MODELS:
        from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

        logger.info("esmfold2: loading %s on %s", model_name, device)
        _MODELS[key] = ESMFold2Model.from_pretrained(model_name).to(device).eval()
    return _MODELS[key]


def fold_one(
    h: str,
    seq: str,
    *,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    num_loops: int = DEFAULT_NUM_LOOPS,
    num_sampling_steps: int = DEFAULT_NUM_SAMPLING_STEPS,
    model_name: str = DEFAULT_MODEL,
    device: str = "cuda",
    seed: int = 0,
) -> Path:
    """Run ESMFold2 on a single protein and write the cache entry."""
    try:
        import torch
        from esm.models.esmfold2 import (
            ESMFold2InputBuilder,
            ProteinInput,
            StructurePredictionInput,
        )
    except ImportError as exc:
        raise RuntimeError(
            "esm / transformers fork not installed — `uv pip install -e '.[fold]'` "
            "into the fold env (see pyproject [fold])."
        ) from exc

    base = cache_path(cache_dir, "esmfold2", h)
    base.mkdir(parents=True, exist_ok=True)
    clean = seq.rstrip("*").upper()

    try:
        model = _load_model(model_name, device)
        spi = StructurePredictionInput(
            sequences=[ProteinInput(id="0", sequence=clean)]
        )
        with torch.no_grad():
            result = ESMFold2InputBuilder().fold(
                model,
                spi,
                num_loops=num_loops,
                num_sampling_steps=num_sampling_steps,
                num_diffusion_samples=1,
                seed=seed,
            )
    except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
        logger.error("esmfold2: OOM for hash=%s len=%d", h, len(clean))
        metrics = derive_metrics(backend="esmfold2", plddt=None, ptm=None, status="oom")
        metrics["length"] = len(clean)
        metrics["error"] = str(exc)[:500]
        write_cache(h, cache_dir=cache_dir, backend="esmfold2", metrics=metrics)
        return base
    except Exception as exc:  # noqa: BLE001
        logger.error("esmfold2: fold failed for hash=%s: %s", h, exc)
        metrics = derive_metrics(backend="esmfold2", plddt=None, ptm=None, status="failed")
        metrics["length"] = len(clean)
        metrics["error"] = str(exc)[:500]
        write_cache(h, cache_dir=cache_dir, backend="esmfold2", metrics=metrics)
        return base

    # ESMFold2 emits pLDDT on a 0-1 scale (verified); to_mmcif() rescales to
    # 0-100 B-factors internally, but result.plddt stays 0-1.
    plddt = [float(v) for v in result.plddt.detach().float().cpu()]
    ptm = float(result.ptm) if result.ptm is not None else None
    iptm = float(result.iptm) if result.iptm is not None else None
    cif_text = result.complex.to_mmcif()

    confidence_payload: dict[str, Any] = {"plddt": plddt, "ptm": ptm, "iptm": iptm}
    with open(base / "confidence.json", "w") as fh:
        json.dump(confidence_payload, fh)

    # PAE (predicted aligned error): ESMFold2 computes the full L×L map every
    # fold — pTM/ipTM are reductions of it — but the fold result otherwise
    # discards it. Persist it as its own file (too big for confidence.json).
    # Cached as float16 (PAE is 0..~32 Å, fp16 is lossless enough) to keep the
    # per-protein cache small; keyed by protein hash like the rest of the entry.
    _write_pae(base, getattr(result, "pae", None), h)

    metrics = derive_metrics(backend="esmfold2", plddt=plddt, ptm=ptm, status="ok")
    metrics["length"] = len(clean)
    write_cache(
        h, cache_dir=cache_dir, backend="esmfold2", metrics=metrics, cif_text=cif_text
    )
    return base


def fold_pae_only(
    h: str,
    seq: str,
    *,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    num_loops: int = DEFAULT_NUM_LOOPS,
    num_sampling_steps: int = DEFAULT_NUM_SAMPLING_STEPS,
    model_name: str = DEFAULT_MODEL,
    device: str = "cuda",
    seed: int = 0,
) -> Path:
    """Re-fold one protein and write ONLY ``pae.npy`` into its existing entry.

    Backfill path for entries folded before PAE capture: it recomputes the fold
    (deterministic, seed-pinned) purely to extract the L×L PAE, and writes just
    ``pae.npy`` — it never touches ``model.cif`` / ``confidence.json`` /
    ``metrics.json``, so it is safe to run against a shared/warm cache. Raises on
    fold error (the caller decides how to count failures).
    """
    import torch
    from esm.models.esmfold2 import (
        ESMFold2InputBuilder,
        ProteinInput,
        StructurePredictionInput,
    )

    base = cache_path(cache_dir, "esmfold2", h)
    base.mkdir(parents=True, exist_ok=True)
    clean = seq.rstrip("*").upper()

    model = _load_model(model_name, device)
    spi = StructurePredictionInput(sequences=[ProteinInput(id="0", sequence=clean)])
    with torch.no_grad():
        result = ESMFold2InputBuilder().fold(
            model,
            spi,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            num_diffusion_samples=1,
            seed=seed,
        )
    _write_pae(base, getattr(result, "pae", None), h)
    return base
