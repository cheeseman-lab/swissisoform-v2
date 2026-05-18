"""Chai-1 single-protein folding wrapper.

Chai-1 (ESM mode) is the optional second-opinion backend. We invoke
``chai_lab.chai1.run_inference`` directly (Python API, not CLI) and
normalise the output into the per-protein cache schema owned by
:mod:`swissisoform.structure.fold`.

Requires ``chai_lab`` in the active env (``swissisoform-v2-fold``).
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from swissisoform.structure.fold import (
    DEFAULT_CACHE_DIR,
    cache_path,
    derive_metrics,
    write_cache,
)

logger = logging.getLogger(__name__)


def fold_one(
    h: str,
    seq: str,
    *,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    num_recycles: int = 3,
    use_esm_embeddings: bool = True,
    device: str = "cuda",
) -> Path:
    """Run Chai-1 on a single protein and write the cache entry."""
    try:
        from chai_lab.chai1 import run_inference
    except ImportError as exc:
        raise RuntimeError(
            "chai_lab not installed — `uv pip install chai_lab` into the fold env."
        ) from exc

    base = cache_path(cache_dir, "chai", h)
    base.mkdir(parents=True, exist_ok=True)

    plddt: list[float] | None = None
    ptm: float | None = None
    status = "ok"

    with tempfile.TemporaryDirectory(dir=str(base)) as workdir:
        work = Path(workdir)
        fasta = work / f"{h}.fasta"
        fasta.write_text(f">protein|name={h}\n{seq.rstrip('*').upper()}\n")

        try:
            output = run_inference(
                fasta_file=fasta,
                output_dir=work / "out",
                num_trunk_recycles=num_recycles,
                device=device,
                use_esm_embeddings=use_esm_embeddings,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("chai: run_inference failed for hash=%s: %s", h, exc)
            metrics = derive_metrics(backend="chai", plddt=None, ptm=None, status="failed")
            metrics["length"] = len(seq.rstrip("*"))
            metrics["error"] = str(exc)[:500]
            write_cache(h, cache_dir=cache_dir, backend="chai", metrics=metrics)
            return base

        # Locate the top-rank CIF + per-residue pLDDT.
        cif_files = sorted((work / "out").rglob("*.cif"))
        if not cif_files:
            cif_files = sorted((work / "out").rglob("*.pdb"))
        if cif_files:
            (base / "model.cif").write_bytes(cif_files[0].read_bytes())
        else:
            status = "failed"

        # Chai's run_inference returns an object whose shape varies by
        # version; pull pTM and per-residue confidence pragmatically.
        if hasattr(output, "ptm"):
            try:
                ptm = float(output.ptm)
            except (TypeError, ValueError):
                ptm = None
        for attr in ("plddt", "per_residue_plddt", "plddt_per_residue"):
            arr = getattr(output, attr, None)
            if arr is not None:
                plddt = [float(v) for v in arr]
                break

    if plddt is not None:
        confidence_payload: dict[str, Any] = {
            "plddt": plddt,
            "ptm": ptm,
            "iptm": None,
        }
        with open(base / "confidence.json", "w") as fh:
            json.dump(confidence_payload, fh)
    elif status == "ok":
        # CIF without pLDDT — partial success.
        logger.warning("chai: no per-residue plddt for hash=%s", h)

    metrics = derive_metrics(backend="chai", plddt=plddt, ptm=ptm, status=status)
    metrics["length"] = len(seq.rstrip("*"))
    write_cache(h, cache_dir=cache_dir, backend="chai", metrics=metrics)
    return base
