"""Boltz-2 single-protein folding wrapper.

Boltz-2 is an open-source AF3-class structure predictor. We invoke it
through its ``boltz predict`` CLI in a subprocess (the canonical install
shape) and normalise the output into the per-protein cache schema owned
by :mod:`swissisoform.structure.fold`.

Requires ``boltz[cuda]`` in the active env (typically ``swissisoform-v2-fold``).
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
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

BOLTZ_TIMEOUT_S = 3600  # per-protein wall clock; ~1024-aa fits inside an A100 hour.


def _write_yaml(seq: str, path: Path) -> None:
    path.write_text(
        "version: 1\nsequences:\n  - protein:\n      id: A\n"
        f"      sequence: {seq}\n"
    )


def _parse_confidence_json(path: Path) -> tuple[list[float] | None, float | None, float | None]:
    """Return (per-residue plddt list, ptm, iptm) from a Boltz confidence_*.json.

    Boltz's exact key names have shifted; we accept ``plddt`` (list),
    ``complex_plddt`` (scalar fallback), ``ptm``, ``iptm``.
    """
    with open(path) as fh:
        conf = json.load(fh)
    plddt = conf.get("plddt") or conf.get("per_residue_plddt") or None
    if plddt is None and "complex_plddt" in conf:
        # No per-residue list available — leave plddt None and synthesise a
        # uniform scalar in derive_metrics. We log so the user knows.
        logger.warning(
            "Boltz confidence at %s lacks per-residue plddt; only complex_plddt=%.3f present.",
            path,
            conf.get("complex_plddt"),
        )
    return plddt, conf.get("ptm"), conf.get("iptm")


def fold_one(
    h: str,
    seq: str,
    *,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    num_recycles: int = 3,
    diffusion_samples: int = 1,
    use_msa_server: bool = True,
    no_kernels: bool = True,
    boltz_bin: str = "boltz",
) -> Path:
    """Run Boltz-2 on a single protein and write the cache entry.

    Args:
        h: protein hash (sha1 of stop-stripped uppercased seq).
        seq: protein sequence.
        cache_dir: root cache directory (per-backend subdirs underneath).
        num_recycles: Boltz ``--recycling_steps``.
        diffusion_samples: Boltz ``--diffusion_samples``.
        use_msa_server: Boltz can fetch its own MSA via the public server.
        no_kernels: disable cuequivariance kernels (compatibility).
        boltz_bin: ``boltz`` executable name (resolved via PATH).

    Returns:
        The cache directory path.

    Raises:
        RuntimeError: if the boltz binary is missing or the subprocess fails.
    """
    if shutil.which(boltz_bin) is None:
        raise RuntimeError(
            f"{boltz_bin!r} not on PATH — install with `uv pip install 'boltz[cuda]'` "
            f"into env {DEFAULT_CACHE_DIR}."
        )

    base = cache_path(cache_dir, "boltz", h)
    base.mkdir(parents=True, exist_ok=True)

    # Boltz writes a directory tree under --out_dir; isolate per-call so
    # concurrent calls don't clobber each other.
    with tempfile.TemporaryDirectory(dir=str(base)) as workdir:
        work = Path(workdir)
        yaml_path = work / f"{h}.yaml"
        _write_yaml(seq.rstrip("*").upper(), yaml_path)

        cmd = [
            boltz_bin,
            "predict",
            str(yaml_path),
            "--out_dir",
            str(work),
            "--recycling_steps",
            str(num_recycles),
            "--diffusion_samples",
            str(diffusion_samples),
            "--output_format",
            "mmcif",
        ]
        if use_msa_server:
            cmd.append("--use_msa_server")
        if no_kernels:
            cmd.append("--no_kernels")

        logger.info("boltz: running for hash=%s len=%d", h, len(seq))
        proc = subprocess.run(  # noqa: S603 — trusted args
            cmd,
            capture_output=True,
            text=True,
            timeout=BOLTZ_TIMEOUT_S,
            check=False,
        )

        status = "ok"
        plddt: list[float] | None = None
        ptm: float | None = None

        if proc.returncode != 0:
            logger.error(
                "boltz: rc=%d for hash=%s. stderr=%s",
                proc.returncode,
                h,
                (proc.stderr or "")[-500:],
            )
            status = "failed"
        else:
            cif_files = list(work.rglob("*.cif"))
            conf_files = list(work.rglob("confidence_*.json"))
            if not cif_files:
                logger.warning("boltz: no CIF produced for hash=%s", h)
                status = "failed"
            else:
                # Copy the top-rank model into the cache.
                cif_files.sort()
                (base / "model.cif").write_bytes(cif_files[0].read_bytes())

            if conf_files:
                conf_files.sort()
                plddt, ptm, _iptm = _parse_confidence_json(conf_files[0])
                if plddt is None:
                    # No per-residue pLDDT; try complex_plddt as uniform fallback
                    with open(conf_files[0]) as _cfh:
                        _cdata = json.load(_cfh)
                    cplddt = _cdata.get("complex_plddt")
                    if cplddt is not None:
                        seq_len = len(seq.rstrip("*"))
                        plddt = [float(cplddt)] * seq_len
                        logger.info(
                            "boltz: using complex_plddt=%.3f as uniform per-residue for %s",
                            cplddt, h,
                        )
                    else:
                        status = "failed"
                if plddt is not None:
                    confidence_payload: dict[str, Any] = {
                        "plddt": [float(v) for v in plddt],
                        "ptm": float(ptm) if ptm is not None else None,
                        "iptm": None,
                    }
                    with open(base / "confidence.json", "w") as fh:
                        json.dump(confidence_payload, fh)
            else:
                logger.warning("boltz: no confidence_*.json for hash=%s", h)
                status = "failed"

    metrics = derive_metrics(backend="boltz", plddt=plddt, ptm=ptm, status=status)
    metrics["length"] = len(seq.rstrip("*"))
    write_cache(h, cache_dir=cache_dir, backend="boltz", metrics=metrics)
    return base
