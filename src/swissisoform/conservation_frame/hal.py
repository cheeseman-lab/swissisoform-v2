"""``hal2maf`` subprocess wrapper.

Each call invokes ``hal2maf`` for one genomic interval and returns the MAF
text, or ``None`` when the binary / HAL / query is unavailable. Graceful
degradation is deliberate: the Zoonomia HAL is 200–600 GB, so the module
must work on machines that don't have it and simply report
``status="not_run"``.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def hal2maf_available(hal2maf_binary: str = "hal2maf") -> bool:
    """Return True if *hal2maf_binary* is resolvable on ``PATH``."""
    return shutil.which(hal2maf_binary) is not None


def extract_maf(
    hal_path: str | Path,
    chrom: str,
    start: int,
    length: int,
    *,
    ref_genome: str = "hg38",
    hal2maf_binary: str = "hal2maf",
    timeout: float = 120.0,
) -> str | None:
    """Run ``hal2maf`` for a single interval and return the MAF text.

    Args:
        hal_path: Path to the Zoonomia HAL file.
        chrom: Reference chromosome name (e.g. ``chr1``).
        start: 0-based start (plus strand).
        length: Number of reference bases to extract.
        ref_genome: Reference genome name in the HAL (default ``hg38``).
        hal2maf_binary: Binary to invoke.
        timeout: Seconds before we abandon the subprocess.

    Returns:
        Raw MAF text on success; ``None`` when the binary is missing, the
        HAL file doesn't exist, the subprocess fails, or the interval is
        empty. All failure modes log at ``warning`` level but never raise.
    """
    if length <= 0:
        return None
    hal_path = Path(hal_path)
    if not hal_path.exists():
        logger.warning("HAL file not found: %s", hal_path)
        return None
    if not hal2maf_available(hal2maf_binary):
        logger.warning("hal2maf binary not on PATH (%s)", hal2maf_binary)
        return None

    cmd = [
        hal2maf_binary,
        str(hal_path),
        "/dev/stdout",
        f"--refGenome={ref_genome}",
        f"--refSequence={chrom}",
        f"--start={start}",
        f"--length={length}",
        "--onlyOrthologs",
    ]
    try:
        completed = subprocess.run(  # noqa: S603 — inputs are constrained above
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("hal2maf failed for %s:%d+%d: %s", chrom, start, length, exc)
        return None
    if completed.returncode != 0:
        logger.warning(
            "hal2maf exit=%d for %s:%d+%d stderr=%s",
            completed.returncode,
            chrom,
            start,
            length,
            completed.stderr.strip()[:200],
        )
        return None
    return completed.stdout
