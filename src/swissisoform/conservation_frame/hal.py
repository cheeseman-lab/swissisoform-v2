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


def extract_tree(
    hal_path: str | Path,
    *,
    halstats_binary: str = "halStats",
    timeout: float = 60.0,
) -> str | None:
    """Run ``halStats --tree`` and return the Newick string.

    The HAL carries the Cactus species tree alongside the alignment; pulling
    it lets us rank species by phylogenetic depth from ``hg38`` without
    hand-maintaining clade tables. Returns ``None`` and logs a warning on
    any failure (missing binary, missing HAL, non-zero exit), so callers
    can fall back to a default depth map.

    Args:
        hal_path: Path to the Zoonomia HAL file.
        halstats_binary: Binary to invoke (default ``"halStats"``).
        timeout: Seconds before we abandon the subprocess.

    Returns:
        Newick tree string on success; ``None`` otherwise.
    """
    hal_path = Path(hal_path)
    if not hal_path.exists():
        logger.warning("HAL file not found: %s", hal_path)
        return None
    if shutil.which(halstats_binary) is None:
        logger.warning("halStats binary not on PATH (%s)", halstats_binary)
        return None

    cmd = [halstats_binary, "--tree", str(hal_path)]
    try:
        completed = subprocess.run(  # noqa: S603 — constrained inputs
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("halStats --tree failed: %s", exc)
        return None
    if completed.returncode != 0:
        logger.warning(
            "halStats --tree exit=%d stderr=%s",
            completed.returncode,
            completed.stderr.strip()[:200],
        )
        return None
    tree = completed.stdout.strip()
    return tree or None


def extract_maf(
    hal_path: str | Path,
    chrom: str,
    start: int,
    length: int,
    *,
    ref_genome: str = "Homo_sapiens",
    hal2maf_binary: str = "hal2maf",
    timeout: float = 600.0,
) -> str | None:
    """Run ``hal2maf`` for a single interval and return the MAF text.

    First-touch of a new chromosome in the 241-way Zoonomia HAL loads
    per-sequence HDF5 metadata and can take ~90-120 s on networked
    storage; subsequent queries on the same chromosome complete in ~5-10 s
    once the kernel page cache is warm. The default timeout is generous
    to accommodate the cold path; raise it further for larger intervals.

    Args:
        hal_path: Path to the Zoonomia HAL file.
        chrom: Reference chromosome name (e.g. ``chr1``).
        start: 0-based start (plus strand).
        length: Number of reference bases to extract.
        ref_genome: Reference genome name in the HAL (default
            ``Homo_sapiens`` for the 241-mammal Cactus).
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

    # hal2maf v2.2 in the cactus v2.9.9 container takes space-separated
    # options only (``--refGenome X``, not ``--refGenome=X``); pass args
    # as separate list items.
    cmd = [
        hal2maf_binary,
        str(hal_path),
        "/dev/stdout",
        "--refGenome",
        ref_genome,
        "--refSequence",
        chrom,
        "--start",
        str(start),
        "--length",
        str(length),
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
