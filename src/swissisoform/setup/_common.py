"""Primitives shared by the setup-time builders.

Every builder under ``swissisoform.setup`` provisions reference data and stamps
a ``_setup.json`` beside it, so they all need the same three things: the repo
root, a repo-relative path for the sidecar, and a content hash of each input.
Those were copied into each module verbatim — three ``_sha256``, two ``_rel``,
four ``ROOT`` — which is how provenance fields drift apart without anyone
noticing that two sidecars now mean slightly different things.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# Provisioning convention: every reference-data directory carries this beside
# its artifacts. `distributions` and `tags.registry` each state the same name
# as part of their own read-side contract — that is deliberate, since a reader
# of those subsystems should not have to import the setup package to learn
# where provenance lives.
SIDECAR_FILE = "_setup.json"

# Bytes hashed per read. Inputs here are genomes and multi-GB parquets, so the
# file is never held whole.
_CHUNK = 1 << 20


def rel_to_root(path: Path) -> str:
    """Repo-relative path when it is inside the repo, else absolute.

    Inputs are routinely outside the tree (a scratch parquet, a ``tmp_path``
    fixture), so a bare ``relative_to`` would fail the build on the provenance
    write rather than on anything that matters.
    """
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's contents, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_parquet(run: str | None, parquet: Path | None) -> tuple[list[Path], str]:
    """Return ``(files, label)`` for a run name or an explicit parquet path.

    One merged ``all_paired.parquet`` if the campaign was merged, else its
    ``{run}_shard_*`` parts in sorted order. Shared because the tag builder
    needs the same lookup to read a run's *schema*, and its own copy of this
    cascade had already drifted into a second spelling of the same error.
    """
    if parquet is not None:
        return [parquet], parquet.parent.name
    run_dir = ROOT / "data" / "output" / (run or "")
    merged = run_dir / "all_paired.parquet"
    if merged.exists():
        return [merged], run or ""
    shards = sorted((ROOT / "data" / "output").glob(f"{run}_shard_*/all_paired.parquet"))
    if shards:
        return shards, run or ""
    raise SystemExit(f"no all_paired.parquet under {run_dir} or {run}_shard_*/")


__all__ = ["ROOT", "SIDECAR_FILE", "rel_to_root", "resolve_parquet", "sha256_file"]
