"""On-disk lifecycle for uploaded VCFs and their scan digests.

**The only module in the site that touches the scan filesystem.** Routes go
through here, so replacing ephemeral container disk with object storage later is
a change to one file rather than a hunt through handlers.

Layout under ``$SWISSISOFORM_SCAN_DIR`` (default ``/tmp/swissisoform-scans``)::

    tokens/<token>.json      {"key", "filename", "created_at", "vcf_sha256"}
    blobs/<key>/
        source.vcf.gz        the upload, always gzipped, never served back
        scan.json            the digest — the only file any page reads

Two identifiers, deliberately separated:

* ``key = sha256(vcf)[:32]-<index_version>`` is the **address**. Content-derived,
  so a re-upload of the same file against the same index finds the finished
  digest and skips the parse entirely. The index version is part of it because a
  digest is only valid for the coordinates it was computed against.
* ``token`` is a 128-bit ``secrets`` value and the **capability** — it is what
  appears in the user's URL. Using the hash there instead would let anyone
  confirm what someone else uploaded by hashing a candidate file, which matters
  for controlled-access genomic data.

The same reasoning binds anything *derived* from a shared blob: a cache-hit flag
or the filename it was first stored under are both facts about a previous
uploader, so neither may reach the client. Per-upload facts — filename, timing —
live on the token and are read from there, never defaulted from the digest.

Expiry is enforced when a token is **read**, not when the sweeper runs, so a
missed sweep can never serve stale results. Deletion is best-effort cleanup.

This container's disk is ephemeral: it is shared by both gunicorn workers but
dies with the container. That is the right trade for data we delete after 24 h —
but it does mean the service must stay at **one replica**, since a second
container would not see the first one's uploads.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import secrets
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, BinaryIO

logger = logging.getLogger(__name__)

DEFAULT_SCAN_DIR = Path("/tmp/swissisoform-scans")
DEFAULT_TTL_HOURS = 24.0

#: Ceiling on total bytes under the scan dir. Eviction is oldest-first on the
#: hourly sweep, so behaviour does not depend on the host's disk quota (which is
#: plan-dependent on Railway and not something we can read).
DEFAULT_BUDGET_BYTES = 2 * 1024**3

#: Minimum gap between sweeps, tracked by a marker file's mtime. Every worker
#: runs the same check; the sweep is idempotent so overlap is harmless.
SWEEP_INTERVAL_SECONDS = 3600

_TOKEN_BYTES = 16
_CHUNK = 1 << 20  # 1 MiB: hashing costs ~4 ms per chunk at 253 MB/s.

SOURCE_NAME = "source.vcf.gz"
DIGEST_NAME = "scan.json"
_SWEEP_MARKER = ".last_sweep"

#: Shape of ``scan.json``. Part of the blob key, so changing the digest's fields
#: invalidates every cached digest instead of serving one the templates can no
#: longer read correctly.
#:
#: **Bump this whenever the digest's structure changes.** Learned the hard way:
#: renaming ``GeneSummary.n_variants`` to ``n_hits`` and adding a true variant
#: count left cached digests in place, so the results page rendered the old
#: field under the new label — plausible numbers, silently wrong.
#:
#: It is written into the **token** as well as the blob key. The key alone only
#: stops new uploads from reusing a stale blob; an existing token keeps pointing at
#: the old blob and would serve its old shape. That is harmless while a redeploy
#: wipes /tmp — but the moment the store outlives a deploy (a volume, a bucket) it
#: becomes a live bug, so the token carries the schema and a mismatch reads as
#: expired.
#:
#: History: d1 initial · d2 per-gene n_hits split from n_variants ·
#: d3 per-hit consequence / aa_ref / aa_alt / hgvsp · d4 hgvsp dropped, and hits now
#: classified by the pipeline's own ConsequenceValidator (so a minus-strand
#: multi-base variant's residue moves one codon, to the first one it changes) ·
#: d5 counts gained ``stopped``, which says the scan ended on a record or
#: wall-clock budget and every count below it is a partial · d6 the start-codon
#: rule turned asymmetric, so an SNV destroying an annotated ATG is start_lost
#: where it used to read as missense, and hits carry the classifier's own note ·
#: d7 indels gained amino acids and their residue moved to the first *changed*
#: base (the VCF padding base is not it), and a wrong-REF indel now reports
#: reference_mismatch instead of a clean frameshift.
DIGEST_SCHEMA = "d7"


class ScanStoreError(RuntimeError):
    """Raised when a store operation cannot complete (e.g. disk full)."""


@dataclass(frozen=True, slots=True)
class SavedUpload:
    """Result of accepting an upload."""

    token: str
    key: str
    vcf_sha256: str
    was_cached: bool
    """True when a finished digest already existed for this (file, index) pair."""


@dataclass(frozen=True, slots=True)
class LoadedScan:
    """A resolved token, or the reason it could not be resolved.

    Exactly one of ``digest`` / ``missing`` / ``expired`` is meaningful, so a
    route can map the three onto 200 / 404 / 410 without guessing.
    """

    digest: dict[str, Any] | None = None
    missing: bool = False
    expired: bool = False
    token: str = ""

    @property
    def ok(self) -> bool:
        """True when a digest was found and is still within its TTL."""
        return self.digest is not None


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


def scan_dir() -> Path:
    """Root of the scan store, from ``SWISSISOFORM_SCAN_DIR``."""
    return Path(os.environ.get("SWISSISOFORM_SCAN_DIR", DEFAULT_SCAN_DIR))


def ttl_hours() -> float:
    """Retention window, from ``SWISSISOFORM_SCAN_TTL_HOURS``.

    ``0`` is honoured (everything is immediately expired) — the tests rely on it
    to exercise the expired path without waiting.
    """
    raw = os.environ.get("SWISSISOFORM_SCAN_TTL_HOURS")
    if raw is None:
        return DEFAULT_TTL_HOURS
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning("ignoring non-numeric SWISSISOFORM_SCAN_TTL_HOURS=%r", raw)
        return DEFAULT_TTL_HOURS


def budget_bytes() -> int:
    """Total-size ceiling, from ``SWISSISOFORM_SCAN_BUDGET_BYTES``."""
    raw = os.environ.get("SWISSISOFORM_SCAN_BUDGET_BYTES")
    if raw is None:
        return DEFAULT_BUDGET_BYTES
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("ignoring non-numeric SWISSISOFORM_SCAN_BUDGET_BYTES=%r", raw)
        return DEFAULT_BUDGET_BYTES


def _tokens_dir() -> Path:
    return scan_dir() / "tokens"


def _blobs_dir() -> Path:
    return scan_dir() / "blobs"


def blob_dir(key: str) -> Path:
    """Directory holding one content-addressed upload and its digest."""
    return _blobs_dir() / key


def source_path(key: str) -> Path:
    """Path to the stored (gzipped) VCF for ``key``."""
    return blob_dir(key) / SOURCE_NAME


def digest_path(key: str) -> Path:
    """Path to the scan digest for ``key``."""
    return blob_dir(key) / DIGEST_NAME


def make_key(vcf_sha256: str, index_version: str) -> str:
    """Compose the blob key from the file, the index version and the digest schema.

    All three matter. The digest depends on the VCF's bytes *and* on the
    coordinates it was resolved against *and* on the shape this code writes, so
    changing any one of them must produce a different key — otherwise a cached
    digest outlives the thing that made it valid.

    ``index_version`` is not optional in spirit: an empty one means the index was
    unversioned, and such a key must not be treated as safely reusable across
    rebuilds. Callers get ``noindex`` so at least the cache never silently mixes
    results from different catalogues under one name.
    """
    return f"{vcf_sha256[:32]}-{index_version or 'noindex'}-{DIGEST_SCHEMA}"


# ----------------------------------------------------------------------
# Write path
# ----------------------------------------------------------------------


def save(stream: BinaryIO, *, index_version: str, filename: str = "") -> SavedUpload:
    """Store an uploaded VCF and mint a token for it.

    Hashing and compression happen in a single pass over the stream — the hash is
    essentially free next to the write it rides along with — into a staging file
    that is only promoted once the key is known. Already-gzipped input is stored
    verbatim rather than double-compressed.

    Args:
        stream: The uploaded file object (Werkzeug has already spooled anything
            large to disk, so this is not held in memory).
        index_version: Version of the ORF index the scan will use.
        filename: Original filename, for display only.

    Returns:
        A :class:`SavedUpload`; ``was_cached`` is True when a finished digest
        already exists and the caller should skip the parse.
    """
    root = scan_dir()
    staging_dir = root / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    _tokens_dir().mkdir(parents=True, exist_ok=True)
    _blobs_dir().mkdir(parents=True, exist_ok=True)

    staging = staging_dir / f"{secrets.token_hex(8)}.vcf.gz"
    digest = hashlib.sha256()
    try:
        head = stream.read(2)
        already_gzipped = head == b"\x1f\x8b"
        if already_gzipped:
            with staging.open("wb") as out:
                digest.update(head)
                out.write(head)
                while chunk := stream.read(_CHUNK):
                    digest.update(chunk)
                    out.write(chunk)
        else:
            with gzip.open(staging, "wb") as out:
                digest.update(head)
                out.write(head)
                while chunk := stream.read(_CHUNK):
                    digest.update(chunk)
                    out.write(chunk)
    except OSError as exc:
        staging.unlink(missing_ok=True)
        raise ScanStoreError(f"could not write upload: {exc}") from exc

    # The hash is over the *original* bytes, so an upload is recognised whether
    # it arrives plain or gzipped only if the bytes match — a gzip of the same
    # VCF is a different file and gets its own blob. That is the honest answer:
    # we cannot know two differently-compressed streams are equal without
    # decompressing both.
    vcf_sha256 = digest.hexdigest()
    key = make_key(vcf_sha256, index_version)
    target = blob_dir(key)
    was_cached = digest_path(key).is_file()

    # The token is written FIRST, before the blob is promoted. ``sweep()`` in the
    # other worker derives its live set from the token files and deletes any blob
    # not in it, so promoting first leaves a window where a sweep can remove the
    # source this caller is about to scan — an unhandled FileNotFoundError, not the
    # 507 the route is prepared for. Ordering it this way inverts the transient
    # state to "token points at a blob that is not there yet", which ``load()``
    # already reports as missing.
    token = secrets.token_urlsafe(_TOKEN_BYTES)
    _write_json(
        _tokens_dir() / f"{token}.json",
        {
            "key": key,
            "filename": filename,
            "vcf_sha256": vcf_sha256,
            "created_at": _now_iso(),
            "schema": DIGEST_SCHEMA,
        },
    )

    if was_cached:
        staging.unlink(missing_ok=True)
    else:
        target.mkdir(parents=True, exist_ok=True)
        # os.replace is atomic within a filesystem, so a concurrent upload of the
        # same file cannot observe a half-written source.
        os.replace(staging, source_path(key))
    logger.info("scan upload stored token=%s key=%s cached=%s", token, key, was_cached)
    return SavedUpload(token=token, key=key, vcf_sha256=vcf_sha256, was_cached=was_cached)


def write_digest(key: str, digest: dict[str, Any]) -> None:
    """Persist a scan digest beside its blob."""
    blob_dir(key).mkdir(parents=True, exist_ok=True)
    _write_json(digest_path(key), digest)


# ----------------------------------------------------------------------
# Read path
# ----------------------------------------------------------------------


#: Parsed digests held per worker, keyed by the file's identity rather than the
#: token. A digest at the hit cap is ~4 MB of JSON, and ``resolve_scan`` memoises
#: only per request — so every page view re-read and re-parsed it, making page
#: latency a function of the uploader's VCF size on pages that consult the scan for
#: nothing more than a breadcrumb. Deliberately small: the entries are large, and
#: one active scan per worker is the ordinary case.
_DIGEST_CACHE_SIZE = 4


@lru_cache(maxsize=_DIGEST_CACHE_SIZE)
def _read_digest_cached(path_str: str, _mtime: float, _size: int) -> dict[str, Any] | None:
    """Parse a digest, keyed on the file's identity so a rewrite cannot be missed.

    ``mtime`` and ``size`` are arguments purely to take part in the cache key: a
    digest rewritten in place produces a different key and a fresh parse, where
    keying on the path alone would serve the previous contents forever.

    The returned dict is **shared between callers** — :func:`load` shallow-copies it
    before stamping per-token fields, and nothing may mutate the nested hit list.
    """
    return _read_json(Path(path_str))


def _read_digest(path: Path) -> dict[str, Any] | None:
    """The digest at ``path``, from cache when the file has not changed."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return _read_digest_cached(str(path), stat.st_mtime, stat.st_size)


def load(token: str) -> LoadedScan:
    """Resolve a token to its digest, or say why it cannot be resolved.

    Expiry is decided here from the token's ``created_at``, independently of
    whether the sweeper has run — so a container that has been idle for days
    still reports expired rather than serving stale hits.
    """
    if not _is_safe_token(token):
        return LoadedScan(missing=True, token=token)

    pointer = _read_json(_tokens_dir() / f"{token}.json")
    if pointer is None:
        return LoadedScan(missing=True, token=token)

    if _age_seconds(pointer.get("created_at")) > ttl_hours() * 3600:
        return LoadedScan(expired=True, token=token)

    # A token minted against an older digest shape points at a blob whose fields the
    # current templates no longer match. Treat it as expired — "upload again" is
    # correct and honest, where rendering the old shape would be quietly wrong.
    if pointer.get("schema", "") != DIGEST_SCHEMA:
        logger.info(
            "scan token=%s was minted under schema %r, current is %r — treating as expired",
            token,
            pointer.get("schema", ""),
            DIGEST_SCHEMA,
        )
        return LoadedScan(expired=True, token=token)

    key = str(pointer.get("key") or "")
    digest = _read_digest(digest_path(key)) if key else None
    if digest is None:
        # Token survived but the blob is gone — a partial sweep, or a crash
        # between minting the token and writing the digest.
        return LoadedScan(missing=True, token=token)

    digest = dict(digest)
    digest["vcf_id"] = token
    # Assigned, never defaulted: the blob is addressed by content and shared by
    # everyone who uploads the same bytes, so a filename stored in it belongs to
    # whoever got there first. Only the token knows what *this* uploader called
    # their file — the same reason created_at and expires_at are read from here.
    digest["filename"] = pointer.get("filename", "")
    digest["created_at"] = pointer.get("created_at", "")
    digest["expires_at"] = _expiry_iso(pointer.get("created_at"))
    return LoadedScan(digest=digest, token=token)


# ----------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------


def sweep(*, force: bool = False) -> dict[str, int]:
    """Delete expired tokens, then orphaned blobs, then evict over budget.

    Rate-limited to :data:`SWEEP_INTERVAL_SECONDS` by a marker file so the two
    gunicorn workers do not both walk the tree on every upload. Deliberately not
    a background thread: threads would be per-worker, would keep running after a
    worker is recycled, and would race each other for no benefit — a lazy
    idempotent sweep on the write path is simpler and sufficient.

    Returns:
        Counts of what was removed (all zero when the sweep was skipped).
    """
    root = scan_dir()
    if not root.is_dir():
        return {"tokens": 0, "blobs": 0, "evicted": 0, "skipped": 1}

    marker = root / _SWEEP_MARKER
    if not force and marker.exists():
        try:
            if time.time() - marker.stat().st_mtime < SWEEP_INTERVAL_SECONDS:
                return {"tokens": 0, "blobs": 0, "evicted": 0, "skipped": 1}
        except OSError:
            pass

    removed_tokens = 0
    live_keys: set[str] = set()
    max_age = ttl_hours() * 3600

    for pointer_path in _tokens_dir().glob("*.json"):
        pointer = _read_json(pointer_path)
        if pointer is None or _age_seconds(pointer.get("created_at")) > max_age:
            pointer_path.unlink(missing_ok=True)
            removed_tokens += 1
        else:
            live_keys.add(str(pointer.get("key") or ""))

    removed_blobs = 0
    for candidate in _blobs_dir().glob("*"):
        if candidate.is_dir() and candidate.name not in live_keys:
            shutil.rmtree(candidate, ignore_errors=True)
            removed_blobs += 1

    # Abandoned staging files from an interrupted upload.
    staging_dir = root / "staging"
    if staging_dir.is_dir():
        for leftover in staging_dir.glob("*"):
            try:
                if time.time() - leftover.stat().st_mtime > SWEEP_INTERVAL_SECONDS:
                    leftover.unlink(missing_ok=True)
            except OSError:
                pass

    evicted = _evict_over_budget()

    # Imported here, not at module scope: throttle reads scan_dir() from this
    # module, so a top-level import would be circular.
    from swissisoform_site import throttle

    throttle.purge()

    try:
        marker.touch()
    except OSError:
        pass

    if removed_tokens or removed_blobs or evicted:
        logger.info(
            "scan sweep removed tokens=%d blobs=%d evicted=%d",
            removed_tokens,
            removed_blobs,
            evicted,
        )
    return {"tokens": removed_tokens, "blobs": removed_blobs, "evicted": evicted, "skipped": 0}


def _evict_over_budget() -> int:
    """Drop oldest blobs until total size is under budget.

    Tokens pointing at an evicted blob then resolve as missing, which the read
    path already renders as "no longer available" — the same state as expiry.
    """
    limit = budget_bytes()
    if limit <= 0:
        return 0

    sized: list[tuple[float, int, Path]] = []
    total = 0
    for candidate in _blobs_dir().glob("*"):
        if not candidate.is_dir():
            continue
        size = sum(f.stat().st_size for f in candidate.rglob("*") if f.is_file())
        total += size
        try:
            mtime = candidate.stat().st_mtime
        except OSError:
            mtime = 0.0
        sized.append((mtime, size, candidate))

    if total <= limit:
        return 0

    evicted = 0
    for _mtime, size, path in sorted(sized):
        if total <= limit:
            break
        shutil.rmtree(path, ignore_errors=True)
        total -= size
        evicted += 1
    logger.warning("scan store over budget; evicted %d oldest blobs", evicted)
    return evicted


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _is_safe_token(token: str) -> bool:
    """Reject anything that is not a plausible ``token_urlsafe`` value.

    Guards the ``tokens/<token>.json`` join against traversal — a token arrives
    straight from the URL.
    """
    if not token or len(token) > 64:
        return False
    return all(c.isalnum() or c in "-_" for c in token)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_iso(raw: Any) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _age_seconds(created_at: Any) -> float:
    """Seconds since ``created_at``; treats an unparseable stamp as ancient."""
    created = _parse_iso(created_at)
    if created is None:
        return float("inf")
    return (datetime.now(UTC) - created).total_seconds()


def _expiry_iso(created_at: Any) -> str:
    created = _parse_iso(created_at)
    if created is None:
        return ""
    return (
        (created + timedelta(hours=ttl_hours()))
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically so a reader never sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, separators=(",", ":")))
        os.replace(tmp, path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise ScanStoreError(f"could not write {path.name}: {exc}") from exc


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None
