"""Per-IP upload throttle for the VCF scan endpoint.

The scan is synchronous and CPU-bound, and the deployment runs two sync gunicorn
workers, so two slow uploads occupy the whole service. The caps in
``swissisoform.variantquery.vcf`` and ``scan()`` bound how bad *one* request can
be; this bounds how many a single client can send.

**Why not flask-limiter.** Its default backend is per-process, and with two
workers that silently permits twice the configured limit. There is no Redis here
and no database — the store is the container's ephemeral disk — so a new
dependency would buy a weaker guarantee than the filesystem pattern
``scanstore`` already uses for exactly this problem.

**Counting is one marker file per request, not a counter.** A read-modify-write
counter loses updates under concurrency, which is precisely the condition this
exists to handle. Creating a file per recorded request cannot lose one, and the
count for a window is the number of files whose mtime falls inside it — a true
sliding window, so a client cannot burst 2N across a fixed-window boundary.

**It fails open.** If the disk refuses a write, ``record`` warns and the request
proceeds. Failing closed would turn a disk hiccup into a total outage of the
endpoint, and the per-request caps still bound the damage in the meantime.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

from swissisoform_site.scanstore import scan_dir

logger = logging.getLogger(__name__)

#: Uploads per IP per rolling hour / day. Generous for exploratory use; a single
#: abusive client still cannot hold a worker for more than a few minutes an hour.
DEFAULT_HOURLY = 5
DEFAULT_DAILY = 20

HOUR = 3600.0
DAY = 24 * HOUR

_SALT_FILE = ".throttle_salt"


@dataclass(frozen=True, slots=True)
class Decision:
    """Whether one client may upload now, and when to try again if not."""

    allowed: bool
    retry_after: int = 0
    #: Which window refused it — ``"hourly"`` or ``"daily"``; empty when allowed.
    scope: str = ""


def _env_int(name: str, default: int) -> int:
    """Read a limit from the environment; ``0`` disables that window."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("ignoring non-numeric %s=%r", name, raw)
        return default


def hourly_limit() -> int:
    """Uploads per IP per hour, from ``SWISSISOFORM_SCAN_RATE_HOURLY``."""
    return _env_int("SWISSISOFORM_SCAN_RATE_HOURLY", DEFAULT_HOURLY)


def daily_limit() -> int:
    """Uploads per IP per day, from ``SWISSISOFORM_SCAN_RATE_DAILY``."""
    return _env_int("SWISSISOFORM_SCAN_RATE_DAILY", DEFAULT_DAILY)


def throttle_dir() -> Path:
    """Root of the throttle markers, inside the scan store."""
    return scan_dir() / "throttle"


def _salt() -> str:
    """Per-deployment salt for the IP hash.

    From ``SWISSISOFORM_THROTTLE_SALT`` when set, otherwise a random value
    persisted beside the markers so both workers agree on it. Without a stable
    salt the two workers would hash the same client into different buckets and
    the limit would double.
    """
    configured = os.environ.get("SWISSISOFORM_THROTTLE_SALT")
    if configured:
        return configured

    path = scan_dir() / _SALT_FILE
    existing = _read_salt(path)
    if existing:
        return existing

    # Published in two steps, because the salt must be created exactly once and
    # never overwritten. Write the content to a uniquely-named temp file, then
    # os.link it into place: link is atomic AND refuses to clobber, so the first
    # writer wins and everyone else adopts its value by re-reading.
    #
    # The two obvious alternatives are both wrong here, and both were tried:
    # open(path, "x") creates the file empty and writes after, so a racing reader
    # sees "" and invents its own salt; os.replace overwrites, so a later writer
    # silently moves every earlier caller into a different bucket. Either way the
    # markers scatter across buckets and the limit multiplies by the number of
    # writers — the exact failure this module exists to prevent.
    salt = secrets.token_hex(16)
    tmp = path.with_name(f"{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(salt)
        try:
            os.link(tmp, path)
        except FileExistsError:
            pass  # Another writer got there first; its salt is the real one.
    except OSError as exc:
        logger.warning("could not persist throttle salt: %s", exc)
        return salt
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass

    return _read_salt(path) or salt


def _read_salt(path: Path) -> str:
    """The persisted salt, or ``""`` if it is absent or not yet complete."""
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _bucket(ip: str) -> Path:
    """Directory holding one client's markers.

    The IP is hashed before it touches disk — this service takes genomic uploads
    and the store is already careful about what it writes down. With a random
    persisted salt this is not reversible by inspection of the directory alone.
    """
    digest = hashlib.sha256(f"{_salt()}:{ip}".encode()).hexdigest()[:16]
    return throttle_dir() / digest


def _stamps(bucket: Path, window: float, now: float) -> list[float]:
    """Marker mtimes inside ``window`` seconds of ``now``."""
    try:
        entries = list(bucket.iterdir())
    except OSError:
        return []

    live: list[float] = []
    for entry in entries:
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        if now - mtime <= window:
            live.append(mtime)
    return live


def _retry_after(oldest: float, window: float, now: float) -> int:
    """Seconds until the oldest marker in a full window ages out."""
    return max(1, int(oldest + window - now) + 1)


def check(ip: str) -> Decision:
    """Ask whether ``ip`` may start a scan now.

    Pure query — call :func:`record` separately, and only once the work has
    actually happened, so a request that costs nothing does not spend budget.
    """
    hourly = hourly_limit()
    daily = daily_limit()
    if not hourly and not daily:
        return Decision(allowed=True)

    now = time.time()
    # One listing over the widest window; the hourly set is a subset of it.
    stamps = _stamps(_bucket(ip), DAY, now)

    if daily and len(stamps) >= daily:
        return Decision(False, _retry_after(min(stamps), DAY, now), "daily")

    if hourly:
        recent = [stamp for stamp in stamps if now - stamp <= HOUR]
        if len(recent) >= hourly:
            return Decision(False, _retry_after(min(recent), HOUR, now), "hourly")

    return Decision(allowed=True)


def record(ip: str) -> None:
    """Charge one upload against ``ip``'s budget."""
    if not hourly_limit() and not daily_limit():
        return

    bucket = _bucket(ip)
    try:
        bucket.mkdir(parents=True, exist_ok=True)
        # Timestamp in the name is for reading the directory by eye; the count
        # itself goes by mtime, which survives a copy the name would not.
        (bucket / f"{int(time.time())}-{secrets.token_hex(4)}").touch()
    except OSError as exc:
        logger.warning("could not record a scan rate marker: %s", exc)


def purge(max_age: float = DAY) -> int:
    """Drop markers older than ``max_age`` and any bucket left empty.

    Called from :func:`swissisoform_site.scanstore.sweep`, which already runs on
    the upload path at most once an hour — no new scheduler.

    Returns:
        How many markers were removed.
    """
    root = throttle_dir()
    if not root.is_dir():
        return 0

    now = time.time()
    removed = 0
    for bucket in root.iterdir():
        if not bucket.is_dir():
            continue
        for marker in bucket.iterdir():
            try:
                if now - marker.stat().st_mtime > max_age:
                    marker.unlink(missing_ok=True)
                    removed += 1
            except OSError:
                continue
        try:
            bucket.rmdir()  # only succeeds when it is empty
        except OSError:
            pass
    return removed
