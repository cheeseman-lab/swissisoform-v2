"""The per-IP upload throttle.

Pure unit tests against a ``tmp_path`` throttle dir — no Flask, no staged data, so
these run everywhere the other scan-route tests are skipped.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from swissisoform_site import throttle


@pytest.fixture(autouse=True)
def root(tmp_path: Path, monkeypatch) -> Path:
    """Point the throttle at a fresh directory and pin the limits per test."""
    monkeypatch.setenv("SWISSISOFORM_THROTTLE_DIR", str(tmp_path))
    monkeypatch.setenv("SWISSISOFORM_SCAN_RATE_HOURLY", "3")
    monkeypatch.setenv("SWISSISOFORM_SCAN_RATE_DAILY", "5")
    return tmp_path


def _only_bucket(root: Path) -> Path:
    """The one client bucket; the salt and purge marker are files beside it."""
    (bucket,) = [p for p in root.iterdir() if p.is_dir()]
    return bucket


def test_the_first_upload_is_allowed() -> None:
    assert throttle.check("1.2.3.4").allowed is True


def test_the_hourly_limit_refuses_the_next_one() -> None:
    for _ in range(3):
        assert throttle.check("1.2.3.4").allowed is True
        throttle.record("1.2.3.4")

    verdict = throttle.check("1.2.3.4")
    assert verdict.allowed is False
    assert verdict.scope == "hourly"
    assert 0 < verdict.retry_after <= throttle.HOUR + 1


def test_the_daily_limit_refuses_past_the_hourly_window(root: Path) -> None:
    """Five old markers: outside the hour, inside the day, so daily is what bites."""
    ip = "1.2.3.4"
    throttle.record(ip)
    bucket = _only_bucket(root)
    stale = time.time() - (2 * throttle.HOUR)
    for i in range(4):
        marker = bucket / f"old-{i}"
        marker.touch()
        os.utime(marker, (stale, stale))
    os.utime(next(m for m in bucket.iterdir() if not m.name.startswith("old-")), (stale, stale))

    verdict = throttle.check(ip)
    assert verdict.allowed is False
    assert verdict.scope == "daily"


def test_markers_outside_both_windows_stop_counting(root: Path) -> None:
    ip = "1.2.3.4"
    for _ in range(3):
        throttle.record(ip)
    assert throttle.check(ip).allowed is False

    bucket = _only_bucket(root)
    ancient = time.time() - (2 * throttle.DAY)
    for marker in bucket.iterdir():
        os.utime(marker, (ancient, ancient))

    assert throttle.check(ip).allowed is True


def test_clients_do_not_share_a_budget() -> None:
    for _ in range(3):
        throttle.record("1.2.3.4")

    assert throttle.check("1.2.3.4").allowed is False
    assert throttle.check("5.6.7.8").allowed is True


def test_zero_disables_the_throttle(monkeypatch) -> None:
    monkeypatch.setenv("SWISSISOFORM_SCAN_RATE_HOURLY", "0")
    monkeypatch.setenv("SWISSISOFORM_SCAN_RATE_DAILY", "0")

    for _ in range(50):
        throttle.record("1.2.3.4")
    assert throttle.check("1.2.3.4").allowed is True


def test_a_disabled_throttle_writes_nothing(root: Path, monkeypatch) -> None:
    """Off means off — no directory, no per-request stat cost."""
    monkeypatch.setenv("SWISSISOFORM_SCAN_RATE_HOURLY", "0")
    monkeypatch.setenv("SWISSISOFORM_SCAN_RATE_DAILY", "0")

    throttle.record("1.2.3.4")
    assert not any(root.iterdir())


def test_a_bad_limit_falls_back_instead_of_raising(monkeypatch) -> None:
    monkeypatch.setenv("SWISSISOFORM_SCAN_RATE_HOURLY", "not-a-number")
    assert throttle.hourly_limit() == throttle.DEFAULT_HOURLY


def test_concurrent_records_do_not_lose_a_count(root: Path) -> None:
    """The reason this counts files instead of incrementing an integer."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: throttle.record("1.2.3.4"), range(40)))

    bucket = _only_bucket(root)
    assert len(list(bucket.iterdir())) == 40


def test_the_ip_is_not_written_to_disk_in_the_clear(root: Path) -> None:
    throttle.record("203.0.113.7")

    names = [p.name for p in root.iterdir()]
    assert names and all("203.0.113.7" not in name for name in names)


def test_both_workers_agree_on_the_bucket(root: Path) -> None:
    """A per-process salt would split one client across two buckets, doubling the limit."""
    first = throttle._bucket("1.2.3.4")
    throttle.record("1.2.3.4")
    second = throttle._bucket("1.2.3.4")
    assert first == second

    # A fresh read of the persisted salt (as a second worker would do) agrees.
    salt = (root / throttle._SALT_FILE).read_text().strip()
    assert salt and throttle._salt() == salt


def test_a_second_writer_never_replaces_an_existing_salt(root) -> None:
    """The salt is create-once. Overwriting it silently rebuckets every client.

    os.replace would do exactly that, and open(path, "x") leaves a window where a
    racing reader sees an empty file and invents its own — both scatter the markers
    across buckets and multiply the effective limit.
    """
    first = throttle._salt()
    salt_file = root / throttle._SALT_FILE
    on_disk = salt_file.read_text().strip()

    # A second caller, with the file already present, must adopt it unchanged.
    assert throttle._salt() == first == on_disk
    assert salt_file.read_text().strip() == on_disk


def test_the_salt_write_leaves_no_temp_files_behind(root) -> None:
    throttle._salt()
    leftovers = [p.name for p in root.iterdir() if p.name.endswith(".tmp")]
    assert not leftovers


def test_a_configured_salt_wins(monkeypatch) -> None:
    monkeypatch.setenv("SWISSISOFORM_THROTTLE_SALT", "pepper")
    assert throttle._salt() == "pepper"


def test_purge_drops_only_what_aged_out(root: Path) -> None:
    ip = "1.2.3.4"
    throttle.record(ip)
    bucket = _only_bucket(root)
    stale = bucket / "stale"
    stale.touch()
    ancient = time.time() - (2 * throttle.DAY)
    os.utime(stale, (ancient, ancient))

    assert throttle.purge() == 1
    assert not stale.exists()
    assert len(list(bucket.iterdir())) == 1


def test_purge_removes_a_bucket_it_empties(root: Path) -> None:
    throttle.record("1.2.3.4")
    bucket = _only_bucket(root)
    ancient = time.time() - (2 * throttle.DAY)
    for marker in bucket.iterdir():
        os.utime(marker, (ancient, ancient))

    throttle.purge()
    assert not bucket.exists()


def test_purge_on_a_store_that_never_throttled_is_a_no_op() -> None:
    assert throttle.purge() == 0


def test_record_purges_aged_markers_at_most_hourly(root: Path) -> None:
    """With scanstore gone there is no sweep to call purge; record() does, rate-limited."""
    throttle.record("1.2.3.4")
    bucket = _only_bucket(root)
    stale = bucket / "0-stale"
    stale.touch()
    old = time.time() - throttle.DAY - 60
    os.utime(stale, (old, old))

    # The first record already purged and stamped the marker, so this one skips.
    throttle.record("1.2.3.4")
    assert stale.exists()

    marker = root / throttle._PURGE_MARKER
    expired = time.time() - throttle.PURGE_INTERVAL_SECONDS - 60
    os.utime(marker, (expired, expired))
    throttle.record("1.2.3.4")
    assert not stale.exists()


def _expire_purge_marker(root: Path) -> None:
    marker = root / throttle._PURGE_MARKER
    if marker.exists():
        expired = time.time() - throttle.PURGE_INTERVAL_SECONDS - 60
        os.utime(marker, (expired, expired))


def test_a_bucket_removed_mid_purge_does_not_fail_the_upload(root: Path, monkeypatch) -> None:
    """Another worker's purge can rmdir a bucket after this one's is_dir() check."""
    throttle.record("1.2.3.4")
    _expire_purge_marker(root)
    listing = Path.iterdir

    def vanished(self: Path):
        if self.parent == root:
            raise FileNotFoundError(self)
        return listing(self)

    monkeypatch.setattr(Path, "iterdir", vanished)
    throttle.record("1.2.3.4")  # must not raise
    monkeypatch.setattr(Path, "iterdir", listing)
    assert len(list(_only_bucket(root).iterdir())) == 2


def test_an_upload_still_counts_when_purge_removed_its_bucket(root: Path, monkeypatch) -> None:
    """The bucket can vanish between record()'s mkdir and its touch; it retries once."""
    touch = Path.touch
    failed = []

    def racy_touch(self: Path, *args, **kwargs):
        if self.parent.parent == root and not failed:
            failed.append(self)
            raise FileNotFoundError(self)
        return touch(self, *args, **kwargs)

    monkeypatch.setattr(Path, "touch", racy_touch)
    throttle.record("1.2.3.4")
    assert failed
    assert len(list(_only_bucket(root).iterdir())) == 1


def test_the_purge_slot_is_claimed_before_purging(root: Path, monkeypatch) -> None:
    """A second worker must see the slot taken while the first is still purging."""
    seen = []

    def purge(max_age: float = throttle.DAY) -> int:
        marker = root / throttle._PURGE_MARKER
        seen.append(marker.exists() and time.time() - marker.stat().st_mtime < 5)
        raise OSError("disk went away")  # and a failing purge must not fail the upload

    monkeypatch.setattr(throttle, "purge", purge)
    throttle.record("1.2.3.4")
    assert seen == [True]
