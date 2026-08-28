"""The per-IP upload throttle.

Pure unit tests against a ``tmp_path`` scan dir — no Flask, no staged data, so
these run everywhere the other scan-route tests are skipped.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from swissisoform_site import throttle


@pytest.fixture(autouse=True)
def scan_dir(tmp_path: Path, monkeypatch) -> Path:
    """Point the store at a fresh directory and pin the limits per test."""
    monkeypatch.setenv("SWISSISOFORM_SCAN_DIR", str(tmp_path))
    monkeypatch.setenv("SWISSISOFORM_SCAN_RATE_HOURLY", "3")
    monkeypatch.setenv("SWISSISOFORM_SCAN_RATE_DAILY", "5")
    return tmp_path


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


def test_the_daily_limit_refuses_past_the_hourly_window(scan_dir: Path) -> None:
    """Five old markers: outside the hour, inside the day, so daily is what bites."""
    ip = "1.2.3.4"
    throttle.record(ip)
    bucket = next((scan_dir / "throttle").iterdir())
    stale = time.time() - (2 * throttle.HOUR)
    for i in range(4):
        marker = bucket / f"old-{i}"
        marker.touch()
        os.utime(marker, (stale, stale))
    os.utime(next(m for m in bucket.iterdir() if not m.name.startswith("old-")), (stale, stale))

    verdict = throttle.check(ip)
    assert verdict.allowed is False
    assert verdict.scope == "daily"


def test_markers_outside_both_windows_stop_counting(scan_dir: Path) -> None:
    ip = "1.2.3.4"
    for _ in range(3):
        throttle.record(ip)
    assert throttle.check(ip).allowed is False

    bucket = next((scan_dir / "throttle").iterdir())
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


def test_a_disabled_throttle_writes_nothing(scan_dir: Path, monkeypatch) -> None:
    """Off means off — no directory, no per-request stat cost."""
    monkeypatch.setenv("SWISSISOFORM_SCAN_RATE_HOURLY", "0")
    monkeypatch.setenv("SWISSISOFORM_SCAN_RATE_DAILY", "0")

    throttle.record("1.2.3.4")
    assert not (scan_dir / "throttle").exists()


def test_a_bad_limit_falls_back_instead_of_raising(monkeypatch) -> None:
    monkeypatch.setenv("SWISSISOFORM_SCAN_RATE_HOURLY", "not-a-number")
    assert throttle.hourly_limit() == throttle.DEFAULT_HOURLY


def test_concurrent_records_do_not_lose_a_count(scan_dir: Path) -> None:
    """The reason this counts files instead of incrementing an integer."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: throttle.record("1.2.3.4"), range(40)))

    bucket = next((scan_dir / "throttle").iterdir())
    assert len(list(bucket.iterdir())) == 40


def test_the_ip_is_not_written_to_disk_in_the_clear(scan_dir: Path) -> None:
    throttle.record("203.0.113.7")

    names = [p.name for p in (scan_dir / "throttle").iterdir()]
    assert names and all("203.0.113.7" not in name for name in names)


def test_both_workers_agree_on_the_bucket(scan_dir: Path) -> None:
    """A per-process salt would split one client across two buckets, doubling the limit."""
    first = throttle._bucket("1.2.3.4")
    throttle.record("1.2.3.4")
    second = throttle._bucket("1.2.3.4")
    assert first == second

    # A fresh read of the persisted salt (as a second worker would do) agrees.
    salt = (scan_dir / throttle._SALT_FILE).read_text().strip()
    assert salt and throttle._salt() == salt


def test_a_second_writer_never_replaces_an_existing_salt(scan_dir) -> None:
    """The salt is create-once. Overwriting it silently rebuckets every client.

    os.replace would do exactly that, and open(path, "x") leaves a window where a
    racing reader sees an empty file and invents its own — both scatter the markers
    across buckets and multiply the effective limit.
    """
    first = throttle._salt()
    salt_file = scan_dir / throttle._SALT_FILE
    on_disk = salt_file.read_text().strip()

    # A second caller, with the file already present, must adopt it unchanged.
    assert throttle._salt() == first == on_disk
    assert salt_file.read_text().strip() == on_disk


def test_the_salt_write_leaves_no_temp_files_behind(scan_dir) -> None:
    throttle._salt()
    leftovers = [p.name for p in scan_dir.iterdir() if p.name.endswith(".tmp")]
    assert not leftovers


def test_a_configured_salt_wins(monkeypatch) -> None:
    monkeypatch.setenv("SWISSISOFORM_THROTTLE_SALT", "pepper")
    assert throttle._salt() == "pepper"


def test_purge_drops_only_what_aged_out(scan_dir: Path) -> None:
    ip = "1.2.3.4"
    throttle.record(ip)
    bucket = next((scan_dir / "throttle").iterdir())
    stale = bucket / "stale"
    stale.touch()
    ancient = time.time() - (2 * throttle.DAY)
    os.utime(stale, (ancient, ancient))

    assert throttle.purge() == 1
    assert not stale.exists()
    assert len(list(bucket.iterdir())) == 1


def test_purge_removes_a_bucket_it_empties(scan_dir: Path) -> None:
    throttle.record("1.2.3.4")
    bucket = next((scan_dir / "throttle").iterdir())
    ancient = time.time() - (2 * throttle.DAY)
    for marker in bucket.iterdir():
        os.utime(marker, (ancient, ancient))

    throttle.purge()
    assert not bucket.exists()


def test_purge_on_a_store_that_never_throttled_is_a_no_op() -> None:
    assert throttle.purge() == 0
