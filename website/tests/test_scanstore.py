"""Scan-store lifecycle: addressing, expiry, sweeping, and the traversal guard.

No parquet and no Flask — every test drives ``scanstore`` directly against a tmp
directory, so the disk semantics are pinned independently of the routes.
"""

from __future__ import annotations

import gzip
import io
import json

import pytest

from swissisoform_site import scanstore

VCF_BODY = b"##fileformat=VCFv4.1\n#CHROM\tPOS\tID\tREF\tALT\n17\t100\t.\tG\tA\n"
INDEX_VERSION = "abc123"


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    """Point the store at a tmp dir and restore the default TTL for each test."""
    monkeypatch.setenv("SWISSISOFORM_SCAN_DIR", str(tmp_path / "scans"))
    monkeypatch.delenv("SWISSISOFORM_SCAN_TTL_HOURS", raising=False)
    monkeypatch.delenv("SWISSISOFORM_SCAN_BUDGET_BYTES", raising=False)
    return tmp_path


def _save(body: bytes = VCF_BODY, *, index_version: str = INDEX_VERSION, name: str = "a.vcf"):
    return scanstore.save(io.BytesIO(body), index_version=index_version, filename=name)


# ----------------------------------------------------------------------
# Addressing: content hash vs capability token
# ----------------------------------------------------------------------


def test_same_file_shares_a_blob_but_gets_a_fresh_token() -> None:
    """The hash is the address; the token is the capability. Two uploads, one blob."""
    first = _save()
    second = _save()
    assert first.key == second.key
    assert first.token != second.token
    assert second.vcf_sha256 == first.vcf_sha256


def test_second_upload_is_only_cached_once_a_digest_exists() -> None:
    """``was_cached`` gates skipping the parse, so it must track the digest."""
    first = _save()
    assert first.was_cached is False
    assert _save().was_cached is False, "no digest yet — the parse must still run"

    scanstore.write_digest(first.key, {"counts": {"hits": 1}})
    assert _save().was_cached is True


def test_filename_lives_on_the_token_not_in_the_shared_blob() -> None:
    """Two uploaders, one blob, two filenames — each must get their own back."""
    first = _save(name="patient_A.vcf")
    scanstore.write_digest(first.key, {"counts": {"hits": 1}})
    second = _save(name="patient_B.vcf")

    assert first.key == second.key
    assert scanstore.load(first.token).digest["filename"] == "patient_A.vcf"
    assert scanstore.load(second.token).digest["filename"] == "patient_B.vcf"


def test_a_sweep_during_save_cannot_delete_the_upload(monkeypatch) -> None:
    """Both gunicorn workers share this directory and sweep on their write path.

    ``save`` used to promote the blob before writing the token, so a sweep landing
    in that window saw an unreferenced blob and removed the very source the caller
    was about to scan — an unhandled FileNotFoundError, not the 507 the route is
    prepared for. Writing the token first makes the key live before the blob exists.
    """
    real_replace = scanstore.os.replace

    def replace_with_a_sweep(src, dst):
        # The other worker, landing exactly in the window.
        scanstore.sweep(force=True)
        return real_replace(src, dst)

    monkeypatch.setattr(scanstore.os, "replace", replace_with_a_sweep)
    saved = _save()
    assert scanstore.source_path(saved.key).is_file()


def test_different_index_version_gets_a_different_key() -> None:
    """A digest is only valid for the coordinates it was computed against."""
    a = _save(index_version="v1")
    b = _save(index_version="v2")
    assert a.key != b.key
    assert a.vcf_sha256 == b.vcf_sha256


def test_unversioned_index_is_marked_rather_than_silently_shared() -> None:
    assert f"-noindex-{scanstore.DIGEST_SCHEMA}" in _save(index_version="").key


def test_digest_schema_is_part_of_the_key(monkeypatch) -> None:
    """Changing the digest's shape must invalidate cached digests.

    Without this, renaming a digest field leaves old blobs in place and the new
    template renders them under the new labels — plausible numbers, silently wrong.
    """
    first = _save()
    scanstore.write_digest(first.key, {"counts": {}})
    assert _save().was_cached is True

    monkeypatch.setattr(scanstore, "DIGEST_SCHEMA", "d999")
    reissued = _save()
    assert reissued.key != first.key
    assert reissued.was_cached is False, "a schema bump must force a fresh scan"


def test_token_is_not_derivable_from_the_file() -> None:
    """Guessing a token by hashing a candidate file must be impossible."""
    saved = _save()
    assert saved.vcf_sha256[:16] not in saved.token
    assert len(saved.token) >= 16


# ----------------------------------------------------------------------
# Storage form
# ----------------------------------------------------------------------


def test_plain_upload_is_stored_gzipped() -> None:
    """Uncompressed VCFs are ~4.5x larger; there is no reason to keep them so."""
    saved = _save()
    stored = scanstore.source_path(saved.key)
    assert stored.read_bytes()[:2] == b"\x1f\x8b"
    assert gzip.decompress(stored.read_bytes()) == VCF_BODY


def test_already_gzipped_upload_is_not_double_compressed() -> None:
    packed = gzip.compress(VCF_BODY)
    saved = _save(packed, name="a.vcf.gz")
    assert gzip.decompress(scanstore.source_path(saved.key).read_bytes()) == VCF_BODY


def test_no_staging_files_are_left_behind() -> None:
    saved = _save()
    scanstore.write_digest(saved.key, {"counts": {}})
    _save()  # cached path: staging must be discarded, not orphaned
    staging = scanstore.scan_dir() / "staging"
    assert not staging.exists() or not list(staging.glob("*"))


# ----------------------------------------------------------------------
# Read path
# ----------------------------------------------------------------------


def test_load_returns_the_digest_with_identity_stamped_on() -> None:
    saved = _save(name="zcc10.vcf")
    scanstore.write_digest(saved.key, {"counts": {"hits": 3}})
    loaded = scanstore.load(saved.token)
    assert loaded.ok
    assert loaded.digest["vcf_id"] == saved.token
    assert loaded.digest["filename"] == "zcc10.vcf"
    assert loaded.digest["expires_at"]


def test_unknown_token_is_missing_not_an_error() -> None:
    loaded = scanstore.load("neverminted")
    assert loaded.missing and not loaded.expired


def test_token_without_a_digest_is_missing() -> None:
    """Crash between minting a token and writing the digest must not 500."""
    saved = _save()
    assert scanstore.load(saved.token).missing is True


def test_expiry_is_decided_at_read_time(monkeypatch) -> None:
    """A missed sweep must never serve stale results."""
    saved = _save()
    scanstore.write_digest(saved.key, {"counts": {}})
    assert scanstore.load(saved.token).ok

    monkeypatch.setenv("SWISSISOFORM_SCAN_TTL_HOURS", "0")
    loaded = scanstore.load(saved.token)
    assert loaded.expired and not loaded.ok
    # The files are still on disk — only the clock decided.
    assert scanstore.digest_path(saved.key).is_file()


def test_corrupt_pointer_reads_as_missing() -> None:
    saved = _save()
    (scanstore.scan_dir() / "tokens" / f"{saved.token}.json").write_text("{not json")
    assert scanstore.load(saved.token).missing is True


@pytest.mark.parametrize(
    "token", ["../../etc/passwd", "a/b", "..", "", "x" * 65, "tok en", "tok.json"]
)
def test_unsafe_tokens_are_refused_before_touching_disk(token: str) -> None:
    """The token comes straight off the URL, so the path join must be guarded."""
    assert scanstore.load(token).missing is True


# ----------------------------------------------------------------------
# Sweep and budget
# ----------------------------------------------------------------------


def test_sweep_removes_expired_tokens_and_their_blobs(monkeypatch) -> None:
    saved = _save()
    scanstore.write_digest(saved.key, {"counts": {}})
    monkeypatch.setenv("SWISSISOFORM_SCAN_TTL_HOURS", "0")

    stats = scanstore.sweep(force=True)
    assert stats["tokens"] == 1
    assert stats["blobs"] == 1
    assert not scanstore.blob_dir(saved.key).exists()


def test_sweep_keeps_live_scans() -> None:
    saved = _save()
    scanstore.write_digest(saved.key, {"counts": {}})
    stats = scanstore.sweep(force=True)
    assert stats == {"tokens": 0, "blobs": 0, "evicted": 0, "skipped": 0}
    assert scanstore.load(saved.token).ok


def test_sweep_drops_a_blob_no_token_points_at(monkeypatch) -> None:
    """Two tokens on one blob: the blob survives until the last token expires."""
    first = _save()
    scanstore.write_digest(first.key, {"counts": {}})
    second = _save()
    assert first.key == second.key

    (scanstore.scan_dir() / "tokens" / f"{first.token}.json").unlink()
    assert scanstore.sweep(force=True)["blobs"] == 0, "still referenced by the 2nd token"

    (scanstore.scan_dir() / "tokens" / f"{second.token}.json").unlink()
    assert scanstore.sweep(force=True)["blobs"] == 1


def test_sweep_is_rate_limited_between_calls() -> None:
    """Both gunicorn workers call this on every upload; it must not walk the tree."""
    _save()
    scanstore.sweep(force=True)
    assert scanstore.sweep()["skipped"] == 1


def test_budget_eviction_drops_oldest_blobs(monkeypatch) -> None:
    """Behaviour must not depend on the host's unverifiable disk quota."""
    keys = []
    for i in range(3):
        saved = _save(VCF_BODY + str(i).encode() * 500)
        scanstore.write_digest(saved.key, {"counts": {}})
        keys.append(saved.key)

    monkeypatch.setenv("SWISSISOFORM_SCAN_BUDGET_BYTES", "1")
    assert scanstore.sweep(force=True)["evicted"] >= 1
    assert sum(scanstore.blob_dir(k).exists() for k in keys) < 3


def test_sweep_on_a_missing_directory_is_a_no_op(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SWISSISOFORM_SCAN_DIR", str(tmp_path / "never-created"))
    assert scanstore.sweep(force=True)["skipped"] == 1


# ----------------------------------------------------------------------
# Config parsing
# ----------------------------------------------------------------------


def test_non_numeric_ttl_falls_back_to_the_default(monkeypatch) -> None:
    monkeypatch.setenv("SWISSISOFORM_SCAN_TTL_HOURS", "banana")
    assert scanstore.ttl_hours() == scanstore.DEFAULT_TTL_HOURS


def test_zero_ttl_is_honoured_not_treated_as_unset(monkeypatch) -> None:
    monkeypatch.setenv("SWISSISOFORM_SCAN_TTL_HOURS", "0")
    assert scanstore.ttl_hours() == 0.0


def test_digest_is_written_atomically() -> None:
    """Readers must never see a half-written digest, so no .tmp may survive."""
    saved = _save()
    scanstore.write_digest(saved.key, {"counts": {"hits": 7}})
    blob = scanstore.blob_dir(saved.key)
    assert not list(blob.glob("*.tmp"))
    assert json.loads(scanstore.digest_path(saved.key).read_text())["counts"]["hits"] == 7


def test_a_token_minted_under_an_older_schema_reads_as_expired() -> None:
    """The key alone does not retire old tokens — they keep pointing at old blobs.

    Harmless while a redeploy wipes /tmp, but a live bug the moment the store
    outlives a deploy. So the token carries the schema and a mismatch is expired.
    """
    saved = _save()
    scanstore.write_digest(saved.key, {"counts": {}})
    assert scanstore.load(saved.token).ok

    pointer = scanstore.scan_dir() / "tokens" / f"{saved.token}.json"
    stale = json.loads(pointer.read_text())
    stale["schema"] = "d1"
    pointer.write_text(json.dumps(stale))

    loaded = scanstore.load(saved.token)
    assert loaded.expired is True
    assert not loaded.ok


def test_a_token_with_no_schema_field_reads_as_expired() -> None:
    """Tokens written before the field existed are also retired, not trusted."""
    saved = _save()
    scanstore.write_digest(saved.key, {"counts": {}})
    pointer = scanstore.scan_dir() / "tokens" / f"{saved.token}.json"
    without = {k: v for k, v in json.loads(pointer.read_text()).items() if k != "schema"}
    pointer.write_text(json.dumps(without))
    assert scanstore.load(saved.token).expired is True
