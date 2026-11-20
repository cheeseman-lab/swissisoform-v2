"""The scan response builder: staging, hit enrichment, derived counts.

Pure unit tests — no Flask and no staged run — so they run everywhere.
"""

from __future__ import annotations

import gzip
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from swissisoform_site.scanview import (
    StagingError,
    derived_counts,
    enrich_hits,
    hgvsp_from_hit,
    scan_file_json,
    size_refusal,
    staged_upload,
    tis_slug,
    with_frame,
)

VCF = b"##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"


@pytest.mark.parametrize("payload", [VCF, gzip.compress(VCF)], ids=["plain", "gzipped"])
def test_staged_copy_is_byte_identical_and_hashed(tmp_path: Path, payload: bytes) -> None:
    with staged_upload(io.BytesIO(payload), directory=tmp_path) as staged:
        assert staged.path.read_bytes() == payload
        assert staged.vcf_sha256 == hashlib.sha256(payload).hexdigest()
        assert staged.path.stat().st_mode & 0o077 == 0  # private to this user
    assert not staged.path.exists()


def test_staged_file_is_removed_when_the_block_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        with staged_upload(io.BytesIO(VCF), directory=tmp_path):
            raise RuntimeError("scan failed")
    assert list(tmp_path.iterdir()) == []


def test_a_failed_copy_is_a_staging_error(tmp_path: Path) -> None:
    class Broken(io.BytesIO):
        def read(self, *args):
            raise OSError("disk gone")

    with pytest.raises(StagingError):
        with staged_upload(Broken(), directory=tmp_path):
            pytest.fail("the block must not run")
    assert list(tmp_path.iterdir()) == []


def test_an_unusable_temp_dir_is_a_staging_error(tmp_path: Path) -> None:
    """mkstemp failing (missing, read-only or full dir) is reported like a failed copy."""
    with pytest.raises(StagingError):
        with staged_upload(io.BytesIO(VCF), directory=tmp_path / "missing"):
            pytest.fail("the block must not run")


def test_a_file_too_large_is_refused_before_it_is_copied(tmp_path: Path, monkeypatch) -> None:
    """Same error the scan gives after reading past the cap, decided from the size alone."""
    import json

    from swissisoform.variantquery.vcf import VcfLimitExceeded, iter_lines

    monkeypatch.setenv("SWISSISOFORM_VCF_MAX_BYTES", "2000000")
    assert size_refusal(2_000_000) == ""
    refusal = json.loads(size_refusal(2_000_001))
    assert refusal["error"] == "expands_too_large"

    big = tmp_path / "big.vcf"
    big.write_bytes(b"x\n" * 1_000_001)
    with pytest.raises(VcfLimitExceeded) as raised:
        list(iter_lines(big))
    assert refusal["message"] == str(raised.value)


class _Index:
    def __init__(self, records: dict[str, SimpleNamespace]):
        self._records = records

    def by_tis_id(self, tis_id: str):
        return self._records.get(tis_id)


#: An isoform-frame ORF that starts 30 nt (10 residues) downstream of the canonical.
_RECORD = SimpleNamespace(
    canonical_len=100,
    isoform_len=90,
    canonical_per_tid_length=100,
    canonical_x_offset_nt=30,
)


def _hit(**overrides):
    hit = {
        "line_no": 3,
        "chrom": "chr1",
        "pos": 1000,
        "ref": "C",
        "alt": "T",
        "vclass": "snv",
        "gene": "G",
        "tis_id": "T1",
        "transcript_id": "ENST1",
        "orf_type": "truncated",
        "frame": "isoform",
        "residue": 4,
        "region": "shared",
        "consequence": "missense_variant",
        "aa_ref": "P",
        "aa_alt": "L",
        "consequence_note": "",
    }
    hit.update(overrides)
    return hit


def test_enrich_adds_x_and_protein_change_and_drops_unread_fields() -> None:
    (row,) = enrich_hits([_hit()], _Index({"T1": _RECORD}))
    assert row["x"] == 4 + 10
    assert row["protein_change"] == "p.Pro5Leu · isoform frame"
    assert "vclass" not in row and "transcript_id" not in row
    assert row["line_no"] == 3 and row["region"] == "shared"


def test_hits_carry_the_isoform_slug_the_site_routes_on() -> None:
    from swissisoform_site import data

    tis_id = "chr17:48071500:+:CTG:ENST00000225603.9"
    (row,) = enrich_hits([_hit(tis_id=tis_id)], _Index({}))
    assert row["tis_slug"] == "chr17-48071500-+-CTG-ENST00000225603-9"
    assert row["tis_slug"] == data.tis_slug(tis_id)


def test_the_slug_rule_matches_the_llm_dispatchers() -> None:
    """Isoform pages read LLM output from directories named by that copy of the rule."""
    _tis_slug = pytest.importorskip("swissisoform.site.llm")._tis_slug

    for tis_id in ("chr1:10:-:ATG:ENST1.2", "chrX:5:+:CTG:ENST9", "", None):
        assert tis_slug(tis_id) == _tis_slug(tis_id)


def test_a_hit_without_a_residue_is_kept_but_unplaced() -> None:
    (row,) = enrich_hits([_hit(residue=None)], _Index({"T1": _RECORD}))
    assert row["x"] is None
    assert row["protein_change"] == ""


def test_a_hit_whose_orf_left_the_index_has_no_x() -> None:
    (row,) = enrich_hits([_hit()], _Index({}))
    assert row["x"] is None
    assert row["protein_change"] == "p.Pro5Leu · isoform frame"


def test_an_indel_says_why_it_has_no_amino_acids() -> None:
    hit = _hit(aa_ref="", aa_alt="", consequence="frameshift_variant", consequence_note="1 nt")
    (row,) = enrich_hits([hit], _Index({"T1": _RECORD}))
    assert row["protein_change"] == (
        "residue 5 — no amino-acid change resolved (1 nt) · isoform frame"
    )


def test_hgvsp_shapes() -> None:
    assert hgvsp_from_hit("R", "E", 252, "missense_variant") == "p.Arg253Glu"
    assert hgvsp_from_hit("R", "R", 252, "synonymous_variant") == "p.Arg253="
    assert hgvsp_from_hit("R", "*", 252, "stop_gained") == "p.Arg253Ter"
    assert hgvsp_from_hit("L", "V", 0, "start_lost") == "p.Leu1?"
    assert hgvsp_from_hit("FE", "SK", 224, "missense_variant") == "p.Phe225_Glu226delinsSerLys"
    assert hgvsp_from_hit("", "", 3, "frameshift_variant") == ""


def test_with_frame() -> None:
    assert with_frame("", "canonical") == ""
    assert with_frame(None, "canonical") == ""
    assert with_frame("p.Met1Val", "") == "p.Met1Val"
    assert with_frame("p.Met1Val", "isoform") == "p.Met1Val · isoform frame"


def test_derived_funnel_is_monotonic_and_clamped() -> None:
    counts = {
        "alleles": 10,
        "skipped_non_pass": 2,
        "off_catalog_contig": 1,
        "no_orf": 3,
        "consequences": {"missense_variant": 8, "reference_mismatch": 12},
    }
    derived = derived_counts(counts)
    assert derived["passing"] == 8
    assert derived["in_orf_alleles"] == 4
    assert derived["assembly"]["suspect"] is True
    assert derived_counts({})["in_orf_alleles"] == 0


def test_a_refused_file_comes_back_in_the_server_routes_error_shape(tmp_path, monkeypatch) -> None:
    """The browser path reports a bad file exactly as POST /api/variants/scan does."""
    import json

    from swissisoform.variantquery.index import OrfIndex

    monkeypatch.setenv("SWISSISOFORM_VCF_MAX_LINE_BYTES", "64")
    vcf = tmp_path / "oneline.vcf"
    vcf.write_bytes(b"A" * 4096)
    body = json.loads(scan_file_json(vcf, OrfIndex.from_records([])))
    assert body["error"] == "line_too_long"
    assert body["message"]
