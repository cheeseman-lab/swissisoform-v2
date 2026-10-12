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
    staged_upload,
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
