"""Interval index and frame arithmetic on synthetic ORFs.

Synthetic so the membership convention and the minus-strand walk are pinned
without needing a pipeline run; the real-data assertions live in
``test_variantquery_fixture.py``.
"""

from __future__ import annotations

from dataclasses import replace

from swissisoform.variantquery.frame import (
    FRAME_CANONICAL,
    FRAME_ISOFORM,
    REGION_SHARED,
    REGION_UNIQUE,
    coding_offset,
    plotly_x,
    region_for,
    resolve_residue,
)
from swissisoform.variantquery.index import OrfIndex, OrfRecord

# Two exons, plus strand: coding offsets 0..29 then 30..59.
PLUS = OrfRecord(
    gene_name="PLUS",
    tis_id="chr1:101:+:ATG:T1",
    transcript_id="T1",
    chrom="chr1",
    strand="+",
    orf_exons=((100, 130), (200, 230)),
    canonical_orf_exons=((100, 130), (200, 230)),
    unique_intervals=((100, 130),),
    shared_intervals=((200, 230),),
    canonical_len=20,
    isoform_len=20,
    diff_space="isoform",
    orf_type="extended",
)

# Same coordinates, minus strand: mRNA order runs (200,230) first, high→low.
MINUS = OrfRecord(
    gene_name="MINUS",
    tis_id="chr1:230:-:ATG:T2",
    transcript_id="T2",
    chrom="chr1",
    strand="-",
    orf_exons=((100, 130), (200, 230)),
    canonical_orf_exons=((100, 130), (200, 230)),
    unique_intervals=((200, 230),),
    shared_intervals=((100, 130),),
    canonical_len=20,
    isoform_len=20,
    diff_space="isoform",
    orf_type="extended",
)


# ----------------------------------------------------------------------
# Membership convention: start < pos <= end
# ----------------------------------------------------------------------


def test_exclusive_start_inclusive_end() -> None:
    """Intervals are 0-based half-open; VCF positions are 1-based."""
    index = OrfIndex.from_records([PLUS])
    assert index.lookup("chr1", 100) == []  # the exclusive start
    assert index.lookup("chr1", 101) == [PLUS]  # first in-ORF base
    assert index.lookup("chr1", 130) == [PLUS]  # last in-ORF base
    assert index.lookup("chr1", 131) == []


def test_intronic_position_between_exons_is_not_a_hit() -> None:
    index = OrfIndex.from_records([PLUS])
    assert index.lookup("chr1", 165) == []


def test_unknown_chromosome_is_distinguishable_from_no_orf() -> None:
    index = OrfIndex.from_records([PLUS])
    assert index.has_chrom("chr1") is True
    assert index.has_chrom("chr9") is False
    assert index.lookup("chr9", 101) == []


def test_lookup_span_catches_a_deletion_reaching_into_an_exon() -> None:
    """A deletion starting in an intron still overlaps the ORF."""
    index = OrfIndex.from_records([PLUS])
    assert index.lookup_span("chr1", 195, 205) == [PLUS]


def test_lookup_span_returns_each_orf_once() -> None:
    """A span covering two exons of one ORF must not double-report it."""
    index = OrfIndex.from_records([PLUS])
    assert index.lookup_span("chr1", 101, 230) == [PLUS]


def test_overlapping_orfs_both_found() -> None:
    """The scan-back must not stop at the first non-matching interval."""
    long_orf = OrfRecord(
        gene_name="LONG",
        tis_id="chr1:1:+:ATG:T3",
        transcript_id="T3",
        chrom="chr1",
        strand="+",
        orf_exons=((0, 500),),
        canonical_orf_exons=((0, 500),),
        unique_intervals=(),
        shared_intervals=((0, 500),),
        canonical_len=166,
        isoform_len=166,
        diff_space="isoform",
        orf_type="extended",
    )
    index = OrfIndex.from_records([PLUS, long_orf])
    assert {r.tis_id for r in index.lookup("chr1", 110)} == {PLUS.tis_id, long_orf.tis_id}


def test_index_deduplicates_shared_exons_between_the_two_orfs() -> None:
    """PLUS's isoform and canonical exons are identical — 2 intervals, not 4."""
    assert OrfIndex.from_records([PLUS]).n_intervals == 2


# ----------------------------------------------------------------------
# Coding offset / residue
# ----------------------------------------------------------------------


def test_plus_strand_offsets_run_ascending() -> None:
    assert coding_offset(PLUS.orf_exons, "+", 101) == 0
    assert coding_offset(PLUS.orf_exons, "+", 130) == 29
    assert coding_offset(PLUS.orf_exons, "+", 201) == 30  # exon 2 continues the frame
    assert coding_offset(PLUS.orf_exons, "+", 165) is None


def test_minus_strand_offsets_start_at_the_highest_position() -> None:
    """On the minus strand mRNA order is reversed, so offset 0 is the max base."""
    assert coding_offset(MINUS.orf_exons, "-", 230) == 0
    assert coding_offset(MINUS.orf_exons, "-", 201) == 29
    assert coding_offset(MINUS.orf_exons, "-", 130) == 30
    assert coding_offset(MINUS.orf_exons, "-", 101) == 59


def test_residue_is_offset_over_three_and_zero_based() -> None:
    residue, frame, gpos = resolve_residue(PLUS, 104, 104)
    assert (residue, frame, gpos) == (1, FRAME_ISOFORM, 104)


def test_truncation_lost_region_resolves_in_canonical_frame() -> None:
    """The lost N-terminus is absent from the isoform, so only canonical numbering exists."""
    truncation = OrfRecord(
        gene_name="TRUNC",
        tis_id="chr2:200:+:ATG:T4",
        transcript_id="T4",
        chrom="chr2",
        strand="+",
        orf_exons=((200, 260),),
        canonical_orf_exons=((100, 260),),
        unique_intervals=((100, 200),),
        shared_intervals=((200, 260),),
        canonical_len=53,
        isoform_len=20,
        diff_space="canonical",
        orf_type="truncated",
    )
    lost_residue, lost_frame, _ = resolve_residue(truncation, 110, 110)
    assert (lost_residue, lost_frame) == (3, FRAME_CANONICAL)

    kept_residue, kept_frame, _ = resolve_residue(truncation, 210, 210)
    assert (kept_residue, kept_frame) == (3, FRAME_ISOFORM)


def test_span_walks_to_the_first_base_inside_the_orf() -> None:
    """A deletion beginning in an intron reports the first translated base."""
    residue, frame, gpos = resolve_residue(PLUS, 195, 205)
    assert (residue, frame, gpos) == (10, FRAME_ISOFORM, 201)


def test_position_outside_both_orfs_has_no_residue() -> None:
    assert resolve_residue(PLUS, 165, 165) == (None, "", None)


# ----------------------------------------------------------------------
# Region + plotly coordinate
# ----------------------------------------------------------------------


def test_region_reads_the_precomputed_intervals() -> None:
    assert region_for(PLUS, 110, 110) == REGION_UNIQUE
    assert region_for(PLUS, 210, 210) == REGION_SHARED


def test_region_prefers_unique_when_a_span_straddles_the_boundary() -> None:
    straddler = replace(PLUS, unique_intervals=((100, 115),), shared_intervals=((115, 130),))
    assert region_for(straddler, 114, 118) == REGION_UNIQUE


def test_plotly_x_shifts_isoform_frame_only() -> None:
    """Canonical-frame residues are already in the figure's coordinate space."""
    record = replace(PLUS, canonical_len=185, isoform_len=239)
    assert plotly_x(record, 21, FRAME_ISOFORM) == -33
    assert plotly_x(record, 21, FRAME_CANONICAL) == 21
