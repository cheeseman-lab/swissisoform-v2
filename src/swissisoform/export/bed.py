"""BED12 export of differential ORF regions for variant intersection.

For every alternative TIS, emit a BED12 record where blocks are the genomic
exon intervals covering the isoform-unique region (extended N-terminus for
Extended/uORFs, removed N-terminus for Truncations). Strand-aware, intron-aware.

Output columns (UCSC BED12):
  chrom, chromStart, chromEnd, name, score, strand,
  thickStart, thickEnd, itemRgb, blockCount, blockSizes, blockStarts

name = "{gene}|{orf_type}|{tis_id}|{cell_lines_csv}"
itemRgb encodes orf_type:
  Extended  = 0,128,0      (green)
  uORF      = 0,0,255      (blue)
  Truncated = 255,128,0    (orange)
  uoORF     = 128,0,128    (purple)
  other     = 128,128,128  (gray)
"""
from __future__ import annotations

ORF_COLORS = {
    "extended": "0,128,0",
    "uORF": "0,0,255",
    "truncated": "255,128,0",
    "uoORF": "128,0,128",
}


def diff_exons(
    orf_exons: list[tuple[int, int]],
    canonical_exons: list[tuple[int, int]],
    strand: str,
    orf_type: str,
) -> list[tuple[int, int]]:
    """Compute exon intervals in the isoform-unique region.

    All inputs are plus-strand 0-based half-open [start, end) intervals.

    Truncated isoforms lose the N-terminal portion (in genomic terms,
    the 5-prime flank on +strand or 3-prime flank on -strand of canonical).
    Extended / uORF isoforms gain N-terminal residues.

    Returns plus-strand half-open intervals covering the differential
    region.
    """
    if not orf_exons or not canonical_exons:
        return []
    orf_set = sorted(orf_exons)
    can_set = sorted(canonical_exons)
    # Simple set-difference at base resolution. orf_exons can include or
    # extend beyond canonical, or be a subset.
    orf_bases: set[int] = set()
    for s, e in orf_set:
        orf_bases.update(range(s, e))
    can_bases: set[int] = set()
    for s, e in can_set:
        can_bases.update(range(s, e))

    # The unique region is the symmetric difference: bases in isoform
    # that aren\'t in canonical, OR bases in canonical that aren\'t in
    # isoform. For truncations, the second set is what we want
    # (canonical-only N-terminus that the isoform lacks).
    iso_only = orf_bases - can_bases
    can_only = can_bases - orf_bases

    if orf_type.lower() in ("extended", "uorf", "uoorf"):
        diff = iso_only
    elif orf_type.lower() == "truncated":
        diff = can_only
    else:
        # internal/3utr — show whichever is non-empty
        diff = iso_only or can_only

    if not diff:
        return []

    # Convert base set back to contiguous intervals
    sorted_bases = sorted(diff)
    intervals: list[tuple[int, int]] = []
    run_start = sorted_bases[0]
    prev = sorted_bases[0]
    for b in sorted_bases[1:]:
        if b == prev + 1:
            prev = b
            continue
        intervals.append((run_start, prev + 1))
        run_start = b
        prev = b
    intervals.append((run_start, prev + 1))
    return intervals


def gene_bed_rows(gene, cell_lines: list[str]) -> list[str]:
    """Return one BED12 line per alternative TIS in *gene*."""
    rows = []
    for site in gene.tis_sites:
        if site.orf_type.value.lower() == "annotated":
            continue
        chrom, pos_strand = site.tis_id.split(":", 1)
        # tis_id format: "chr:pos:strand:codon:Tid"
        parts = site.tis_id.split(":")
        if len(parts) < 5:
            continue
        chrom, _, strand, codon, tid = parts[0], parts[1], parts[2], parts[3], ":".join(parts[4:])

        ot_str = site.orf_type.value if hasattr(site.orf_type, 'value') else str(site.orf_type)
        diff = diff_exons(
            site.orf_exons,
            site.canonical_orf_exons,
            strand,
            ot_str,
        )
        if not diff:
            continue

        diff.sort()
        chrom_start = diff[0][0]
        chrom_end = diff[-1][1]
        block_count = len(diff)
        block_sizes = ",".join(str(e - s) for s, e in diff) + ","
        block_starts = ",".join(str(s - chrom_start) for s, _ in diff) + ","

        active = [
            cl for cl in cell_lines
            if cl in site.expression
        ]
        active_csv = ",".join(active) if active else "?"
        name = f"{gene.gene_name}|{ot_str}|{tid}|{codon}|{active_csv}"
        # score field gets clamped 0-1000; use 1000
        score = 1000
        rgb = ORF_COLORS.get(ot_str.lower(), "128,128,128")
        rows.append(
            "\t".join([
                chrom,
                str(chrom_start),
                str(chrom_end),
                name,
                str(score),
                strand,
                str(chrom_start),
                str(chrom_end),
                rgb,
                str(block_count),
                block_sizes,
                block_starts,
            ])
        )
    return rows
