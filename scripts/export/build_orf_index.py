r"""Build ``orf_index.parquet`` — the ORF coordinates + CDS for variant queries.

A variant→ORF lookup needs only a handful of ``all_paired.parquet``'s columns, which
are 2.43 MB of the full catalogue's 2.09 GB (0.116%). So the website can carry a
*whole-catalogue* index (3,371 genes / 6,462 isoforms) while still displaying
whichever small run is deployed — indexing off the deployed run instead would mean
every real VCF returns zero hits.

The read must be column-projected: ``all_paired.parquet`` is a single row group, so
an unprojected ``pd.read_parquet`` materialises all 2 GB.

The index also carries two things derived here from reference data the container
does not have:

* each ORF's **coding sequence**, read out of the genome — without the codon the
  scan cannot tell missense from synonymous from stop_gained. Measured on
  ``full_catalog``: 4.93 MB total, of which ~3.5 MB is sequence, built in ~77 s.
* ``canonical_x_offset_nt``, the figure's isoform-to-canonical x shift in mRNA
  nucleotides, walked over the GTF's transcript exons. Both drawing paths read it,
  which is what stops the bar and its variant markers from being placed by two
  formulas that disagree for ORFs with no shared region.

Usage:
    python scripts/export/build_orf_index.py --run full_catalog
    python scripts/export/build_orf_index.py --run cheeseman_test
    python scripts/export/build_orf_index.py --run full_catalog --no-cds   # coords only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from swissisoform.coords import start_offset_nt
from swissisoform.variantquery.index import INDEX_COLUMNS, OPTIONAL_COLUMNS
from swissisoform.variantquery.load import VERSION_METADATA_KEY

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "output"
DEFAULT_GENOME = ROOT / "data" / "reference" / "Gencode_v49_GRCh38.primary_assembly.genome.fa"
DEFAULT_GTF = ROOT / "data" / "reference" / "gencode.v49.primary_assembly.annotation.gtf"

logger = logging.getLogger("build_orf_index")


def compute_index_version(table: pa.Table) -> str:
    """Fingerprint the index by its **coordinates**, not its bytes.

    Keyed on sorted ``(tis_id, orf_exons, canonical_orf_exons)`` so a rebuild
    that changes only compression or column order keeps the same version, while
    any change to an ORF boundary produces a new one. Cached scan digests are
    keyed on this, so a stale version would silently report hits against
    isoforms that no longer exist.
    """
    rows = table.select(["tis_id", "orf_exons", "canonical_orf_exons"]).to_pylist()
    payload = sorted(
        (
            str(r["tis_id"]),
            [[int(a), int(b)] for a, b in (r["orf_exons"] or [])],
            [[int(a), int(b)] for a, b in (r["canonical_orf_exons"] or [])],
        )
        for r in rows
    )
    digest = hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode())
    return digest.hexdigest()[:16]


def extract_cds(table: pa.Table, genome_fasta: Path) -> tuple[pa.Table, int]:
    """Add ``orf_cds`` / ``canonical_cds``, read out of the genome per ORF.

    The container cannot do this itself — it has neither the 3 GB FASTA nor pysam —
    so the sequence is extracted once here and shipped as data. Without it the scan
    can still classify by length (frameshift vs in-frame) but not distinguish
    missense from synonymous from stop_gained, which needs the codon.

    Uses ``build_coding_sequence_from_orf``, which fetches **per exon**
    (~108k slices for the full catalogue). Its sibling ``build_coding_sequence``
    fetches per base and would be 20 M calls for the same result.

    Both sequences are length-checked, against **different** proteins:
    ``orf_cds`` against ``isoform_len``, and ``canonical_cds`` against
    ``canonical_per_tid_length`` — ``canonical_orf_exons`` describes that Tid's own
    canonical, not the gene-level representative ``canonical_len`` counts.

    Returns:
        The table with both columns appended, and the number of rows whose
        translated length disagreed with the protein it should encode (should be 0).
    """
    from swissisoform.clinical.validate import ConsequenceValidator

    validator = ConsequenceValidator(genome_fasta=str(genome_fasta))
    wanted = ["tis_id", "chrom", "strand", "orf_exons", "canonical_orf_exons", "isoform_len"]
    if "canonical_per_tid_length" in table.schema.names:
        wanted.append("canonical_per_tid_length")
    else:
        logger.warning(
            "canonical_per_tid_length is absent — canonical_cds cannot be length-checked, "
            "so every canonical-frame residue number rests on an unverified exon walk"
        )
    rows = table.select(wanted).to_pylist()

    orf_cds: list[str] = []
    canonical_cds: list[str] = []
    mismatched = 0
    for row in rows:
        chrom, strand = row["chrom"], row["strand"]
        orf = validator.build_coding_sequence_from_orf(
            [tuple(e) for e in (row["orf_exons"] or [])], strand, chrom
        )
        canon = validator.build_coding_sequence_from_orf(
            [tuple(e) for e in (row["canonical_orf_exons"] or [])], strand, chrom
        )
        # The invariant that makes downstream translation trustworthy: an ORF is a
        # whole number of codons, and exactly as many as the protein has residues.
        # A mismatch means the exon walk and the recorded length disagree, which
        # would silently shift every residue number for that ORF. Checked on the
        # canonical too: a truncation's lost N-terminus is classified against
        # canonical_cds, so an unverified walk there mis-numbers exactly the
        # variants this index exists to place.
        for seq, length_key, column in (
            (orf, "isoform_len", "orf_cds"),
            (canon, "canonical_per_tid_length", "canonical_cds"),
        ):
            expected = row.get(length_key)
            if seq and expected is not None and len(seq) != int(expected) * 3:
                mismatched += 1
                logger.error(
                    "%s: %s is %d nt but %s is %s (expected %d nt)",
                    row["tis_id"],
                    column,
                    len(seq),
                    length_key,
                    expected,
                    int(expected) * 3,
                )
        orf_cds.append(orf)
        canonical_cds.append(canon)

    return (
        table.append_column("orf_cds", pa.array(orf_cds, pa.string())).append_column(
            "canonical_cds", pa.array(canonical_cds, pa.string())
        ),
        mismatched,
    )


def derive_x_offsets(table: pa.Table, gtf_path: Path) -> tuple[pa.Table, int]:
    """Add ``canonical_x_offset_nt`` — the figure's isoform→canonical x shift, in mRNA nt.

    Derived here rather than projected because it is a *rendering* coordinate, not a
    pipeline result: keeping it out of ``all_paired.parquet`` means no pipeline
    re-run to deploy it. Same arrangement as ``orf_cds`` above, which is likewise
    derived from reference data the container does not carry.

    Both drawing paths read this one column — ``variantquery.frame.canonical_x`` for
    uploaded VCF markers, the site's figure adapter for bars, domains and annotated
    variants — replacing the two copies of ``canonical_len - isoform_len`` that
    disagreed for ORFs with no shared region (uORFs, altORFs: 449 of 6,462).

    Two coordinate facts are composed here:

    1. :func:`~swissisoform.coords.start_offset_nt` — mRNA distance from the
       **per-transcript** canonical start codon to this ORF's, walked over the
       transcript's exons so the 5'UTR separating a uORF from the canonical ATG is
       counted and the introns are not.
    2. ``canonical_len - canonical_per_tid_length`` — the figure draws **one**
       canonical bar per gene, the gene-level representative protein, which for
       1,670 of 6,462 ORFs is not the canonical of that ORF's own transcript. Step 1
       is exact in transcript space; this shifts it into the space of the bar
       actually drawn.

    The composition reproduces ``canonical_len - isoform_len`` for 6,006 of the
    6,013 ORFs that share a C-terminus, so the figure is unchanged wherever it was
    already right. The 7 exceptions are selenoprotein / readthrough genes (SELENOT,
    TXNRD1/2, GPATCH4) whose isoform stops early: they have no shared C-terminus to
    align on, and right-alignment was placing them flush against a canonical
    C-terminus they never reach.

    Only ``exons`` is taken from the skeletons — ``build_skeletons`` derives
    ``cds_start`` as a plus-strand ``min()``, which is not the strand-aware start
    codon, and the canonical anchor comes from ``canonical_orf_exons`` anyway.

    Returns:
        The table with the column appended, and the number of rows whose offset
        differs from the old right-alignment shift.
    """
    from swissisoform.site.skeletons import build_skeletons

    wanted = [
        "transcript_id", "strand", "orf_exons", "canonical_orf_exons",
        "canonical_len", "isoform_len",
    ]
    if "canonical_per_tid_length" in table.schema.names:
        wanted.append("canonical_per_tid_length")
    rows = table.select(wanted).to_pylist()
    skeletons = build_skeletons(gtf_path, {r["transcript_id"] for r in rows})

    offsets: list[int | None] = []
    unresolved = 0
    differs = 0
    for row in rows:
        skeleton = skeletons.get(row["transcript_id"])
        offset = None
        if skeleton is not None:
            offset = start_offset_nt(
                skeleton["exons"],
                row["strand"],
                [tuple(e) for e in (row["orf_exons"] or [])],
                [tuple(e) for e in (row["canonical_orf_exons"] or [])],
            )
        if offset is None:
            unresolved += 1
        else:
            gene_len = row.get("canonical_len") or 0
            # Absent column (older parquet) means we cannot tell the two canonicals
            # apart; assuming they agree leaves the offset in transcript space,
            # which is what it already was.
            per_tid_len = row.get("canonical_per_tid_length")
            if per_tid_len is None:
                per_tid_len = gene_len
            offset += 3 * (int(gene_len) - int(per_tid_len))
            if offset != 3 * (int(gene_len) - int(row.get("isoform_len") or 0)):
                differs += 1
        offsets.append(offset)

    if unresolved:
        logger.warning(
            "%d ORFs have no canonical_x_offset_nt (missing skeleton, or a start "
            "outside it) — the figure falls back to right-alignment for those",
            unresolved,
        )
    logger.info(
        "canonical_x_offset_nt: %d of %d rows differ from canonical_len - isoform_len "
        "(the ORFs with no shared C-terminus, which right-alignment cannot place)",
        differs, len(rows),
    )
    return (
        table.append_column("canonical_x_offset_nt", pa.array(offsets, pa.int64())),
        differs,
    )


def build(
    paired_path: Path,
    out_path: Path,
    genome_fasta: Path | None = None,
    gtf_path: Path | None = None,
) -> tuple[int, str]:
    """Project the coordinate columns out of ``all_paired.parquet`` and write them."""
    available = set(pq.ParquetFile(paired_path).schema_arrow.names)
    missing = [c for c in INDEX_COLUMNS if c not in available]
    if missing:
        raise SystemExit(
            f"{paired_path} is missing required columns: {missing}. "
            "It predates the ORF-interval writer (io/parquet.py) — re-run the pipeline."
        )

    # ``canonical_per_tid_length`` is both used here and shipped: derive_x_offsets
    # needs it to express the isoform offset against the gene-level canonical the
    # figure draws, and frame.canonical_x needs it at render time to do the same for
    # a canonical-frame residue, which is numbered against the per-transcript one.
    extra = [c for c in OPTIONAL_COLUMNS if c in available]
    table = pq.read_table(paired_path, columns=list(INDEX_COLUMNS) + extra)
    # Computed before the CDS is attached, so adding sequence does NOT change the
    # version: it fingerprints coordinates, and cached scan digests keyed on it stay
    # valid across this change.
    version = compute_index_version(table)

    if gtf_path is not None:
        table, _differs = derive_x_offsets(table, gtf_path)

    if genome_fasta is not None:
        table, mismatched = extract_cds(table, genome_fasta)
        if mismatched:
            raise SystemExit(
                f"{mismatched} ORFs have a CDS length inconsistent with the protein it "
                "encodes (orf_cds vs isoform_len, or canonical_cds vs "
                "canonical_per_tid_length); refusing to write an index that would "
                "mis-number residues."
            )

    # Replace, not merge: the inherited pandas metadata describes all 533 columns
    # of all_paired.parquet and dwarfs the actual data (217 KB of footer for 4 KB
    # of intervals). Nothing reads this index through pandas.
    table = table.replace_schema_metadata({VERSION_METADATA_KEY: version.encode()})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="zstd")
    return table.num_rows, version


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="full_catalog", help="Pipeline run under data/output/.")
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--paired", type=Path, help="Explicit all_paired.parquet path.")
    ap.add_argument("--out", type=Path, help="Explicit output path.")
    ap.add_argument(
        "--genome",
        type=Path,
        default=DEFAULT_GENOME,
        help=(
            "Genome FASTA. Adds orf_cds/canonical_cds, which the scan needs to tell "
            "missense from synonymous from stop_gained."
        ),
    )
    ap.add_argument(
        "--gtf",
        type=Path,
        default=DEFAULT_GTF,
        help=(
            "GENCODE GTF. Adds start_offset_nt, the figure's isoform-to-canonical x "
            "shift; without it uORF/altORF bars and markers fall back to "
            "right-alignment, which is undefined for them."
        ),
    )
    ap.add_argument(
        "--no-x-offset",
        action="store_true",
        help="Skip canonical_x_offset_nt; the figure then right-aligns everything.",
    )
    ap.add_argument(
        "--no-cds",
        action="store_true",
        help="Skip sequence extraction; consequence then limited to length-based classes.",
    )
    args = ap.parse_args()

    run_dir = args.output_root / args.run
    paired = args.paired or run_dir / "all_paired.parquet"
    out = args.out or run_dir / "orf_index.parquet"

    if not paired.exists():
        raise SystemExit(f"all_paired.parquet not found: {paired}")

    genome: Path | None = None
    if not args.no_cds:
        if not args.genome.exists():
            raise SystemExit(
                f"genome FASTA not found: {args.genome}\n"
                "  Pass --genome, or --no-cds to build a coordinates-only index."
            )
        genome = args.genome

    gtf: Path | None = None
    if not args.no_x_offset:
        if not args.gtf.exists():
            raise SystemExit(
                f"GTF not found: {args.gtf}\n"
                "  Pass --gtf, or --no-x-offset to build without the x offset."
            )
        gtf = args.gtf

    n_rows, version = build(paired, out, genome, gtf)
    size_mb = out.stat().st_size / 1e6
    logger.info(
        "wrote %s (%d isoforms, %.2f MB, index_version=%s, cds=%s, x_offset=%s)",
        out,
        n_rows,
        size_mb,
        version,
        "yes" if genome else "no",
        "yes" if gtf else "no",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
