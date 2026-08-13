r"""Build ``orf_index.parquet`` — the ORF coordinates + CDS for variant queries.

A variant→ORF lookup needs only a handful of ``all_paired.parquet``'s columns, which
are 2.43 MB of the full catalogue's 2.09 GB (0.116%). So the website can carry a
*whole-catalogue* index (3,371 genes / 6,462 isoforms) while still displaying
whichever small run is deployed — indexing off the deployed run instead would mean
every real VCF returns zero hits.

The read must be column-projected: ``all_paired.parquet`` is a single row group, so
an unprojected ``pd.read_parquet`` materialises all 2 GB.

The index also carries each ORF's **coding sequence**, extracted here from the
genome, because the container has neither the 3 GB FASTA nor pysam and cannot tell
missense from synonymous from stop_gained without the codon. Measured on
``full_catalog``: 4.93 MB total, of which ~3.5 MB is sequence, built in ~77 s.

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

from swissisoform.variantquery.index import INDEX_COLUMNS
from swissisoform.variantquery.load import VERSION_METADATA_KEY

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "output"
DEFAULT_GENOME = ROOT / "data" / "reference" / "Gencode_v49_GRCh38.primary_assembly.genome.fa"

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

    Returns:
        The table with both columns appended, and the number of rows whose
        translated length disagreed with ``isoform_len`` (should be 0).
    """
    from swissisoform.clinical.validate import ConsequenceValidator

    validator = ConsequenceValidator(genome_fasta=str(genome_fasta))
    rows = table.select(
        ["tis_id", "chrom", "strand", "orf_exons", "canonical_orf_exons", "isoform_len"]
    ).to_pylist()

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
        # would silently shift every residue number for that isoform.
        expected = row.get("isoform_len")
        if orf and expected is not None and len(orf) != int(expected) * 3:
            mismatched += 1
            logger.error(
                "%s: orf_cds is %d nt but isoform_len is %s (expected %d nt)",
                row["tis_id"],
                len(orf),
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


def build(paired_path: Path, out_path: Path, genome_fasta: Path | None = None) -> tuple[int, str]:
    """Project the coordinate columns out of ``all_paired.parquet`` and write them."""
    available = set(pq.ParquetFile(paired_path).schema_arrow.names)
    missing = [c for c in INDEX_COLUMNS if c not in available]
    if missing:
        raise SystemExit(
            f"{paired_path} is missing required columns: {missing}. "
            "It predates the ORF-interval writer (io/parquet.py) — re-run the pipeline."
        )

    table = pq.read_table(paired_path, columns=list(INDEX_COLUMNS))
    # Computed before the CDS is attached, so adding sequence does NOT change the
    # version: it fingerprints coordinates, and cached scan digests keyed on it stay
    # valid across this change.
    version = compute_index_version(table)

    if genome_fasta is not None:
        table, mismatched = extract_cds(table, genome_fasta)
        if mismatched:
            raise SystemExit(
                f"{mismatched} ORFs have a CDS length inconsistent with isoform_len; "
                "refusing to write an index that would mis-number residues."
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

    n_rows, version = build(paired, out, genome)
    size_mb = out.stat().st_size / 1e6
    logger.info(
        "wrote %s (%d isoforms, %.2f MB, index_version=%s, cds=%s)",
        out,
        n_rows,
        size_mb,
        version,
        "yes" if genome else "no",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
