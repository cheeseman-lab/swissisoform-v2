r"""Build ``orf_index.parquet`` — the slim ORF-coordinate index for variant queries.

A variant→ORF lookup needs only the 13 coordinate columns of ``all_paired.parquet``.
Those are **2.43 MB of the full catalogue's 2.09 GB** (0.116%) and write out to
~1.6 MB, so the website can carry a *whole-catalogue* index (3,371 genes) while
still displaying whichever small run is deployed. Indexing off the deployed run
instead would mean every real VCF returns zero hits.

The read must be column-projected: ``all_paired.parquet`` is a single row group,
so an unprojected ``pd.read_parquet`` materialises all 2 GB.

Usage:
    python scripts/export/build_orf_index.py --run full_catalog
    python scripts/export/build_orf_index.py --run cheeseman_test
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


def build(paired_path: Path, out_path: Path) -> tuple[int, str]:
    """Project the coordinate columns out of ``all_paired.parquet`` and write them."""
    available = set(pq.ParquetFile(paired_path).schema_arrow.names)
    missing = [c for c in INDEX_COLUMNS if c not in available]
    if missing:
        raise SystemExit(
            f"{paired_path} is missing required columns: {missing}. "
            "It predates the ORF-interval writer (io/parquet.py) — re-run the pipeline."
        )

    table = pq.read_table(paired_path, columns=list(INDEX_COLUMNS))
    version = compute_index_version(table)

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
    args = ap.parse_args()

    run_dir = args.output_root / args.run
    paired = args.paired or run_dir / "all_paired.parquet"
    out = args.out or run_dir / "orf_index.parquet"

    if not paired.exists():
        raise SystemExit(f"all_paired.parquet not found: {paired}")

    n_rows, version = build(paired, out)
    size_mb = out.stat().st_size / 1e6
    logger.info("wrote %s (%d isoforms, %.2f MB, index_version=%s)", out, n_rows, size_mb, version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
