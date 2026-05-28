"""Build per-transcript exon-structure parquet from the upstream GTF.

The output drives the V2 site's transcript graph and gives downstream consumers
(genome LM training, ad-hoc DNA-window extraction) a clean columnar copy of the
transcript structure without re-parsing the GTF.

Coordinates are 0-based half-open plus-strand throughout, matching the
``orf_exons`` convention already used by ``all_paired.parquet``.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

TRANSCRIPT_ID_RE = re.compile(r'transcript_id "([^"]+)"')
GENE_NAME_RE = re.compile(r'gene_name "([^"]+)"')


def _parse_attr(attr: str, regex: re.Pattern[str]) -> str | None:
    m = regex.search(attr)
    return m.group(1) if m else None


def build_skeletons(gtf_path: Path, transcript_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Read GTF and return per-transcript skeletons keyed by transcript_id.

    Each value is ``{chrom, strand, exons, cds_start, cds_end, gene_name}``.
    Transcripts not in ``transcript_ids`` are skipped.
    """
    wanted = set(transcript_ids)
    by_tx: dict[str, dict[str, Any]] = {}

    with open(gtf_path) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, _src, feature, start, end, _score, strand, _frame, attrs = parts
            tx_id = _parse_attr(attrs, TRANSCRIPT_ID_RE)
            if tx_id is None or tx_id not in wanted:
                continue

            entry = by_tx.setdefault(
                tx_id,
                {
                    "chrom": chrom,
                    "strand": strand,
                    "gene_name": _parse_attr(attrs, GENE_NAME_RE),
                    "exons": [],
                    "cds_start": None,
                    "cds_end": None,
                },
            )
            if entry["chrom"] != chrom or entry["strand"] != strand:
                raise ValueError(
                    f"Transcript {tx_id!r} has inconsistent chrom/strand across GTF rows: "
                    f"existing {entry['chrom']}/{entry['strand']} vs new {chrom}/{strand}"
                )
            # Coordinate conversion: GTF is 1-based inclusive → 0-based half-open.
            s, e = int(start) - 1, int(end)
            if feature == "exon":
                entry["exons"].append((s, e))
            elif feature == "CDS":
                entry["cds_start"] = s if entry["cds_start"] is None else min(entry["cds_start"], s)
                entry["cds_end"] = e if entry["cds_end"] is None else max(entry["cds_end"], e)

    for entry in by_tx.values():
        entry["exons"].sort()

    return by_tx


def write_skeletons_parquet(gtf_path: Path, transcript_ids: Iterable[str], out_path: Path) -> int:
    """Build skeletons and write the V2-shaped parquet. Returns row count."""
    skeletons = build_skeletons(gtf_path, transcript_ids)
    rows = []
    for tx_id, entry in sorted(skeletons.items()):
        length_nt = sum(e - s for s, e in entry["exons"])
        cds_start = entry["cds_start"]
        cds_end = entry["cds_end"]
        # length_aa is the spliced CDS / 3 — sum exon ∩ [cds_start, cds_end), NOT the
        # genomic span (cds_end - cds_start), which would include intronic nucleotides
        # and overstates length by 1-2 orders of magnitude on multi-exon transcripts.
        if cds_start is not None and cds_end is not None:
            cds_nt = sum(max(0, min(e, cds_end) - max(s, cds_start)) for s, e in entry["exons"])
            length_aa = cds_nt // 3 if cds_nt > 0 else None
        else:
            length_aa = None
        rows.append(
            {
                "transcript_id": tx_id,
                "gene_name": entry["gene_name"],
                "chrom": entry["chrom"],
                "strand": entry["strand"],
                "exons": [{"start": s, "end": e} for s, e in entry["exons"]],
                "cds_start": cds_start,
                "cds_end": cds_end,
                "length_nt": length_nt,
                "length_aa": length_aa,
            }
        )
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return len(df)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gtf", required=True, type=Path, help="GTF used by the upstream pipeline."
    )
    parser.add_argument(
        "--parquet",
        required=True,
        type=Path,
        help="all_paired.parquet (transcript_id column drives the subset).",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output parquet path.")
    args = parser.parse_args()

    if not args.gtf.exists():
        raise SystemExit(f"GTF not found: {args.gtf}")
    if not args.parquet.exists():
        raise SystemExit(f"Parquet not found: {args.parquet}")
    import pyarrow.parquet as pq

    schema_names = pq.read_schema(args.parquet).names
    if "transcript_id" not in schema_names:
        raise SystemExit(
            f"Parquet at {args.parquet} has no 'transcript_id' column "
            f"(found: {schema_names[:5]}{'...' if len(schema_names) > 5 else ''})"
        )

    df = pd.read_parquet(args.parquet, columns=["transcript_id"])
    transcript_ids = set(df["transcript_id"].dropna().astype(str).tolist())
    n = write_skeletons_parquet(args.gtf, transcript_ids, args.out)
    print(f"Wrote {n} transcript skeletons → {args.out}")


if __name__ == "__main__":
    main()
