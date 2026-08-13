"""CLI for local checks: resolve a VCF against a run's ORF index.

    python -m swissisoform.variantquery scan <vcf> --run cheeseman_test

Falls back to building the index straight from ``all_paired.parquet`` when
``orf_index.parquet`` has not been built yet, so the scan is usable before the
export step has run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from swissisoform.variantquery.load import load_index, load_index_from_paired
from swissisoform.variantquery.scan import scan

DEFAULT_OUTPUT_ROOT = Path("data/output")


def _resolve_index(args: argparse.Namespace):
    if args.index:
        return load_index(args.index), Path(args.index)
    run_dir = Path(args.output_root) / args.run
    index_path = run_dir / "orf_index.parquet"
    if index_path.is_file():
        return load_index(index_path), index_path
    paired = run_dir / "all_paired.parquet"
    if not paired.is_file():
        sys.exit(f"neither {index_path} nor {paired} exists — check --run/--output-root")
    print(f"[variantquery] {index_path.name} not built; indexing {paired}", file=sys.stderr)
    return load_index_from_paired(paired), paired


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run one scan, print the funnel and the hits."""
    parser = argparse.ArgumentParser(prog="python -m swissisoform.variantquery")
    sub = parser.add_subparsers(dest="command", required=True)

    scan_parser = sub.add_parser("scan", help="resolve a VCF against an ORF index")
    scan_parser.add_argument("vcf", help="VCF path (plain or gzipped)")
    scan_parser.add_argument("--run", default="cheeseman_test", help="pipeline run name")
    scan_parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    scan_parser.add_argument("--index", help="explicit orf_index.parquet path")
    scan_parser.add_argument(
        "--all-filters",
        action="store_true",
        help="include records whose FILTER is not PASS",
    )
    scan_parser.add_argument("--json", help="write the full result to this path")
    scan_parser.add_argument("--limit", type=int, default=20, help="hits to print (0 for all)")

    args = parser.parse_args(argv)

    index, source = _resolve_index(args)
    print(f"[variantquery] index: {source}  {index!r}", file=sys.stderr)

    result = scan(args.vcf, index, pass_only=not args.all_filters)

    counts = result.counts
    print(
        f"lines={counts.lines} alleles={counts.alleles} "
        f"non_pass={counts.skipped_non_pass} off_contig={counts.off_catalog_contig} "
        f"no_orf={counts.no_orf} hits={counts.hits} genes={counts.genes_hit}"
    )
    if counts.rejected:
        print("rejected:", ", ".join(f"{k}={v}" for k, v in sorted(counts.rejected.items())))

    shown = result.hits if args.limit == 0 else result.hits[: args.limit]
    for hit in shown:
        residue = "-" if hit.residue is None else hit.residue
        print(
            f"  L{hit.line_no}\t{hit.chrom}:{hit.pos} {hit.ref}>{hit.alt}\t"
            f"{hit.gene}\t{hit.orf_type}\t{hit.frame}\tres={residue}\t{hit.region}"
        )
    if args.limit and len(result.hits) > len(shown):
        print(f"  ... {len(result.hits) - len(shown)} more")

    if args.json:
        payload = result.to_dict()
        payload["index_version"] = result.index_version
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"[variantquery] wrote {args.json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
