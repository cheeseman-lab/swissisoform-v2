r"""Run SwissIsoform on externally supplied start codons instead of Ribo-TISH calls.

The input is a MAF-like table of start positions (``Hugo_Symbol``, ``Chromosome``,
``Start_Position``, ``End_Position``, ``Strand``, ``Transcript_ID``, ``Type``,
optionally ``Reference_Allele`` + ``Provenance``), e.g. the merged Ly 2024 / Chen
2020 table a base-editing screen was designed from. ``swissisoform.io.external_tis``
turns each start into a Ribo-TISH-format row (ORF span, mRNA offset, protein) on its
GENCODE transcript, imputes each transcript's canonical, and the rows go through the
normal pipeline via ``RunSpec.restricted_df``. The shared Ribo-TISH combined catalog
is never read or rebuilt.

Read evidence is optional. Without ``--evidence`` the rows carry none, so the
expression criteria (D1/D2, initiation efficiency, TIS p-values) are not evaluable.
``--evidence`` takes a CSV of ``sample,predict_file`` (one row per replicate) naming
Ribo-TISH predict outputs; each start is matched to them by genomic site, not by
transcript, and lands in the same ``present_{sample}`` / ``{sample}_TISCounts`` /
``{sample}_FisherQvalue`` columns the combined catalog uses (see
``external_tis.attach_evidence``). Without matching RNA-seq there is no normalised
count, so initiation efficiency stays not evaluable. Everything derived from the
position, transcript and protein is computed as usual.

Transcript placement, per start: ``--transcript-override GENE=ENST…`` if given;
else the source's transcript at the GTF's version; else, if that transcript is
retired, the gene's MANE transcript. A placement is kept only when the codon lies on
the transcript's exons and reads the source's codon.

Usage:
    # Every start in the table
    python scripts/run_external_starts.py --starts starts.tsv --run-name my_starts

    # Only some of them (CSV with gene, codon_start, codon_end; 1-based inclusive)
    python scripts/run_external_starts.py --starts starts.tsv --select hits.csv \\
        --transcript-override PSMB2=ENST00000373237.4 --run-name BE_nc_hits

    # With Ribo-TISH evidence (CSV: sample,predict_file; replicates share a sample)
    python scripts/run_external_starts.py --starts starts.tsv --select hits.csv \
        --evidence ly2024_evidence.csv --run-name BE_nc_hits

    # Rows only (check placements before a multi-hour run)
    python scripts/run_external_starts.py --starts starts.tsv --select hits.csv \\
        --run-name BE_nc_hits --rows-only

Writes ``data/output/<run-name>/external_tis_rows.csv`` (the built rows) and
``external_tis_placements.csv`` (how each start was placed or why it was skipped),
then the usual run outputs beside them.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from swissisoform import runner  # noqa: E402
from swissisoform.config import ScoringConfig  # noqa: E402
from swissisoform.io.external_tis import (  # noqa: E402
    attach_evidence,
    build_tis_rows,
    read_maf_starts,
)
from swissisoform.io.gtf import load_transcript_annotations  # noqa: E402

logger = logging.getLogger("run_external_starts")


def _overrides(pairs: list[str]) -> dict[str, str]:
    out = {}
    for p in pairs:
        gene, _, tid = p.partition("=")
        if not gene or not tid:
            raise SystemExit(f"--transcript-override expects GENE=TRANSCRIPT, got {p!r}")
        out[gene] = tid
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--starts", type=Path, required=True, help="MAF-like start table (TSV)")
    ap.add_argument(
        "--select",
        type=Path,
        help="CSV of gene,codon_start,codon_end to keep (default: every start)",
    )
    ap.add_argument(
        "--transcript-override",
        action="append",
        default=[],
        metavar="GENE=TID",
        help="Place GENE's start on this transcript (repeatable)",
    )
    ap.add_argument(
        "--evidence",
        type=Path,
        help="CSV of sample,predict_file: Ribo-TISH predict outputs to attach (by site)",
    )
    ap.add_argument(
        "--min-cell-lines",
        type=int,
        default=ScoringConfig().min_cell_lines,
        help="D1 threshold: samples that must call the start (default: the scoring default)",
    )
    ap.add_argument("--run-name", required=True, help="Output subdirectory under data/output/")
    ap.add_argument(
        "--rows-only",
        action="store_true",
        help="Write the rows + placements and stop before annotation",
    )
    ap.add_argument("--skip-modules", default="", help="Comma-separated modules to skip")
    ap.add_argument("--tag-registry", default=runner.TAG_REGISTRY_DEFAULT)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    import pysam

    sites = read_maf_starts(args.starts)
    if args.select:
        sel = pd.read_csv(args.select)
        keys = set(
            zip(
                sel["gene"],
                sel["codon_start"].astype(int),
                sel["codon_end"].astype(int),
                strict=True,
            )
        )
        sites = [s for s in sites if (s.gene, s.codon_start, s.codon_end) in keys]
        found = {(s.gene, s.codon_start, s.codon_end) for s in sites}
        for k in sorted(keys - found):
            logger.warning("selected start %s not in %s", k, args.starts)
    if not sites:
        logger.error("no starts to run")
        return 1

    ref = runner.UpstreamReference.load(
        gtf_path=runner.GTF, genome_fasta=runner.GENOME, protein_fasta=runner.PROTEIN
    )
    tx = load_transcript_annotations(str(runner.GTF))
    rows, notes = build_tis_rows(
        sites,
        ref,
        pysam.FastaFile(str(runner.GENOME)),
        tx,
        overrides=_overrides(args.transcript_override),
    )
    samples = ["external"]
    if args.evidence and not rows.empty:
        manifest = pd.read_csv(args.evidence)
        missing = {"sample", "predict_file"} - set(manifest.columns)
        if missing:
            raise SystemExit(f"{args.evidence}: missing columns {sorted(missing)}")
        rows = attach_evidence(rows, manifest)
        samples = list(dict.fromkeys(manifest["sample"]))
    out = runner.OUT / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(notes).to_csv(out / "external_tis_placements.csv", index=False)
    if rows.empty:
        logger.error("no start could be placed; see %s", out / "external_tis_placements.csv")
        return 1
    rows.to_csv(out / "external_tis_rows.csv", index=False)
    n_hit = int((~rows["Imputed"]).sum())
    print(
        f"{n_hit} starts placed (+{int(rows['Imputed'].sum())} imputed canonicals) of "
        f"{len(sites)}; placements: {out / 'external_tis_placements.csv'}"
    )
    if args.rows_only:
        return 0

    spec = runner.RunSpec(
        gene_names=sorted(rows["Symbol"].unique()),
        restricted_df=rows,
        cell_lines=samples,
        single_sample=False,
        min_cell_lines=args.min_cell_lines,
        skip={m.strip() for m in args.skip_modules.split(",") if m.strip()},
        run_name=args.run_name,
        # Per-run FASTA, never the shared data/cache/proteins.fa the GPU jobs read.
        fasta_out=out / "proteins.fa",
        spot_check_limit=None,
        tag_registry_version=args.tag_registry,
    )
    return runner.run(spec)


if __name__ == "__main__":
    sys.exit(main())
