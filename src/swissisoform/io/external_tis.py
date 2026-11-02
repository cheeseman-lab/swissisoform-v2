"""Ribo-TISH-shaped TIS rows from bare start-codon positions.

Some start sites come from outside our Ribo-TISH calls: a published list, a base-
editing screen, a merged BED of several datasets. They carry a position, a codon,
a type and a transcript, but none of the columns the pipeline is built on: the ORF
span (``GenomePos``), the mRNA offset (``Start``) or the protein (``AASeq``).
This module derives those from the GENCODE skeleton + genome, producing rows in the
same per-sample schema ``run_sample`` emits, so ``runner.run`` takes them through
``restricted_df`` unchanged.

The rows have no read evidence. ``TISCounts`` / ``TISPvalue`` / ``RiboPvalue`` /
``FisherQvalue`` are NaN, so the expression criteria are not evaluable (None), never
failed. Each transcript's canonical is imputed from its GENCODE CDS, as
``run_sample`` does for transcripts Ribo-TISH did not call.

Coordinates: inputs are 1-based inclusive codon ranges (``codon_start <= codon_end``
on the plus strand, whatever the transcript strand), as BED-derived MAF tables write
them. ``GenomePos`` is emitted in Ribo-TISH's convention (``chrom:a-b:strand``, the
TIS at 0-based ``a`` on ``+`` and at 1-based ``b`` on ``-``), which
:func:`swissisoform.assembly._row_to_tis` reads back as ``TIS.position``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from Bio.Seq import Seq

from swissisoform.coords import iter_coding_positions
from swissisoform.models import TranscriptCoordinates
from swissisoform.sourceresolve.mrna import build_transcript_mrna, start_codon_index

logger = logging.getLogger(__name__)

_STOPS = frozenset({"TAA", "TAG", "TGA"})

#: Columns :func:`read_maf_starts` requires (the MAF-like layout of merged BEDs).
MAF_COLUMNS = (
    "Hugo_Symbol",
    "Chromosome",
    "Start_Position",
    "End_Position",
    "Strand",
    "Transcript_ID",
    "Type",
)


@dataclass(frozen=True)
class StartSite:
    """One externally supplied start codon.

    Attributes:
        gene: Gene symbol.
        chrom: Chromosome as the genome FASTA names it (``chr1``).
        strand: ``"+"`` or ``"-"``.
        codon_start: 1-based first plus-strand base of the codon.
        codon_end: 1-based last plus-strand base of the codon (``codon_start + 2``).
        transcript_id: Transcript the source assigned the start to, any version.
        tis_type: Ribo-TISH-style type (``Truncated``, ``Extended``, ``uORF``, …).
        codon: Codon the source reports, checked against the genome when given.
        source: Free-text provenance carried into the row (e.g. a dataset list).
    """

    gene: str
    chrom: str
    strand: str
    codon_start: int
    codon_end: int
    transcript_id: str
    tis_type: str
    codon: str | None = None
    source: str | None = None


def read_maf_starts(path: str | Path) -> list[StartSite]:
    """Read a MAF-like start table (``Hugo_Symbol``, ``Start_Position``, …).

    ``Chromosome`` gains a ``chr`` prefix when it lacks one; ``Strand`` may be
    ``+``/``-`` or ``1``/``-1``. ``Reference_Allele`` (the codon) and
    ``Provenance`` are optional.
    """
    df = pd.read_csv(path, sep="\t", dtype=str)
    missing = [c for c in MAF_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    sites = []
    for r in df.itertuples(index=False):
        chrom = str(r.Chromosome)
        strand = {"1": "+", "-1": "-"}.get(str(r.Strand), str(r.Strand))
        sites.append(
            StartSite(
                gene=str(r.Hugo_Symbol),
                chrom=chrom if chrom.startswith("chr") else f"chr{chrom}",
                strand=strand,
                codon_start=int(r.Start_Position),
                codon_end=int(r.End_Position),
                transcript_id=str(r.Transcript_ID),
                tis_type=str(r.Type),
                codon=_opt(getattr(r, "Reference_Allele", None)),
                source=_opt(getattr(r, "Provenance", None)),
            )
        )
    return sites


def _opt(v: Any) -> str | None:
    return None if v is None or (isinstance(v, float) and pd.isna(v)) or v == "" else str(v)


def resolve_transcript(
    site: StartSite,
    by_base_id: Mapping[str, str],
    mane_by_gene: Mapping[str, str],
    overrides: Mapping[str, str] | None = None,
) -> tuple[str | None, str]:
    """Pick the annotation transcript a start is placed on.

    In order: an explicit per-gene override; the source's transcript at the
    annotation's version (matched on the unversioned ENST); the gene's MANE
    transcript when the source's transcript was retired. Every choice is only a
    candidate: :func:`tis_row_for_start` still rejects it unless the codon lies on its
    exons and reads the expected codon.

    Returns:
        ``(transcript_id or None, how)``, ``how`` one of ``override`` / ``same`` /
        ``version_changed`` / ``retired_to_mane`` / ``unresolved``.
    """
    if overrides and site.gene in overrides:
        return overrides[site.gene], "override"
    tid = by_base_id.get(site.transcript_id.split(".")[0])
    if tid is not None:
        return tid, "same" if tid == site.transcript_id else "version_changed"
    mane = mane_by_gene.get(site.gene)
    if mane is not None:
        return mane, "retired_to_mane"
    return None, "unresolved"


def tis_row_for_start(
    site: StartSite,
    transcript_id: str,
    skeleton: TranscriptCoordinates,
    genome: Any,
    tx_meta: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Build one Ribo-TISH-format row for ``site`` on ``skeleton``.

    Walks the spliced mRNA from the start codon to the first in-frame stop.
    ``AASeq`` holds the residue the start codon encodes (``L`` for CTG) and no stop,
    as Ribo-TISH writes it; assembly installs the initiator Met.

    Returns:
        ``(row, None)``, or ``(None, reason)`` when the codon is not exonic on the
        transcript, does not match ``site.codon``, or no stop precedes the 3' end.
    """
    if site.codon_end - site.codon_start != 2:
        return None, f"codon range {site.codon_start}-{site.codon_end} is not 3 nt"
    # The start codon's first base in mRNA order, in start_codon_index's convention.
    position = site.codon_start - 1 if site.strand == "+" else site.codon_end
    idx = start_codon_index(skeleton, position)
    if idx is None:
        return None, f"start is not on the exons of {transcript_id}"
    mrna = build_transcript_mrna(skeleton, genome)
    codon = mrna[idx : idx + 3]
    if site.codon and codon != site.codon.upper():
        return None, f"codon {codon} on {transcript_id} != source {site.codon}"
    stop = next((i for i in range(idx, len(mrna) - 2, 3) if mrna[i : i + 3] in _STOPS), None)
    if stop is None:
        return None, f"no in-frame stop before the 3' end of {transcript_id}"
    protein = str(Seq(mrna[idx:stop]).translate())
    # 1-based genomic position of every mRNA base, in mRNA order.
    gpos = [g for g, _ in iter_coding_positions(sorted(skeleton.exons), skeleton.strand)]
    first, last = gpos[idx], gpos[stop + 2]
    if skeleton.strand == "+":
        genome_pos = f"{skeleton.chrom}:{first - 1}-{last}:+"
    else:
        genome_pos = f"{skeleton.chrom}:{last - 1}-{first}:-"
    meta = dict(tx_meta or {})
    nan = float("nan")
    row = {
        "Gid": meta.get("gene_id"),
        "Tid": transcript_id,
        "Symbol": site.gene,
        "GeneType": meta.get("gene_type"),
        "GenomePos": genome_pos,
        "StartCodon": codon,
        "Start": float(idx),
        "TisType": site.tis_type,
        "RecatTISType": site.tis_type,
        "AALen": float(len(protein)),
        "AASeq": protein,
        "MANE_Select": bool(meta.get("MANE_Select", False)),
        "transcript_support_level": meta.get("transcript_support_level"),
        "TISCounts": nan,
        "NormTISCounts": nan,
        "TISPvalue": nan,
        "RiboPvalue": nan,
        "FisherQvalue": nan,
        "GeneRNASeqCounts": nan,
        "TotalRNASeqCounts": nan,
        "source_transcript_id": site.transcript_id,
        "source_provenance": site.source,
    }
    return row, None


def build_tis_rows(
    sites: Iterable[StartSite],
    reference: Any,
    genome: Any,
    tx_annotations: pd.DataFrame,
    *,
    overrides: Mapping[str, str] | None = None,
    sample: str = "external",
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Rows for ``sites`` plus an imputed canonical per transcript.

    Args:
        sites: Start codons to place.
        reference: :class:`swissisoform.pipeline.UpstreamReference` (exon skeletons
            and the canonical-imputation tables).
        genome: Indexed genome, e.g. ``pysam.FastaFile``.
        tx_annotations: :func:`swissisoform.io.gtf.load_transcript_annotations`
            output (``transcript_id``, ``gene_id``, ``gene_name``, ``MANE_Select``,
            …), for transcript metadata and the MANE fallback.
        overrides: ``{gene: transcript_id}`` placing a gene's start on a transcript
            other than the source's (e.g. to match a screen's canonical).
        sample: Value of the ``Sample`` column.

    Returns:
        ``(rows, notes)``: ``rows`` in ``run_sample``'s schema, ``Imputed`` marking
        the canonical rows; ``notes`` one dict per site with how it was placed or why
        it was skipped.
    """
    from swissisoform.io.canonical import impute_missing_canonical_starts

    tx = tx_annotations.set_index("transcript_id", drop=False)
    by_base = {t.split(".")[0]: t for t in reference.exon_skeletons}
    mane = tx[tx["MANE_Select"].astype(bool)]
    mane_by_gene = dict(zip(mane["gene_name"], mane["transcript_id"], strict=False))

    rows: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []
    for site in sites:
        tid, how = resolve_transcript(site, by_base, mane_by_gene, overrides)
        note = {
            "gene": site.gene,
            "source_transcript": site.transcript_id,
            "transcript": tid,
            "placement": how,
            "skipped": None,
        }
        skeleton = reference.exon_skeletons.get(tid) if tid else None
        if skeleton is None:
            note["skipped"] = f"no GENCODE skeleton for {tid or site.transcript_id}"
        else:
            meta = tx.loc[tid].to_dict() if tid in tx.index else {}
            row, why = tis_row_for_start(site, tid, skeleton, genome, meta)
            if row is None:
                note["skipped"] = why
            else:
                rows.append(row)
        notes.append(note)
        if note["skipped"]:
            logger.warning(
                "start %s %s skipped: %s", site.gene, site.transcript_id, note["skipped"]
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df, notes
    df["Sample"] = sample
    imputed = impute_missing_canonical_starts(
        df,
        genome_pos=reference.genome_pos,
        utr_lengths=reference.utr_lengths,
        start_codons=reference.start_codons,
        protein_products=reference.protein_products,
    )
    imputed["Sample"] = sample
    df = pd.concat([df.assign(Imputed=False), imputed], ignore_index=True)
    no_canonical = set(df["Tid"]) - set(df.loc[df["RecatTISType"] == "Annotated", "Tid"])
    for note in notes:
        if note["transcript"] in no_canonical and not note["skipped"]:
            note["skipped"] = f"{note['transcript']} has no GENCODE canonical CDS"
    return df[~df["Tid"].isin(no_canonical)].reset_index(drop=True), notes


# ---------------------------------------------------------------------------
# Read evidence from Ribo-TISH predict files
# ---------------------------------------------------------------------------

#: Ribo-TISH ``predict`` columns, in file order. Some outputs are written without a
#: header row (e.g. the Ly 2024 per-replicate HeLa files); those are read by
#: position against this list. Extra trailing columns are ignored.
PREDICT_COLUMNS = (
    "Gid", "Tid", "Symbol", "GeneType", "GenomePos", "StartCodon", "Start", "Stop",
    "TisType", "TISGroup", "TISCounts", "TISPvalue", "RiboPvalue", "RiboPStatus",
    "FisherPvalue", "TISQvalue", "FrameQvalue", "FisherQvalue", "AALen", "Seq", "AASeq",
)  # fmt: skip

#: Per-sample metrics attached as ``{sample}_{metric}``, as in the combined catalog.
EVIDENCE_METRICS = ("TISCounts", "TISPvalue", "RiboPvalue", "FisherQvalue")


def site_key(genome_pos: str) -> tuple[str, str, int]:
    """The genomic start site of a Ribo-TISH ``GenomePos``: ``(chrom, strand, pos)``.

    ``pos`` is the start codon's first base in mRNA order (0-based ``a`` on ``+``,
    1-based ``b`` on ``-``), so every transcript and annotation version that carries
    one start codon gives the same key.
    """
    from swissisoform.io.ribotish import parse_genome_pos

    chrom, a, b, strand = parse_genome_pos(genome_pos)
    return chrom, strand, a if strand == "+" else b


def _num(v: str) -> float:
    try:
        return float(v)
    except ValueError:  # "None" in a Ribo-TISH column
        return float("nan")


def read_predict_sites(
    path: str | Path, keys: set[tuple[str, str, int]]
) -> dict[tuple[str, str, int], dict[str, float]]:
    """One replicate's call at each of ``keys``: the best row over its transcripts.

    Ribo-TISH reports a start once per transcript containing it. ``TISCounts`` is the
    site's count (the max is taken in case transcripts disagree); the p- and
    q-values are the smallest across transcripts, i.e. the transcript Ribo-TISH found
    the start most significant on. Streams the file; only rows at ``keys`` are kept.
    """
    out: dict[tuple[str, str, int], dict[str, float]] = {}
    with open(path) as fh:
        first = fh.readline().rstrip("\n").split("\t")
        if first[0] == "Gid":
            cols = {c: i for i, c in enumerate(first)}
            lines: Iterable[str] = fh
        else:  # headerless: the first line is data
            cols = {c: i for i, c in enumerate(PREDICT_COLUMNS)}
            lines = [("\t".join(first)), *fh]
        for line in lines:
            f = line.rstrip("\n").split("\t")
            try:
                key = site_key(f[cols["GenomePos"]])
            except (ValueError, IndexError):
                continue
            if key not in keys:
                continue
            vals = {m: _num(f[cols[m]]) for m in EVIDENCE_METRICS}
            best = out.get(key)
            if best is None:
                out[key] = vals
                continue
            best["TISCounts"] = max(best["TISCounts"], vals["TISCounts"])
            for m in EVIDENCE_METRICS[1:]:
                if pd.isna(best[m]) or (not pd.isna(vals[m]) and vals[m] < best[m]):
                    best[m] = vals[m]
    return out


def attach_evidence(rows: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Add per-sample Ribo-TISH evidence to ``rows`` as combined-catalog columns.

    ``manifest`` has ``sample`` and ``predict_file``; a sample listed on several rows
    is several replicates. Each row's start is matched by genomic site
    (:func:`site_key`), not by transcript, so calls made against another annotation
    version or another transcript still count. Per sample: present when any
    replicate called the site; ``TISCounts`` summed over the replicates that did;
    p/q the smallest. ``{sample}_NormTISCounts`` stays NaN: without the sample's
    RNA-seq totals it cannot be computed, so initiation efficiency is not evaluable.

    The predict files are taken as given: a filtered call set (only significant
    rows) makes "absent" mean "not called", not "not translated".

    Returns:
        A copy of ``rows`` with ``present_{sample}``, ``{sample}_{metric}``,
        ``samples`` and ``n_samples`` columns (the per-sample-mode ``TISPvalue`` /
        ``RiboPvalue`` / ``FisherQvalue`` columns are dropped, as in the combined
        catalog, so assembly reads the wide ones).
    """
    out = rows.drop(
        columns=[
            c
            for c in ("TISPvalue", "RiboPvalue", "FisherQvalue", "TISCounts", "NormTISCounts")
            if c in rows
        ]
    )
    keys = [site_key(gp) for gp in out["GenomePos"]]
    wanted = set(keys)
    samples = list(dict.fromkeys(manifest["sample"]))
    for sample in samples:
        reps = [
            read_predict_sites(f, wanted)
            for f in manifest.loc[manifest["sample"] == sample, "predict_file"]
        ]
        present, cols = [], {m: [] for m in EVIDENCE_METRICS}
        for key in keys:
            calls = [r[key] for r in reps if key in r]
            present.append(bool(calls))
            cols["TISCounts"].append(
                float(sum(c["TISCounts"] for c in calls)) if calls else float("nan")
            )
            for m in EVIDENCE_METRICS[1:]:
                vals = [c[m] for c in calls if not pd.isna(c[m])]
                cols[m].append(min(vals) if vals else float("nan"))
        out[f"present_{sample}"] = present
        for m, v in cols.items():
            out[f"{sample}_{m}"] = v
        out[f"{sample}_NormTISCounts"] = float("nan")
        logger.info(
            "evidence %s: %d replicate file(s), %d of %d starts called",
            sample,
            len(reps),
            sum(present),
            len(keys),
        )
    present_cols = [f"present_{s}" for s in samples]
    out["samples"] = [
        [s for s, p in zip(samples, flags, strict=True) if p]
        for flags in out[present_cols].itertuples(index=False)
    ]
    out["n_samples"] = out["samples"].map(len)
    return out
