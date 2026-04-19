"""Canonical TIS imputation and reference-sequence helpers.

When Ribo-TISH fails to emit an ``Annotated`` row for a transcript
(common for non-MANE transcripts, or short/UTR-less CDS), every
downstream comparison against that transcript's canonical silently
uses the wrong reference. This module rebuilds the missing canonical
rows from GTF CDS/UTR/start_codon features and the GENCODE protein
FASTA, matching the upstream reference's ``impute_missing_canonical_starts``.

The public entry point is :func:`impute_missing_canonical_starts`;
the helpers are exported so callers can cache them across samples.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from swissisoform.io.gtf import load_transcript_annotations

logger = logging.getLogger(__name__)


def get_canonical_genome_positions(cds_df: pd.DataFrame) -> pd.DataFrame:
    """Compute one ``GenomePos`` string per transcript from its CDS rows.

    The outermost CDS start/end define the canonical CDS span. Returned
    as a DataFrame with columns ``Tid``, ``chromosome``, ``start``,
    ``end``, ``strand``, ``GenomePos``.
    """
    sorted_cds = cds_df.sort_values(["transcript_id", "start"])
    first = sorted_cds.drop_duplicates(subset=["transcript_id"])
    starts = sorted_cds.groupby("transcript_id")["start"].first()
    ends = sorted_cds.groupby("transcript_id")["end"].last()
    out = pd.concat(
        [
            first.set_index("transcript_id")["chromosome"],
            starts,
            ends,
            first.set_index("transcript_id")["strand"],
        ],
        axis=1,
    )
    out["GenomePos"] = out.apply(
        lambda r: f"{r['chromosome']}:{r['start']}-{r['end']}:{r['strand']}",
        axis=1,
    )
    out.index.name = "Tid"
    return out.reset_index()


def get_utr_lengths(cds_df: pd.DataFrame, utr_df: pd.DataFrame) -> pd.DataFrame:
    """Total 5'-UTR length per transcript.

    This equals the canonical ``Start`` offset (bases between the
    spliced-transcript 5' end and the first CDS base).
    """
    fwd_idx = (
        cds_df[cds_df["strand"] == "+"]
        .reset_index()
        .sort_values(["transcript_id", "start"])
        .groupby("transcript_id")
        .first()["index"]
    )
    rev_idx = (
        cds_df[cds_df["strand"] == "-"]
        .reset_index()
        .sort_values(["transcript_id", "start"], ascending=[False, True])
        .groupby("transcript_id")
        .first()["index"]
    )
    first_cds = cds_df.loc[list(fwd_idx) + list(rev_idx)]

    merged = utr_df.merge(
        first_cds[["start", "transcript_id"]].rename(columns={"start": "cds_start"}),
        on="transcript_id",
    )
    upstream = merged[
        ((merged["strand"] == "+") & (merged["end"] < merged["cds_start"]))
        | ((merged["strand"] == "-") & (merged["end"] > merged["cds_start"]))
    ].copy()
    upstream["utr_length"] = upstream["end"] - upstream["start"] + 1
    totals = upstream.groupby("transcript_id")["utr_length"].sum().rename("Start")
    return totals.reset_index().rename(columns={"transcript_id": "Tid"})


def get_start_codons(
    start_codon_df: pd.DataFrame,
    genome_fasta: str | Path,
) -> pd.DataFrame:
    """Fetch each transcript's canonical start-codon trinucleotide.

    GENCODE emits one ``start_codon`` feature per transcript (split across
    rows for split codons). We concatenate those rows in genomic order,
    then reverse-complement for minus-strand transcripts.
    """
    from Bio.Seq import Seq
    from pyfaidx import Fasta

    genome = Fasta(str(genome_fasta))

    per_tid: dict[str, str] = {}
    for row in start_codon_df.itertuples(index=False):
        seq = Seq(str(genome[row.chromosome][row.start - 1 : row.end]))
        if row.strand == "-":
            seq = seq.reverse_complement()
        per_tid[row.transcript_id] = per_tid.get(row.transcript_id, "") + str(seq)

    return pd.DataFrame({"Tid": list(per_tid.keys()), "StartCodon": list(per_tid.values())})


def get_protein_products(protein_fasta: str | Path) -> pd.DataFrame:
    """Parse GENCODE ``pc_translations.fa`` into a ``Tid → AASeq`` table.

    Each FASTA header is pipe-delimited; the transcript ID is the token
    starting with ``ENST``.
    """
    from Bio import SeqIO

    logger.info("Reading protein sequences from %s", protein_fasta)
    rows: list[dict] = []
    for rec in SeqIO.parse(str(protein_fasta), "fasta"):
        tid = next((t for t in rec.id.split("|") if t.startswith("ENST")), None)
        if tid is None:
            continue
        rows.append({"Tid": tid, "AASeq": str(rec.seq), "AALen": len(rec.seq)})
    return pd.DataFrame(rows)


def load_reference_features(gtf_path: str | Path) -> dict[str, pd.DataFrame]:
    """Load CDS, UTR, and start_codon feature tables in one GTF pass.

    Returns a dict keyed by feature name (``"CDS"``, ``"UTR"``,
    ``"start_codon"``). Each value has the standard transcript
    annotation columns.
    """
    logger.info("Loading CDS/UTR/start_codon features from %s", gtf_path)
    out: dict[str, pd.DataFrame] = {}
    for feat in ("CDS", "UTR", "start_codon"):
        out[feat] = load_transcript_annotations(str(gtf_path), feature_type=feat)
    return out


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

_IMPUTED_STATIC: dict[str, object] = {
    "TisType": "Annotated",
    "RecatTISType": "Annotated",
    "TISGroup": 0,
    "TISCounts": 0,
    "NormTISCounts": 0,
}


def impute_missing_canonical_starts(
    tis_df: pd.DataFrame,
    *,
    transcript_ids: list[str] | None = None,
    gtf_path: str | Path | None = None,
    genome_fasta: str | Path | None = None,
    protein_fasta: str | Path | None = None,
    features: dict[str, pd.DataFrame] | None = None,
    genome_pos: pd.DataFrame | None = None,
    utr_lengths: pd.DataFrame | None = None,
    start_codons: pd.DataFrame | None = None,
    protein_products: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build synthetic Annotated rows for transcripts missing them.

    For every transcript present in *tis_df* that has no
    ``RecatTISType == 'Annotated'`` row, synthesises one using GTF CDS
    coordinates, the 5'-UTR length (→ ``Start``), the genomic
    start-codon sequence, and the GENCODE protein product.

    Provide precomputed tables (*genome_pos* / *utr_lengths* /
    *start_codons* / *protein_products*) to amortise across samples.

    Returns a DataFrame of the *new* imputed rows only — caller concats
    onto *tis_df*. Rows missing any of ``GenomePos``, ``Start``,
    ``StartCodon``, ``AALen`` are dropped.
    """
    if transcript_ids is None:
        transcript_ids = tis_df["Tid"].unique().tolist()

    has_annotated = (
        tis_df.assign(_ann=lambda x: x["RecatTISType"] == "Annotated").groupby("Tid")["_ann"].sum()
    )
    missing = set(has_annotated[has_annotated == 0].index) & set(transcript_ids)
    if not missing:
        logger.info("No transcripts require canonical imputation")
        return tis_df.iloc[0:0].assign(Imputed=True)

    logger.info("Imputing canonical starts for %d transcripts", len(missing))

    if any(x is None for x in (genome_pos, utr_lengths, start_codons, protein_products)):
        if features is None:
            if gtf_path is None:
                raise ValueError(
                    "Provide either precomputed tables, a `features` dict, or `gtf_path`"
                )
            features = load_reference_features(gtf_path)

    if genome_pos is None:
        genome_pos = get_canonical_genome_positions(features["CDS"])
    if utr_lengths is None:
        utr_lengths = get_utr_lengths(features["CDS"], features["UTR"])
    if start_codons is None:
        if genome_fasta is None:
            raise ValueError("genome_fasta required to derive start codons")
        start_codons = get_start_codons(features["start_codon"], genome_fasta)
    if protein_products is None:
        if protein_fasta is None:
            raise ValueError("protein_fasta required to derive AASeq/AALen")
        protein_products = get_protein_products(protein_fasta)

    keep_cols = [
        c
        for c in [
            "Gid",
            "Tid",
            "Symbol",
            "GeneType",
            "gene_type",
            "transcript_type",
            "MANE_Select",
            "transcript_support_level",
            "Chromosome",
            "Strand",
            "GeneRNASeqCounts",
            "TotalRNASeqCounts",
            "Sample",
        ]
        if c in tis_df.columns
    ]
    base = tis_df[tis_df["Tid"].isin(missing)].drop_duplicates(subset=["Tid"]).loc[:, keep_cols]

    gp = genome_pos[["Tid", "GenomePos"]].copy()
    gp["GenomeStart"] = gp["GenomePos"].apply(_extract_genome_start)

    imputed = (
        base.merge(gp, on="Tid", how="left")
        .merge(utr_lengths, on="Tid", how="left")
        .merge(start_codons, on="Tid", how="left")
        .merge(protein_products[["Tid", "AASeq", "AALen"]], on="Tid", how="left")
    )
    # No 5'UTR annotation = CDS starts at the transcript 5' end; Start=0 is
    # correct.  Only drop rows missing the hard requirements (CDS coords,
    # start codon sequence, or protein product).
    cds_present = imputed["GenomePos"].notna()
    imputed.loc[cds_present & imputed["Start"].isna(), "Start"] = 0
    before = len(imputed)
    imputed = imputed.dropna(subset=["GenomePos", "StartCodon", "AALen"])
    dropped = before - len(imputed)
    if dropped:
        logger.warning(
            "Dropped %d imputation candidates missing GenomePos/Start/StartCodon/AALen",
            dropped,
        )

    for k, v in _IMPUTED_STATIC.items():
        imputed[k] = v
    imputed["Imputed"] = True
    return imputed


def _extract_genome_start(genome_pos: str) -> str:
    """Return the strand-appropriate start coordinate as ``chr:pos``."""
    chrom, coords, strand = genome_pos.rsplit(":", 2)
    left, right = coords.split("-")
    return f"{chrom}:{left}" if strand == "+" else f"{chrom}:{right}"
