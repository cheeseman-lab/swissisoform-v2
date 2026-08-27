"""Walk a VCF against an :class:`OrfIndex` and emit hits plus a funnel.

The funnel is not decoration. Measured against the full catalogue, 28 of 34,706
PASS variants in a real somatic VCF land inside an annotated ORF (~0.08%), so
"no hits" is the *normal* outcome and has to be explainable — how many records
were read, how many passed the filter, how many were on contigs the catalogue
covers at all. A bare zero is indistinguishable from a broken scan.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from swissisoform.clinical.validate import ConsequenceValidator
from swissisoform.variantquery.consequence import OTHER, classify_without_sequence
from swissisoform.variantquery.frame import region_for, resolve_residue
from swissisoform.variantquery.index import OrfIndex
from swissisoform.variantquery.spec import Rejection, VariantSpec, parse_line
from swissisoform.variantquery.vcf import iter_data_lines

logger = logging.getLogger(__name__)

#: Hard ceiling on stored hits, so a germline whole-exome VCF cannot fill the
#: disk. Exceeding it sets ``truncated`` rather than failing the scan.
DEFAULT_MAX_HITS = 20_000

#: Ceiling on data lines read. The hit cap above bounds *output*; this bounds
#: *work*, which is the thing that occupies a worker — the scan classifies every
#: allele against every overlapping ORF whether or not the hit is stored.
DEFAULT_MAX_RECORDS = 5_000_000

#: Wall-clock ceiling, in seconds. Deliberately under the deployment's 60 s
#: gunicorn timeout (``website/entrypoint.sh``): past that the worker is killed
#: and the uploader gets a proxy error, where stopping here returns a partial
#: answer that says so.
DEFAULT_MAX_SECONDS = 45.0

#: Classifier terms that carry no amino acids and read as a failure without a
#: reason attached. Everything else explains itself, or carries its own ``note``
#: from the classifier (start-codon calls do).
_TERM_NOTES = {
    # Two ways to be intronic: the variant maps nowhere in this frame, or only part
    # of a multi-base span does. Neither can be translated, so one wording covers it.
    "intronic": "not inside this ORF's coding sequence",
    "reference_mismatch": "REF does not match this ORF's reference sequence",
}
_UNCLASSIFIED_NOTE = "could not be classified against this ORF's coding sequence"


def _hit_fields(result: dict[str, Any]) -> tuple[str, str, str, str]:
    """``classify_against_orf`` output → the four fields a hit records.

    ``consequence=None`` means the classifier could not answer — no position map, no
    sequence, or a codon running past the end of it — which is distinct from a term
    it declines to refine, so it gets its own note rather than a bare ``other``.
    """
    term = result.get("consequence")
    if not term:
        return OTHER, "", "", _UNCLASSIFIED_NOTE
    return (
        term,
        result.get("aa_ref") or "",
        result.get("aa_alt") or "",
        # The classifier's own note wins: it knows *why* — a start that got stronger
        # rather than lost reads as ordinary missense without it. The table below is
        # the fallback for terms it emits with nothing to add.
        result.get("note") or _TERM_NOTES.get(term, ""),
    )


@dataclass(frozen=True, slots=True)
class VariantHit:
    """One (variant, ORF) pair. A position in N isoforms yields N hits.

    ``residue`` is **0-based**, matching ``protein_pos`` elsewhere in the pipeline —
    so TP53 p.R248 reports ``residue=247`` — and is numbered against the protein
    named by ``frame``. The same nucleotide is a different residue in every ORF
    containing it, so the two fields have to travel together.

    Everything below ``region`` comes from the pipeline's own classifier
    (:meth:`~swissisoform.clinical.validate.ConsequenceValidator.classify_against_orf`),
    so an uploaded variant is described exactly the way an annotated one is: the
    consequence term, the reference and alternate amino acids, and nothing else. No
    HGVS notation is produced here — the pipeline never derives one either, it copies
    what VEP / ClinVar / COSMIC supplied, and an uploaded VCF has no such source.
    """

    line_no: int
    chrom: str
    pos: int
    ref: str
    alt: str
    vclass: str
    gene: str
    tis_id: str
    transcript_id: str
    orf_type: str
    frame: str
    residue: int | None
    region: str
    #: One of the figure's consequence terms; ``other`` when unclassifiable.
    consequence: str = OTHER
    #: Reference and alternate amino acids. More than one letter when a multi-base
    #: substitution straddles a codon boundary; empty for indels, which the
    #: classifier resolves by length without reading sequence.
    aa_ref: str = ""
    aa_alt: str = ""
    #: Why the amino acids are absent, when the term alone does not say (e.g. a span
    #: that leaves the coding sequence, or a REF the reference disagrees with).
    consequence_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form for the scan digest."""
        return asdict(self)


@dataclass
class ScanCounts:
    """Funnel from raw lines down to hits."""

    lines: int = 0
    alleles: int = 0
    skipped_non_pass: int = 0
    off_catalog_contig: int = 0
    no_orf: int = 0
    hits: int = 0
    genes_hit: int = 0
    truncated: bool = False
    #: Why the scan stopped before EOF: ``""`` (it didn't), ``"records"`` or
    #: ``"time"``. Distinct from ``truncated``, which means the hit *list* hit its
    #: cap while the counts below stayed complete. When this is set the counts
    #: themselves are partial, which is a different claim and has to read as one.
    stopped: str = ""
    rejected: dict[str, int] = field(default_factory=dict)
    #: Hit records per consequence term — the figure's row breakdown, and the
    #: quickest way to see from the logs whether classification ran at all.
    consequences: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        """Tally one rejected allele under ``reason``."""
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form for the scan digest."""
        return asdict(self)


@dataclass
class GeneSummary:
    """Per-gene rollup for the results list.

    Two counts, because they differ and the difference is the interesting part: a
    single variant inside three isoforms of one gene is **1 variant, 3 hits**.
    ``n_unique``/``n_shared`` are hit-record counts too — the same nucleotide can be
    isoform-unique in one ORF and shared in another, which is a real observation
    rather than a bookkeeping artifact.
    """

    gene: str
    n_hits: int = 0
    n_unique: int = 0
    n_shared: int = 0
    tis_ids: set[str] = field(default_factory=set)
    #: (line_no, alt) — identifies one allele occurrence in the source file, so the
    #: same position on two VCF lines counts twice while one line in N isoforms
    #: counts once.
    variant_keys: set[tuple[int, str]] = field(default_factory=set)

    @property
    def n_variants(self) -> int:
        """Distinct alleles from the VCF that hit this gene."""
        return len(self.variant_keys)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form; the two sets collapse to counts."""
        return {
            "gene": self.gene,
            "n_variants": self.n_variants,
            "n_hits": self.n_hits,
            "n_unique": self.n_unique,
            "n_shared": self.n_shared,
            "n_isoforms": len(self.tis_ids),
        }


@dataclass
class ScanResult:
    """Everything one scan produced: hits, per-gene rollup, and the funnel."""

    counts: ScanCounts
    hits: list[VariantHit]
    genes: list[GeneSummary]
    index_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form for the scan digest."""
        return {
            "counts": self.counts.to_dict(),
            "genes": [g.to_dict() for g in self.genes],
            "hits": [h.to_dict() for h in self.hits],
        }


def scan(
    path: str | Path,
    index: OrfIndex,
    *,
    pass_only: bool = True,
    max_hits: int = DEFAULT_MAX_HITS,
    max_records: int = DEFAULT_MAX_RECORDS,
    max_seconds: float = DEFAULT_MAX_SECONDS,
) -> ScanResult:
    """Resolve every variant in ``path`` against ``index``.

    Args:
        path: VCF, plain or gzipped (detected by magic bytes).
        index: The ORF interval index to resolve against.
        pass_only: Drop records whose FILTER is neither ``PASS`` nor ``.``.
        max_hits: Stop *recording* hits past this many; counting continues.
        max_records: Stop the scan past this many data lines; ``0`` disables.
        max_seconds: Stop the scan past this much wall clock; ``0`` disables.

    Returns:
        A :class:`ScanResult`. Distinct negatives are kept apart in the counts:
        ``off_catalog_contig`` means the chromosome is absent from the index
        (usually a naming mismatch or a scaffold), ``no_orf`` means the contig is
        covered but the position is intronic, UTR or intergenic.

        When ``counts.stopped`` is set the scan ended before EOF and **every
        count is a partial**, describing only the records that were read.

    Raises:
        VcfLimitExceeded: If the file breaches a reader cap (see :mod:`vcf`).
            Not caught here — the caps are about the input, and only the caller
            knows how to report a bad input.
    """
    counts = ScanCounts()
    hits: list[VariantHit] = []
    genes: dict[str, GeneSummary] = {}
    # One validator for the whole scan, so its position-map cache is reused across
    # every variant that lands in the same ORF. Constructed with no ``cds_df`` and no
    # genome: the coding sequence comes from the index, so nothing here opens a FASTA.
    validator = ConsequenceValidator()
    deadline = time.monotonic() + max_seconds if max_seconds else None

    for line_no, line in iter_data_lines(path):
        # Checked per line, not per allele: at millions of records a clock read
        # per allele is measurable, and one line of overshoot costs nothing.
        if max_records and counts.lines >= max_records:
            counts.stopped = "records"
            break
        if deadline is not None and time.monotonic() > deadline:
            counts.stopped = "time"
            break
        counts.lines += 1
        for parsed in parse_line(line):
            if isinstance(parsed, Rejection):
                counts.reject(parsed.reason)
                continue
            counts.alleles += 1
            spec: VariantSpec = parsed
            if pass_only and not spec.is_pass:
                counts.skipped_non_pass += 1
                continue
            if not index.has_chrom(spec.chrom):
                counts.off_catalog_contig += 1
                continue

            start, end = spec.span
            records = index.lookup_span(spec.chrom, start, end)
            if not records:
                counts.no_orf += 1
                continue

            for record in records:
                residue, frame, _gpos = resolve_residue(record, start, end)
                region = region_for(record, start, end)
                # The frame decides which of the two CDSs and exon sets to use — a
                # truncation's lost N-terminus is numbered against the canonical.
                cds = record.cds_for(frame)
                if cds:
                    result = validator.classify_against_orf(
                        orf_exons=[tuple(exon) for exon in record.exons_for(frame)],
                        strand=record.strand,
                        cds=cds,
                        genomic_pos=spec.pos,
                        ref=spec.ref,
                        alt=spec.alt,
                        orf_key=(record.tis_id, frame),
                        context=f"{record.tis_id}:{frame}",
                    )
                    term, aa_ref, aa_alt, note = _hit_fields(result)
                    # One residue number, and it comes from the classifier.
                    # ``resolve_residue`` walks the REF span in ascending *genomic*
                    # order, which on the minus strand reaches the span's last
                    # translated base first — so it lands a codon late on a
                    # multi-base variant. It still decides the frame.
                    if result["protein_pos"] is not None:
                        residue = result["protein_pos"]
                else:
                    # Index built without a genome: length still gives the class,
                    # but missense/synonymous/stop_gained are indistinguishable.
                    term, note = classify_without_sequence(spec.ref, spec.alt)
                    aa_ref = aa_alt = ""
                counts.consequences[term] = counts.consequences.get(term, 0) + 1
                counts.hits += 1
                summary = genes.setdefault(record.gene_name, GeneSummary(record.gene_name))
                summary.n_hits += 1
                summary.tis_ids.add(record.tis_id)
                summary.variant_keys.add((line_no, spec.alt))
                if region == "unique":
                    summary.n_unique += 1
                elif region == "shared":
                    summary.n_shared += 1
                if len(hits) < max_hits:
                    hits.append(
                        VariantHit(
                            line_no=line_no,
                            chrom=spec.chrom,
                            pos=spec.pos,
                            ref=spec.ref,
                            alt=spec.alt,
                            vclass=spec.vclass,
                            gene=record.gene_name,
                            tis_id=record.tis_id,
                            transcript_id=record.transcript_id,
                            orf_type=record.orf_type,
                            frame=frame,
                            residue=residue,
                            region=region,
                            consequence=term,
                            aa_ref=aa_ref,
                            aa_alt=aa_alt,
                            consequence_note=note,
                        )
                    )
                else:
                    counts.truncated = True

    counts.genes_hit = len(genes)
    if counts.stopped:
        logger.warning(
            "scan stopped early (%s) after %d records — counts are partial",
            counts.stopped,
            counts.lines,
        )
    # Rank by distinct variants first — that is what "most affected gene" means —
    # then by hit records, then alphabetically for a stable order.
    ordered = sorted(genes.values(), key=lambda g: (-g.n_variants, -g.n_hits, g.gene))
    return ScanResult(counts=counts, hits=hits, genes=ordered, index_version=index.version)
