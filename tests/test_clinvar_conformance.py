"""ClinVar conformance: both arms, every consequence class, both strands.

**Conformance, not regression.** Every expectation here comes from ClinVar's own
published notation, so a failure breaks a claim the NCBI made rather than one this
codebase made about itself. ``test_variantquery_fixture.py`` cannot do that — its
expectations are computed by the code under test.

Four things are asserted:

* **arm vs arm, end to end** — ``scan()`` over ``orf_index.parquet`` and the
  pipeline's batch ``validate_variants_against_orf`` must agree on every hit, in the
  frame the scan chose. The two reach the same classifier by different routes (VCF
  parsing, multi-allelic split, index lookup and frame choice on one side; the
  variant-dict batch writer on the other) and from *different coding sequence*: the
  scan reads ``orf_cds`` out of the index, the pipeline extracts it from the genome
  FASTA. Agreement therefore also proves the index was built from the genome it
  claims.
* **arm vs arm, canonical frame** — the scan prefers the isoform frame, where an
  extension's canonical ``p.Met1`` is an ordinary residue, so start-loss would never
  be compared end to end. The canonical frame is compared explicitly, and must
  produce ``start_lost``.
* **both arms vs ClinVar** — term and residue against the source's ``p.`` string,
  wherever the frame permits. "Where the frame permits" is decided on the
  *nucleotide*: our coding offset for the variant must land on ClinVar's ``c.``
  position (within the window a repeat lets an indel slide). A row that agrees there
  and is still called ``intronic`` or ``reference_mismatch`` is a failure, not a skip
  — which is how an intronic VCF padding base once hid a coding frameshift.
* **coverage** — every (class, strand) cell is populated, including indels whose
  padding base sits outside the exon they edit, or names why it cannot be.

Selection is a **query over provisioned reference data**, not a checked-in fixture:
``data/reference/clinvar/variant_summary.parquet`` is already on disk, and rows are
chosen deterministically — ordered by a fixed hash of the variant, not by position,
so a cell is not all chr1 — and a run is reproducible without a second artifact.

The index is a ``full_catalog`` build rather than ``cheeseman_test``: of the 877
ClinVar variants landing inside a cheeseman_test ORF, **zero** are minus-strand
insertions and **zero** are ATG start-loss. It is a run output, so it lives in
whichever checkout ran the catalogue; point ``SWISSISO_ORF_INDEX`` at it when it is
not this one's.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from swissisoform import coords

REPO = Path(__file__).resolve().parents[1]
CLINVAR = REPO / "data" / "reference" / "clinvar" / "variant_summary.parquet"
GENOME = REPO / "data" / "reference" / "Gencode_v49_GRCh38.primary_assembly.genome.fa"
INDEX = Path(
    os.environ.get(
        "SWISSISO_ORF_INDEX", REPO / "data" / "output" / "full_catalog" / "orf_index.parquet"
    )
)

#: Rows per (class, strand) cell. Small: each row costs a genome-backed CDS
#: extraction in the pipeline arm, and the point is coverage of the matrix rather
#: than statistical weight. Large enough that the frame filter leaves every class
#: something to compare.
PER_CELL = 12

pytestmark = pytest.mark.skipif(
    not (CLINVAR.is_file() and INDEX.is_file()),
    reason=(
        "needs the provisioned ClinVar parquet "
        "(python -m swissisoform.setup.databases clinvar) and a built full_catalog "
        "orf_index.parquet (python scripts/export/build_orf_index.py --run full_catalog, "
        "or SWISSISO_ORF_INDEX=<path>)"
    ),
)
needs_genome = pytest.mark.skipif(
    not GENOME.is_file(), reason="needs the genome FASTA for the pipeline arm"
)

#: The GNB1 frameshift whose VCF padding base is the last intronic base before the
#: exon it deletes from. Pinned because the matrix may not sample it.
GNB1_EXON_EDGE = ("chr1", 1789052, "CCT", "C")


def _clinvar_class(protein: str, ref: str, alt: str) -> str:
    """The class ClinVar's own p. string claims, which is both selector and oracle.

    The parquet carries no consequence column — only ``Name``, an HGVS-ish string —
    so the class is read out of the notation rather than looked up.
    """
    if "fs" in protein:
        return "frameshift"
    if protein.endswith("="):
        return "synonymous"
    if "delins" in protein:
        return "delins"
    if "Ter" in protein and not protein.startswith("Ter"):
        return "nonsense"
    if re.match(r"^Met1[A-Z?]", protein):
        return "start_lost"
    if "del" in protein:
        return "deletion"
    if "dup" in protein or "ins" in protein:
        return "insertion"
    if len(ref) == len(alt) == 1:
        return "missense"
    if len(ref) == len(alt) > 1:
        return "mnv"
    return "other"


def _clinvar_residue(protein: str) -> int | None:
    """The 1-based residue ClinVar's p. string names first, or None if it names none.

    ``Met1Val`` -> 1, ``Tyr96fs`` -> 96, ``Ser83_Leu85del`` -> 83. Notations without a
    leading residue (repeat expansions like ``3250VP[2]``) return None.
    """
    match = re.match(r"^[A-Z][a-z]{2}(\d+)", protein)
    return int(match.group(1)) if match else None


def _clinvar_cdot(name: str) -> tuple[int, int | None, str] | None:
    """``(first, last, event)`` from the ``c.`` part of a ClinVar ``Name``.

    Only plain exonic coding positions are returned. Intronic (``c.123+1``), UTR
    (``c.-5``, ``c.*3``) and uncertain positions return None: they carry no coding
    offset to compare against.
    """
    match = re.search(r":c\.(\d+)(?:_(\d+))?([a-z>A-Z].*?)(?:\s|$)", name)
    if not match:
        return None
    first, last, rest = match.groups()
    return int(first), int(last) if last else None, rest


def _slide_window(seq: str, start: int, length: int) -> tuple[int, int]:
    """How far the segment ``seq[start:start+length]`` can slide without changing seq.

    VCF places an indel leftmost in genomic coordinates, HGVS 3'-most in transcript
    coordinates, so inside a repeat the two name one event at different places.
    Returns ``(toward_5prime, toward_3prime)`` in bases.
    """
    d3 = 0
    while start + length + d3 < len(seq) and seq[start + d3] == seq[start + length + d3]:
        d3 += 1
    d5 = 0
    while start - 1 - d5 >= 0 and seq[start - 1 - d5] == seq[start + length - 1 - d5]:
        d5 += 1
    return d5, d3


def _our_event(record, entry) -> dict | None:
    """Where the variant sits in the canonical CDS, independent of the classifier.

    Computed from exons and alleles alone so it can judge the classifier: a row whose
    changed bases are coding here is coding, whatever the classifier said.
    Returns the ClinVar-comparable ``c.`` anchor (first changed base for a
    substitution or deletion, the left flank for an insertion) and the window a
    repeat lets it slide over. None when no changed base is coding.
    """
    exons = record.exons_for("canonical")
    cds = record.cds_for("canonical")
    if not exons or not cds:
        return None
    pos_map = dict(coords.iter_coding_positions(exons, record.strand))
    start, ref_changed, alt_changed = coords.changed_bases(entry["pos"], entry["ref"], entry["alt"])
    if ref_changed:
        offsets = [pos_map.get(p) for p in range(start, start + len(ref_changed))]
        if any(o is None for o in offsets):
            return None
        first = min(offsets)
        anchor = first + 1
        if len(ref_changed) == len(alt_changed):
            window = (0, 0)
        else:
            window = _slide_window(cds, first, len(ref_changed) - len(alt_changed))
        return {
            "anchor": anchor,
            "window": window,
            "kind": "del_or_sub",
            "codons": (first // 3, (first + len(ref_changed) - 1) // 3),
        }
    left, right = pos_map.get(start - 1), pos_map.get(start)
    if left is None or right is None:
        return None
    point = min(left, right) + 1  # inserted before 0-based ``point``
    inserted = alt_changed if record.strand == "+" else coords.revcomp(alt_changed)
    mutant = cds[:point] + inserted + cds[point:]
    return {
        "anchor": point,
        "window": _slide_window(mutant, point, len(inserted)),
        "kind": "ins",
        "codons": (point // 3, point // 3),
    }


def _frame_agrees(event: dict | None, cdot: tuple[int, int | None, str] | None) -> bool:
    """Our nucleotide anchor lands on ClinVar's ``c.`` position, modulo repeat slide."""
    if event is None or cdot is None:
        return False
    first, last, rest = cdot
    if event["kind"] == "ins":
        # HGVS names an insertion by its flanks (c.X_Yins: X is the left one) and a
        # duplication by the copied bases, whose last base is the left flank of the
        # 3'-most placement.
        clinvar_anchor = (last or first) if "dup" in rest else first
    else:
        clinvar_anchor = first
    d5, d3 = event["window"]
    return event["anchor"] - d5 <= clinvar_anchor <= event["anchor"] + d3


@pytest.fixture(scope="module")
def index():
    from swissisoform.variantquery.load import load_index

    return load_index(INDEX)


@pytest.fixture(scope="module")
def matrix(index):
    """ClinVar rows landing inside an ORF, stratified by (class, strand).

    The parquet is read through a pyarrow column projection and then filtered to the
    index's own chromosome/position envelope before any per-row work, so the 4.4M-row
    table is never walked in Python. Rows are ordered by a fixed hash of the variant
    rather than by position, so the first N of a cell are spread over the genome.
    """
    import pandas as pd
    import pyarrow.parquet as pq

    cv = pq.read_table(
        CLINVAR,
        columns=[
            "Chromosome",
            "PositionVCF",
            "ReferenceAlleleVCF",
            "AlternateAlleleVCF",
            "Name",
            "ClinicalSignificance",
        ],
    ).to_pandas()
    cv["PositionVCF"] = pd.to_numeric(cv["PositionVCF"], errors="coerce")
    cv = cv.dropna(subset=["PositionVCF"])
    cv = cv[cv["Name"].str.contains(r"\(p\.", regex=True, na=False)]
    cv["chrom"] = "chr" + cv["Chromosome"].astype(str)

    bounds: dict[str, list[int]] = {}
    for record in index.records:
        exons = record.exons_for("isoform")
        lo, hi = min(s for s, _ in exons), max(e for _, e in exons)
        span = bounds.setdefault(record.chrom, [lo, hi])
        span[0], span[1] = min(span[0], lo), max(span[1], hi)

    keep = pd.Series(False, index=cv.index)
    for chrom, (lo, hi) in bounds.items():
        keep |= (cv["chrom"] == chrom) & (cv["PositionVCF"] >= lo) & (cv["PositionVCF"] <= hi)
    cv = cv[keep].drop_duplicates(
        subset=["chrom", "PositionVCF", "ReferenceAlleleVCF", "AlternateAlleleVCF"]
    )
    order = pd.util.hash_pandas_object(
        cv[["chrom", "PositionVCF", "ReferenceAlleleVCF", "AlternateAlleleVCF"]], index=False
    )
    cv = cv.assign(_order=order.values).sort_values("_order")

    cells: dict[tuple[str, str], list[dict]] = {}
    for row in cv.itertuples():
        ref = str(row.ReferenceAlleleVCF or "")
        alt = str(row.AlternateAlleleVCF or "")
        if not ref or not alt or set(ref + alt) - set("ACGTN"):
            continue
        pos = int(row.PositionVCF)
        records = index.lookup_span(row.chrom, pos, pos + max(len(ref), 1) - 1)
        if not records:
            continue
        record = records[0]
        name = str(row.Name)
        protein = name.split("(p.")[-1].rstrip(")")
        entry = {
            "chrom": row.chrom,
            "pos": pos,
            "ref": ref,
            "alt": alt,
            "name": name,
            "protein": protein,
            "record": record,
            "significance": str(row.ClinicalSignificance),
        }
        keys = [(_clinvar_class(protein, ref, alt), record.strand)]
        if len(ref) != len(alt):
            exons = record.exons_for("canonical")
            if exons and coords.coding_offset(exons, record.strand, pos) is None:
                keys.append(("padding_outside_exon", record.strand))
        for key in keys:
            bucket = cells.setdefault(key, [])
            if len(bucket) < PER_CELL:
                bucket.append(entry)
    return cells


def _pipeline_arm(validator, record, frame: str, entry) -> dict:
    """The pipeline's own batch writer, reading the CDS out of the genome FASTA."""
    variant = {"genomic_pos": entry["pos"], "ref": entry["ref"], "alt": entry["alt"]}
    validator.validate_variants_against_orf(
        [variant],
        orf_exons=[tuple(e) for e in record.exons_for(frame)],
        strand=record.strand,
        chrom=record.chrom,
        orf_key=(record.tis_id, frame, "pipeline"),
        field_prefix="arm",
    )
    return {
        "consequence": variant["arm_consequence"],
        "protein_pos": variant["arm_protein_pos"],
        "aa_ref": variant["arm_aa_ref"],
        "aa_alt": variant["arm_aa_alt"],
    }


def _index_arm(validator, record, frame: str, entry) -> dict:
    """The classifier over the CDS the index stores — what ``scan()`` calls per hit."""
    return validator.classify_against_orf(
        orf_exons=[tuple(e) for e in record.exons_for(frame)],
        strand=record.strand,
        cds=record.cds_for(frame),
        genomic_pos=entry["pos"],
        ref=entry["ref"],
        alt=entry["alt"],
        orf_key=(record.tis_id, frame, "index"),
    )


def _entries(matrix) -> list[dict]:
    seen: set[tuple] = set()
    out = []
    for _cell, entries in sorted(matrix.items()):
        for entry in entries:
            key = (entry["chrom"], entry["pos"], entry["ref"], entry["alt"])
            if key not in seen:
                seen.add(key)
                out.append(entry)
    return out


# ----------------------------------------------------------------------
# 1. The two arms must agree
# ----------------------------------------------------------------------


@needs_genome
def test_scan_and_the_pipeline_agree_on_every_hit(matrix, index, tmp_path):
    """``scan()`` end to end against the pipeline's batch writer, hit by hit.

    Also a check on the index itself: the arms disagree if ``orf_cds`` in the parquet
    ever drifts from the genome it was built from, which nothing else would notice.
    """
    from swissisoform.clinical.validate import ConsequenceValidator
    from swissisoform.variantquery.consequence import OTHER
    from swissisoform.variantquery.scan import scan

    entries = _entries(matrix)
    vcf = tmp_path / "conformance.vcf"
    header = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    vcf.write_text(
        header
        + "".join(
            f"{e['chrom']}\t{e['pos']}\t.\t{e['ref']}\t{e['alt']}\t.\tPASS\t.\n" for e in entries
        )
    )
    result = scan(vcf, index, max_hits=10**7, max_records=0, max_seconds=0)
    assert not result.counts.rejected, result.counts.rejected

    validator = ConsequenceValidator(genome_fasta=str(GENOME))
    by_line = {line_no: entry for line_no, entry in enumerate(entries, start=3)}
    compared = 0
    for hit in result.hits:
        entry = by_line[hit.line_no]
        record = index.by_tis_id(hit.tis_id)
        pipeline = _pipeline_arm(validator, record, hit.frame, entry)
        where = f"{hit.tis_id}:{hit.frame} {hit.chrom}:{hit.pos} {hit.ref}>{hit.alt}"
        assert hit.consequence == (pipeline["consequence"] or OTHER), where
        assert hit.aa_ref == (pipeline["aa_ref"] or ""), where
        assert hit.aa_alt == (pipeline["aa_alt"] or ""), where
        if pipeline["protein_pos"] is not None:
            assert hit.residue == pipeline["protein_pos"], where
        compared += 1
    assert compared >= len(entries), f"only {compared} hits for {len(entries)} variants"


@needs_genome
def test_the_arms_agree_in_canonical_frame_and_reach_start_loss(matrix):
    """The frame the scan does not pick for an extension's canonical Met1."""
    from swissisoform.clinical.validate import ConsequenceValidator

    validator = ConsequenceValidator(genome_fasta=str(GENOME))
    compared = 0
    start_lost = {"+": 0, "-": 0}
    for entry in _entries(matrix):
        record = entry["record"]
        if not record.exons_for("canonical"):
            continue
        index_arm = _index_arm(validator, record, "canonical", entry)
        pipeline = _pipeline_arm(validator, record, "canonical", entry)
        where = f"{entry['chrom']}:{entry['pos']} {entry['ref']}>{entry['alt']}"
        for field in ("consequence", "protein_pos", "aa_ref", "aa_alt"):
            assert index_arm[field] == pipeline[field], (
                f"{where}: arms disagree on {field} — "
                f"index={index_arm[field]!r} pipeline={pipeline[field]!r}"
            )
        compared += 1
        if pipeline["consequence"] == "start_lost":
            start_lost[record.strand] += 1
    assert compared >= 20, f"only {compared} variants compared across both arms"
    assert all(start_lost.values()), f"start_lost never produced on a strand: {start_lost}"


# ----------------------------------------------------------------------
# 2. Our call must match what ClinVar published
# ----------------------------------------------------------------------


def _expected(klass: str, entry: dict) -> str | None:
    protein, ref, alt = entry["protein"], entry["ref"], entry["alt"]
    if protein.startswith("Met1") and klass in ("deletion", "frameshift", "delins"):
        # Removing or rewriting the initiator is start-loss whatever the length
        # delta — ClinVar writes these p.Met1? or p.Met1del.
        return "start_lost"
    if klass in ("mnv", "delins"):
        if len(ref) != len(alt):
            return None  # a complex length change: the class is not comparable
        if protein.endswith("="):
            return "synonymous_variant"
        return "stop_gained" if "Ter" in protein else "missense_variant"
    return {
        "frameshift": "frameshift_variant",
        "synonymous": "synonymous_variant",
        "nonsense": "stop_gained",
        "missense": "missense_variant",
        "start_lost": "start_lost",
        "deletion": "inframe_deletion",
        "insertion": "inframe_insertion",
    }.get(klass)


def _residue_matches(klass: str, entry: dict, event: dict, ours: int) -> bool:
    clinvar = _clinvar_residue(entry["protein"])
    if clinvar is None or event["window"] != (0, 0):
        # Inside a repeat both residues are right; the nucleotide check already
        # placed the event.
        return True
    first_codon, last_codon = event["codons"]
    if ours != first_codon:
        # We number the first codon the change touches, on every class.
        return False
    if klass == "frameshift":
        # ClinVar names the first residue whose letter changes, which can be a codon
        # after the first one the event touches.
        return ours + 1 <= clinvar
    if klass == "insertion" and "dup" not in entry["protein"]:
        # We number the first inserted residue — HGVS's right-hand flank.
        return clinvar + 1 in (ours + 1, ours + 2)
    # A change straddling a codon boundary touches two codons, and ClinVar names the
    # one whose residue actually changes (p.Pro73Ser for c.216_217delinsTT, where
    # codon 72 is silent). Either codon is the same event.
    return first_codon + 1 <= clinvar <= last_codon + 1


@pytest.mark.parametrize("arm", ["index", pytest.param("pipeline", marks=needs_genome)])
def test_both_arms_match_clinvar(matrix, arm):
    """Class and residue against ClinVar wherever the nucleotide frame agrees."""
    from swissisoform.clinical.validate import ConsequenceValidator

    validator = ConsequenceValidator(genome_fasta=str(GENOME) if arm == "pipeline" else None)
    classify = _pipeline_arm if arm == "pipeline" else _index_arm
    compared: dict[tuple[str, str], int] = {}
    skipped_frame = 0
    failures: list[str] = []
    for (klass, strand), entries in sorted(matrix.items()):
        if klass in ("other", "padding_outside_exon"):
            continue
        for entry in entries:
            record = entry["record"]
            event = _our_event(record, entry)
            if not _frame_agrees(event, _clinvar_cdot(entry["name"])):
                # ClinVar's transcript is not this ORF's canonical: different c.
                # numbering, so the residue and often the frame are not shared.
                skipped_frame += 1
                continue
            out = classify(validator, record, "canonical", entry)
            where = f"{entry['chrom']}:{entry['pos']} {entry['ref']}>{entry['alt']} {entry['name']}"
            expected = _expected(klass, entry)
            if out["consequence"] in (None, "intronic", "reference_mismatch"):
                failures.append(f"{where}: coding in ClinVar's frame, we say {out['consequence']}")
                continue
            if (
                expected == "stop_gained"
                and (len(entry["alt"]) - len(entry["ref"])) % 3
                and out["consequence"] == "frameshift_variant"
            ):
                # HGVS writes a frameshift whose first new residue is a stop as a
                # nonsense change (c.948dup, p.Lys317Ter). Both are loss of function.
                expected = "frameshift_variant"
            if expected and out["consequence"] != expected:
                failures.append(f"{where}: ClinVar says {expected}, we say {out['consequence']}")
                continue
            if not _residue_matches(klass, entry, event, out["protein_pos"]):
                failures.append(f"{where}: residue {out['protein_pos'] + 1} vs ClinVar")
                continue
            compared[(klass, strand)] = compared.get((klass, strand), 0) + 1
    assert not failures, "\n".join(failures[:25])
    missing = [
        f"{klass}{strand}"
        for klass in REQUIRED_CLASSES
        for strand in ("+", "-")
        if not compared.get((klass, strand))
    ]
    assert not missing, (
        f"no row comparable in ClinVar's frame for {missing} "
        f"(skipped for frame: {skipped_frame}); the test would assert less than it reads"
    )


@pytest.mark.parametrize("arm", ["index", pytest.param("pipeline", marks=needs_genome)])
def test_an_indel_padded_outside_its_exon_is_classified_on_what_it_changes(matrix, index, arm):
    """The VCF padding base is not coding; the bases the indel edits are.

    Reading POS first called every one of these ``intronic`` — including the GNB1
    frameshift pinned below — so a coding frameshift missed the LoF gate.
    """
    from swissisoform.clinical.validate import ConsequenceValidator

    validator = ConsequenceValidator(genome_fasta=str(GENOME) if arm == "pipeline" else None)
    classify = _pipeline_arm if arm == "pipeline" else _index_arm
    entries = matrix.get(("padding_outside_exon", "+"), []) + matrix.get(
        ("padding_outside_exon", "-"), []
    )
    checked = 0
    for entry in entries:
        if _our_event(entry["record"], entry) is None:
            continue  # nothing it changes is coding here either
        out = classify(validator, entry["record"], "canonical", entry)
        assert out["consequence"] not in (None, "intronic"), entry["name"]
        checked += 1
    assert checked, "no exon-edge indel with coding changed bases was sampled"

    chrom, pos, ref, alt = GNB1_EXON_EDGE
    records = [
        r for r in index.lookup_span(chrom, pos, pos + len(ref) - 1) if r.gene_name == "GNB1"
    ]
    if records:
        pinned = {"pos": pos, "ref": ref, "alt": alt}
        outs = {classify(validator, r, "canonical", pinned)["consequence"] for r in records}
        assert "frameshift_variant" in outs, outs


# ----------------------------------------------------------------------
# 3. The matrix has to actually be covered
# ----------------------------------------------------------------------

#: Cells ClinVar cannot supply, with the reason. A hole with a recorded reason is a
#: documented limit; a hole without one is an untested quadrant nobody noticed —
#: which is exactly how the minus-strand indel bug survived.
UNREACHABLE = {
    "near_cognate_start": (
        "ClinVar numbers against canonical transcripts, which begin at ATG, so a "
        "near-cognate start variant has no ClinVar representation. Covered "
        "synthetically in test_clinical_validate_orf.py."
    ),
}

REQUIRED_CLASSES = (
    "missense",
    "synonymous",
    "nonsense",
    "frameshift",
    "deletion",
    "insertion",
    "start_lost",
)


def test_every_class_is_covered_on_both_strands(matrix):
    """A cell that silently empties is the failure this suite exists to prevent."""
    missing = [
        f"{klass} on {strand} strand"
        for klass in (*REQUIRED_CLASSES, "mnv", "padding_outside_exon")
        for strand in ("+", "-")
        if not matrix.get((klass, strand))
    ]
    assert not missing, (
        "empty cells: " + ", ".join(missing) + ". Either the index lost coverage or "
        "the class parser stopped recognising them; both are real."
    )


def test_the_matrix_is_not_one_chromosome(matrix):
    chroms = {entry["chrom"] for entry in _entries(matrix)}
    assert len(chroms) >= 5, chroms


def test_the_unreachable_cells_are_named_rather_than_silently_absent():
    assert UNREACHABLE["near_cognate_start"]


def test_start_loss_is_reachable_at_all(matrix):
    """The variants PR #29 gate 4 exists for.

    Before the asymmetric rule no SNV at an annotated ATG could ever be start_lost —
    NEAR_COGNATE_STARTS is exactly ATG plus its nine single-base neighbours, so the
    membership test could not fire. Every one of these was called missense and missed
    the loss-of-function gate.
    """
    rows = matrix.get(("start_lost", "+"), []) + matrix.get(("start_lost", "-"), [])
    assert rows, "no ClinVar ATG start-loss variant landed in the catalogue"
    pathogenic = [r for r in rows if "athogenic" in r["significance"]]
    assert pathogenic, f"none of the {len(rows)} start-loss rows is pathogenic"
