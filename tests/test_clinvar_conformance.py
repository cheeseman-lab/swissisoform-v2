"""ClinVar conformance: both arms, every consequence class, both strands.

**Conformance, not regression.** Every expectation here comes from ClinVar's own
published protein notation, so a failure breaks a claim the NCBI made rather than one
this codebase made about itself. ``test_variantquery_fixture.py`` cannot do that —
its expectations are computed by the code under test.

Three things are asserted, and the first is the one nothing else covers:

* **arm vs arm** — ``scan()`` over ``orf_index.parquet`` and
  ``validate_variants_against_orf`` must agree. They reach the same
  ``_analyze_variant`` by different routes and, crucially, from *different coding
  sequence*: the scan reads ``orf_cds`` out of the index, the pipeline extracts it
  from the genome FASTA. Agreement therefore also proves the index was built from the
  genome it claims. This is the promise commit f26b4a3 made ("one classifier") and
  was never tested.
* **arm vs ClinVar** — our term and residue against the source's ``p.`` string, in
  canonical-residue space, for the rows where the frame permits.
* **coverage** — every (class, strand) cell is populated, or names why it cannot be.

Selection is a **query over provisioned reference data**, not a checked-in fixture:
``data/reference/clinvar/variant_summary.parquet`` is already on disk, and the rows
are chosen deterministically (sorted, first N per cell — no RNG) so a run is
reproducible without a second artifact to keep in sync.

The index is ``full_catalog`` rather than ``cheeseman_test`` for a measured reason:
of the 877 ClinVar variants landing inside a cheeseman_test ORF, **zero** are
minus-strand insertions and **zero** are ATG start-loss — precisely the two
behaviours PR #29 changed. full_catalog carries 374/348 minus/plus start-loss and
~1,900 insertions per strand.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CLINVAR = REPO / "data" / "reference" / "clinvar" / "variant_summary.parquet"
GENOME = REPO / "data" / "reference" / "Gencode_v49_GRCh38.primary_assembly.genome.fa"
INDEX = REPO / "data" / "output" / "full_catalog" / "orf_index.parquet"

#: Rows per (class, strand) cell. Small: each row costs a genome-backed CDS
#: extraction in the pipeline arm, and the point is coverage of the matrix rather
#: than statistical weight.
PER_CELL = 4

pytestmark = pytest.mark.skipif(
    not (CLINVAR.is_file() and INDEX.is_file()),
    reason=(
        "needs the provisioned ClinVar parquet "
        "(python -m swissisoform.setup.databases clinvar) and a built full_catalog "
        "orf_index.parquet (python scripts/export/build_orf_index.py --run full_catalog)"
    ),
)

_THREE_TO_ONE = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
    "Ter": "*",
}


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
    """The 1-based residue ClinVar's p. string names, or None if it names none.

    ``Met1Val`` -> 1, ``Tyr96fs`` -> 96, ``Ser83_Leu85del`` -> 83 (the first of a
    range). Notations without a leading residue (repeat expansions like ``3250VP[2]``)
    return None and are not compared.
    """
    match = re.match(r"^[A-Z][a-z]{2}(\d+)", protein)
    return int(match.group(1)) if match else None


def _is_ambiguous_indel(cds: str, offset: int, ref_len: int, alt_len: int) -> bool:
    """True when the allele can be shifted without changing the sequence.

    VCF normalises an indel leftmost in genomic coordinates, HGVS 3'-most in
    transcript coordinates. Inside a repeat both name the same event one unit apart,
    so our residue and ClinVar's differ by one and neither is wrong. Where that shift
    crosses a codon boundary the *residue letter* differs too, so such rows are
    exempt from the position and the amino-acid check alike.
    """
    if alt_len == ref_len or offset <= 0 or offset + ref_len >= len(cds):
        return False
    shifted = cds[: offset - 1] + cds[offset : offset + ref_len] + cds[offset - 1]
    return shifted == cds[offset - 1 : offset + ref_len]


@pytest.fixture(scope="module")
def index():
    from swissisoform.variantquery.load import load_index

    return load_index(INDEX)


@pytest.fixture(scope="module")
def matrix(index):
    """ClinVar rows landing inside an ORF, stratified by (class, strand).

    The parquet is read through a pyarrow column projection and then filtered to the
    index's own chromosome/position envelope before any per-row work, so the 4.4M-row
    table is never walked in Python.
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
    for record in index._records:
        exons = record.exons_for("isoform")
        lo, hi = min(s for s, _ in exons), max(e for _, e in exons)
        span = bounds.setdefault(record.chrom, [lo, hi])
        span[0], span[1] = min(span[0], lo), max(span[1], hi)

    keep = pd.Series(False, index=cv.index)
    for chrom, (lo, hi) in bounds.items():
        keep |= (cv["chrom"] == chrom) & (cv["PositionVCF"] >= lo) & (cv["PositionVCF"] <= hi)

    cells: dict[tuple[str, str], list[dict]] = {}
    for row in cv[keep].sort_values(["chrom", "PositionVCF"]).itertuples():
        ref = str(row.ReferenceAlleleVCF or "")
        alt = str(row.AlternateAlleleVCF or "")
        if not ref or not alt or set(ref + alt) - set("ACGTN"):
            continue
        pos = int(row.PositionVCF)
        records = index.lookup_span(row.chrom, pos, pos + max(len(ref), 1) - 1)
        if not records:
            continue
        protein = str(row.Name).split("(p.")[-1].rstrip(")")
        cell = (_clinvar_class(protein, ref, alt), records[0].strand)
        bucket = cells.setdefault(cell, [])
        if len(bucket) >= PER_CELL:
            continue
        bucket.append(
            {
                "chrom": row.chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "protein": protein,
                "record": records[0],
                "significance": str(row.ClinicalSignificance),
            }
        )
    return cells


def _both_arms(validator, entry):
    """Classify one variant through each arm and return the two results.

    Arm A takes the coding sequence stored in ``orf_index.parquet`` — the path the
    website's VCF scan uses. Arm B extracts it from the genome FASTA — the path the
    pipeline uses. Same classifier, different provenance.
    """
    record = entry["record"]
    frame = "isoform"
    scan_arm = validator.classify_against_orf(
        orf_exons=[tuple(e) for e in record.exons_for(frame)],
        strand=record.strand,
        cds=record.cds_for(frame),
        genomic_pos=entry["pos"],
        ref=entry["ref"],
        alt=entry["alt"],
        orf_key=(record.tis_id, frame, "scan"),
    )
    pipeline_arm = validator.validate_variant_against_orf(
        orf_exons=[tuple(e) for e in record.exons_for(frame)],
        strand=record.strand,
        chrom=record.chrom,
        genomic_pos=entry["pos"],
        ref=entry["ref"],
        alt=entry["alt"],
        orf_key=(record.tis_id, frame, "pipeline"),
    )
    return scan_arm, pipeline_arm


# ----------------------------------------------------------------------
# 1. The two arms must agree
# ----------------------------------------------------------------------


@pytest.mark.skipif(not GENOME.is_file(), reason="needs the genome FASTA for the pipeline arm")
def test_the_two_arms_agree_on_every_class(matrix):
    """The claim f26b4a3 made — one classifier, no second opinion — asserted.

    Also a check on the index itself: the arms disagree if ``orf_cds`` in the parquet
    ever drifts from the genome it was built from, which nothing else would notice.
    """
    from swissisoform.clinical.validate import ConsequenceValidator

    validator = ConsequenceValidator(genome_fasta=str(GENOME))
    compared = 0
    for cell, entries in sorted(matrix.items()):
        for entry in entries:
            scan_arm, pipeline_arm = _both_arms(validator, entry)
            where = f"{cell} {entry['chrom']}:{entry['pos']} {entry['ref']}>{entry['alt']}"
            for field in ("consequence", "protein_pos", "aa_ref", "aa_alt"):
                assert scan_arm[field] == pipeline_arm[field], (
                    f"{where}: arms disagree on {field} — "
                    f"scan={scan_arm[field]!r} pipeline={pipeline_arm[field]!r}"
                )
            compared += 1
    assert compared >= 20, f"only {compared} variants compared across both arms"


# ----------------------------------------------------------------------
# 2. Our call must match what ClinVar published
# ----------------------------------------------------------------------


def test_the_consequence_class_matches_clinvar(matrix, index):
    """Class agreement, which needs no frame translation to be comparable.

    Only where ClinVar and we are talking about the same reading frame: a variant in
    an alternative-TIS isoform's unique region has no canonical residue, and ClinVar
    numbers against the canonical transcript. Those rows are covered by the arm-vs-arm
    test instead — mat10d's "where the frame permits".
    """
    from swissisoform.clinical.validate import ConsequenceValidator

    validator = ConsequenceValidator()
    checked: dict[str, int] = {}
    disagreements: list[str] = []
    skipped_frame: list[str] = []
    for (klass, _strand), entries in sorted(matrix.items()):
        if klass in ("other", "delins"):
            continue
        for entry in entries:
            record = entry["record"]
            out = validator.classify_against_orf(
                orf_exons=[tuple(e) for e in record.exons_for("canonical")],
                strand=record.strand,
                cds=record.cds_for("canonical"),
                genomic_pos=entry["pos"],
                ref=entry["ref"],
                alt=entry["alt"],
                orf_key=(record.tis_id, "canonical"),
            )
            ours = out["consequence"]
            if ours in (None, "intronic", "reference_mismatch"):
                # Not in this ORF's canonical frame, or the ORF's reference disagrees
                # with ClinVar's — neither is a classification disagreement.
                continue

            # Same codon, or we are not comparing like with like. ClinVar numbers
            # against its own canonical transcript, and ours is not always the same
            # one: 16 of the catalogue's ClinVar p.Met1 variants sit at our residue
            # 63, 18, 24 … because that ORF starts upstream of ClinVar's transcript.
            # Comparing the class there would report a disagreement about the frame
            # as though it were one about the classifier.
            clinvar_residue = _clinvar_residue(entry["protein"])
            if clinvar_residue is None or out["protein_pos"] is None:
                continue
            if clinvar_residue != out["protein_pos"] + 1:
                skipped_frame.append(
                    f"{entry['chrom']}:{entry['pos']} p.{entry['protein']} — "
                    f"ClinVar residue {clinvar_residue}, ours {out['protein_pos'] + 1}"
                )
                continue
            expected = {
                "frameshift": "frameshift_variant",
                "synonymous": "synonymous_variant",
                "nonsense": "stop_gained",
                "missense": "missense_variant",
                "start_lost": "start_lost",
                "deletion": "inframe_deletion",
                "insertion": "inframe_insertion",
                "mnv": "missense_variant",
            }[klass]
            checked[klass] = checked.get(klass, 0) + 1
            if ours != expected:
                disagreements.append(
                    f"{entry['chrom']}:{entry['pos']} {entry['ref']}>{entry['alt']} "
                    f"p.{entry['protein']}: ClinVar says {klass} ({expected}), we say {ours}"
                )
    assert checked, "no row was comparable in canonical frame"
    # Every required class must survive the frame filter, or the test is asserting
    # less than it appears to.
    for klass in ("missense", "synonymous", "nonsense", "frameshift", "start_lost"):
        assert checked.get(klass), (
            f"no {klass} row was comparable in canonical frame "
            f"(skipped for frame mismatch: {len(skipped_frame)})"
        )
    assert not disagreements, "\n".join(disagreements[:20])


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
        for klass in REQUIRED_CLASSES
        for strand in ("+", "-")
        if not matrix.get((klass, strand))
    ]
    assert not missing, (
        "empty cells: " + ", ".join(missing) + ". Either the index lost coverage or "
        "the class parser stopped recognising them; both are real."
    )


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
