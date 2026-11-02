"""Ribo-TISH-format rows from external start-codon positions (``io/external_tis.py``).

The geometry is checked on a synthetic spliced transcript on both strands: the stop
codon sits in the second exon, so the ORF span crosses the intron. On real data the
same conversion reproduces Ribo-TISH's own rows (GenomePos, Start, AALen, AASeq)
for CDC73, RRM2 and PSMB2; that check needs the genome and is not a unit test.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from swissisoform.coords import iter_coding_positions
from swissisoform.io.external_tis import (
    StartSite,
    build_tis_rows,
    read_maf_starts,
    resolve_transcript,
    tis_row_for_start,
)
from swissisoform.models import TranscriptCoordinates

_COMP = str.maketrans("ACGT", "TGCA")

# 50-nt mRNA: CTG start at index 5, three Ala codons, TAA stop at index 17.
MRNA = "GGGGG" + "CTG" + "GCA" * 3 + "TAA" + "C" * 30
assert len(MRNA) == 50 and MRNA[17:20] == "TAA"


class FakeGenome:
    """``pysam.FastaFile``-like stub over one in-memory chromosome."""

    def __init__(self, seq: str) -> None:
        """Store the chromosome sequence."""
        self.seq = seq

    def fetch(self, chrom: str, start: int, end: int) -> str:
        return self.seq[start:end]


def _place(exons, strand):
    """Genome whose transcript on ``exons``/``strand`` reads ``MRNA``; and its gpos."""
    genome = ["N"] * 100
    gpos = [g for g, _ in iter_coding_positions(exons, strand)]
    for base, g in zip(MRNA, gpos, strict=True):
        genome[g - 1] = base if strand == "+" else base.translate(_COMP)
    return FakeGenome("".join(genome)), gpos


def _site(strand, gpos, **kw):
    lo, hi = sorted((gpos[5], gpos[7]))
    return StartSite(
        gene="G",
        chrom="chrT",
        strand=strand,
        codon_start=lo,
        codon_end=hi,
        transcript_id="T.1",
        tis_type="Truncated",
        **kw,
    )


PLUS_EXONS = [(10, 22), (40, 78)]  # 12 + 38 nt; the stop falls in exon 2
MINUS_EXONS = [(10, 48), (60, 72)]  # mRNA order: (60, 72) first


class TestTisRowForStart:
    def test_plus_strand_row(self):
        genome, gpos = _place(PLUS_EXONS, "+")
        sk = TranscriptCoordinates(transcript_id="T.1", chrom="chrT", strand="+", exons=PLUS_EXONS)
        row, why = tis_row_for_start(_site("+", gpos, codon="CTG"), "T.1", sk, genome)
        assert why is None
        # A at 0-based 15; stop TAA at mRNA 17-19 = genomic 46-48 (1-based), past the intron.
        assert row["GenomePos"] == "chrT:15-48:+"
        assert row["StartCodon"] == "CTG"
        assert row["Start"] == 5.0
        assert row["AASeq"] == "LAAA" and row["AALen"] == 4.0
        assert pd.isna(row["TISPvalue"]) and pd.isna(row["TISCounts"])

    def test_minus_strand_row(self):
        genome, gpos = _place(MINUS_EXONS, "-")
        sk = TranscriptCoordinates(transcript_id="T.1", chrom="chrT", strand="-", exons=MINUS_EXONS)
        row, why = tis_row_for_start(_site("-", gpos, codon="CTG"), "T.1", sk, genome)
        assert why is None
        # Ribo-TISH's minus convention: 0-based low end of the stop, 1-based A.
        assert row["GenomePos"] == f"chrT:{gpos[19] - 1}-{gpos[5]}:-"
        assert row["AASeq"] == "LAAA" and row["Start"] == 5.0

    def test_codon_mismatch_rejected(self):
        genome, gpos = _place(PLUS_EXONS, "+")
        sk = TranscriptCoordinates(transcript_id="T.1", chrom="chrT", strand="+", exons=PLUS_EXONS)
        row, why = tis_row_for_start(_site("+", gpos, codon="ATG"), "T.1", sk, genome)
        assert row is None and "CTG" in why

    def test_intronic_start_rejected(self):
        genome, _ = _place(PLUS_EXONS, "+")
        sk = TranscriptCoordinates(transcript_id="T.1", chrom="chrT", strand="+", exons=PLUS_EXONS)
        site = StartSite("G", "chrT", "+", 31, 33, "T.1", "Truncated")  # inside the intron
        row, why = tis_row_for_start(site, "T.1", sk, genome)
        assert row is None and "not on the exons" in why

    def test_no_stop_rejected(self):
        genome = FakeGenome("A" * 100)  # AAA… never reads a stop
        sk = TranscriptCoordinates(transcript_id="T.1", chrom="chrT", strand="+", exons=[(10, 40)])
        row, why = tis_row_for_start(
            StartSite("G", "chrT", "+", 12, 14, "T.1", "Extended"), "T.1", sk, genome
        )
        assert row is None and "no in-frame stop" in why


class TestResolveTranscript:
    site = StartSite("G", "chr1", "+", 1, 3, "ENST1.3", "Truncated")

    def test_same_version(self):
        assert resolve_transcript(self.site, {"ENST1": "ENST1.3"}, {}) == ("ENST1.3", "same")

    def test_version_changed(self):
        assert resolve_transcript(self.site, {"ENST1": "ENST1.5"}, {}) == (
            "ENST1.5",
            "version_changed",
        )

    def test_retired_falls_back_to_mane(self):
        assert resolve_transcript(self.site, {}, {"G": "ENST9.1"}) == ("ENST9.1", "retired_to_mane")

    def test_override_wins(self):
        got = resolve_transcript(
            self.site, {"ENST1": "ENST1.3"}, {"G": "ENST9.1"}, {"G": "ENST7.2"}
        )
        assert got == ("ENST7.2", "override")

    def test_unresolved(self):
        assert resolve_transcript(self.site, {}, {}) == (None, "unresolved")


def test_read_maf_starts_normalises_chrom_and_strand(tmp_path):
    p = tmp_path / "starts.tsv"
    p.write_text(
        "Hugo_Symbol\tChromosome\tStart_Position\tEnd_Position\tVariant_Type\tReference_Allele"
        "\tTumor_Seq_Allele2\tStrand\tTranscript_ID\tEntrez_Transcript_ID\tType\tProvenance\n"
        "RPL35A\t3\t197951160\t197951162\t\tCTG\t\t1\tENST00000464167.5\tNM_1\tTruncated\tipsc\n"
        "PSMB2\tchr1\t35605239\t35605241\t\tCTG\t\t-\tENST00000621781.4\tNM_2\tTruncated\thela\n"
    )
    a, b = read_maf_starts(p)
    assert (a.chrom, a.strand, a.codon, a.source) == ("chr3", "+", "CTG", "ipsc")
    assert (b.chrom, b.strand, b.codon_start, b.codon_end) == ("chr1", "-", 35605239, 35605241)


def test_read_maf_starts_requires_columns(tmp_path):
    p = tmp_path / "bad.tsv"
    p.write_text("Hugo_Symbol\tChromosome\nX\t1\n")
    with pytest.raises(ValueError, match="missing columns"):
        read_maf_starts(p)


def test_build_tis_rows_skips_and_imputes(monkeypatch):
    genome, gpos = _place(PLUS_EXONS, "+")
    sk = TranscriptCoordinates(transcript_id="T.1", chrom="chrT", strand="+", exons=PLUS_EXONS)
    ref = SimpleNamespace(
        exon_skeletons={"T.1": sk},
        genome_pos=None,
        utr_lengths=None,
        start_codons=None,
        protein_products=None,
    )
    tx = pd.DataFrame(
        [
            {
                "transcript_id": "T.1",
                "gene_id": "GENE1",
                "gene_name": "G",
                "gene_type": "protein_coding",
                "MANE_Select": True,
                "transcript_support_level": "1",
            }
        ]
    )

    def fake_impute(df, **_):
        return pd.DataFrame(
            [
                {
                    "Gid": "GENE1",
                    "Tid": "T.1",
                    "Symbol": "G",
                    "TisType": "Annotated",
                    "RecatTISType": "Annotated",
                    "AASeq": "MXXXX",
                    "Imputed": True,
                }
            ]
        )

    monkeypatch.setattr("swissisoform.io.canonical.impute_missing_canonical_starts", fake_impute)
    sites = [_site("+", gpos, codon="CTG"), _site("+", gpos, codon="GTG")]
    rows, notes = build_tis_rows(sites, ref, genome, tx)
    assert list(rows["Imputed"]) == [False, True]
    assert rows.loc[0, "Gid"] == "GENE1" and bool(rows.loc[0, "MANE_Select"]) is True
    assert notes[0]["skipped"] is None and "!= source GTG" in notes[1]["skipped"]


# --- Ribo-TISH evidence ------------------------------------------------------------

_HDR = "\t".join(
    ["Gid", "Tid", "Symbol", "GeneType", "GenomePos", "StartCodon", "Start", "Stop",
     "TisType", "TISGroup", "TISCounts", "TISPvalue", "RiboPvalue", "RiboPStatus",
     "FisherPvalue", "TISQvalue", "FrameQvalue", "FisherQvalue", "AALen", "Seq", "AASeq",
     "InFrameCount"]
)  # fmt: skip


def _predict_line(genome_pos, tid, counts, tisp, ribop, q):
    f = ["G", tid, "S", "pc", genome_pos, "CTG", "5", "20", "Truncated", "1", str(counts),
         str(tisp), str(ribop), "T", "0.1", "0.1", "0.1", str(q), "4", "ACGT", "LAAA"]  # fmt: skip
    return "\t".join(f)


def test_site_key_strands():
    from swissisoform.io.external_tis import site_key

    assert site_key("chr1:15-48:+") == ("chr1", "+", 15)
    assert site_key("chr1:35603266-35605241:-") == ("chr1", "-", 35605241)


def test_attach_evidence_by_site_across_replicates_and_transcripts(tmp_path):
    from swissisoform.io.external_tis import attach_evidence

    # Replicate 1 has a header and an extra column; it calls the site on two
    # transcripts (other annotation versions) with different p-values.
    r1 = tmp_path / "r1.txt"
    r1.write_text("\n".join([
        _HDR,
        _predict_line("chrT:15-48:+", "T.1", 10, 0.004, 0.001, 0.02) + "\t3",
        _predict_line("chrT:15-60:+", "T.9", 10, 0.0001, 0.01, 0.003) + "\t3",
    ]) + "\n")  # fmt: skip
    # Replicate 2 is headerless and calls it once more; sample B never calls it.
    r2 = tmp_path / "r2.txt"
    r2.write_text(_predict_line("chrT:15-48:+", "T.1", 6, 0.002, "None", 0.01) + "\n")
    b = tmp_path / "b.txt"
    b.write_text(_predict_line("chrT:99-120:+", "T.1", 50, 0.001, 0.001, 0.001) + "\n")
    rows = pd.DataFrame(
        {"GenomePos": ["chrT:15-48:+", "chrT:200-230:+"], "TISPvalue": [float("nan")] * 2}
    )
    manifest = pd.DataFrame({"sample": ["A", "A", "B"], "predict_file": [r1, r2, b]})

    out = attach_evidence(rows, manifest)
    assert "TISPvalue" not in out  # per-sample-mode column replaced by the wide ones
    assert list(out["present_A"]) == [True, False] and list(out["present_B"]) == [False, False]
    # Counts: max over transcripts within a replicate, summed over replicates.
    assert out.loc[0, "A_TISCounts"] == 16.0
    # p/q: the smallest over every call of the site; "None" is not a value.
    assert out.loc[0, "A_TISPvalue"] == 0.0001
    assert out.loc[0, "A_RiboPvalue"] == 0.001
    assert out.loc[0, "A_FisherQvalue"] == 0.003
    assert pd.isna(out.loc[0, "A_NormTISCounts"]) and pd.isna(out.loc[1, "A_TISCounts"])
    assert list(out["samples"]) == [["A"], []] and list(out["n_samples"]) == [1, 0]


def test_expression_kept_without_normalised_count():
    # External evidence has no RNA-seq totals: the call still counts toward D1.
    from swissisoform.assembly import _expression_from_row

    row = pd.Series(
        {
            "present_A": True,
            "A_TISCounts": 16.0,
            "A_NormTISCounts": float("nan"),
            "A_FisherQvalue": 0.003,
            "present_B": False,
        }
    )
    expr = _expression_from_row(row, ["A", "B"])
    assert list(expr) == ["A"]
    assert expr["A"].raw_count == 16 and expr["A"].cpm is None
    assert expr["A"].initiation_efficiency is None
