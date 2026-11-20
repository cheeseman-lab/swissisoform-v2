"""The stdlib classifier: translation pinned to Biopython, and a light import chain.

The classifier moved out of ``clinical/validate.py`` (issue #33, item 2) so the
VCF scan can run without pandas and Biopython — in the website image and in the
browser under Pyodide. Its behaviour is pinned by the existing classifier suites
(``test_clinical_validate*.py``, ``test_variantquery_*.py``), which still go
through ``ConsequenceValidator``; this file pins the two things the move itself
could break.
"""

from __future__ import annotations

import itertools
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from swissisoform.variantclass import (
    CODON_TABLE,
    PROTEIN_LETTERS_1TO3,
    OrfClassifier,
    position_map,
    translate,
)

SRC = Path(__file__).resolve().parents[1] / "src"
IUPAC = "ACGTRYSWKMBDHVN"


def test_the_codon_table_is_complete() -> None:
    assert len(CODON_TABLE) == 64
    assert CODON_TABLE["ATG"] == "M"
    assert {c for c, aa in CODON_TABLE.items() if aa == "*"} == {"TAA", "TAG", "TGA"}


def test_translation_matches_biopython_on_every_iupac_codon() -> None:
    """Including ambiguity codes, which Biopython resolves (GCN→A, RAY→B, TAN→X)."""
    seq_mod = pytest.importorskip("Bio.Seq")
    mismatches = {}
    for codon in map("".join, itertools.product(IUPAC, repeat=3)):
        expected = str(seq_mod.Seq(codon).translate())
        if translate(codon) != expected:
            mismatches[codon] = (translate(codon), expected)
    assert not mismatches, dict(itertools.islice(mismatches.items(), 10))


def test_translation_matches_biopython_on_real_length_sequences() -> None:
    seq_mod = pytest.importorskip("Bio.Seq")
    import random

    rng = random.Random(33)
    for length in (0, 1, 2, 3, 300, 301, 302, 999):
        seq = "".join(rng.choice("ACGTacgtN") for _ in range(length))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Biopython warns on a partial codon
            assert translate(seq) == str(seq_mod.Seq(seq).translate()), length


def test_three_letter_codes_match_biopython() -> None:
    iupac_data = pytest.importorskip("Bio.Data.IUPACData")
    assert PROTEIN_LETTERS_1TO3 == iupac_data.protein_letters_1to3


def test_orf_classifier_caches_one_map_per_orf() -> None:
    clf = OrfClassifier()
    exons = [(100, 112)]
    first = clf.build_position_map_from_orf(exons, "+", orf_key="orf1")
    assert clf.build_position_map_from_orf(exons, "+", orf_key="orf1") is first
    assert first == position_map(exons, "+")
    assert first[101] == 0 and first[112] == 11


def test_orf_classifier_keeps_a_bounded_cache_of_recent_maps() -> None:
    """An exome-scale scan must not keep one map per ORF it ever touched."""
    clf = OrfClassifier(max_cached_maps=3)
    maps = [clf.build_position_map_from_orf([(i, i + 6)], "+", orf_key=i) for i in range(3)]
    # Touching ORF 0 again makes ORF 1 the least recently used, so ORF 3 evicts it.
    assert clf.build_position_map_from_orf([(0, 6)], "+", orf_key=0) is maps[0]
    clf.build_position_map_from_orf([(3, 9)], "+", orf_key=3)
    assert clf.build_position_map_from_orf([(0, 6)], "+", orf_key=0) is maps[0]
    rebuilt = clf.build_position_map_from_orf([(1, 7)], "+", orf_key=1)
    assert rebuilt is not maps[1] and rebuilt == maps[1]
    assert len(clf._position_maps) == 3


def test_the_pipeline_validator_classifies_through_the_scan_classifier() -> None:
    """One cache and one code path for the ORF frame, so the two cannot drift."""
    validate = pytest.importorskip("swissisoform.clinical.validate")
    validator = validate.ConsequenceValidator()
    assert isinstance(validator._orf, OrfClassifier)
    kwargs = dict(orf_exons=[(100, 106)], strand="+", cds="ATGGCC", genomic_pos=105)
    assert validator.classify_against_orf(**kwargs, ref="C", alt="A") == (
        OrfClassifier().classify_against_orf(**kwargs, ref="C", alt="A")
    )
    pos_map = validator.build_position_map_from_orf([(100, 106)], "+")
    assert validator._orf.build_position_map_from_orf([(100, 106)], "+") is pos_map


def test_orf_classifier_calls_a_missense() -> None:
    """ATG GCC → ATG GAC at the second codon: Ala2Asp."""
    out = OrfClassifier().classify_against_orf(
        orf_exons=[(100, 106)], strand="+", cds="ATGGCC", genomic_pos=105, ref="C", alt="A"
    )
    assert (out["consequence"], out["protein_pos"], out["aa_ref"], out["aa_alt"]) == (
        "missense_variant",
        1,
        "A",
        "D",
    )


def test_the_scan_path_imports_no_heavy_dependency() -> None:
    """The point of the move: pandas, numpy, Biopython, pyarrow and pysam stay out.

    A fresh interpreter, since this test session has long since imported them.
    """
    probe = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import swissisoform.variantquery, swissisoform.variantquery.scan; "
        "heavy = {'pandas', 'numpy', 'Bio', 'pyarrow', 'pysam'}; "
        "print(sorted(heavy & {m.split('.')[0] for m in sys.modules}))"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(SRC)], capture_output=True, text=True, check=True
    )
    assert done.stdout.strip() == "[]", done.stdout + done.stderr
