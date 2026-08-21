"""Where an isoform bar sits, and whether its variant markers sit on it.

``_make_gene_protein_view`` (bars, domains, motifs, annotated variants) and
``variantquery.frame.plotly_x`` (uploaded VCF markers) both convert an isoform
residue into the figure's canonical-residue space. They used to do it with two
copies of ``canonical_len - isoform_len``, which is exact only when the two proteins
share a C-terminus — for a uORF the adapter fell back to ``offset = 0`` while
``plotly_x`` applied the shift anyway, so the marker was drawn hundreds of residues
from its own bar. Both now read ``canonical_x_offset_nt`` off the ORF index.

Lives under ``website/tests`` rather than ``tests/`` deliberately: this repo's
``swissisoform_site`` is installed editable from a *sibling* checkout, and only
``website/tests/conftest.py`` puts the local ``website/src`` first on ``sys.path``.
The coordinate assertions in ``tests/test_plots_protein.py`` therefore exercise
whichever copy is installed, not this one.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from swissisoform_site.app import _make_gene_protein_view

CAN_LEN = 100
UORF_LEN = 17
#: ABHD2's real uORF: 184 nt upstream of the canonical ATG, and not a whole number
#: of codons — it does not read in the canonical frame.
UORF_OFFSET_NT = -184
UORF_TIS = "chr15:89088485:+:ATG:ENST1"
EXT_LEN = 130
EXT_TIS = "chr1:100:+:CTG:ENST2"


class _FakeIndex:
    """Just enough ``OrfIndex`` for the adapter and ``plotly_x``."""

    def __init__(self, records: dict[str, tuple[int, int | None]]):
        self._records = records

    def by_tis_id(self, tis_id: str):
        entry = self._records.get(tis_id)
        if entry is None:
            return None
        isoform_len, offset_nt = entry
        return SimpleNamespace(
            canonical_len=CAN_LEN, isoform_len=isoform_len, canonical_x_offset_nt=offset_nt
        )


def _isoform(orf_type, diff_space, isoform_len, diff_end, tis_id):
    return SimpleNamespace(
        orf_type=orf_type,
        diff_space=diff_space,
        diff_end=diff_end,
        isoform_len=isoform_len,
        canonical_len=CAN_LEN,
        start_codon="ATG",
        tis_id=tis_id,
        raw={},
        variants_all=[],
    )


def _gene(*isoforms):
    return SimpleNamespace(canonical_len=CAN_LEN, isoforms=list(isoforms))


def _uorf_gene():
    # diff_end == isoform_len: the whole isoform is differential, i.e. no shared region.
    return _gene(_isoform("uorf", "isoform", UORF_LEN, UORF_LEN, UORF_TIS))


def _bar(view, orf_type):
    return next(b for b in view.bars if b["orf_type"] == orf_type)


@pytest.fixture
def index(monkeypatch):
    fake = _FakeIndex({
        UORF_TIS: (UORF_LEN, UORF_OFFSET_NT),
        EXT_TIS: (EXT_LEN, (CAN_LEN - EXT_LEN) * 3),
    })
    monkeypatch.setattr("swissisoform_site.app.load_orf_index", lambda: fake)
    return fake


def test_uorf_bar_sits_upstream_of_the_canonical_start(index) -> None:
    """A uORF is in the 5'UTR, so its whole bar belongs left of x = 0.

    Right-alignment cannot express that — there is no shared C-terminus to align on —
    and the old fallback drew the bar at 0..isoform_len-1, on top of the canonical
    N-terminus, asserting a residue correspondence that does not exist.
    """
    bar = _bar(_make_gene_protein_view(_uorf_gene()), "uorf")
    assert bar["x0"] == UORF_OFFSET_NT / 3
    assert bar["x1"] < 0
    assert bar["diff_on_canonical"] is False  # none of it lies on the canonical bar


def test_uorf_x_is_fractional_when_it_reads_out_of_frame(index) -> None:
    """A uORF residue has no canonical counterpart; rounding would invent one."""
    bar = _bar(_make_gene_protein_view(_uorf_gene()), "uorf")
    assert bar["x0"] != int(bar["x0"])


def test_extension_placement_is_unchanged_by_the_offset(index) -> None:
    """The offset must reproduce right-alignment wherever right-alignment applied.

    An extension shares the canonical C-terminus, so its bar still ends flush with
    it — this is the no-op half of the change, and covers the ~93% of the catalogue
    that already drew correctly.
    """
    gene = _gene(_isoform("extended", "isoform", EXT_LEN, 30, EXT_TIS))
    bar = _bar(_make_gene_protein_view(gene), "extended")
    assert bar["x0"] == -(EXT_LEN - CAN_LEN)
    assert bar["x1"] == CAN_LEN - 1


def test_falls_back_to_the_old_placement_without_an_index(monkeypatch) -> None:
    """No index staged, or one predating the column → previous behaviour, not a crash."""
    monkeypatch.setattr("swissisoform_site.app.load_orf_index", lambda: None)
    assert _bar(_make_gene_protein_view(_uorf_gene()), "uorf")["x0"] == 0


def test_uploaded_marker_lands_on_the_bar_it_belongs_to(index) -> None:
    """The actual defect: the bar and its markers must come from the same number."""
    residue = 5
    hit = {
        "tis_id": UORF_TIS,
        "residue": residue,
        "frame": "isoform",
        "chrom": "chr15",
        "pos": 89088500,
        "ref": "C",
        "alt": "T",
        "consequence": "missense_variant",
        "aa_ref": "P",
        "aa_alt": "L",
        "region": "unique",
        "line_no": 1,
    }
    view = _make_gene_protein_view(_uorf_gene(), uploaded=[hit])
    bar = _bar(view, "uorf")
    marker = next(v for v in view.variants if v.get("source") == "uploaded")
    assert marker["pos"] == bar["x0"] + residue
    assert bar["x0"] <= marker["pos"] <= bar["x1"]
