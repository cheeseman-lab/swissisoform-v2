"""Canonical-start site annotations (pipeline-followups §1).

The canonical (Annotated) start gets the symmetric twins of the alt-start Kozak
features + point conservation, emitted as ``canonical_initiation_context_*`` /
``canonical_conservation_*``. These cover the parts testable without BigWigs;
the BigWig conservation path is exercised by the full run.
"""

from __future__ import annotations

import pandas as pd

from swissisoform.assembly import canonical_start_position
from swissisoform.config import PipelineConfig
from swissisoform.io.canonical import get_canonical_genome_positions
from swissisoform.modules.initiation_context import InitiationContextModule

_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


class TestCanonicalGenomePosition:
    """Imputed canonical ``GenomePos`` must be 0-based half-open at the codon.

    GTF coords are 1-based; the +-strand start was used without the −1, which
    shifted imputed canonicals one base 3' of the actual start codon.
    """

    def test_plus_strand_start_is_zero_based_on_the_codon(self):
        # ATG occupies 0-based genome[100:103]; GTF (1-based) CDS start is 101.
        genome = "N" * 100 + "ATG" + "C" * 47
        cds = pd.DataFrame(
            {
                "transcript_id": ["T1"],
                "chromosome": ["chr1"],
                "start": [101],
                "end": [150],
                "strand": ["+"],
            }
        )
        gp = get_canonical_genome_positions(cds).set_index("Tid").loc["T1", "GenomePos"]
        lo = int(gp.rsplit(":", 2)[1].split("-")[0])
        assert genome[lo : lo + 3] == "ATG"  # 0-based start anchors on the codon

    def test_minus_strand_anchor_is_zero_based_on_the_codon(self):
        # minus transcript: the mRNA start codon ATG is revcomp(CAT); plus-strand
        # genome carries CAT at the high CDS end. GTF (1-based inclusive) end E
        # → 0-based half-open hi = E; the anchor base sits at hi-1.
        genome = "N" * 100 + "CAT" + "N" * 47  # CAT at 0-based [100:103]
        cds = pd.DataFrame(
            {
                "transcript_id": ["T1"],
                "chromosome": ["chr1"],
                "start": [60],
                "end": [103],
                "strand": ["-"],
            }
        )
        gp = get_canonical_genome_positions(cds).set_index("Tid").loc["T1", "GenomePos"]
        hi = int(gp.rsplit(":", 2)[1].split("-")[1])
        assert genome[hi - 3 : hi].translate(_COMPLEMENT)[::-1] == "ATG"


class TestCanonicalStartPosition:
    def test_plus_strand_is_first_interval_start(self):
        coe = [(531931, 532108), (540000, 540100)]
        assert canonical_start_position(coe, "+") == 531931

    def test_minus_strand_is_last_interval_end(self):
        coe = [(48071437, 48071579), (48076864, 48077004)]
        assert canonical_start_position(coe, "-") == 48077004

    def test_empty_returns_none(self):
        assert canonical_start_position([], "+") is None
        assert canonical_start_position(None, "-") is None


class TestInitiationContextCanonical:
    def _site(self, canon_kozak):
        from types import SimpleNamespace

        return SimpleNamespace(kozak_context="GCCGCCGCCATGG", canonical_kozak_context=canon_kozak)

    def test_canonical_kozak_features(self):
        mod = InitiationContextModule(PipelineConfig())
        out = mod.annotate_canonical_site(self._site("GCCGCCGCCATGG"))
        assert out is not None
        assert out["kozak_context"] == "GCCGCCGCCATGG"
        assert out["kozak_hamming_full"] is not None
        assert out["kozak_window_gc_content"] is not None

    def test_none_when_no_canonical_kozak(self):
        mod = InitiationContextModule(PipelineConfig())
        assert mod.annotate_canonical_site(self._site(None)) is None
