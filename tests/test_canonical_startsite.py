"""Canonical-start site annotations (pipeline-followups §1).

The canonical (Annotated) start gets the symmetric twins of the alt-start Kozak
features + point conservation, emitted as ``canonical_initiation_context_*`` /
``canonical_conservation_*``. These cover the parts testable without BigWigs;
the BigWig conservation path is exercised by the full run.
"""

from __future__ import annotations

from swissisoform.assembly import canonical_start_position
from swissisoform.config import PipelineConfig
from swissisoform.modules.initiation_context import InitiationContextModule


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
