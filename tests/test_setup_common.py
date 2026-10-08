"""The provenance primitives shared by the setup builders and the LLM stamps."""

from __future__ import annotations

from swissisoform.setup._common import ROOT, code_provenance


def test_code_provenance_reads_the_checkout():
    prov = code_provenance()
    assert set(prov) == {"commit", "dirty"}
    assert isinstance(prov["commit"], str) and len(prov["commit"]) == 40
    assert isinstance(prov["dirty"], bool)
    assert code_provenance(ROOT) == prov


def test_code_provenance_is_none_outside_a_checkout(tmp_path):
    """Not a repo: say so with Nones rather than raising into the build."""
    assert code_provenance(tmp_path) == {"commit": None, "dirty": None}
