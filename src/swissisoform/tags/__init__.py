"""Tag vocabulary — candidate proposal, review, and (later) the frozen registry.

Issue #30 replaces the per-category ``interesting / neutral / not_interesting``
verdict with per-category tags: tri-state (on / off / not-evaluable), each firing
on roughly 10-60% of the corpus, each carrying one cited number. Most are fired by
code; only genuine judgment calls go through the LLM tool loop.

Today this package holds the *proposal* half — the hand-curated seeds
(:mod:`.seeds`) and the sweep that turns the frozen distributions into a
reviewable candidate table (:mod:`.candidates`). The accepted rows of that table
become the registry.

How a named quantity is computed from the parquet lives one level up, in
:mod:`swissisoform.metrics`: both this package and the distributions builder
consume it, and neither owns it, so putting it here would have made the lower
layer import the higher one.

Submodules are imported explicitly rather than re-exported, so importing the
seeds does not drag in the sweep.
"""
