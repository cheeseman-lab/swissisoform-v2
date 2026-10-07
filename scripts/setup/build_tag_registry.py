"""Thin CLI for the frozen tag-registry build.

Logic lives in ``swissisoform.setup.tags``. Re-runs the candidate sweep against a
frozen distributions version, keeps the rows the reviewer did not reject in
``figures/tag_vocab/tag_candidates.csv``, and writes the vocabulary plus each
tag's cutoff to ``data/reference/tags/<version>/``. The runtime read-side is
``swissisoform.tags.registry``; the evaluator is ``swissisoform.tags.evaluate``.

The version is frozen on purpose: every parquet stamps the registry version that
fired its tags, so rebuilding in place changes what an already-written column
means. Bump ``--version`` instead, and keep ``--force`` for genuine do-overs.

Build ``--cutoffs config`` first. It cuts the criterion tags at their live
``ScoringConfig`` thresholds, which is what makes the layer verifiable: the
criterion tags must reproduce ``isoform_scoring_criteria`` row for row before any
cutoff is allowed to move.

v1 and v2 are historical: rebuilding either name today writes today's vocabulary
under it. Name a new version (v4, ...) for a new build.

Usage:
    python scripts/setup/build_tag_registry.py --version v4 --cutoffs config
    python scripts/setup/build_tag_registry.py --version v4 --dist-version v4 \
        --cutoffs distribution
"""

from __future__ import annotations

from swissisoform.setup.tags import main

if __name__ == "__main__":
    raise SystemExit(main())
