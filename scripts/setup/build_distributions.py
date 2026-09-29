"""Thin CLI for the frozen metric-distributions build.

Logic lives in ``swissisoform.setup.distributions``. Profiles every metric in a
paired-TIS run — scored and unscored — stratified by ``orf_type``, and writes the
frozen tables to ``data/reference/distributions/<version>/``. The runtime
read-side (``percentile`` / ``value_at``) lives in ``swissisoform.distributions``.

The version is frozen on purpose: tag cutoffs are derived from it once and
recorded as scalars, so rebuilding in place changes what every existing tag means.
Bump ``--version`` instead, and keep ``--force`` for genuine do-overs.

Usage:
    python scripts/setup/build_distributions.py --run full_catalog --version v1
    python scripts/setup/build_distributions.py --run cheeseman50 --out /tmp/$USER/dist_smoke
"""

from __future__ import annotations

from swissisoform.setup.distributions import main

if __name__ == "__main__":
    raise SystemExit(main())
