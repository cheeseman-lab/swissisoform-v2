"""Thin CLI for the ESM Atlas SAE feature-term fetch.

Logic lives in ``swissisoform.setup.sae_atlas``. Idempotent network fetch of the
ESM-C 6B layer-60 SAE feature dictionary → ``data/reference/sae_atlas/``; skips
when already present (``--force`` to refetch). The runtime read-side
(``load_atlas`` / ``atlas_provenance``) lives in ``swissisoform.plm.atlas``.

Usage:
    python scripts/setup/fetch_sae_atlas.py            # fetch if missing
    python scripts/setup/fetch_sae_atlas.py --force     # refetch
"""

from __future__ import annotations

from swissisoform.setup.sae_atlas import main

if __name__ == "__main__":
    raise SystemExit(main())
