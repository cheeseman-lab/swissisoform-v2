"""Thin CLI for the Affinage generef fetch.

Logic lives in ``swissisoform.setup.generef``. Queries the Affinage API for each
gene symbol (function/localization/keywords) plus a minimal UniProtKB accession
lookup, and writes ``data/reference/generef/generef.json``.

Usage:
    python scripts/setup/fetch_generef.py --genes CBX1 CDC34 TP53
    python scripts/setup/fetch_generef.py --gene-list genes.txt --merge
    python scripts/setup/fetch_generef.py --combined        # every gene in the catalog (slow)
"""

from __future__ import annotations

from swissisoform.setup.generef import main

if __name__ == "__main__":
    raise SystemExit(main())
