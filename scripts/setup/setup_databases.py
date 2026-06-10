"""Thin CLI for reference-database setup.

Logic lives in ``swissisoform.setup.databases``. Idempotent fetch + build of
the external reference databases the expensive annotation modules depend on
(diamond, gnomad, clinvar, cosmic, alphamissense, deeploc, pepquery, hal,
gencode, signalp, targetp, interproscan).

Usage:
    python scripts/setup/setup_databases.py all
    python scripts/setup/setup_databases.py diamond
    python scripts/setup/setup_databases.py clinvar --refresh
"""

from __future__ import annotations

import sys

from swissisoform.setup.databases import main

if __name__ == "__main__":
    sys.exit(main())
