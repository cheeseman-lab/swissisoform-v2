"""Module entry point: ``python -m swissisoform run ...``."""

from __future__ import annotations

import sys

from swissisoform.cli import main

if __name__ == "__main__":
    sys.exit(main())
