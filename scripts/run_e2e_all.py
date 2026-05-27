"""Full-catalog run — thin wrapper around run.py --all.

Preserved for muscle memory.  All logic lives in run.py.
"""

from __future__ import annotations

import sys

from run import main

if __name__ == "__main__":
    sys.exit(main(["--all", *sys.argv[1:]]))
