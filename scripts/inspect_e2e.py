"""5-gene diagnostic — thin wrapper around run.py --preset 5gene.

Preserved for muscle memory and existing sbatch scripts.  All logic
lives in run.py.
"""

from __future__ import annotations

import sys

from run import main

if __name__ == "__main__":
    sys.exit(main(["--preset", "5gene", *sys.argv[1:]]))
