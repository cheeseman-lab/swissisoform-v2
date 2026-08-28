"""Put the site package and the backend modules on the path.

The website is a separate deployable with its own ``pyproject.toml``, normally run
with ``PYTHONPATH=/app/src`` set by the Dockerfile. These tests reproduce that plus
the repo's ``src/``, so ``swissisoform.variantquery`` resolves without running
``prepare_deploy.sh`` first.

**Order matters, and getting it wrong is silent.** ``prepare_deploy.sh`` stages a
*copy* of the backend modules at ``website/src/swissisoform/`` for the Docker build.
If that directory precedes the repo's ``src/`` on ``sys.path``, the whole suite
tests the staged snapshot instead of the working tree — so an edit to
``swissisoform/variantquery/`` appears to have no effect and the tests keep passing
against stale code. The repo's ``src/`` therefore goes first; ``swissisoform_site``
exists only under ``website/src`` so it is unaffected by the ordering.
"""

from __future__ import annotations

import sys
from pathlib import Path

WEBSITE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WEBSITE_ROOT.parent

# Insert in reverse: each insert(0) puts its path in front, so the last one wins.
for path in (WEBSITE_ROOT / "src", REPO_ROOT / "src"):
    entry = str(path)
    if entry in sys.path:
        sys.path.remove(entry)
    sys.path.insert(0, entry)
