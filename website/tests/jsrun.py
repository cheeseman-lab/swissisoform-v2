"""Run the site's browser JS under Node, for tests that pin its behaviour.

There is no JS toolchain in this repo (no package.json, no bundler), so the scripts
are loaded the way a browser loads them — as plain files evaluated against a
``window`` — inside Node's ``vm`` module. Tests that need this skip when ``node`` is
not on ``PATH``; ``conda install -c conda-forge nodejs`` in any env provides it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

JS_DIR = Path(__file__).resolve().parents[1] / "src" / "swissisoform_site" / "static" / "js"
NODE = shutil.which("node")

needs_node = pytest.mark.skipif(NODE is None, reason="node is not on PATH")

_OVERLAY = """
const fs = require("fs"), vm = require("vm");
const ctx = { window: {} };
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), ctx);
const input = JSON.parse(fs.readFileSync(0, "utf8"));
process.stdout.write(JSON.stringify(ctx.window.applyUploadedHits(input.fig, input.hits)));
"""


def apply_uploaded_hits(fig: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """``gene_overlay.js``'s ``applyUploadedHits(fig, hits)``, run in Node."""
    done = subprocess.run(
        [NODE, "-e", _OVERLAY, str(JS_DIR / "gene_overlay.js")],
        input=json.dumps({"fig": fig, "hits": hits}),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(done.stdout)


def run_script(source: str, *args: str) -> str:
    """Run a Node program (e.g. a test written against ``scanstate.js``)."""
    done = subprocess.run([NODE, "-e", source, *args], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise AssertionError(done.stderr or done.stdout)
    return done.stdout
