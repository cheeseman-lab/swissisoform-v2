"""The single-file HTML export's pure helpers (``scripts/export/export_static_html.py``).

Loaded by path: ``scripts/export`` is not a package. Rendering real pages needs the
staged site and is exercised by running the script, not here.
"""

from __future__ import annotations

import base64
import gzip
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "export_static_html", ROOT / "scripts" / "export" / "export_static_html.py"
)
exp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp)


def test_script_safe_breaks_closing_tags():
    assert "</script" not in exp.script_safe('x="</script>"; y="</SCRIPT>"')


def test_page_paths_from_landing():
    landing = {
        "genes": [
            {"n": "PSMB2", "i": [{"s": "chr1-1---CTG-T1"}]},
            {"n": "WDR5", "i": [{"s": "a"}, {"s": "b"}]},
        ]
    }
    assert exp.page_paths_from_landing(landing) == [
        "/genes/PSMB2",
        "/genes/PSMB2/isoforms/chr1-1---CTG-T1",
        "/genes/WDR5",
        "/genes/WDR5/isoforms/a",
        "/genes/WDR5/isoforms/b",
    ]


def test_inline_page(tmp_path):
    (tmp_path / "css").mkdir()
    (tmp_path / "js").mkdir()
    (tmp_path / "css/site.css").write_text("body{color:red}")
    (tmp_path / "js/app.js").write_text("var s='</script>';")
    page = (
        '<html><head><link rel="stylesheet" href="/static/css/site.css?v=1"></head><body>'
        '<script src="https://cdn.plot.ly/plotly-basic-2.35.2.min.js" defer></script>'
        '<script src="/static/js/app.js"></script>'
        '<div data-cif="/structures/A__canonical__10aa.cif"></div>'
        '<a href="https://www.uniprot.org/x">u</a>'
        '<script>window.location = "/genes/A";</script></body></html>'
    )
    fetched = []

    def fetch(url):
        fetched.append(url)
        return "CIFDATA"

    out = exp.inline_page(page, static_root=tmp_path, fetch=fetch, pages=["/", "/genes/A"])
    assert "<style>body{color:red}</style>" in out
    assert "<\\/script>" in out and "/static/js/app.js" not in out
    assert '<script src="__VENDOR_plotly-basic-2.35.2.min.js__"></script>' in out
    assert fetched == ["/structures/A__canonical__10aa.cif"]
    embeds = json.loads(re.search(r"var E = (\{.*?\});\s*var PAGES", out).group(1))
    assert embeds == {"/structures/A__canonical__10aa.cif": "CIFDATA"}
    assert 'window.__swissisoGo = "/genes/A"' in out
    assert '<a target="_blank" rel="noopener" href="https://www.uniprot.org/x">' in out
    assert out.index("var E =") < out.index("<style>")  # the shim runs first


def test_build_shell_store_round_trips():
    pages = {
        "/": "<html>landing</html>",
        "/genes/A": "<html>gene</html>",
        "/about": "<html>a</html>",
    }
    shell = exp.build_shell(pages, {"lib.js": "var L=1;"}, "T", "note")
    store = json.loads(
        re.search(r'<script id="pages" type="application/json">(.*?)</script>', shell, re.S).group(
            1
        )
    )
    assert {k: gzip.decompress(base64.b64decode(v)).decode() for k, v in store.items()} == pages
    nav = re.findall(r'data-p="([^"]+)"', shell)
    assert nav == ["/", "/genes/A", "/about"]
    assert 'data-name="lib.js">var L=1;</script>' in shell
