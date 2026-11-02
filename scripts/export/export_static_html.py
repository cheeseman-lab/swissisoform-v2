r"""Export a staged site as ONE self-contained, offline HTML file.

For sharing a run without deploying it. Renders every page of the real Flask app
(landing, about, each gene, each isoform) with its test client against a staged
``website/data/`` (``prepare_deploy.sh`` output), then packs them into one file:

- the site's CSS/JS are inlined, and the two CDN libraries the pages load (3Dmol,
  plotly) are embedded once and handed to every page as blob URLs;
- the files the pages fetch at runtime (landing JSON, structure CIFs, colour maps,
  PAE maps) are embedded behind a ``fetch()`` shim;
- each page lives in its own ``<iframe srcdoc>``, so ids and scripts never collide;
  a sidebar lists every page and in-site links (``<a href>`` and
  ``window.location`` assignments) switch to the matching page;
- pages are stored gzip + base64 (CIF text compresses about 5x) and inflated on
  demand with the browser's ``DecompressionStream``.

What the server would do and a file cannot — the VCF scan, CIF downloads, the JSON
dump — answers 404 inside the export.

Usage (from the repo root, after ``cd website && RUN=<run> bash prepare_deploy.sh``):
    python scripts/export/export_static_html.py --data-dir website/data \\
        --out data/output/<run>/<run>.html --title "<run>"

The CDN libraries are read from ``data/reference/site_vendor/`` and downloaded there
(pinned versions, the same URLs the templates name) on first use.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import html
import json
import os
import re
import sys
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VENDOR_DIR = ROOT / "data" / "reference" / "site_vendor"
#: The outer shell and the per-page shim, kept as files rather than Python strings.
ASSETS = Path(__file__).resolve().parent / "static_html"
#: CDN script URL (exactly as the templates name it) → local file name.
VENDOR_FILES = {
    "https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js": "3Dmol-min.js",
    "https://cdn.plot.ly/plotly-basic-2.35.2.min.js": "plotly-basic-2.35.2.min.js",
}
_FETCHED = re.compile(
    r'(/(?:structures|structure-colors|structure-pae)/[^"\'\s<>?#]+|/api/landing\.json[^"\'\s<>#]*)'
)


def script_safe(s: str) -> str:
    """Make ``s`` safe inside a ``<script>`` element (no early ``</script>``)."""
    return s.replace("</script", "<\\/script").replace("</SCRIPT", "<\\/SCRIPT")


def page_paths_from_landing(landing: object) -> list[str]:
    """Gene + isoform page paths named by the landing payload (``landing.py``).

    Genes carry ``n`` (name) and ``i`` (isoforms); each isoform ``s`` (its slug).
    The landing table and gene pages build these links client-side, so they cannot
    be crawled from the HTML.
    """
    out: list[str] = []

    def walk(o: object) -> None:
        if isinstance(o, dict):
            if isinstance(o.get("n"), str) and isinstance(o.get("i"), list):
                out.append(f"/genes/{o['n']}")
                for iso in o["i"]:
                    if isinstance(iso, dict) and isinstance(iso.get("s"), str):
                        out.append(f"/genes/{o['n']}/isoforms/{iso['s']}")
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(landing)
    return out


def _fill(template: str, values: Mapping[str, str]) -> str:
    """Substitute ``__KEY__`` placeholders in one pass (a value is never re-scanned)."""
    return re.sub(r"__([A-Z]+)__", lambda m: values.get(m.group(1), m.group(0)), template)


def page_shim(embeds: Mapping[str, str], pages: list[str]) -> str:
    """The script each page gets first: embedded ``fetch()`` + in-site navigation."""
    js = (ASSETS / "page_shim.js").read_text()
    values = {"EMBEDS": json.dumps(embeds), "PAGES": json.dumps(sorted(pages))}
    return f"<script>{script_safe(_fill(js, values))}</script>"


def inline_page(
    text: str,
    *,
    static_root: Path,
    fetch: Callable[[str], str | None],
    pages: list[str],
) -> str:
    """Inline one rendered page's assets and runtime fetches."""

    def css(m: re.Match) -> str:
        f = static_root / m.group(1)
        return f"<style>{f.read_text()}</style>" if f.is_file() else m.group(0)

    def js(m: re.Match) -> str:
        f = static_root / m.group(1)
        return f"<script>{script_safe(f.read_text())}</script>" if f.is_file() else m.group(0)

    text = re.sub(r'<link[^>]+href="/static/([^"?]+)[^"]*"[^>]*>', css, text)
    text = re.sub(r'<script[^>]*src="/static/([^"?]+)[^"]*"[^>]*>\s*</script>', js, text)
    for url, name in VENDOR_FILES.items():
        text = re.sub(
            r'<script[^>]*src="' + re.escape(url) + r'"[^>]*>\s*</script>',
            f'<script src="__VENDOR_{name}__"></script>',
            text,
        )
    # Navigation by assignment can't be intercepted; route it through the shim.
    text = re.sub(r"window\.location(?:\.href)?\s*=(?!=)", "window.__swissisoGo =", text)
    embeds: dict[str, str] = {}
    for m in _FETCHED.findall(text):
        key = urlsplit(html.unescape(m)).path
        if key not in embeds:
            body = fetch(html.unescape(m))
            if body is not None:
                embeds[key] = body
    text = text.replace("<head>", "<head>" + page_shim(embeds, pages), 1)
    return re.sub(
        r'<a (?![^>]*target=)([^>]*href="https?://)', r'<a target="_blank" rel="noopener" \1', text
    )


def page_label(path: str) -> str:
    """Sidebar label for a page path."""
    if path == "/":
        return "Overview (all genes)"
    if path == "/about":
        return "About / criteria"
    parts = [unquote(x) for x in path.strip("/").split("/")]
    return f"{parts[1]} — gene" if len(parts) == 2 else f"{parts[1]} — isoform {parts[3]}"


def build_shell(pages: Mapping[str, str], vendor: Mapping[str, str], title: str, note: str) -> str:
    """The outer document: sidebar, one iframe, the gzipped page store, the libraries."""
    order = ["/"] + sorted(p for p in pages if p.startswith("/genes/")) + ["/about"]
    nav = "\n".join(
        f'<a href="#" data-p="{html.escape(p)}" class="{"iso" if "/isoforms/" in p else "top"}">'
        f"{html.escape(page_label(p))}</a>"
        for p in order
        if p in pages
    )
    store = json.dumps(
        {k: base64.b64encode(gzip.compress(v.encode(), 9)).decode() for k, v in pages.items()}
    )
    vendor_tags = "\n".join(
        f'<script type="text/x-vendor" data-name="{html.escape(n)}">{script_safe(src)}</script>'
        for n, src in vendor.items()
    )
    values = {
        "TITLE": html.escape(title),
        "NOTE": html.escape(note),
        "NAV": nav,
        "VENDOR": vendor_tags,
        "STORE": script_safe(store),
    }
    return _fill((ASSETS / "shell.html").read_text(), values)


def load_vendor(vendor_dir: Path) -> dict[str, str]:
    """The pinned CDN libraries, downloading any that are not cached yet."""
    out = {}
    for url, name in VENDOR_FILES.items():
        f = vendor_dir / name
        if not f.is_file():
            vendor_dir.mkdir(parents=True, exist_ok=True)
            print(f"downloading {url} -> {f}")
            with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 (pinned https URL)
                f.write_bytes(r.read())
        out[name] = f.read_text()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="staged site data (website/data after prepare_deploy.sh)",
    )
    ap.add_argument("--out", type=Path, required=True, help="the .html file to write")
    ap.add_argument("--title", default="SwissIsoform export")
    ap.add_argument("--note", default="", help="one line under the title in the sidebar")
    ap.add_argument("--vendor-dir", type=Path, default=DEFAULT_VENDOR_DIR)
    ap.add_argument(
        "--site-src",
        type=Path,
        default=ROOT / "website" / "src",
        help="the site package to render with (default: this checkout's website/src)",
    )
    args = ap.parse_args(argv)

    # The staged site imports exactly what the deployed image would: website/src
    # (with prepare_deploy's vendored swissisoform copy) on the path, nothing else.
    os.environ["SWISSISOFORM_DATA_DIR"] = str(args.data_dir.resolve())
    sys.path.insert(0, str(args.site_src.resolve()))
    from swissisoform_site.app import app

    client = app.test_client()
    static_root = Path(app.static_folder)

    def get(path: str):
        return client.get(path, headers={"Accept-Encoding": "identity"})

    landing = get("/api/landing.json")
    if landing.status_code != 200:
        raise SystemExit(
            f"/api/landing.json returned {landing.status_code}; is {args.data_dir} staged?"
        )
    queue = ["/", "/about", *page_paths_from_landing(json.loads(landing.get_data()))]
    rendered: dict[str, str] = {}
    for path in queue:
        if path in rendered:
            continue
        r = get(path)
        if r.status_code != 200 or "html" not in r.headers.get("Content-Type", ""):
            print(f"skip {path}: {r.status_code}")
            continue
        rendered[path] = r.get_data(as_text=True)

    def fetch(url: str) -> str | None:
        r = get(url)
        return r.get_data(as_text=True) if r.status_code == 200 else None

    paths = list(rendered)
    pages = {
        p: inline_page(t, static_root=static_root, fetch=fetch, pages=paths)
        for p, t in rendered.items()
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        build_shell(pages, load_vendor(args.vendor_dir), args.title, args.note), encoding="utf-8"
    )
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB, {len(pages)} pages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
