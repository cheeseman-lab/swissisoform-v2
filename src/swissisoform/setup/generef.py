"""Fetch gene-reference data for the generef module from Affinage (no auth).

Queries the Affinage API (``https://affinage.wi.mit.edu/api``) for each gene
symbol and writes ``data/reference/generef/generef.json`` =
``{gene: {uniprot_id, uniprot_function, subcellular_location, keywords}}``.
``run.py`` loads this and passes it to ``GeneRefModule`` as a gene-level module.

Affinage provides literature-grounded, mechanism-focused annotation (it is itself
LLM-generated). One ``GET /api/gene/{symbol}`` supplies everything we keep:

* ``uniprot_function``      ← ``narrative.mechanistic_narrative`` (PMID-cited prose)
* ``subcellular_location``  ← ``mechanism_profile.localization`` term labels
* ``keywords``              ← ``mechanism_profile.molecular_activity + pathway`` labels

The field *keys* are kept (legacy ``uniprot_`` prefix) so ``GeneRefModule`` and
every downstream consumer are unchanged. ``uniprot_id`` is still sourced from a
minimal UniProtKB accession lookup — Affinage exposes no accession, and the site
links out to the UniProt entry.

Driven by the thin CLI at ``scripts/setup/fetch_generef.py``.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "reference" / "generef" / "generef.json"
AFFINAGE_API = "https://affinage.wi.mit.edu/api"
UNIPROT_API = "https://rest.uniprot.org/uniprotkb/search"

# Which mechanism_profile axes become the functional "keywords" facet.
# molecular_activity (GO MF) + pathway (Reactome) are the functional analog of the
# old UniProt keyword strip; localization is its own field, and complexes/partners
# are interaction gene-names rather than functional terms.
_KEYWORD_AXES = ("molecular_activity", "pathway")


def _get_json(url: str, *, timeout: int = 30, retries: int = 3) -> Any | None:
    """GET + parse JSON, with retry/backoff. Returns None on 404 or repeated error.

    The retry covers Affinage's Railway host, which can idle-sleep (a cold start of
    tens of seconds) even though warm requests return in <1s. A 404 is a definitive
    "not found" and is not retried.
    """
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None  # gene not in the source — definitive
            last = e
        except Exception as e:  # noqa: BLE001 — network hiccup / cold start
            last = e
        if attempt < retries - 1:
            time.sleep(2 * (attempt + 1))  # 2s, 4s backoff
    print(f"  [warn] giving up on {url}: {last}")
    return None


def _term_labels(profile: dict[str, Any], axes: tuple[str, ...]) -> list[str]:
    """Collect unique ``term_label`` values across the given mechanism_profile axes."""
    labels: list[str] = []
    for axis in axes:
        for entry in profile.get(axis) or []:
            if isinstance(entry, dict):
                lab = entry.get("term_label")
                if lab:
                    labels.append(lab)
    return list(dict.fromkeys(labels))


def _uniprot_accession(gene: str) -> str | None:
    """Minimal reviewed-human UniProt accession lookup (for the ID + entry link only)."""
    params = {
        "query": f"gene:{gene} AND organism_id:9606 AND reviewed:true",
        "fields": "accession",
        "format": "json",
        "size": "1",
    }
    data = _get_json(f"{UNIPROT_API}?{urllib.parse.urlencode(params)}", retries=2)
    results = (data or {}).get("results") or []
    return results[0].get("primaryAccession") if results else None


def fetch_one(gene: str) -> dict | None:
    """Return {uniprot_id, uniprot_function, subcellular_location, keywords} or None.

    Function + localization + keywords come from Affinage; ``uniprot_id`` is a
    minimal UniProt accession lookup (Affinage exposes no accession). Returns None
    when Affinage has no record for the gene (so downstream columns stay null,
    exactly as an unmatched gene did under the old UniProt-only fetch).
    """
    data = _get_json(f"{AFFINAGE_API}/gene/{urllib.parse.quote(gene)}")
    if not isinstance(data, dict):
        return None
    narrative = data.get("narrative") or {}
    function = narrative.get("mechanistic_narrative")
    profile = narrative.get("mechanism_profile") or {}

    locations = _term_labels(profile, ("localization",))
    keywords = _term_labels(profile, _KEYWORD_AXES)

    return {
        "uniprot_id": _uniprot_accession(gene),
        "uniprot_function": function or None,
        "subcellular_location": "; ".join(locations) or None,
        "keywords": "; ".join(keywords) or None,
    }


def main(argv: list[str] | None = None) -> int:
    """Fetch generef records for a gene set and write/merge ``generef.json``."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--genes", nargs="+", help="Gene symbols")
    g.add_argument("--gene-list", type=Path, help="File with one symbol per line")
    g.add_argument("--combined", action="store_true", help="All genes in the combined catalog")
    p.add_argument("--merge", action="store_true", help="Merge into the existing generef.json")
    args = p.parse_args(argv)

    if args.genes:
        genes = args.genes
    elif args.gene_list:
        genes = [
            ln.strip() for ln in args.gene_list.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    else:
        import pandas as pd

        combined = ROOT / "data" / "output" / "filtered" / "all_samples_combined.parquet"
        genes = sorted(pd.read_parquet(combined, columns=["Symbol"])["Symbol"].dropna().unique())

    data: dict[str, dict] = {}
    if args.merge and OUT.exists():
        data = json.loads(OUT.read_text())

    n_ok = 0
    for i, gene in enumerate(genes):
        if gene in data:
            continue
        rec = fetch_one(gene)
        if rec and rec.get("uniprot_id"):
            data[gene] = rec
            n_ok += 1
        time.sleep(0.2)  # be polite to the UniProt endpoint
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(genes)} …")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2, sort_keys=True))
    print(f"wrote {OUT}: {len(data)} genes total ({n_ok} newly fetched)")
    return 0
