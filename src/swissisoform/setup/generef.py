"""Fetch UniProt gene-reference data for the generef module (no auth).

Queries UniProtKB (reviewed, human) for each gene symbol and writes
``data/reference/generef/generef.json`` = ``{gene: {uniprot_id,
uniprot_function, subcellular_location}}``. ``run.py`` loads this and passes it
to ``GeneRefModule`` as a gene-level module.

Driven by the thin CLI at ``scripts/setup/fetch_generef.py``.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "reference" / "generef" / "generef.json"
API = "https://rest.uniprot.org/uniprotkb/search"


def fetch_one(gene: str) -> dict | None:
    """Return {uniprot_id, uniprot_function, subcellular_location} or None."""
    params = {
        "query": f"gene:{gene} AND organism_id:9606 AND reviewed:true",
        "fields": "accession,cc_function,cc_subcellular_location",
        "format": "json",
        "size": "1",
    }
    url = f"{API}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.load(resp)
    except Exception:  # noqa: BLE001 — network hiccup → treat as not found
        return None
    results = data.get("results") or []
    if not results:
        return None
    res = results[0]
    function = None
    locations: list[str] = []
    for c in res.get("comments", []):
        if c.get("commentType") == "FUNCTION" and c.get("texts"):
            function = c["texts"][0].get("value")
        elif c.get("commentType") == "SUBCELLULAR LOCATION":
            for sl in c.get("subcellularLocations", []):
                val = (sl.get("location") or {}).get("value")
                if val:
                    locations.append(val)
    return {
        "uniprot_id": res.get("primaryAccession"),
        "uniprot_function": function,
        "subcellular_location": "; ".join(dict.fromkeys(locations)) or None,
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
