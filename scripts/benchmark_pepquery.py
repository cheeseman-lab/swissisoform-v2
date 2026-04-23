"""Benchmark PepQuery2 wall-clock across dataset tags.

Runs precompute_pepquery on a small hand-picked peptide set against a
few PepQueryDB ``-b`` tags so we can see how long ``-b all`` would take
before committing to it on our full unique-peptide set.

Usage:
    python scripts/benchmark_pepquery.py
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from swissisoform.modules.massspec import precompute_pepquery

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Five hand-picked peptides from well-known human proteins — all should
# have abundant MS/MS evidence, so a 0-hit result means the run is broken.
BENCH_PEPTIDES = {
    "TP53_hits": {
        "SVTCTYSPALNK",      # TP53 (known tryptic)
        "FEVRVCACPGR",       # TP53
    },
    "VEGFA_hits": {
        "QIMRIKPHQGQHIGEMSFLQHNK",  # VEGFA
    },
    "CTNND1_hits": {
        "LLVNAVSPDR",        # CTNND1
        "DHILSVVR",          # CTNND1
    },
}


def run(dataset: str) -> float:
    t0 = time.time()
    validated = precompute_pepquery(
        BENCH_PEPTIDES,
        dataset=dataset,
        cache_dir=Path("./data/cache/pepquery_bench"),
    )
    dt = time.time() - t0
    total_hits = sum(len(v) for v in validated.values())
    print(f"dataset={dataset!r:20s} time={dt:7.1f}s  validated_hits={total_hits}")
    return dt


if __name__ == "__main__":
    for tag in ("w",):          # global proteome — fastest
        run(tag)
    for tag in ("CPTAC",):      # all CPTAC tumor proteomes
        run(tag)
    # Only run -b all if the user un-comments it: this is the slow one.
    # for tag in ("all",):
    #     run(tag)
