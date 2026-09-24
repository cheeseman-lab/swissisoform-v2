"""Content hashes used as cache keys.

One definition, imported everywhere. :func:`protein_hash` keys every
GPU-precompute and external-tool cache in the pipeline — ESM embeddings, Boltz
and ESMFold structures, DeepLoc, SignalP, TargetP, InterProScan — so a second
copy of it is not a duplicated three-liner but a second answer to "where is this
protein's result". Editing one copy silently orphans one cache and not the
others, and the symptom is a cache that quietly recomputes forever.

Deliberately dependency-free (``hashlib`` only) so the light annotation modules
can import it without pulling in torch through ``swissisoform.plm``.
"""

from __future__ import annotations

import hashlib


def protein_hash(protein: str) -> str:
    """Stable hash of a protein sequence (stop codon stripped, uppercased)."""
    seq = protein.rstrip("*").upper()
    return hashlib.sha1(seq.encode("ascii"), usedforsecurity=False).hexdigest()


__all__ = ["protein_hash"]
