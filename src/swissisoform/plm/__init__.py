"""PLM (protein language model) subpackage.

Houses ESM-2 embedding extraction and downstream analyses:

- ``embed.py``: per-residue ESM-2 embeddings + masked-marginal LLR.
- ``vep.py``: variant-effect helpers built on cached LLR scores.

Pattern: precompute (GPU, expensive) writes per-protein cache files
keyed by sha1 of the stop-stripped, uppercased sequence. Modules then
do hash-keyed lookups at pipeline runtime.
"""

from swissisoform.plm.embed import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MODEL_ID,
    PLM_CONDA_ENV,
    load_cache,
    precompute_plm_esm2,
    protein_hash,
)

__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_MODEL_ID",
    "PLM_CONDA_ENV",
    "load_cache",
    "precompute_plm_esm2",
    "protein_hash",
]
