"""PLM (protein language model) subpackage.

Houses ESM-C embedding extraction and downstream analyses:

- ``embed.py``: per-residue ESM-C embeddings + masked-marginal LLR.
- ``module.py``: PLM VEP SiteModule built on the cached LLR scores.

Pattern: precompute (GPU, expensive) writes per-protein cache files
keyed by sha1 of the stop-stripped, uppercased sequence. Modules then
do hash-keyed lookups at pipeline runtime.
"""

from swissisoform.plm.embed import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_SIZE,
    ESMC_MODEL_IDS,
    PLM_CONDA_ENV,
    load_cache,
    precompute_plm,
    precompute_plm_esm2,
    protein_hash,
    resolve_model_id,
)

__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_MODEL_ID",
    "DEFAULT_MODEL_SIZE",
    "ESMC_MODEL_IDS",
    "PLM_CONDA_ENV",
    "load_cache",
    "precompute_plm",
    "precompute_plm_esm2",
    "protein_hash",
    "resolve_model_id",
]
