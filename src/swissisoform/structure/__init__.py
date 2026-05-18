"""Structure prediction subpackage.

Houses Boltz-2 / Chai-1 wrappers + structure comparison.

Pattern: precompute (GPU, expensive) writes per-protein cache directories
keyed by sha1(stop-stripped, uppercased seq). The pipeline-side
``StructureModule`` does cache lookups + diff-region pLDDT slicing +
optional TM-score/RMSD/contacts (lazy imports, degrade gracefully).

Cache layout::

    data/cache/structure/<backend>/<hash>/
        model.cif         # top-rank predicted structure
        confidence.json   # {"plddt": [...L floats 0-100], "ptm": float, "iptm": float | None}
        metrics.json      # {"status": "ok"|"too_long"|"failed"|"oom",
                          #  "backend": str, "length": int,
                          #  "plddt_mean": float, "plddt_std": float, "ptm": float}
"""

from swissisoform.structure.fold import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MAX_SEQ_LEN,
    FOLD_CONDA_ENV,
    cache_path,
    load_cache,
    precompute_fold,
    protein_hash,
    write_cache,
)

__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_MAX_SEQ_LEN",
    "FOLD_CONDA_ENV",
    "cache_path",
    "load_cache",
    "precompute_fold",
    "protein_hash",
    "write_cache",
]
