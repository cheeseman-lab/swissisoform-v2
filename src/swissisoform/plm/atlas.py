"""ESM Atlas feature descriptions — runtime read-side (index→term lookup).

Loads the per-feature functional descriptions for the SAE codebook from a local
JSON cache so the SAE feature module can attach human-readable labels offline
(consistent with the three-phase, no-network-at-annotate-time design). This
module is import-safe and network-free; the one-off network fetch that populates
the cache lives in :mod:`swissisoform.setup.sae_atlas` (CLI:
``scripts/setup/fetch_sae_atlas.py``), wired into the pipeline via
``scripts/slurm/run.sbatch``.

PROVENANCE: the Atlas describes the **6B layer-60** SAE dictionary
(``esmc-6b-2024-12-sae-layer60-k64-codebook16384``). When the pipeline runs the
**6B-layer60** SAE (the default), feature #N IS that dictionary's #N — the labels
are correct. For any other SAE (e.g. the legacy 600M-layer27 dictionary) the
indices are NOT aligned, so labels are only a placeholder. Use
:func:`atlas_provenance` to stamp each label with the right caveat for the SAE
that produced the feature; every attached label carries that as ``label_source``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ATLAS_BASE_URL = "https://biohub.ai/esm/protein/api/v1alpha1/features"
ATLAS_SOURCE_SAE = "esmc-6b-2024-12-sae-layer60-k64-codebook16384"
# The Atlas IS this dictionary. Aligned (correct) provenance vs the placeholder
# stamp used when a different SAE produced the features.
ATLAS_PROVENANCE_ALIGNED = ATLAS_SOURCE_SAE
ATLAS_PROVENANCE_PLACEHOLDER = (
    f"{ATLAS_SOURCE_SAE} (PLACEHOLDER — not aligned to this SAE's indices)"
)
# Back-compat default: the aligned string (pipeline default SAE is 6B-layer60).
ATLAS_PROVENANCE = ATLAS_PROVENANCE_ALIGNED


def atlas_provenance(model_size: str = "6b") -> str:
    """Label provenance for the SAE that produced the features.

    Aligned (the Atlas describes exactly this dictionary) for the 6B-layer60 SAE;
    placeholder for any other SAE size, whose indices are not Atlas-aligned.
    """
    return (
        ATLAS_PROVENANCE_ALIGNED
        if (model_size or "").lower() == "6b"
        else ATLAS_PROVENANCE_PLACEHOLDER
    )


DEFAULT_CODEBOOK_DIM = 16384
DEFAULT_ATLAS_PATH = (
    Path(__file__).resolve().parents[3]
    / "data" / "reference" / "sae_atlas" / "esmc6b_layer60_features.json"
)
# Fields the bulk ``/features`` listing returns per item. (The per-feature
# ``/features/{idx}`` endpoint adds summary/category/uniref stats, but the bulk
# list — one request for all 16,384 — only carries label + description, which
# are the functional terms we need.)
TERM_FIELDS = ("label", "description")


def load_atlas(path: Path | str = DEFAULT_ATLAS_PATH) -> dict[int, dict[str, Any]]:
    """Load the local Atlas term dictionary as ``{feature_index: {label, ...}}``.

    Returns an empty dict if the cache file doesn't exist (module then degrades
    to index-only output).
    """
    path = Path(path)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    feats = payload.get("features", payload)
    return {int(k): v for k, v in feats.items()}
