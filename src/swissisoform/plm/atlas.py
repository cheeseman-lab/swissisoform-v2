"""ESM Atlas feature descriptions — local index→term cache.

Fetches per-feature functional descriptions from the public Biohub ESM Atlas
(``https://biohub.ai/esm/protein/api/v1alpha1/features``) once and stores them
locally so the SAE feature module can attach human-readable labels offline
(consistent with the three-phase, no-network-at-annotate-time design).

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


def fetch_atlas(
    out_path: Path | str = DEFAULT_ATLAS_PATH,
    *,
    codebook_dim: int = DEFAULT_CODEBOOK_DIM,
    timeout: int = 120,
) -> dict[str, Any]:
    """Fetch all feature terms in one bulk request and write them as JSON.

    The bare ``/features`` endpoint returns every feature in a single
    ``{"data": [{feature_index, label, description}, ...]}`` response (all
    16,384, no pagination), so this is one GET — not one-per-feature.

    Returns the written payload ``{"_meta": {...}, "features": {idx: {...}}}``.
    """
    import requests

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        r = session.get(ATLAS_BASE_URL, timeout=timeout)
        r.raise_for_status()
        data = r.json()["data"]

    features: dict[str, Any] = {
        str(item["feature_index"]): {k: item.get(k) for k in TERM_FIELDS}
        for item in data
        if item.get("feature_index") is not None
    }

    payload = {
        "_meta": {
            "source_sae": ATLAS_SOURCE_SAE,
            "provenance": ATLAS_PROVENANCE,
            "base_url": ATLAS_BASE_URL,
            "codebook_dim": codebook_dim,
            "n_features": len(features),
            "fields": list(TERM_FIELDS),
        },
        "features": features,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote %d feature terms to %s", len(features), out_path)
    return payload


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


def main(argv: list[str] | None = None) -> int:
    """CLI: fetch the Atlas term dictionary to the local JSON cache."""
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_ATLAS_PATH)
    p.add_argument("--codebook-dim", type=int, default=DEFAULT_CODEBOOK_DIM)
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    payload = fetch_atlas(args.out, codebook_dim=args.codebook_dim)
    print(f"wrote {payload['_meta']['n_features']} terms to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
