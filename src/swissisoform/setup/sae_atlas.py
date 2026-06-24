"""Fetch the ESM Atlas SAE feature-term dictionary (setup-time, network).

One-off setup step: downloads the per-feature functional descriptions for the
ESM-C 6B layer-60 SAE codebook from the public Biohub ESM Atlas and writes them
to ``data/reference/sae_atlas/esmc6b_layer60_features.json`` — the
index→{label, description} dictionary that
:class:`~swissisoform.plm.sae_module.SAEFeatureModule` attaches as human-readable
feature labels at annotate time.

Idempotent: skips the fetch when the local JSON already exists (pass ``--force``
to refetch). The runtime read-side (``load_atlas`` / ``atlas_provenance``) lives
in :mod:`swissisoform.plm.atlas`; only this network fetch is setup code.

Driven by the thin CLI ``scripts/setup/fetch_sae_atlas.py`` and wired into
``scripts/slurm/run.sbatch`` (best-effort fetch before annotation).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from swissisoform.plm.atlas import (
    ATLAS_BASE_URL,
    ATLAS_PROVENANCE,
    ATLAS_SOURCE_SAE,
    DEFAULT_ATLAS_PATH,
    DEFAULT_CODEBOOK_DIM,
    TERM_FIELDS,
)

logger = logging.getLogger(__name__)


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


def main(argv: list[str] | None = None) -> int:
    """CLI: idempotently fetch the Atlas term dictionary to the local JSON cache."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", type=Path, default=DEFAULT_ATLAS_PATH)
    p.add_argument("--codebook-dim", type=int, default=DEFAULT_CODEBOOK_DIM)
    p.add_argument(
        "--force",
        action="store_true",
        help="Refetch even if the local JSON already exists.",
    )
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    # Idempotent: skip the network fetch when the artifact is already present.
    if args.out.exists() and args.out.stat().st_size > 0 and not args.force:
        print(f"SAE Atlas already present: {args.out} (--force to refetch)")
        return 0

    payload = fetch_atlas(args.out, codebook_dim=args.codebook_dim)
    print(f"wrote {payload['_meta']['n_features']} terms to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
