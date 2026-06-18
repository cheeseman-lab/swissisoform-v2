"""ESM-C SAE encode step — sparse feature activations per residue.

Consumes the layer-27 residual-stream representation cached by
:mod:`swissisoform.plm.embed` (the ``embedding_sae`` key) and runs it through the
ESM-C Top-K sparse autoencoder (``biohub/ESMC-600M-sae-k64-codebook16384``,
layer 27). This is a cheap encode (one linear layer + Top-K), decoupled from the
GPU ESM-C forward — it reads the cached representation and never re-runs ESM-C,
so the SAE can be swapped or re-encoded without touching the embedding cache.

Cache layout (under ``cache_dir``, default ``data/cache/sae_esmc/``)::

    cache_dir/
      <hash>.npz       # idx: (L, K) int32   — top-K feature indices per residue,
                       # val: (L, K) float16 — their activation magnitudes,
                       # recon_loss: () float32 — SAE reconstruction MSE (QC)
      manifest.tsv     # hash<TAB>L  (audit log)

The features are the *raw* Top-K magnitudes (post z-score-normalize, post Top-K,
pre TF-IDF) — the SAE's intrinsic output. Optional TF-IDF down-weighting is a
downstream choice left to the feature module.

``precompute_sae`` is the public surface; tests can pre-populate the cache so no
GPU/model dependency is needed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from swissisoform.plm.embed import (
    DEFAULT_CACHE_DIR as PLM_CACHE_DIR,
)
from swissisoform.plm.embed import (
    SAE_LAYER,
    load_cache,
    protein_hash,
)

logger = logging.getLogger(__name__)

DEFAULT_SAE_REPO = "biohub/ESMC-600M-sae-k64-codebook16384"
DEFAULT_SAE_CACHE_DIR = (
    Path(__file__).resolve().parents[3] / "data" / "cache" / "sae_esmc"
)

# Lazy singleton: one materialized SAE layer per (repo, layer, device).
_SAE_LAYERS: dict[tuple[str, int, str], Any] = {}


def _sae_path(cache_dir: Path, h: str) -> Path:
    return cache_dir / f"{h}.npz"


def load_sae_cache(
    h: str, cache_dir: Path | str = DEFAULT_SAE_CACHE_DIR
) -> dict[str, Any] | None:
    """Load cached sparse SAE features by protein hash, or ``None`` if absent.

    Returns ``{"idx": (L, K) int, "val": (L, K) float, "recon_loss": float}``.
    """
    import numpy as np

    p = _sae_path(Path(cache_dir), h)
    if not p.exists():
        return None
    with np.load(p) as npz:
        out: dict[str, Any] = {}
        for key in ("idx", "val", "recon_loss"):
            if key in npz.files:
                out[key] = npz[key]
    return out or None


def _save_sae_cache(
    h: str, cache_dir: Path, *, idx: Any, val: Any, recon_loss: float
) -> None:
    import numpy as np

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        _sae_path(cache_dir, h),
        idx=np.asarray(idx, dtype=np.int32),
        val=np.asarray(val, dtype=np.float16),
        recon_loss=np.asarray(recon_loss, dtype=np.float32),
    )


def _append_manifest(cache_dir: Path, h: str, length: int) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = cache_dir / "manifest.tsv"
    if manifest.exists():
        with open(manifest) as fh:
            seen = {line.split("\t", 1)[0] for line in fh if line.strip()}
        if h in seen:
            return
    with open(manifest, "a") as fh:
        fh.write(f"{h}\t{length}\n")


def _load_sae_layer(repo: str, layer_idx: int, device: str) -> Any:
    """Materialize (and cache) one SAE layer via the HF ``ESMCSAEModel``.

    Only the requested ``layer_{idx}.safetensors`` is downloaded; the repo's
    other layers are skipped.
    """
    key = (repo, layer_idx, device)
    if key in _SAE_LAYERS:
        return _SAE_LAYERS[key]
    from transformers.models.esmc.modeling_esmc_sae import ESMCSAEModel

    sae = ESMCSAEModel.from_pretrained(
        repo, allow_patterns=["config.json", f"layer_{layer_idx}.safetensors"]
    )
    sae.initialize_layers([layer_idx])
    layer = sae.layers[str(layer_idx)].to(device).eval()
    _SAE_LAYERS[key] = layer
    logger.info("loaded SAE %s layer %d on %s (k=%d, codebook=%d)",
                repo, layer_idx, device, layer.params.k, layer.params.codebook_dim)
    return layer


def encode_one(embedding_sae: Any, layer: Any) -> dict[str, Any]:
    """Encode one protein's (L, d_model) layer rep into sparse Top-K features.

    Returns ``{"idx": (L, K) int32, "val": (L, K) float16, "recon_loss": float}``.
    K is the SAE's ``k`` (e.g. 64); the forward already zeroes all but the
    top-K activations, so the Top-K read-out recovers exactly the nonzeros.
    """
    import numpy as np
    import torch

    k = int(layer.params.k)
    x = np.asarray(embedding_sae, dtype=np.float32)
    L = x.shape[0]
    if L == 0:
        return {
            "idx": np.zeros((0, k), dtype=np.int32),
            "val": np.zeros((0, k), dtype=np.float16),
            "recon_loss": 0.0,
        }
    device = next(layer.parameters()).device
    xt = torch.from_numpy(x).to(device)
    with torch.no_grad():
        out = layer(xt)
        fm = out.feature_magnitudes  # (L, codebook_dim), only k nonzero per row
        vals, idx = fm.topk(k, dim=-1)
        recon = float(out.reconstruction_loss.mean())
    return {
        "idx": idx.cpu().numpy().astype(np.int32),
        "val": vals.cpu().numpy().astype(np.float16),
        "recon_loss": recon,
    }


def precompute_sae(
    proteins: dict[str, str] | list[str],
    *,
    plm_cache_dir: Path | str = PLM_CACHE_DIR,
    cache_dir: Path | str = DEFAULT_SAE_CACHE_DIR,
    sae_repo: str = DEFAULT_SAE_REPO,
    sae_layer: int = SAE_LAYER,
    device: str = "cpu",
    inline: bool = True,
    skip_missing: bool = True,
) -> dict[str, dict[str, Any]]:
    """Encode (or load cached) sparse SAE features for a batch of proteins.

    Reads each protein's ``embedding_sae`` (layer-27 rep) from the PLM cache and
    runs the SAE encode. Pure cache-read + small matmul — no ESM-C inference.

    Args:
        proteins: ``{label: sequence}`` or list of sequences; keyed by hash.
        plm_cache_dir: Source of ``embedding_sae`` (``data/cache/plm_esmc``).
        cache_dir: Destination for the sparse-feature ``<hash>.npz`` files.
        sae_repo: HuggingFace SAE repo id.
        sae_layer: Backbone layer the SAE is trained against (must match the
            layer cached as ``embedding_sae``).
        device: Torch device for the encode (CPU is fine; cheap).
        inline: If True, encode uncached proteins now (needs torch + the SAE).
            If False, skip them — used when a GPU/batch job runs separately.
        skip_missing: When True, omit proteins with no cached features.

    Returns:
        ``{protein_hash: {"idx", "val", "recon_loss"}}``.
    """
    plm_cache_dir = Path(plm_cache_dir)
    cache_dir = Path(cache_dir)

    if isinstance(proteins, list):
        proteins = {f"seq_{i}": s for i, s in enumerate(proteins)}

    hash_to_seq: dict[str, str] = {}
    for _label, seq in proteins.items():
        if not seq:
            continue
        hash_to_seq.setdefault(protein_hash(seq), seq.rstrip("*").upper())

    logger.info(
        "precompute_sae: %d inputs → %d unique sequences (sae=%s L%d, cache=%s, inline=%s)",
        len(proteins), len(hash_to_seq), sae_repo, sae_layer, cache_dir, inline,
    )

    result: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for h in hash_to_seq:
        cached = load_sae_cache(h, cache_dir)
        if cached is not None:
            result[h] = cached
        else:
            missing.append(h)

    if missing and inline:
        layer = _load_sae_layer(sae_repo, sae_layer, device)
        for h in missing:
            plm = load_cache(h, plm_cache_dir)
            if plm is None or plm.get("embedding_sae") is None:
                logger.warning(
                    "precompute_sae: no embedding_sae for hash=%s — run the "
                    "plm embed backfill first; skipping.", h,
                )
                continue
            feats = encode_one(plm["embedding_sae"], layer)
            _save_sae_cache(
                h, cache_dir,
                idx=feats["idx"], val=feats["val"], recon_loss=feats["recon_loss"],
            )
            _append_manifest(cache_dir, h, int(feats["idx"].shape[0]))
            result[h] = feats
    elif missing:
        logger.info("precompute_sae: %d/%d missing (inline=False)", len(missing), len(hash_to_seq))

    if not skip_missing:
        for h in missing:
            result.setdefault(h, {"idx": None, "val": None, "recon_loss": None})

    return result


def main(argv: list[str] | None = None) -> int:
    """CLI: encode SAE features for a FASTA (default ``data/cache/proteins.fa``)."""
    import argparse
    import sys

    from swissisoform.plm.cli import _read_fasta

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fasta", type=Path, nargs="?",
                        default=Path("data/cache/proteins.fa"),
                        help="FASTA of proteins to encode (default %(default)s).")
    parser.add_argument("--plm-cache-dir", type=Path, default=PLM_CACHE_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_SAE_CACHE_DIR)
    parser.add_argument("--sae-repo", default=DEFAULT_SAE_REPO)
    parser.add_argument("--sae-layer", type=int, default=SAE_LAYER)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if not args.fasta.exists():
        print(f"FASTA not found: {args.fasta}", file=sys.stderr)
        return 2
    seqs = _read_fasta(args.fasta)
    if not seqs:
        print(f"No sequences in {args.fasta}", file=sys.stderr)
        return 2

    res = precompute_sae(
        seqs, plm_cache_dir=args.plm_cache_dir, cache_dir=args.cache_dir,
        sae_repo=args.sae_repo, sae_layer=args.sae_layer, device=args.device,
        inline=True,
    )
    print(f"Wrote/loaded {len(res)} SAE feature caches to {args.cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
