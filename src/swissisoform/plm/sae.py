"""ESM-C SAE encode step — sparse feature activations per residue.

Consumes the SAE-target-layer residual-stream representation cached by
:mod:`swissisoform.plm.embed` (the ``embedding_sae`` key) and runs it through the
matching ESM-C Top-K sparse autoencoder (per model size: 6B →
``biohub/ESMC-6B-sae-layer60-k64-codebook16384`` layer 60; 600M →
``biohub/ESMC-600M-sae-k64-codebook16384`` layer 27). This is a cheap encode
(one linear layer + Top-K), decoupled from the GPU ESM-C forward — it reads the
cached representation and never re-runs ESM-C, so the SAE can be swapped or
re-encoded without touching the embedding cache.

Cache layout (under ``cache_dir`` — per size, e.g. ``data/cache/sae_esmc/6B/``)::

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
    DEFAULT_MODEL_SIZE,
    load_cache,
    protein_hash,
    sae_layer_for,
)
from swissisoform.plm.embed import plm_cache_dir as plm_cache_dir_for

logger = logging.getLogger(__name__)

# ESM-C SAE repos per model size. The 600M repo bundles all 37 layers; the 6B
# repo ships only its layer-60 SAE (a single ``layer_60.safetensors``). Both are
# Top-K k=64, codebook 16384, and load identically (d_model is read from config).
SAE_REPOS: dict[str, str] = {
    "600m": "biohub/ESMC-600M-sae-k64-codebook16384",
    "6b": "biohub/ESMC-6B-sae-layer60-k64-codebook16384",
}
_SAE_CACHE_ROOT = Path(__file__).resolve().parents[3] / "data" / "cache" / "sae_esmc"


def sae_cache_dir(model_size: str = DEFAULT_MODEL_SIZE) -> Path:
    """Per-size SAE feature cache dir, e.g. ``data/cache/sae_esmc/6B``."""
    return _SAE_CACHE_ROOT / (model_size or DEFAULT_MODEL_SIZE).upper()


def sae_repo_for(model_size: str = DEFAULT_MODEL_SIZE) -> str:
    """SAE HuggingFace repo id for a model size."""
    return SAE_REPOS[(model_size or DEFAULT_MODEL_SIZE).lower()]


DEFAULT_SAE_REPO = sae_repo_for(DEFAULT_MODEL_SIZE)
DEFAULT_SAE_CACHE_DIR = sae_cache_dir(DEFAULT_MODEL_SIZE)

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
    model_size: str = DEFAULT_MODEL_SIZE,
    plm_cache_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
    sae_repo: str | None = None,
    sae_layer: int | None = None,
    device: str = "cpu",
    inline: bool = True,
    skip_missing: bool = True,
) -> dict[str, dict[str, Any]]:
    """Encode (or load cached) sparse SAE features for a batch of proteins.

    Reads each protein's ``embedding_sae`` (the SAE-target layer rep) from the
    PLM cache and runs the SAE encode. Pure cache-read + small matmul — no ESM-C
    inference. The SAE repo, target layer, and both cache dirs default to the
    per-size values for ``model_size`` (so the 6B SAE reads the 6B PLM cache and
    writes the 6B SAE cache), each overridable.

    Args:
        proteins: ``{label: sequence}`` or list of sequences; keyed by hash.
        model_size: ESM-C size selecting SAE repo/layer + cache dirs (default 6b).
        plm_cache_dir: Source of ``embedding_sae`` (default
            ``data/cache/plm_esmc/<SIZE>``).
        cache_dir: Destination for the sparse-feature ``<hash>.npz`` files
            (default ``data/cache/sae_esmc/<SIZE>``).
        sae_repo: HuggingFace SAE repo id (default per size).
        sae_layer: Backbone layer the SAE is trained against (default per size;
            must match the layer cached as ``embedding_sae``).
        device: Torch device for the encode (CPU is fine; cheap).
        inline: If True, encode uncached proteins now (needs torch + the SAE).
            If False, skip them — used when a GPU/batch job runs separately.
        skip_missing: When True, omit proteins with no cached features.

    Returns:
        ``{protein_hash: {"idx", "val", "recon_loss"}}``.
    """
    size = (model_size or DEFAULT_MODEL_SIZE).lower()
    plm_cache_dir = (
        plm_cache_dir_for(size) if plm_cache_dir is None else Path(plm_cache_dir)
    )
    cache_dir = sae_cache_dir(size) if cache_dir is None else Path(cache_dir)
    sae_repo = sae_repo_for(size) if sae_repo is None else sae_repo
    sae_layer = sae_layer_for(size) if sae_layer is None else sae_layer

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
    parser.add_argument(
        "--model-size", default=DEFAULT_MODEL_SIZE,
        help="ESM-C size selecting SAE repo/layer + cache dirs (default %(default)s).",
    )
    parser.add_argument("--plm-cache-dir", type=Path, default=None,
                        help="Override the per-size PLM cache dir.")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Override the per-size SAE feature cache dir.")
    parser.add_argument("--sae-repo", default=None, help="Override the SAE repo id.")
    parser.add_argument("--sae-layer", type=int, default=None,
                        help="Override the SAE target layer.")
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
        seqs, model_size=args.model_size,
        plm_cache_dir=args.plm_cache_dir, cache_dir=args.cache_dir,
        sae_repo=args.sae_repo, sae_layer=args.sae_layer, device=args.device,
        inline=True,
    )
    resolved_dir = args.cache_dir or sae_cache_dir(args.model_size)
    print(f"Wrote/loaded {len(res)} SAE feature caches to {resolved_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
