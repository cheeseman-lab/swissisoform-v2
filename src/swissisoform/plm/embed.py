"""ESM-2 embedding extraction + masked-marginal LLR (per protein).

Precompute is GPU-expensive (~1-30s per protein for ESM-2 650M). Results
are cached on disk as ``.npz`` files keyed by the sha1 of the
stop-stripped, uppercased protein sequence so duplicates dedupe and
re-runs are no-ops.

Cache layout (under ``cache_dir``)::

    cache_dir/
      <hash>.npz                # llr: (L,), embedding: (L, hidden_dim)
      manifest.tsv              # hash<TAB>seq  (audit log)

The :func:`precompute_plm_esm2` entrypoint is the only public surface.
It dedupes inputs, reads existing cache files, and (optionally) runs
ESM-2 inline or via a conda-env subprocess for the missing hashes.
Tests pre-populate the cache, so no GPU dep is required for unit tests.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "facebook/esm2_t33_650M_UR50D"
DEFAULT_LAYER = 33  # last layer for ESM-2 650M
DEFAULT_INTERPLM_LAYER = 18  # SAE-trained layer (Elana/InterPLM-esm2-650m)
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache" / "plm_esm2"
PLM_CONDA_ENV = "swissisoform-v2-plm"


def protein_hash(protein: str) -> str:
    """Stable hash of a protein sequence (stop codon stripped, uppercased)."""
    seq = protein.rstrip("*").upper()
    return hashlib.sha1(seq.encode("ascii"), usedforsecurity=False).hexdigest()


def _cache_path(cache_dir: Path, h: str) -> Path:
    return cache_dir / f"{h}.npz"


def load_cache(h: str, cache_dir: Path | str = DEFAULT_CACHE_DIR) -> dict[str, Any] | None:
    """Load a cached PLM result by protein hash.

    Returns ``{"llr": np.ndarray (L,), "embedding": np.ndarray (L, hidden)}``
    or ``None`` if the cache file doesn't exist.
    """
    import numpy as np

    p = _cache_path(Path(cache_dir), h)
    if not p.exists():
        return None
    with np.load(p) as npz:
        out: dict[str, Any] = {}
        if "llr" in npz.files:
            out["llr"] = npz["llr"]
        if "embedding" in npz.files:
            out["embedding"] = npz["embedding"]
        if "embedding_layer18" in npz.files:
            out["embedding_layer18"] = npz["embedding_layer18"]
    return out or None


def _save_cache(
    h: str,
    cache_dir: Path,
    *,
    llr: Any,
    embedding: Any | None = None,
    embedding_layer18: Any | None = None,
) -> None:
    import numpy as np

    cache_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"llr": np.asarray(llr, dtype=np.float32)}
    if embedding is not None:
        payload["embedding"] = np.asarray(embedding, dtype=np.float16)
    if embedding_layer18 is not None:
        payload["embedding_layer18"] = np.asarray(embedding_layer18, dtype=np.float16)
    np.savez_compressed(_cache_path(cache_dir, h), **payload)


def _append_manifest(cache_dir: Path, h: str, seq: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = cache_dir / "manifest.tsv"
    if manifest.exists():
        with open(manifest) as fh:
            seen = {line.split("\t", 1)[0] for line in fh if line.strip()}
        if h in seen:
            return
    with open(manifest, "a") as fh:
        fh.write(f"{h}\t{seq}\n")


# ---------------------------------------------------------------------------
# Inline ESM-2 forward (per-residue embeddings + masked-marginal LLR)
# ---------------------------------------------------------------------------


def _esm2_forward_one(
    seq: str,
    model: Any,
    tokenizer: Any,
    device: str,
    *,
    extract_layers: tuple[int, ...] = (DEFAULT_LAYER, DEFAULT_INTERPLM_LAYER),
) -> dict[str, Any]:
    """Single forward pass: per-residue embeddings at requested layers + per-position LLR.

    LLR uses masked marginals: mask each position in turn, take the log-prob
    of the wild-type AA. Constant-time per position, O(L) forward passes per
    sequence, so this is the hot path. Runs them in batches of size B.
    """
    import numpy as np
    import torch

    seq = seq.rstrip("*").upper()
    L = len(seq)
    if L == 0:
        return {
            "llr": np.zeros(0, dtype=np.float32),
            "embedding": np.zeros((0, model.config.hidden_size), dtype=np.float16),
            "embedding_layer18": np.zeros((0, model.config.hidden_size), dtype=np.float16),
        }

    encoded = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(device)  # (1, L+2) — BOS, residues, EOS
    attention_mask = encoded["attention_mask"].to(device)

    # Pass 1: full forward, grab hidden states.
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
    hidden = out.hidden_states  # tuple of (1, L+2, hidden) per layer
    # Strip BOS/EOS — residues live at indices 1..L
    embeddings = {ly: hidden[ly][0, 1 : L + 1, :].float().cpu().numpy() for ly in extract_layers}

    # Pass 2: masked marginals for LLR.
    # Build a (L, L+2) batch where row i has position (i+1) replaced by <mask>.
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise RuntimeError("Tokenizer has no mask token; cannot compute masked marginals.")
    base = input_ids.expand(L, -1).clone()  # (L, L+2)
    for i in range(L):
        base[i, i + 1] = mask_token_id
    mask_batch_attn = attention_mask.expand(L, -1)

    # Run in chunks to fit GPU memory.
    chunk = 16
    llr = np.zeros(L, dtype=np.float32)
    for start in range(0, L, chunk):
        stop = min(start + chunk, L)
        with torch.no_grad():
            mlm_out = model(
                input_ids=base[start:stop],
                attention_mask=mask_batch_attn[start:stop],
                return_dict=True,
            )
        # ESM-2 model has lm_head; if logits aren't returned (encoder-only
        # AutoModel), fall back to AutoModelForMaskedLM upstream.
        logits = getattr(mlm_out, "logits", None)
        if logits is None:
            raise RuntimeError(
                "Loaded model does not expose `logits`; load with AutoModelForMaskedLM "
                "to compute masked marginals."
            )
        # Per-row, position (start+k+1) is the masked one; gather log-prob of wt.
        log_probs = torch.log_softmax(logits, dim=-1)
        for k in range(stop - start):
            i = start + k
            wt_id = input_ids[0, i + 1].item()
            llr[i] = float(log_probs[k, i + 1, wt_id].item())

    return {
        "llr": llr,
        "embedding": embeddings.get(DEFAULT_LAYER),
        "embedding_layer18": embeddings.get(DEFAULT_INTERPLM_LAYER),
    }


def _load_esm2(model_id: str, device: str, dtype: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype, torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    model = model.to(device)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Public precompute entrypoint
# ---------------------------------------------------------------------------


def precompute_plm_esm2(
    proteins: dict[str, str] | list[str],
    *,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    device: str = "cuda",
    dtype: str = "float16",
    inline: bool = True,
    skip_missing: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run (or load cached) ESM-2 embeddings + LLR for a batch of proteins.

    Args:
        proteins: ``{label: sequence}`` or list of sequences. Labels discarded;
            output is keyed on ``protein_hash``.
        model_id: HuggingFace ESM-2 model ID. Default is 650M.
        cache_dir: Directory holding ``<hash>.npz`` cache files.
        device: PyTorch device string.
        dtype: Inference precision.
        inline: If True, run any uncached proteins inline (requires torch +
            transformers). If False, skip uncached entries silently — useful
            when the GPU job is run separately via ``scripts/slurm/run_plm_embed.sbatch``.
        skip_missing: When True, returns only proteins with a cache file.
            When False, missing entries get ``None`` values.

    Returns:
        ``{protein_hash: {"llr": np.ndarray, "embedding": np.ndarray | None,
        "embedding_layer18": np.ndarray | None}}``. Missing entries either
        absent (skip_missing=True) or have ``None`` values.
    """
    cache_dir = Path(cache_dir)

    if isinstance(proteins, list):
        proteins = {f"seq_{i}": s for i, s in enumerate(proteins)}

    hash_to_seq: dict[str, str] = {}
    for _label, seq in proteins.items():
        if not seq:
            continue
        h = protein_hash(seq)
        hash_to_seq.setdefault(h, seq.rstrip("*").upper())

    logger.info(
        "precompute_plm_esm2: %d inputs → %d unique sequences (cache=%s, inline=%s)",
        len(proteins),
        len(hash_to_seq),
        cache_dir,
        inline,
    )

    # Pass 1: load whatever's cached.
    result: dict[str, dict[str, Any]] = {}
    missing: list[tuple[str, str]] = []
    for h, seq in hash_to_seq.items():
        cached = load_cache(h, cache_dir)
        if cached is not None:
            result[h] = cached
        else:
            missing.append((h, seq))

    if missing:
        logger.info(
            "precompute_plm_esm2: %d/%d sequences missing from cache",
            len(missing),
            len(hash_to_seq),
        )

    # Pass 2: optionally compute the missing ones inline.
    if missing and inline:
        try:
            model, tokenizer = _load_esm2(model_id, device, dtype)
        except Exception as exc:  # noqa: BLE001 — graceful fallback when GPU stack absent
            logger.warning(
                "precompute_plm_esm2: could not load %s (%s); skipping inline compute. "
                "Run scripts/slurm/run_plm_embed.sbatch to populate the cache offline.",
                model_id,
                exc,
            )
            model = None
            tokenizer = None

        if model is not None and tokenizer is not None:
            for h, seq in missing:
                try:
                    payload = _esm2_forward_one(seq, model, tokenizer, device)
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "precompute_plm_esm2: forward failed for hash=%s len=%d: %s",
                        h,
                        len(seq),
                        exc,
                    )
                    continue
                _save_cache(
                    h,
                    cache_dir,
                    llr=payload["llr"],
                    embedding=payload.get("embedding"),
                    embedding_layer18=payload.get("embedding_layer18"),
                )
                _append_manifest(cache_dir, h, seq)
                result[h] = payload

    if not skip_missing:
        for h, _ in missing:
            result.setdefault(h, {"llr": None, "embedding": None, "embedding_layer18": None})

    return result
