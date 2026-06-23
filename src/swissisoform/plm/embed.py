"""ESM-C embedding extraction + masked-marginal LLR (per protein).

Precompute is GPU-expensive (one full forward + L masked forwards per
protein). Results are cached on disk as ``.npz`` files keyed by the sha1
of the stop-stripped, uppercased protein sequence so duplicates dedupe
and re-runs are no-ops.

Cache layout (under ``cache_dir`` — per model size, e.g.
``data/cache/plm_esmc/6B/``)::

    cache_dir/
      <hash>.npz                # llr: (L,), aa_logprobs: (L, 20),
                                # embedding: (L, hidden)  [final layer],
                                # embedding_sae: (L, hidden)  [SAE-target layer]
      manifest.tsv              # hash<TAB>seq  (audit log)

Model: ESM-C (EvolutionaryScale / Biohub), selectable by size — ``300m``,
``600m``, or ``6b`` (default) — via ``model_size``. ESM-C exposes the
``ESMCForMaskedLM`` architecture, so it loads through the same
HuggingFace ``AutoModelForMaskedLM`` interface as the old ESM-2 path; the
masked-marginal LLR machinery is unchanged apart from the model handle,
tokenizer vocab, and embedding dim (300M→960, 600M→1152, 6B→2560).

The :func:`precompute_plm` entrypoint is the only public surface. It
dedupes inputs, reads existing cache files, and (optionally) runs ESM-C
inline for the missing hashes. Tests pre-populate the cache, so no GPU
dependency is required for unit tests. ``precompute_plm_esm2`` is kept as
a backward-compatible alias.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ESM-C model sizes → HuggingFace repo ids (Biohub). 6B is the pipeline default;
# in bf16 its ~12 GB of weights fit the 48 GB A6000 / ~80 GB A100 nodes. The 6B
# HF weights are fp32 (~28 GB) — point ``--model-id`` at a local clone (see
# run_plm_embed.sbatch) to avoid re-downloading on offline compute nodes. 300M /
# 600M stay selectable via the size flag.
#
# NB: use the HF-transformers-fork repos (capitalized ``ESMC-6B``), NOT the raw
# EvolutionaryScale SDK checkpoint ``esmc-6b-2024-12`` — the SDK checkpoint names
# its qkv/ffn LayerNorms differently (``layernorm_qkv.0`` vs
# ``layernorm_qkv.layer_norm_weight``), so ``ESMCForMaskedLM.from_pretrained``
# silently drops them and re-inits them randomly → NaN activations.
ESMC_MODEL_IDS: dict[str, str] = {
    "300m": "biohub/ESMC-300M",
    "600m": "biohub/ESMC-600M",
    "6b": "biohub/ESMC-6B",
}
DEFAULT_MODEL_SIZE = "6b"
DEFAULT_MODEL_ID = ESMC_MODEL_IDS[DEFAULT_MODEL_SIZE]
PLM_CONDA_ENV = "swissisoform-v2-plm"

_CACHE_ROOT = Path(__file__).resolve().parents[3] / "data" / "cache"


def _size_dir(model_size: str) -> str:
    """Cache-subdir name for a model size (``"6b"`` → ``"6B"``)."""
    size = (model_size or DEFAULT_MODEL_SIZE).lower()
    if size not in ESMC_MODEL_IDS:
        raise ValueError(
            f"unknown ESM-C model_size {size!r}; choose from {sorted(ESMC_MODEL_IDS)}"
        )
    return size.upper()


def plm_cache_dir(model_size: str = DEFAULT_MODEL_SIZE) -> Path:
    """Per-size PLM cache dir, e.g. ``data/cache/plm_esmc/6B``.

    Each model size gets its own subdir so 6B and 600M caches (keyed only by
    protein hash, with no size tag) never collide.
    """
    return _CACHE_ROOT / "plm_esmc" / _size_dir(model_size)


DEFAULT_CACHE_DIR = plm_cache_dir(DEFAULT_MODEL_SIZE)

# Backbone layer whose residual-stream activation the ESM-C SAE is trained
# against, per model size: 600M-sae → layer 27, 6B-sae → layer 60 (300M-sae →
# layer 23). ``output_hidden_states`` returns a tuple of length ``n_layers + 1``
# (index 0 = embeddings, 1..n = block outputs), so ``hidden_states[layer]`` is
# the output of that block — the tensor the SAE consumes. Cached as the
# ``embedding_sae`` key so the SAE encode (plm/sae.py) can run offline without
# re-running ESM-C.
SAE_LAYERS: dict[str, int] = {"300m": 23, "600m": 27, "6b": 60}


def sae_layer_for(model_size: str = DEFAULT_MODEL_SIZE) -> int:
    """Backbone layer the SAE targets for a given model size."""
    return SAE_LAYERS[(model_size or DEFAULT_MODEL_SIZE).lower()]


# Backward-compatible scalar for the default size (importers wanting one int).
SAE_LAYER = sae_layer_for(DEFAULT_MODEL_SIZE)

# Fixed column order for the per-position masked-marginal distribution stored as
# ``aa_logprobs`` (L, 20). Allele-specific variant effect is then a pure cache
# lookup: ``aa_logprobs[pos, AA_TO_COL[alt]] - aa_logprobs[pos, AA_TO_COL[ref]]``.
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_COL: dict[str, int] = {aa: i for i, aa in enumerate(STANDARD_AA)}


def resolve_model_id(model_size: str | None = None, model_id: str | None = None) -> str:
    """Resolve a HuggingFace model id from a size flag or an explicit id.

    ``model_id`` (when given) wins. Otherwise ``model_size`` — one of
    ``300m`` / ``600m`` / ``6b`` — maps to the matching Biohub repo.
    """
    if model_id:
        return model_id
    size = (model_size or DEFAULT_MODEL_SIZE).lower()
    if size not in ESMC_MODEL_IDS:
        raise ValueError(
            f"unknown ESM-C model_size {size!r}; choose from {sorted(ESMC_MODEL_IDS)}"
        )
    return ESMC_MODEL_IDS[size]


def aa_column(aa: str | None) -> int | None:
    """Return the ``aa_logprobs`` column for a one-letter residue, or ``None``.

    ``None`` for empty input or non-standard residues (``X``, ``U``, ``*``,
    multi-character indels) — those have no masked-marginal column.
    """
    if not aa or len(aa) != 1:
        return None
    return AA_TO_COL.get(aa.upper())


def protein_hash(protein: str) -> str:
    """Stable hash of a protein sequence (stop codon stripped, uppercased)."""
    seq = protein.rstrip("*").upper()
    return hashlib.sha1(seq.encode("ascii"), usedforsecurity=False).hexdigest()


def _cache_path(cache_dir: Path, h: str) -> Path:
    return cache_dir / f"{h}.npz"


def load_cache(h: str, cache_dir: Path | str = DEFAULT_CACHE_DIR) -> dict[str, Any] | None:
    """Load a cached PLM result by protein hash.

    Returns ``{"llr": (L,), "aa_logprobs": (L, 20), "embedding": (L, hidden),
    "embedding_sae": (L, hidden)}`` (whichever keys are present) or ``None`` if
    the cache file doesn't exist.
    """
    import numpy as np

    p = _cache_path(Path(cache_dir), h)
    if not p.exists():
        return None
    with np.load(p) as npz:
        out: dict[str, Any] = {}
        if "llr" in npz.files:
            out["llr"] = npz["llr"]
        if "aa_logprobs" in npz.files:
            out["aa_logprobs"] = npz["aa_logprobs"]
        if "embedding" in npz.files:
            out["embedding"] = npz["embedding"]
        if "embedding_sae" in npz.files:
            out["embedding_sae"] = npz["embedding_sae"]
    return out or None


def _save_cache(
    h: str,
    cache_dir: Path,
    *,
    llr: Any | None = None,
    aa_logprobs: Any | None = None,
    embedding: Any | None = None,
    embedding_sae: Any | None = None,
) -> None:
    import numpy as np

    cache_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    if llr is not None:
        payload["llr"] = np.asarray(llr, dtype=np.float32)
    if aa_logprobs is not None:
        payload["aa_logprobs"] = np.asarray(aa_logprobs, dtype=np.float32)
    # Store residual-stream reps as fp32, not fp16: deep models (esp. ESM-C 6B)
    # have "massive activation" dims that exceed fp16's 65504 max → inf, which
    # then poisons the SAE z-score normalize into NaN. fp32 is safe; compressed
    # npz keeps the size reasonable.
    if embedding is not None:
        payload["embedding"] = np.asarray(embedding, dtype=np.float32)
    if embedding_sae is not None:
        payload["embedding_sae"] = np.asarray(embedding_sae, dtype=np.float32)
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
# Inline ESM-C forward (per-residue embeddings + masked-marginal LLR)
# ---------------------------------------------------------------------------


def _hidden_dim(model: Any) -> int:
    """Hidden size, tolerant of ``hidden_size`` (HF) vs ``d_model`` (ESM-C)."""
    return int(
        getattr(model.config, "hidden_size", None)
        or getattr(model.config, "d_model", 0)
    )


def _esmc_forward_one(
    seq: str,
    model: Any,
    tokenizer: Any,
    device: str,
    *,
    sae_layer: int = SAE_LAYER,
    mask_chunk: int = 16,
    compute_llr: bool = True,
) -> dict[str, Any]:
    """Single forward pass: per-residue last-layer embeddings + per-position LLR.

    LLR uses masked marginals: mask each position in turn, take the log-prob
    of the wild-type AA. O(L) forward passes per sequence (the hot path),
    batched in chunks of ``mask_chunk``. Embeddings come from the final hidden
    layer; the SAE-target representation comes from block ``sae_layer``.

    When ``compute_llr`` is False, Pass 2 (the masked-marginal sweep) is skipped
    entirely — only the single embedding forward runs, and ``llr``/``aa_logprobs``
    come back ``None``. This is the embeddings/SAE-only path (no variant-effect
    LLR), which costs one forward per protein instead of ``1 + L``.
    """
    import numpy as np
    import torch

    seq = seq.rstrip("*").upper()
    L = len(seq)
    hidden_dim = _hidden_dim(model)
    if L == 0:
        return {
            "llr": np.zeros(0, dtype=np.float32) if compute_llr else None,
            "aa_logprobs": (
                np.zeros((0, len(STANDARD_AA)), dtype=np.float32) if compute_llr else None
            ),
            "embedding": np.zeros((0, hidden_dim), dtype=np.float16),
            "embedding_sae": np.zeros((0, hidden_dim), dtype=np.float16),
        }

    encoded = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(device)  # (1, L+2) — BOS, residues, EOS
    attention_mask = encoded["attention_mask"].to(device)

    # Pass 1: full forward, grab the final hidden layer.
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
    # hidden_states is a tuple (embeddings + one per layer); take the last and
    # strip BOS/EOS — residues live at indices 1..L.
    embedding = out.hidden_states[-1][0, 1 : L + 1, :].float().cpu().numpy()
    # Also keep the SAE-target layer (size-dependent: 600M→27, 6B→60) — the
    # residual stream the ESM-C SAE encodes. Same BOS/EOS strip. Computed for
    # free from the same forward; we were previously discarding every layer but
    # the last.
    embedding_sae = (
        out.hidden_states[sae_layer][0, 1 : L + 1, :].float().cpu().numpy()
    )

    # Embeddings/SAE-only path: skip the O(L) masked-marginal sweep entirely.
    if not compute_llr:
        return {
            "llr": None,
            "aa_logprobs": None,
            "embedding": embedding,
            "embedding_sae": embedding_sae,
        }

    # Pass 2: masked marginals for LLR.
    # Build a (L, L+2) batch where row i has position (i+1) replaced by <mask>.
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise RuntimeError("Tokenizer has no mask token; cannot compute masked marginals.")
    base = input_ids.expand(L, -1).clone()  # (L, L+2)
    for i in range(L):
        base[i, i + 1] = mask_token_id
    mask_batch_attn = attention_mask.expand(L, -1)

    # Token ids of the 20 standard AAs, in STANDARD_AA column order, so we can
    # gather the full per-position substitution distribution in one index_select.
    aa_token_ids = torch.tensor(
        [tokenizer.convert_tokens_to_ids(aa) for aa in STANDARD_AA], device=device
    )

    # Run in chunks to fit GPU memory.
    chunk = mask_chunk
    llr = np.zeros(L, dtype=np.float32)
    aa_logprobs = np.zeros((L, len(STANDARD_AA)), dtype=np.float32)
    for start in range(0, L, chunk):
        stop = min(start + chunk, L)
        with torch.no_grad():
            mlm_out = model(
                input_ids=base[start:stop],
                attention_mask=mask_batch_attn[start:stop],
                return_dict=True,
            )
        logits = getattr(mlm_out, "logits", None)
        if logits is None:
            raise RuntimeError(
                "Loaded model does not expose `logits`; load with AutoModelForMaskedLM "
                "to compute masked marginals."
            )
        # Per-row, position (start+k+1) is the masked one. Gather log-prob of the
        # wild-type token (llr) and the full 20-AA distribution (aa_logprobs).
        log_probs = torch.log_softmax(logits, dim=-1)
        aa_lp = log_probs.index_select(-1, aa_token_ids)  # (B, L+2, 20)
        for k in range(stop - start):
            i = start + k
            wt_id = input_ids[0, i + 1].item()
            llr[i] = float(log_probs[k, i + 1, wt_id].item())
            aa_logprobs[i] = aa_lp[k, i + 1, :].float().cpu().numpy()

    return {
        "llr": llr,
        "aa_logprobs": aa_logprobs,
        "embedding": embedding,
        "embedding_sae": embedding_sae,
    }


def _load_esmc(model_id: str, device: str, dtype: str) -> tuple[Any, Any]:
    """Load an ESM-C model + tokenizer via HuggingFace ``AutoModelForMaskedLM``.

    Requires the Biohub ``esm`` install (which pins a transformers fork with
    native ``ESMCForMaskedLM`` support):
    ``pip install esm@git+https://github.com/Biohub/esm.git@main``.
    """
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype, torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # transformers 4.57+ renamed `torch_dtype` → `dtype`.
    model = AutoModelForMaskedLM.from_pretrained(model_id, dtype=torch_dtype)
    model = model.to(device)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Public precompute entrypoint
# ---------------------------------------------------------------------------


def precompute_plm(
    proteins: dict[str, str] | list[str],
    *,
    model_size: str = DEFAULT_MODEL_SIZE,
    model_id: str | None = None,
    cache_dir: Path | str | None = None,
    device: str = "cuda",
    dtype: str | None = None,
    inline: bool = True,
    skip_missing: bool = True,
    require_aa_logprobs: bool = False,
    require_embedding_sae: bool = False,
    mask_chunk: int | None = None,
    compute_llr: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run (or load cached) ESM-C embeddings + LLR for a batch of proteins.

    Args:
        proteins: ``{label: sequence}`` or list of sequences. Labels discarded;
            output is keyed on ``protein_hash``.
        model_size: ESM-C size — ``300m`` / ``600m`` / ``6b``. Default ``6b``.
        model_id: Explicit HuggingFace repo id (or local path); overrides
            ``model_size``.
        cache_dir: Directory holding ``<hash>.npz`` cache files. Defaults to the
            per-size dir ``data/cache/plm_esmc/<SIZE>`` when not given.
        device: PyTorch device string.
        dtype: Inference precision. ``None`` → size-aware default (bf16 for 6B to
            avoid fp16 overflow; fp16 for 300M/600M).
        inline: If True, run any uncached proteins inline (requires torch +
            the Biohub ``esm``/transformers stack). If False, skip uncached
            entries silently — used when the GPU job runs separately via
            ``scripts/slurm/run_plm_embed.sbatch``.
        skip_missing: When True, returns only proteins with a cache file.
            When False, missing entries get ``None`` values.
        require_aa_logprobs: When True, a cache file lacking the per-position
            ``aa_logprobs`` distribution is treated as missing and recomputed.
        require_embedding_sae: When True, a cache file lacking the SAE-target
            layer embedding (``embedding_sae``) is treated as missing and
            recomputed. Used to backfill the key into older cache files.
        mask_chunk: Masked-marginal batch size. ``None`` → size-aware default
            (16 for ≤600M, 8 for 6B to bound VRAM on long proteins).
        compute_llr: When True (default) run the masked-marginal LLR sweep
            (``1 + L`` forwards/protein). When False, run only the single
            embedding forward (embeddings + SAE layer); ``llr``/``aa_logprobs``
            are omitted from the cache. The embeddings/SAE-only fast path.

    Returns:
        ``{protein_hash: {"llr": np.ndarray, "aa_logprobs": np.ndarray | None,
        "embedding": np.ndarray | None}}``. Missing entries either absent
        (skip_missing=True) or have ``None`` values.
    """
    size = (model_size or DEFAULT_MODEL_SIZE).lower()
    # Default to the per-size cache dir so an explicit model_size never writes
    # into another size's cache (the .npz files are keyed by protein hash only).
    cache_dir = plm_cache_dir(size) if cache_dir is None else Path(cache_dir)
    resolved_id = resolve_model_id(size, model_id)
    sae_layer = sae_layer_for(size)
    if mask_chunk is None:
        # 6B is ~10× heavier per forward (80 layers, d_model 2560); shrink the
        # masked-marginal batch so long proteins stay within VRAM.
        mask_chunk = 8 if size == "6b" else 16
    if dtype is None:
        # 6B needs bf16: its activations overflow fp16's ±65504 range (→ inf/NaN).
        # bf16 has fp32's exponent range. 300M/600M are fine in fp16.
        dtype = "bfloat16" if size == "6b" else "float16"

    if isinstance(proteins, list):
        proteins = {f"seq_{i}": s for i, s in enumerate(proteins)}

    hash_to_seq: dict[str, str] = {}
    for _label, seq in proteins.items():
        if not seq:
            continue
        h = protein_hash(seq)
        hash_to_seq.setdefault(h, seq.rstrip("*").upper())

    logger.info(
        "precompute_plm: %d inputs → %d unique sequences (model=%s, cache=%s, inline=%s)",
        len(proteins),
        len(hash_to_seq),
        resolved_id,
        cache_dir,
        inline,
    )

    # Pass 1: load whatever's cached.
    result: dict[str, dict[str, Any]] = {}
    missing: list[tuple[str, str]] = []
    for h, seq in hash_to_seq.items():
        cached = load_cache(h, cache_dir)
        stale = (require_aa_logprobs and (cached or {}).get("aa_logprobs") is None) or (
            require_embedding_sae and (cached or {}).get("embedding_sae") is None
        )
        if cached is not None and not stale:
            result[h] = cached
        else:
            missing.append((h, seq))

    if missing:
        logger.info(
            "precompute_plm: %d/%d sequences missing from cache",
            len(missing),
            len(hash_to_seq),
        )

    # Pass 2: optionally compute the missing ones inline.
    if missing and inline:
        try:
            model, tokenizer = _load_esmc(resolved_id, device, dtype)
        except Exception as exc:  # noqa: BLE001 — graceful fallback when GPU stack absent
            logger.warning(
                "precompute_plm: could not load %s (%s); skipping inline compute. "
                "Run scripts/slurm/run_plm_embed.sbatch to populate the cache offline.",
                resolved_id,
                exc,
            )
            model = None
            tokenizer = None

        if model is not None and tokenizer is not None:
            for h, seq in missing:
                try:
                    payload = _esmc_forward_one(
                        seq, model, tokenizer, device,
                        sae_layer=sae_layer, mask_chunk=mask_chunk,
                        compute_llr=compute_llr,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "precompute_plm: forward failed for hash=%s len=%d: %s",
                        h,
                        len(seq),
                        exc,
                    )
                    continue
                _save_cache(
                    h,
                    cache_dir,
                    llr=payload["llr"],
                    aa_logprobs=payload.get("aa_logprobs"),
                    embedding=payload.get("embedding"),
                    embedding_sae=payload.get("embedding_sae"),
                )
                _append_manifest(cache_dir, h, seq)
                result[h] = payload

    if not skip_missing:
        for h, _ in missing:
            result.setdefault(
                h,
                {"llr": None, "aa_logprobs": None, "embedding": None, "embedding_sae": None},
            )

    return result


# Backward-compatible alias for the previous ESM-2 entrypoint name.
precompute_plm_esm2 = precompute_plm
