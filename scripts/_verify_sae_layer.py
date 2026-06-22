"""SAE-layer verification: confirm ``embedding_sae`` holds the layer the SAE expects.

Loads the size-appropriate SAE (per ``--model-size``: 6B → layer 60, 600M →
layer 27), fetching only that layer's ``layer_<n>.safetensors``, and runs it on
the cached SAE-target rep (``embedding_sae``) vs the cached final-layer rep
(``embedding``). The SAE's own reconstruction loss must be far lower on the
SAE-target layer — that decisively confirms we captured the right layer, without
trusting the index convention.

Run *after* the ``embedding_sae`` backfill so the cache carries both keys.
Usage: python scripts/_verify_sae_layer.py [--model-size 6b]
"""

from __future__ import annotations

import argparse
import glob

import numpy as np
import torch
from transformers.models.esmc.modeling_esmc_sae import ESMCSAEModel

from swissisoform.plm.embed import DEFAULT_MODEL_SIZE, plm_cache_dir, sae_layer_for
from swissisoform.plm.sae import sae_repo_for


def main(argv: list[str] | None = None) -> int:
    """Compare SAE reconstruction loss on the SAE-target layer vs the final layer."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-size", default=DEFAULT_MODEL_SIZE)
    p.add_argument("--sae-repo", default=None, help="Override the SAE repo id.")
    p.add_argument("--sae-layer", type=int, default=None, help="Override the SAE layer.")
    p.add_argument("--n", type=int, default=5, help="How many cached proteins to check.")
    args = p.parse_args(argv)

    size = args.model_size.lower()
    repo = args.sae_repo or sae_repo_for(size)
    layer_idx = args.sae_layer if args.sae_layer is not None else sae_layer_for(size)
    cache_dir = plm_cache_dir(size)

    sae = ESMCSAEModel.from_pretrained(
        repo, allow_patterns=["config.json", f"layer_{layer_idx}.safetensors"]
    )
    sae.initialize_layers([layer_idx])
    layer = sae.layers[str(layer_idx)].eval()

    files = sorted(glob.glob(str(cache_dir / "*.npz")))[: args.n]
    if not files:
        print(f"no cached .npz in {cache_dir} — run the embed step first")
        return 1
    print(
        f"checking {len(files)} cached proteins in {cache_dir} "
        f"against the layer-{layer_idx} SAE ({repo})\n"
    )
    sae_losses, final_losses = [], []
    for f in files:
        with np.load(f) as z:
            if "embedding_sae" not in z.files:
                print(f"  {f.split('/')[-1]}: NO embedding_sae key — run the backfill first")
                return 1
            x_sae = torch.from_numpy(z["embedding_sae"].astype(np.float32))
            x_final = torch.from_numpy(z["embedding"].astype(np.float32))
        with torch.no_grad():
            loss_sae = float(layer(x_sae).reconstruction_loss.mean())
            loss_final = float(layer(x_final).reconstruction_loss.mean())
        sae_losses.append(loss_sae)
        final_losses.append(loss_final)
        print(
            f"  {f.split('/')[-1][:12]}  L={x_sae.shape[0]:4d}  "
            f"recon(L{layer_idx})={loss_sae:.4f}  recon(final)={loss_final:.4f}  "
            f"ratio={loss_final / max(loss_sae, 1e-9):.1f}x"
        )

    m_sae, m_final = float(np.mean(sae_losses)), float(np.mean(final_losses))
    print(f"\nmean recon loss: layer{layer_idx}={m_sae:.4f}  final={m_final:.4f}")
    assert m_sae < m_final, (
        f"layer-{layer_idx} recon loss ({m_sae:.4f}) is NOT lower than the final "
        f"layer ({m_final:.4f}) — wrong layer captured in embedding_sae!"
    )
    print(
        f"PASS: layer-{layer_idx} reconstructs {m_final / m_sae:.1f}x better than the "
        "final layer — embedding_sae is the layer the SAE expects."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
