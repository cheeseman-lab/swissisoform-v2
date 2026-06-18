"""Step-1 verification: confirm hidden_states[27] is the layer the ESM-C SAE
reconstructs.

Loads the layer-27 SAE (``biohub/ESMC-600M-sae-k64-codebook16384``, fetching
only ``layer_27.safetensors``) and runs it on the cached layer-27 rep
(``embedding_sae``) vs the cached final-layer rep (``embedding``). The SAE's own
reconstruction loss must be far lower on layer 27 — that decisively confirms we
captured the right layer, without trusting the index convention.

Run *after* the ``embedding_sae`` backfill so the cache carries both keys.
Usage: python scripts/_verify_sae_layer27.py
"""

from __future__ import annotations

import glob

import numpy as np
import torch
from transformers.models.esmc.modeling_esmc_sae import ESMCSAEModel

SAE_REPO = "biohub/ESMC-600M-sae-k64-codebook16384"
SAE_LAYER = 27


def main() -> int:
    """Compare SAE reconstruction loss on layer 27 vs the final layer."""
    sae = ESMCSAEModel.from_pretrained(
        SAE_REPO, allow_patterns=["config.json", f"layer_{SAE_LAYER}.safetensors"]
    )
    sae.initialize_layers([SAE_LAYER])
    layer = sae.layers[str(SAE_LAYER)].eval()

    files = sorted(glob.glob("data/cache/plm_esmc/*.npz"))[:5]
    print(f"checking {len(files)} cached proteins against the layer-{SAE_LAYER} SAE\n")
    l27, l36 = [], []
    for f in files:
        with np.load(f) as z:
            if "embedding_sae" not in z.files:
                print(f"  {f.split('/')[-1]}: NO embedding_sae key — run the backfill first")
                return 1
            x27 = torch.from_numpy(z["embedding_sae"].astype(np.float32))
            x36 = torch.from_numpy(z["embedding"].astype(np.float32))
        with torch.no_grad():
            loss27 = float(layer(x27).reconstruction_loss.mean())
            loss36 = float(layer(x36).reconstruction_loss.mean())
        l27.append(loss27)
        l36.append(loss36)
        print(
            f"  {f.split('/')[-1][:12]}  L={x27.shape[0]:4d}  "
            f"recon(L27)={loss27:.4f}  recon(L36)={loss36:.4f}  "
            f"ratio={loss36 / max(loss27, 1e-9):.1f}x"
        )

    m27, m36 = float(np.mean(l27)), float(np.mean(l36))
    print(f"\nmean recon loss: layer27={m27:.4f}  layer36={m36:.4f}")
    assert m27 < m36, (
        f"layer-27 recon loss ({m27:.4f}) is NOT lower than layer-36 ({m36:.4f}) "
        "— wrong layer captured in embedding_sae!"
    )
    print(
        f"PASS: layer-27 reconstructs {m36 / m27:.1f}x better than layer-36 "
        "— embedding_sae is the layer the SAE expects."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
