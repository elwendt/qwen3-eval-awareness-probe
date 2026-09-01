"""
Extract activations for the register-crossed spot-check set. Same extraction
logic as extract_activations.py (TransformerBridge, last-token resid, all
layers) -- just pointed at spotcheck_prompts.py instead of the main dataset.
Only 14 prompts, so this runs in seconds.

Run from the same directory as spotcheck_prompts.py (i.e. src/):
    python extract_spotcheck.py
"""

import numpy as np
import torch
from transformer_lens.model_bridge import TransformerBridge

from spotcheck_prompts import SPOTCHECK_PROMPTS

MODEL_NAME = "Qwen/Qwen3-8B"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading {MODEL_NAME} on {device}...")
model = TransformerBridge.boot_transformers(MODEL_NAME, device=device, dtype=torch.bfloat16)
model.eval()

n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
print(f"Model loaded: {n_layers} layers, d_model={d_model}")

RESID_HOOKS = [f"blocks.{layer}.hook_out" for layer in range(n_layers)]


def extract_last_token_activations(text: str) -> np.ndarray:
    tokens = model.to_tokens(text)
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: name in RESID_HOOKS,
        )
    layer_vecs = []
    for layer in range(n_layers):
        acts = cache[f"blocks.{layer}.hook_out"]
        last_token_vec = acts[0, -1, :]
        layer_vecs.append(last_token_vec.float().cpu().numpy())
    return np.stack(layer_vecs, axis=0)


if __name__ == "__main__":
    n = len(SPOTCHECK_PROMPTS)
    activations = np.zeros((n_layers, n, d_model), dtype=np.float32)
    true_styles = np.empty(n, dtype=object)
    task_types = np.empty(n, dtype=object)
    texts = []

    for i, (text, true_style, task_type) in enumerate(SPOTCHECK_PROMPTS):
        print(f"  {i}/{n}: {text[:50]}...")
        activations[:, i, :] = extract_last_token_activations(text)
        true_styles[i] = true_style
        task_types[i] = task_type
        texts.append(text)

    np.savez(
        "activations_spotcheck.npz",
        activations=activations,
        true_styles=true_styles,
        task_types=task_types,
        texts=np.array(texts, dtype=object),
    )
    print(f"Saved activations_spotcheck.npz: {activations.shape}")