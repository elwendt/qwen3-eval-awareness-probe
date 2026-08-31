"""
Day B, step 1: extract resid_post activations for every prompt in
eval_awareness_prompts.py, mean-pooled per layer, and save them to disk.

This script needs the GPU/model. It does NOT train anything — that's
train_probe.py, deliberately kept separate so you can iterate on the probe
without re-running this (slow, GPU-bound) step every time.

Run from the same directory as eval_awareness_prompts.py:
    python extract_activations.py
"""

import numpy as np
import torch
from transformer_lens import HookedTransformer

from eval_awareness_prompts import PROMPTS, IN_DISTRIBUTION_TASK_TYPES, HELD_OUT_TASK_TYPES

# ---------------------------------------------------------------------------
# IMPORTANT: set this to the EXACT model name/loading call that worked in
# your Day 1 model_loader.py. Don't guess a new one here — reuse whatever
# already loaded successfully, since that's the one confirmed-working path.
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-8B"  # <-- replace with your confirmed-working value if different

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading {MODEL_NAME} on {device}...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
model.eval()

n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
print(f"Model loaded: {n_layers} layers, d_model={d_model}")


def extract_last_token_activations(text: str) -> np.ndarray:
    """Run one prompt through the model, return a (n_layers, d_model) array
    of resid_post activations AT THE LAST PROMPT TOKEN, one row per layer.

    Last-token, not mean-pooled: the residual stream at the last position is
    what directly determines the next-token logits, so it's the
    representation most directly implicated in downstream behavior --
    mean-pooling smears that signal across every token in the prompt.
    Because prompts are processed one at a time here (no batching), "last
    token" is simply index -1 -- no attention-mask bookkeeping needed to
    find the last REAL token under padding, which a batched version would
    require."""
    tokens = model.to_tokens(text)
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: name.endswith("resid_post"),
        )
    layer_vecs = []
    for layer in range(n_layers):
        acts = cache["resid_post", layer]  # shape: [1, seq_len, d_model]
        last_token_vec = acts[0, -1, :]  # shape: [d_model]
        assert last_token_vec.shape == (d_model,), f"unexpected shape at layer {layer}: {last_token_vec.shape}"
        layer_vecs.append(last_token_vec.cpu().numpy())
    return np.stack(layer_vecs, axis=0)  # shape: [n_layers, d_model]


def extract_split(prompts_subset, split_name: str):
    n = len(prompts_subset)
    activations = np.zeros((n_layers, n, d_model), dtype=np.float32)
    labels = np.empty(n, dtype=object)
    task_types = np.empty(n, dtype=object)

    for i, (text, label, task_type) in enumerate(prompts_subset):
        if i % 10 == 0:
            print(f"  [{split_name}] {i}/{n}")
        vec = extract_last_token_activations(text)  # [n_layers, d_model]
        activations[:, i, :] = vec
        labels[i] = label
        task_types[i] = task_type

    out_path = f"activations_{split_name}.npz"
    np.savez(
        out_path,
        activations=activations,  # [n_layers, n_prompts, d_model]
        labels=labels,
        task_types=task_types,
    )
    print(f"Saved {out_path}: activations {activations.shape}")
    return out_path


if __name__ == "__main__":
    indist_prompts = [p for p in PROMPTS if p[2] in IN_DISTRIBUTION_TASK_TYPES]
    heldout_prompts = [p for p in PROMPTS if p[2] in HELD_OUT_TASK_TYPES]

    print(f"In-distribution prompts: {len(indist_prompts)}")
    print(f"Held-out prompts: {len(heldout_prompts)}")

    extract_split(indist_prompts, "indist")
    extract_split(heldout_prompts, "heldout")

    print("Done. Run train_probe.py next — it does not need the GPU instance's model loaded.")
