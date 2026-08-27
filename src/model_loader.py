"""
Day 1: Environment setup and model loading sanity check.

Goal: confirm Qwen3-8B loads cleanly through TransformerLens's HookedTransformer
wrapper, and that its residual-stream activations have the shape you expect,
before you build anything on top of them.

Why this matters (and isn't just boilerplate):
TransformerLens re-implements each model's forward pass internally, using its
own attention/MLP modules, so it can expose every intermediate activation as
a hook point. That means "does TransformerLens support model X" is a
different question from "does the transformers library support model X" --
TL needs a matching from-HF conversion function for the architecture family.
Qwen3 adds QK-norm relative to Qwen2, so even if Qwen2 loads fine in your
installed TL version, Qwen3 support is a separate thing to confirm, not an
assumption to inherit.

Run this directly on the Vast.ai instance:
    python src/model_loader.py
"""

import torch
from transformer_lens import HookedTransformer

MODEL_NAME = "Qwen/Qwen3-8B"


def load_model(device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> HookedTransformer:
    """
    Load Qwen3-8B through TransformerLens.

    Fails loudly and specifically rather than silently falling back -- a
    silent fallback (e.g. to CPU, or to a different dtype) would hide
    exactly the kind of infra problem Day 1 exists to catch before you've
    spent hours building on top of it.
    """
    print(f"Loading {MODEL_NAME} via HookedTransformer ...")
    try:
        model = HookedTransformer.from_pretrained(
            MODEL_NAME,
            device=device,
            dtype=dtype,
        )
    except Exception as e:
        raise RuntimeError(
            f"HookedTransformer.from_pretrained('{MODEL_NAME}') failed.\n"
            f"Original error: {e}\n\n"
            "Before assuming you need the fallback, check:\n"
            "  1. `pip show transformer_lens` -- are you on the latest PyPI release?\n"
            "  2. Try installing from source, which sometimes has model support\n"
            "     ahead of the last PyPI release:\n"
            "       pip install git+https://github.com/TransformerLensOrg/TransformerLens.git\n"
            "  3. If a KeyError mentions OFFICIAL_MODEL_NAMES or similar, that's TL\n"
            "     telling you plainly it doesn't know this architecture yet.\n\n"
            "If none of that works, see FALLBACK.md -- raw `transformers` +\n"
            "manual forward hooks. More setup work, but the underlying idea\n"
            "(register a hook, cache the residual stream) is identical to what\n"
            "you already did with GPT-2, just without TransformerLens's naming."
        ) from e
    return model


def sanity_check(model: HookedTransformer) -> dict:
    """
    Run one forward pass and confirm activation shapes match what the config
    claims, before trusting any downstream number.

    Mirrors the Phase 0 discipline of a sum/reconstruction sanity check
    before interpreting any decomposition: verify the plumbing before
    reading meaning into the numbers that come out of it.
    """
    test_prompt = "The capital of France is"
    tokens = model.to_tokens(test_prompt)
    logits, cache = model.run_with_cache(tokens)

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    seq_len = tokens.shape[1]

    report = {
        "n_layers_config": n_layers,
        "d_model_config": d_model,
        "seq_len": seq_len,
        "logits_shape": tuple(logits.shape),
    }

    # Check resid_post at an arbitrary mid layer -- this is the tensor
    # you'll actually be extracting activations from for the probe later.
    mid_layer = n_layers // 2
    resid_key = f"blocks.{mid_layer}.hook_resid_post"
    assert resid_key in cache, f"Expected hook point {resid_key} not found in cache."
    resid = cache[resid_key]
    report["resid_post_shape"] = tuple(resid.shape)

    expected_shape = (1, seq_len, d_model)
    assert resid.shape == expected_shape, (
        f"resid_post shape {tuple(resid.shape)} != expected {expected_shape}. "
        "If d_model is off, check whether GQA (grouped-query attention) is "
        "being represented correctly -- Qwen3 uses GQA, and conversion bugs "
        "in other libraries have shown up exactly here before."
    )

    print("Sanity check passed.")
    for k, v in report.items():
        print(f"  {k}: {v}")
    return report


if __name__ == "__main__":
    model = load_model()
    sanity_check(model)
