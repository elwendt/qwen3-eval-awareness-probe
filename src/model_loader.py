"""Day 1: Environment setup and model loading sanity check.

Goal: confirm Qwen3-8B loads cleanly through TransformerLens's
TransformerBridge wrapper, and that its residual-stream activations have
the expected shape before building anything on top of them.
"""

import torch
from transformer_lens.model_bridge import TransformerBridge

MODEL_NAME = "Qwen/Qwen3-8B"


def load_model(
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> TransformerBridge:
    """
    Load Qwen3-8B through TransformerLens's TransformerBridge.
    """
    print(f"Loading {MODEL_NAME} via TransformerBridge ...")

    try:
        model = TransformerBridge.boot_transformers(
            MODEL_NAME,
            device=device,
            dtype=dtype,
        )
    except Exception as e:
        raise RuntimeError(
            f"TransformerBridge.boot_transformers('{MODEL_NAME}') failed.\n"
            f"Original error: {e}"
        ) from e

    print("Model loaded successfully.")
    print(f"Model dtype: {model.dtype}")
    print(f"Model device: {model.cfg.device}")

    if torch.cuda.is_available():
        print(
            f"GPU memory allocated: "
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )
        
    return model


def sanity_check(model: TransformerBridge) -> dict:
    """
    Run one forward pass and confirm activation shapes match what the
    model configuration claims.
    """

    test_prompt = "The capital of France is"

    logits, cache = model.run_with_cache(test_prompt)

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    seq_len = model.to_tokens(test_prompt).shape[1]

    report = {
        "n_layers_config": n_layers,
        "d_model_config": d_model,
        "seq_len": seq_len,
        "logits_shape": tuple(logits.shape),
    }

    # Use the canonical TransformerBridge residual-stream hook.
    mid_layer = n_layers // 2
    resid_key = f"blocks.{mid_layer}.hook_out"

    assert resid_key in cache, (
        f"Expected hook point {resid_key} not found in cache."
    )

    resid = cache[resid_key]

    report["resid_post_shape"] = tuple(resid.shape)

    expected_shape = (1, seq_len, d_model)

    assert resid.shape == expected_shape, (
        f"Residual stream shape {tuple(resid.shape)} != "
        f"expected {expected_shape}."
    )

    print("Sanity check passed.")

    for k, v in report.items():
        print(f"  {k}: {v}")

    return report


if __name__ == "__main__":
    model = load_model()
    sanity_check(model)
