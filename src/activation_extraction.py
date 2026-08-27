"""
Day 4a: Extract residual-stream activations for each prompt.

Same run_with_cache pattern as model_loader.sanity_check, applied across the
whole prompt set. What's new relative to Phase 0/4B: you're probing the
*prompt* representation only, single forward pass -- so the token position
you extract at is the LAST token of the prompt (the position that actually
determines what gets generated next), not an average over positions.

Why last-token and not mean-pooled: the residual stream at the last position
is what directly determines the next-token logits, so it's the
representation most directly implicated in downstream behavior. Mean-pooling
is a reasonable ablation to test later, but last-token is the right default
-- it's also what makes "single forward pass, no generation" well-defined
in the first place.
"""

from dataclasses import dataclass

import torch
from transformer_lens import HookedTransformer

from data_schema import PromptExample


@dataclass
class ActivationSet:
    """
    activations[layer] has shape (n_examples, d_model) -- one vector per
    example, per layer, at the last real prompt token.
    """
    activations: dict[int, torch.Tensor]
    labels: list[int]            # 1 = eval, 0 = deployment -- keep this convention everywhere downstream
    task_categories: list[str]
    pair_ids: list[str]
    example_ids: list[str]


def extract_activations(
    model: HookedTransformer,
    examples: list[PromptExample],
    layers: list[int] | None = None,
    batch_size: int = 8,
) -> ActivationSet:
    """
    TODO (Day 4): implement batched extraction.

    Steps:
      1. Tokenize each prompt. Watch padding: with a batch of different-length
         prompts you need the LAST REAL token's position per example, not a
         fixed index -- get it from the attention mask, not by assuming no
         padding.
      2. model.run_with_cache(tokens, names_filter=...) -- use names_filter
         to cache only `blocks.{l}.hook_resid_post` for l in `layers`.
         Caching every hook point on an 8B model across your whole dataset
         will not fit in memory.
      3. Pull out the last-real-token vector at each requested layer.
      4. Stack into one tensor per layer: (n_examples, d_model).

    If `layers` is None, default to every layer (range(model.cfg.n_layers))
    for the initial sweep -- narrow down after seeing the layer-accuracy
    curve, don't guess the good layer up front.
    """
    raise NotImplementedError
