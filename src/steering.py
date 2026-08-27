"""
Day 6 stretch goal: probe-gated conditional steering.

Not part of the core deliverable -- only touch this file if Days 1-5 finish
with time to spare. Conceptually this extends your Phase 4B steering-controls
work directly: inject a scaled direction into the residual stream, check the
causal effect on behavior, and use controls (unconditional steering,
random-direction steering) to rule out "any perturbation of this magnitude
does this" as the explanation for whatever effect shows up.

"Probe-gated" is the new part: only steer when the probe's own confidence on
a given prompt is above some threshold, instead of steering unconditionally
on every input. That tests whether the direction is causally relevant
specifically where the probe thinks eval-framing is present -- a stronger
claim than "this direction has *some* effect on behavior in general."
"""


def steer_with_direction(model, prompt: str, direction, scale: float, layer: int):
    """
    TODO: same LayerNorm-aware, norm-scaled injection pattern as Phase 4B --
    recall that scaling resid_final does not linearly scale output, so
    calibrate `scale` empirically for Qwen3-8B. Don't assume a multiplier
    that worked for GPT-2's residual-stream norm transfers directly; the
    scale of resid_post differs a lot between models and even between
    layers of the same model.
    """
    raise NotImplementedError
