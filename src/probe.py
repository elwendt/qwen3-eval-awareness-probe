"""
Day 4b: Train a linear probe per layer, sweep layers, extract direction.

Same tool as GPT-2 practice (`LogisticRegression`), same idea (get an
L2-normalized weight vector you can later use for steering) -- what's new is
doing it across 36 layers of an 8B model, reading the resulting curve for a
task you don't have a ground-truth circuit for yet.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from activation_extraction import ActivationSet


@dataclass
class LayerProbeResult:
    layer: int
    train_acc: float
    test_acc: float          # in-distribution held-out SPLIT -- a learnability sanity check,
                              # NOT the generalization test (that's evaluate_generalization.py)
    direction: np.ndarray     # L2-normalized weight vector, shape (d_model,)


def train_probe_at_layer(
    activations: ActivationSet,
    layer: int,
    test_size: float = 0.2,
    random_state: int = 0,
) -> LayerProbeResult:
    """
    TODO (Day 4):
      1. Train/test split WITHIN the in-distribution set.
      2. Fit LogisticRegression on activations.activations[layer].
      3. Extract coef_, L2-normalize it -- this is your steering direction if
         you reach the Day 6 stretch goal.
      4. Report both train_acc and test_acc -- a large gap at a given layer
         is itself worth noting, not just noise to average away.

    Pre-register before running: what layer-accuracy shape do you expect?
    ("low early, peaks mid-late, plateaus" is the typical shape for a linear
    probe on residual stream -- state your prediction before you see the
    curve, per your usual practice.)
    """
    raise NotImplementedError


def layer_sweep(activations: ActivationSet, layers: list[int]) -> list[LayerProbeResult]:
    """Run train_probe_at_layer across all requested layers."""
    raise NotImplementedError
