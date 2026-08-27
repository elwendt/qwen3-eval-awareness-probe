"""
Day 5a: Baselines the probe result has to beat to mean anything.

Two baselines, per the brief:
  1. Random-direction / random-label probe -- the floor a probe could hit by
     chance, given your sample size and class balance.
  2. Keyword classifier -- does the prompt contain "test" / "evaluate" (or
     similar)? If this already gets high accuracy, "linear probe detects
     eval-awareness" is a much weaker finding: the model may just be
     pattern-matching a keyword, and a residual-stream direction that
     encodes that keyword isn't evidence of an internal "am I being
     evaluated" representation -- just evidence it encoded the word "test."
     Same trap Phase 0 caught in the attention-vs-causal-role gap, one level
     up the stack.
"""

from dataclasses import dataclass


@dataclass
class BaselineResult:
    name: str
    test_acc: float
    heldout_category_acc: float   # run against the SAME held-out set the real probe is tested on


def random_direction_baseline(activations, labels, layer: int, seed: int = 0) -> BaselineResult:
    """
    TODO: sample a random unit vector in d_model, classify by sign of
    projection (threshold fit on the train split), report accuracy on both
    the in-distribution test split and the held-out-category set.
    """
    raise NotImplementedError


def keyword_baseline(examples, keywords: tuple[str, ...] = ("test", "evaluat")) -> BaselineResult:
    """
    TODO: classify eval/deployment purely from substring match on prompt
    text. No activations, no model forward pass -- that's the point; this
    baseline tells you what's recoverable from surface text alone.
    """
    raise NotImplementedError
