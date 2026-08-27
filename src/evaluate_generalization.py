"""
Day 5b: The actual generalization test -- this is the deliverable.

Train on in-distribution task categories, test on held-out categories never
seen during training. This is stricter than the random train/test split in
probe.py (which is only a learnability sanity check) because a random split
lets the probe partially key off template-specific surface features that
happen to correlate with eval-framing in your specific prompt set. A probe
that only works on categories it's seen isn't evidence the model has a
general internal eval-awareness representation -- it's evidence the probe
memorized your templates.

Read the result honestly (the brief says this explicitly for Day 5): a clean
null -- probe generalizes at chance to held-out categories -- is a
publishable, honest finding, not a failed sprint. Don't let a positive
result's greater excitement bias how carefully you check a null one.
"""


def evaluate_on_heldout(
    probe,                # a trained LayerProbeResult / sklearn model, trained on in-dist only
    heldout_activations,   # ActivationSet built from the held-out task-category prompts
) -> dict:
    """
    TODO (Day 5):
      1. Apply the in-distribution-trained probe to held-out-category
         activations. No retraining, no peeking.
      2. Report accuracy AND a per-category breakdown -- "generalizes to
         some categories, not others" is more informative than one number,
         and is the same "distrust the summary statistic" move the SAE
         work's bimodal-L0 investigation modeled.
      3. Compare against both baselines from baselines.py on the identical
         held-out set -- the generalization claim only means something
         relative to what a keyword-matcher or random direction gets on the
         same prompts.
    """
    raise NotImplementedError
