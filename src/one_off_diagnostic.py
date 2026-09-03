"""
One-off diagnostic: compares one point's residualized activation vector
under pooled beta (fit on the full set) vs. a beta fit with that point
excluded (fold-style), to check whether the two residualization methods
produce meaningfully different feature vectors -- not just different
downstream accuracy numbers.

Imports fit_wordcount_beta/apply_wordcount_beta/load_split directly from
cv_diagnostics.py rather than redefining them here, so there's still only
one copy of the actual regression logic in the repo.

Run from the same directory as cv_diagnostics.py:
    python one_off_diagnostic.py

Note: this predates and motivated vector_diff_by_layer.py. Point 0's
relative diff here (0.0390) matches that script's later finding that the
same point's relative diff is constant across all 36 layers -- this
script instead shows the diff varies across points (0.0190-0.0723) at a
single fixed layer, consistent with leverage depending on each point's
word count, never on layer.
"""

import numpy as np

from cv_diagnostics import load_split, fit_wordcount_beta, apply_wordcount_beta
from eval_awareness_prompts import PROMPTS, IN_DISTRIBUTION_TASK_TYPES

indist_acts, indist_labels = load_split("indist")
indist_texts = [text for text, label, task_type in PROMPTS if task_type in IN_DISTRIBUTION_TASK_TYPES]
word_counts = np.array([len(t.split()) for t in indist_texts])

layer = 0
X = indist_acts[layer]

beta_pooled = fit_wordcount_beta(X, word_counts)
X_resid_pooled = apply_wordcount_beta(X, word_counts, beta_pooled)

for i in [0, 1, 2, 3, 4]:
    train_idx = np.array([j for j in range(len(word_counts)) if j != i])
    beta_fold = fit_wordcount_beta(X[train_idx], word_counts[train_idx])
    x_i_resid_fold = apply_wordcount_beta(X[i:i+1], word_counts[i:i+1], beta_fold)
    x_i_resid_pooled = X_resid_pooled[i:i+1]

    diff = np.linalg.norm(x_i_resid_fold - x_i_resid_pooled)
    scale = np.linalg.norm(x_i_resid_pooled)
    print(f"point {i}: ||fold - pooled|| = {diff:.4f}   ||pooled|| = {scale:.4f}   relative diff = {diff/scale:.4f}")