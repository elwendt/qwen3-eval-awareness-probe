"""
Robustness check: leave-one-out cross-validated accuracy (instead of a single
80/20 split) for both the raw probe and the word-count-residualized probe,
across all layers.

Why this matters: the original layer sweep used one 80/20 split, which for
54 in-distribution prompts means a test set of only ~11 examples -- coarse
enough that 8/11 correct (0.727) is not statistically distinguishable from
chance (binomial test, one-sided p ~ 0.11). LOO-CV uses every one of the 54
prompts as a held-out test point exactly once, giving a far more defensible
accuracy estimate from the same data, no new prompts needed.

Run from the same directory as train_probe.py:
    python cv_diagnostics.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut

from eval_awareness_prompts import PROMPTS, IN_DISTRIBUTION_TASK_TYPES

LABEL_TO_INT = {"deployment": 0, "testing": 1}


def load_split(split_name):
    data = np.load(f"activations_{split_name}.npz", allow_pickle=True)
    activations = data["activations"]
    labels = np.array([LABEL_TO_INT[l] for l in data["labels"]])
    return activations, labels


def residualize_on_wordcount(X, word_counts):
    wc = word_counts.reshape(-1, 1).astype(float)
    design = np.hstack([wc, np.ones_like(wc)])
    beta, *_ = np.linalg.lstsq(design, X, rcond=None)
    return X - design @ beta


def loo_accuracy(X, y):
    loo = LeaveOneOut()
    correct = 0
    for train_idx, test_idx in loo.split(X):
        clf = LogisticRegression(max_iter=1000).fit(X[train_idx], y[train_idx])
        correct += clf.score(X[test_idx], y[test_idx]) * len(test_idx)
    return correct / len(y)


def main():
    indist_acts, indist_labels = load_split("indist")
    n_layers = indist_acts.shape[0]
    indist_texts = [text for text, label, task_type in PROMPTS if task_type in IN_DISTRIBUTION_TASK_TYPES]
    word_counts = np.array([len(t.split()) for t in indist_texts])

    print(f"Leave-one-out cross-validated accuracy (n={len(indist_labels)}), per layer:")
    print(f"{'layer':>6}  {'raw acc':>8}  {'resid acc':>10}")
    for layer in range(n_layers):
        X = indist_acts[layer]
        raw_acc = loo_accuracy(X, indist_labels)
        X_resid = residualize_on_wordcount(X, word_counts)
        resid_acc = loo_accuracy(X_resid, indist_labels)
        print(f"{layer:6d}  {raw_acc:8.3f}  {resid_acc:10.3f}")


if __name__ == "__main__":
    main()