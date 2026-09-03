"""
Does the pooled-vs-perfold gap shrink under stronger regularization?
If it does, that's direct evidence the gap is an artifact of an
under-regularized, underdetermined (n<<d) classifier being unstable to
small input perturbations -- not a real difference in signal between the
two residualization methods.

Run from the same directory as cv_diagnostics.py:
    python regularization_sensitivity_check.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut

from cv_diagnostics import load_split, fit_wordcount_beta, apply_wordcount_beta
from eval_awareness_prompts import PROMPTS, IN_DISTRIBUTION_TASK_TYPES

C_VALUES = [1.0, 0.1, 0.01, 0.001]
LAYERS_TO_CHECK = [0, 20]


def loo_accuracy_pooled(X, y, word_counts, C):
    beta = fit_wordcount_beta(X, word_counts)
    X_resid = apply_wordcount_beta(X, word_counts, beta)
    loo = LeaveOneOut()
    correct = 0
    for train_idx, test_idx in loo.split(X_resid):
        clf = LogisticRegression(max_iter=2000, C=C).fit(X_resid[train_idx], y[train_idx])
        correct += clf.score(X_resid[test_idx], y[test_idx]) * len(test_idx)
    return correct / len(y)


def loo_accuracy_perfold(X, y, word_counts, C):
    loo = LeaveOneOut()
    correct = 0
    for train_idx, test_idx in loo.split(X):
        beta = fit_wordcount_beta(X[train_idx], word_counts[train_idx])
        X_train_resid = apply_wordcount_beta(X[train_idx], word_counts[train_idx], beta)
        X_test_resid = apply_wordcount_beta(X[test_idx], word_counts[test_idx], beta)
        clf = LogisticRegression(max_iter=2000, C=C).fit(X_train_resid, y[train_idx])
        correct += clf.score(X_test_resid, y[test_idx]) * len(test_idx)
    return correct / len(y)


def main():
    indist_acts, indist_labels = load_split("indist")
    indist_texts = [text for text, label, task_type in PROMPTS if task_type in IN_DISTRIBUTION_TASK_TYPES]
    word_counts = np.array([len(t.split()) for t in indist_texts])

    print(f"{'layer':>6}  {'C':>7}  {'pooled':>8}  {'perfold':>8}  {'gap':>6}")
    for layer in LAYERS_TO_CHECK:
        X = indist_acts[layer]
        for C in C_VALUES:
            pooled = loo_accuracy_pooled(X, indist_labels, word_counts, C)
            perfold = loo_accuracy_perfold(X, indist_labels, word_counts, C)
            print(f"{layer:6d}  {C:7.3f}  {pooled:8.3f}  {perfold:8.3f}  {perfold - pooled:6.3f}")


if __name__ == "__main__":
    main()