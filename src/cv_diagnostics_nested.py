"""
Final significance check: raw vs. word-count-residualized LOO-CV accuracy,
per layer, with C (L2 regularization strength) selected via NESTED
cross-validation on the training fold only -- not hand-picked from the
earlier C-sweep, since using that sweep's result to fix a single C now
would itself leak information about this exact data into the choice.

Residualization stays per-fold throughout (the corrected version) -- the
pooled version is confirmed leaky and isn't used anywhere here.

Expect this to take longer than previous runs: 36 layers x 54 outer LOO
folds x an inner CV grid search per fold. No GPU needed, pure CPU on
small data, but likely tens of minutes rather than seconds -- see the
note below on running it detached from the SSH session.

Run from the same directory as cv_diagnostics.py:
    python cv_diagnostics_nested.py
"""

import sys

import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import LeaveOneOut

from cv_diagnostics import load_split, fit_wordcount_beta, apply_wordcount_beta
from eval_awareness_prompts import PROMPTS, IN_DISTRIBUTION_TASK_TYPES

ALPHA = 0.05
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
INNER_CV = 5


def loo_accuracy_nested_cv(X, y):
    loo = LeaveOneOut()
    correct = 0
    for train_idx, test_idx in loo.split(X):
        clf = LogisticRegressionCV(
            Cs=C_GRID, cv=INNER_CV, scoring="accuracy", max_iter=2000
        ).fit(X[train_idx], y[train_idx])
        correct += clf.score(X[test_idx], y[test_idx]) * len(test_idx)
    return correct / len(y)


def loo_accuracy_residualized_nested_cv(X, y, word_counts):
    """Residualization AND C selection both fit on the training fold only,
    every outer LOO iteration -- no leakage in either step."""
    loo = LeaveOneOut()
    correct = 0
    for train_idx, test_idx in loo.split(X):
        beta = fit_wordcount_beta(X[train_idx], word_counts[train_idx])
        X_train_resid = apply_wordcount_beta(X[train_idx], word_counts[train_idx], beta)
        X_test_resid = apply_wordcount_beta(X[test_idx], word_counts[test_idx], beta)
        clf = LogisticRegressionCV(
            Cs=C_GRID, cv=INNER_CV, scoring="accuracy", max_iter=2000
        ).fit(X_train_resid, y[train_idx])
        correct += clf.score(X_test_resid, y[test_idx]) * len(test_idx)
    return correct / len(y)


def main():
    indist_acts, indist_labels = load_split("indist")
    n_layers = indist_acts.shape[0]
    n = len(indist_labels)
    indist_texts = [text for text, label, task_type in PROMPTS if task_type in IN_DISTRIBUTION_TASK_TYPES]
    word_counts = np.array([len(t.split()) for t in indist_texts])

    bonferroni_alpha = ALPHA / n_layers
    print(f"LOO-CV accuracy (n={n}), nested C selection, per layer.")
    print(f"Bonferroni threshold: p < {bonferroni_alpha:.5f}\n")
    print(f"{'layer':>6}  {'raw':>7}  {'resid':>7}  {'correct':>9}  {'p':>8}  {'sig?':>5}")

    for layer in range(n_layers):
        X = indist_acts[layer]
        raw_acc = loo_accuracy_nested_cv(X, indist_labels)
        resid_acc = loo_accuracy_residualized_nested_cv(X, indist_labels, word_counts)

        resid_correct = round(resid_acc * n)
        test = binomtest(resid_correct, n, p=0.5, alternative="greater")
        sig = "*" if test.pvalue < bonferroni_alpha else ""

        print(f"{layer:6d}  {raw_acc:7.3f}  {resid_acc:7.3f}  {resid_correct:>6d}/{n:<3d}  {test.pvalue:8.4f}  {sig:>5}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()