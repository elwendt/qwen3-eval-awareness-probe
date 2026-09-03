"""
For every layer: average relative difference between pooled and per-fold
residualized activation vectors (over a full LOO sweep, not just 5 points),
alongside the pooled/perfold accuracy gap at C=1.0 -- to check whether
vector-level difference magnitude actually tracks the accuracy gap, rather
than assuming the layer-0 finding (small, ~4%) generalizes.

Run from the same directory as cv_diagnostics.py:
    python vector_diff_by_layer.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut

from cv_diagnostics import load_split, fit_wordcount_beta, apply_wordcount_beta
from eval_awareness_prompts import PROMPTS, IN_DISTRIBUTION_TASK_TYPES


def mean_relative_resid_diff(X, word_counts):
    beta_pooled = fit_wordcount_beta(X, word_counts)
    X_resid_pooled = apply_wordcount_beta(X, word_counts, beta_pooled)

    loo = LeaveOneOut()
    rel_diffs = []
    for train_idx, test_idx in loo.split(X):
        beta_fold = fit_wordcount_beta(X[train_idx], word_counts[train_idx])
        x_i_fold = apply_wordcount_beta(X[test_idx], word_counts[test_idx], beta_fold)
        x_i_pooled = X_resid_pooled[test_idx]
        diff = np.linalg.norm(x_i_fold - x_i_pooled)
        scale = np.linalg.norm(x_i_pooled)
        rel_diffs.append(diff / scale)
    return np.mean(rel_diffs)


def loo_accuracy_pooled(X, y, word_counts):
    beta = fit_wordcount_beta(X, word_counts)
    X_resid = apply_wordcount_beta(X, word_counts, beta)
    loo = LeaveOneOut()
    correct = 0
    for train_idx, test_idx in loo.split(X_resid):
        clf = LogisticRegression(max_iter=2000).fit(X_resid[train_idx], y[train_idx])
        correct += clf.score(X_resid[test_idx], y[test_idx]) * len(test_idx)
    return correct / len(y)


def loo_accuracy_perfold(X, y, word_counts):
    loo = LeaveOneOut()
    correct = 0
    for train_idx, test_idx in loo.split(X):
        beta = fit_wordcount_beta(X[train_idx], word_counts[train_idx])
        X_train_resid = apply_wordcount_beta(X[train_idx], word_counts[train_idx], beta)
        X_test_resid = apply_wordcount_beta(X[test_idx], word_counts[test_idx], beta)
        clf = LogisticRegression(max_iter=2000).fit(X_train_resid, y[train_idx])
        correct += clf.score(X_test_resid, y[test_idx]) * len(test_idx)
    return correct / len(y)


def main():
    indist_acts, indist_labels = load_split("indist")
    n_layers = indist_acts.shape[0]
    indist_texts = [text for text, label, task_type in PROMPTS if task_type in IN_DISTRIBUTION_TASK_TYPES]
    word_counts = np.array([len(t.split()) for t in indist_texts])

    print(f"{'layer':>6}  {'mean_rel_diff':>14}  {'pooled_acc':>11}  {'perfold_acc':>12}  {'gap':>6}")
    for layer in range(n_layers):
        X = indist_acts[layer]
        rel_diff = mean_relative_resid_diff(X, word_counts)
        pooled_acc = loo_accuracy_pooled(X, indist_labels, word_counts)
        perfold_acc = loo_accuracy_perfold(X, indist_labels, word_counts)
        print(f"{layer:6d}  {rel_diff:14.4f}  {pooled_acc:11.3f}  {perfold_acc:12.3f}  {perfold_acc-pooled_acc:6.3f}")


if __name__ == "__main__":
    main()