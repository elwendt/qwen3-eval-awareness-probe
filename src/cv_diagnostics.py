"""
[... same docstring, plus:]

CHANGE from the original version: word-count residualization is now refit
on each LOO training fold and applied to the held-out point using that
fold's beta -- not fit once on the full 54-prompt set before the LOO loop.
The earlier version let each held-out point's own word-count/activation
relationship leak into the transform used to residualize it, an optimistic
bias on exactly the number the write-up's significance claim rests on.
Also adds the binomial-test + Bonferroni-correction step that wasn't
previously in this file.
"""

import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut

from eval_awareness_prompts import PROMPTS, IN_DISTRIBUTION_TASK_TYPES

LABEL_TO_INT = {"deployment": 0, "testing": 1}
ALPHA = 0.05


def load_split(split_name):
    data = np.load(f"activations_{split_name}.npz", allow_pickle=True)
    activations = data["activations"]
    labels = np.array([LABEL_TO_INT[l] for l in data["labels"]])
    return activations, labels


def fit_wordcount_beta(X, word_counts):
    """Fit [wordcount, intercept] -> activation on X alone, so the
    resulting beta can be applied to OTHER data without that data
    influencing the fit."""
    wc = word_counts.reshape(-1, 1).astype(float)
    design = np.hstack([wc, np.ones_like(wc)])
    beta, *_ = np.linalg.lstsq(design, X, rcond=None)
    return beta


def apply_wordcount_beta(X, word_counts, beta):
    wc = word_counts.reshape(-1, 1).astype(float)
    design = np.hstack([wc, np.ones_like(wc)])
    return X - design @ beta


def loo_accuracy(X, y):
    loo = LeaveOneOut()
    correct = 0
    for train_idx, test_idx in loo.split(X):
        clf = LogisticRegression(max_iter=1000).fit(X[train_idx], y[train_idx])
        correct += clf.score(X[test_idx], y[test_idx]) * len(test_idx)
    return correct / len(y)


def loo_accuracy_residualized(X, y, word_counts):
    """Same LOO structure, but residualization is refit per-fold on
    training data only -- the held-out point never influences the
    transform used to residualize itself."""
    loo = LeaveOneOut()
    correct = 0
    for train_idx, test_idx in loo.split(X):
        beta = fit_wordcount_beta(X[train_idx], word_counts[train_idx])
        X_train_resid = apply_wordcount_beta(X[train_idx], word_counts[train_idx], beta)
        X_test_resid = apply_wordcount_beta(X[test_idx], word_counts[test_idx], beta)
        clf = LogisticRegression(max_iter=1000).fit(X_train_resid, y[train_idx])
        correct += clf.score(X_test_resid, y[test_idx]) * len(test_idx)
    return correct / len(y)

def loo_accuracy_residualized_pooled(X, y, word_counts):
    """DIAGNOSTIC ONLY -- reproduces the original (leaky) version, where
    residualization is fit once on the full set before the LOO loop.
    Built from fit_wordcount_beta/apply_wordcount_beta (the same helpers
    the per-fold version uses) rather than a separate residualize_on_wordcount
    function, so there's only one definition of the underlying regression
    logic in this file, not two that could drift apart."""
    beta = fit_wordcount_beta(X, word_counts)
    X_resid = apply_wordcount_beta(X, word_counts, beta)
    return loo_accuracy(X_resid, y)


def main():
    indist_acts, indist_labels = load_split("indist")
    n_layers = indist_acts.shape[0]
    n = len(indist_labels)
    indist_texts = [text for text, label, task_type in PROMPTS if task_type in IN_DISTRIBUTION_TASK_TYPES]
    word_counts = np.array([len(t.split()) for t in indist_texts])

    bonferroni_alpha = ALPHA / n_layers
    print(f"LOO-CV accuracy (n={n}), per layer. Bonferroni threshold: p < {bonferroni_alpha:.5f}\n")
    header = f"{'layer':>6}  {'raw':>6}  {'pooled':>7}  {'p_pool':>8}  {'sig_p':>5}  {'perfold':>8}  {'p_fold':>8}  {'sig_f':>5}"
    print(header)

    for layer in range(n_layers):
        X = indist_acts[layer]
        raw_acc = loo_accuracy(X, indist_labels)

        resid_pooled = loo_accuracy_residualized_pooled(X, indist_labels, word_counts)
        correct_pooled = round(resid_pooled * n)
        test_pooled = binomtest(correct_pooled, n, p=0.5, alternative="greater")
        sig_pooled = "*" if test_pooled.pvalue < bonferroni_alpha else ""

        resid_perfold = loo_accuracy_residualized(X, indist_labels, word_counts)
        correct_perfold = round(resid_perfold * n)
        test_perfold = binomtest(correct_perfold, n, p=0.5, alternative="greater")
        sig_perfold = "*" if test_perfold.pvalue < bonferroni_alpha else ""

        print(f"{layer:6d}  {raw_acc:6.3f}  {resid_pooled:7.3f}  {test_pooled.pvalue:8.4f}  {sig_pooled:>5}  "
              f"{resid_perfold:8.3f}  {test_perfold.pvalue:8.4f}  {sig_perfold:>5}")


if __name__ == "__main__":
    main()