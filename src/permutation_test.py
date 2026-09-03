"""
Permutation test: shuffle eval/deployment labels (breaking any true
relationship to the activations) and rerun the per-fold-residualized
LOO-CV pipeline. If accuracy under permuted (definitionally null) labels
also comes out significantly above chance, the pipeline itself -- not
eval-awareness -- is producing high accuracy in this n=54, d=4096 regime,
and neither the original nor the newly-corrected numbers can be trusted as
evidence about the model without further work. If permuted accuracy sits
at chance, that's strong, final evidence the corrected numbers reflect
real signal.

Uses a FIXED C=0.1, not nested CV, to keep runtime tractable for repeated
permutations -- justified because the nested-CV run already showed C
selection doesn't change the story much at layer 0 (0.815 either way) or
layer 20 (0.852 nested vs ~0.870 at fixed low C earlier). Pragmatic
simplification for a sanity check, not a fully rigorous permuted p-value --
flag it as such if this goes in the write-up.

Checks layers 0 and 20 only, to fit the remaining time budget. Expand
LAYERS_TO_CHECK if there's time and the result looks worth confirming
more broadly.

Run from the same directory as cv_diagnostics.py:
    python permutation_test.py
"""

import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut

from cv_diagnostics import load_split, fit_wordcount_beta, apply_wordcount_beta
from eval_awareness_prompts import PROMPTS, IN_DISTRIBUTION_TASK_TYPES

FIXED_C = 0.1
LAYERS_TO_CHECK = [0, 20]
N_PERMUTATIONS = 100
SEED = 42


def loo_accuracy_residualized_fixed_c(X, y, word_counts, C):
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

    rng = np.random.default_rng(SEED)

    print(f"Permutation test, fixed C={FIXED_C}, {N_PERMUTATIONS} shuffles per layer\n")
    print(f"{'layer':>6}  {'real_acc':>9}  " + "  ".join(f"perm{i}" for i in range(N_PERMUTATIONS)))

    for layer in LAYERS_TO_CHECK:
        X = indist_acts[layer]
        real_acc = loo_accuracy_residualized_fixed_c(X, indist_labels, word_counts, FIXED_C)

        perm_accs = []
        for p in range(N_PERMUTATIONS):
            shuffled_labels = rng.permutation(indist_labels)
            perm_acc = loo_accuracy_residualized_fixed_c(X, shuffled_labels, word_counts, FIXED_C)
            perm_accs.append(perm_acc)

        perm_str = "  ".join(f"{a:.3f}" for a in perm_accs)
        print(f"{layer:6d}  {real_acc:9.3f}  {perm_str}")
        sys.stdout.flush()

    print("\nIf permuted accuracies cluster near 0.5, the real accuracy reflects genuine signal.")
    print("If permuted accuracies are also well above 0.5, the pipeline itself is the source.")


if __name__ == "__main__":
    main()