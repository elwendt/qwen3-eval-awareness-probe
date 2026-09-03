"""
register_confound_check.py

Purpose
-------
Test whether the word-count-residualized eval/deployment signal at layers 0
and 20 (see cv_diagnostics.py) is explained by an independent, objectively
computable register proxy -- second-person direct address -- rather than
(or in addition to) raw word count.

This is NOT a new experiment requiring GPU extraction. It reuses the
activations you already have saved in .npz files and the prompt texts in
eval_awareness_prompts.py. It generalizes the residualization step that's
already in cv_diagnostics.py to a new covariate.

PRE-REGISTERED INTERPRETATION (write this down before you look at results):
  - If residualizing on second-person address ALSO drives layers 0/20 back
    toward chance -> the signal is register-general, not length-specific.
    This strengthens the "probing setups for eval-awareness are more
    confounded than they look" claim.
  - If layers 0/20 SURVIVE this residualization too -> you've ruled out two
    named surface proxies (length, direct address), not established
    eval-awareness. Report it as "survives two register controls," not as
    support for the eval-awareness interpretation -- that inference still
    needs a causal test (steering), which is out of scope here.

ASSUMPTIONS TO VERIFY AGAINST YOUR ACTUAL FILES BEFORE TRUSTING OUTPUT
-----------------------------------------------------------------------
  1. eval_awareness_prompts.py exposes something you can iterate to get, for
     each of the 54 in-distribution prompts: the text, and the eval/deployment
     label. Field names below (`text`, `style`) are placeholders -- adapt to
     your actual PROMPTS structure.
  2. The order/indexing of prompts as loaded here must exactly match the
     order/indexing of rows in your saved activation .npz arrays. This
     script does NOT independently verify that alignment -- it inherits the
     same "not directly verified" risk already flagged in your Limitations
     section. Before trusting any number below, print 5 prompt texts next
     to their word counts and spot-check them against what you'd expect.
  3. Activations are assumed to be last-token residual stream, shape
     (n_prompts, d_model) per layer, matching your extraction method.

Run the SANITY CHECK first (covariate = word_count) and confirm you recover
your known numbers (~0.815 at layer 0, ~0.852-0.870 at layer 20) before
trusting the register-feature run. If it doesn't reproduce those numbers,
the bug is in this reimplementation, not in your original finding -- don't
report anything from this script until that check passes.
"""

import re
import numpy as np
from sklearn.linear_model import LogisticRegression


# ----------------------------------------------------------------------
# Part A -- cheap, descriptive, no activations needed. Run this first.
# ----------------------------------------------------------------------

def compute_register_features(text: str) -> dict:
    """Independent, objective register proxies computed from raw text.
    None of these ARE word count, though some will correlate with it --
    checking that correlation is the point of Part A."""
    words = text.split()
    lower = text.lower()
    return {
        "n_words": len(words),
        "second_person_count": len(re.findall(r"\byou\b|\byour\b|\byou're\b", lower)),
        "please_or_could": int(bool(re.search(r"\bplease\b|\bcould you\b|\bcan you\b", lower))),
        "question_marks": text.count("?"),
        "n_sentences": max(1, len(re.findall(r"[.!?]+", text))),
    }


def part_a_descriptive(prompts):
    """
    prompts: list of (text, style) tuples -- style is whatever string your
    dataset uses (confirmed: 'testing' is one of the two values; the other
    is auto-detected below rather than hardcoded, in case it isn't literally
    'deployment').
    """
    import pandas as pd

    rows = []
    for text, style in prompts:
        feats = compute_register_features(text)
        feats["style"] = style
        rows.append(feats)
    df = pd.DataFrame(rows)

    print("=== Correlation of each register feature with raw word count ===")
    print("(high correlation with n_words means it's not an independent check)")
    for col in df.columns:
        if col in ("n_words", "style"):
            continue
        r = np.corrcoef(df["n_words"], df[col])[0, 1]
        print(f"  {col:22s} r = {r:+.3f}")

    groups = sorted(df["style"].unique())
    if len(groups) != 2:
        print(f"\nWARNING: expected exactly 2 style values, found {len(groups)}: {groups}")
        print("Stop and check your category filter before trusting anything below.")
        return df
    g0, g1 = groups

    print(f"\n=== Does each feature separate '{g0}' vs '{g1}' prompts? ===")
    a = df[df["style"] == g0]
    b = df[df["style"] == g1]
    for col in df.columns:
        if col == "style":
            continue
        print(f"  {col:22s} {g0} mean={a[col].mean():6.2f}   {g1} mean={b[col].mean():6.2f}")

    return df


# ----------------------------------------------------------------------
# Part B -- residualize + LOO-CV at layers 0 and 20 only (pre-registered
# layers, not a re-sweep). Run only after Part A confirms the feature is
# both label-separating and not just a repaint of word count.
# ----------------------------------------------------------------------

def _residualize_dimension(y_train, y_test, cov_train, cov_test):
    """OLS-regress y on the covariate using TRAIN data only; apply the
    train-fit line to both train and held-out test point. Fitting on train
    only (never touching the test point) is the leak fix already
    established in cv_diagnostics.py -- do not relax this here."""
    A = np.vstack([cov_train, np.ones_like(cov_train)]).T
    beta, *_ = np.linalg.lstsq(A, y_train, rcond=None)
    y_train_pred = A @ beta
    A_test = np.array([[cov_test, 1.0]])
    y_test_pred = (A_test @ beta)[0]
    return y_train - y_train_pred, y_test - y_test_pred


def loo_cv_accuracy(X, y, covariate, C=0.1):
    """
    X: (n, d_model) raw activations at ONE layer
    y: (n,) binary eval/deployment labels
    covariate: (n,) the variable to residualize out (word_count OR the new
               register feature -- never both at once with n=54; see note
               in the module docstring about avoiding joint multivariate
               control at this sample size)
    """
    n = X.shape[0]
    correct = 0
    for i in range(n):
        train_idx = np.array([j for j in range(n) if j != i])
        X_train_resid = np.zeros((n - 1, X.shape[1]))
        X_test_resid = np.zeros(X.shape[1])
        for d in range(X.shape[1]):
            tr_r, te_r = _residualize_dimension(
                X[train_idx, d], X[i, d],
                covariate[train_idx], covariate[i],
            )
            X_train_resid[:, d] = tr_r
            X_test_resid[d] = te_r
        clf = LogisticRegression(C=C, max_iter=1000)
        clf.fit(X_train_resid, y[train_idx])
        pred = clf.predict(X_test_resid.reshape(1, -1))[0]
        correct += int(pred == y[i])
    return correct / n


def permutation_test(X, y, covariate, C=0.1, n_shuffles=100, seed=0):
    """Same logic as your existing permutation test: shuffle labels, rerun
    the full LOO-CV pipeline, build a null distribution. Resolution is
    capped at 1/(n_shuffles+1), same caveat as your original 100-shuffle
    runs -- don't report p-values finer than that."""
    rng = np.random.default_rng(seed)
    real_acc = loo_cv_accuracy(X, y, covariate, C=C)
    null_accs = []
    for _ in range(n_shuffles):
        y_shuffled = rng.permutation(y)
        null_accs.append(loo_cv_accuracy(X, y_shuffled, covariate, C=C))
    null_accs = np.array(null_accs)
    p_value = (np.sum(null_accs >= real_acc) + 1) / (n_shuffles + 1)
    return {
        "real_accuracy": real_acc,
        "null_mean": null_accs.mean(),
        "null_max": null_accs.max(),
        "p_value": p_value,
    }


# ----------------------------------------------------------------------
# Part B data loading -- confirmed layout:
#   activations_indist.npz: activations (36, 54, 4096) float32 [n_layers, n_prompts, d_model]
#                            labels (54,) object  -- style string, e.g. 'testing'/'deployment'
#                            task_types (54,) object -- category string
# ----------------------------------------------------------------------

def load_indist_data(npz_path="activations_indist.npz"):
    """Load the 54-prompt in-distribution activations and INDEPENDENTLY VERIFY
    their order matches PROMPTS filtered the same way as Part A, by comparing
    both style and category, element-by-element, against the saved labels/
    task_types arrays. This closes the "prompt/activation alignment not
    directly verified" gap flagged in your Limitations -- for this script's
    own use of the data. It does not retroactively verify train_probe.py's or
    cv_diagnostics.py's independent re-filtering; each does its own, and
    agreement here is evidence about those too but isn't identical to
    checking them directly.
    """
    from eval_awareness_prompts import PROMPTS

    data = np.load(npz_path, allow_pickle=True)
    activations = data["activations"]        # (36, 54, 4096) -- axis 0 = layer
    saved_labels = data["labels"]             # (54,)
    saved_task_types = data["task_types"]     # (54,)

    in_dist = [(text, style, category) for (text, style, category) in PROMPTS
               if category != "creative_writing"]

    if len(in_dist) != activations.shape[1]:
        raise AssertionError(
            f"Prompt count mismatch: {len(in_dist)} prompts from PROMPTS vs "
            f"{activations.shape[1]} activation rows. Stop -- do not proceed "
            f"until this is resolved."
        )

    style_mismatches = [i for i, (_, style, _) in enumerate(in_dist)
                         if style != saved_labels[i]]
    category_mismatches = [i for i, (_, _, category) in enumerate(in_dist)
                            if category != saved_task_types[i]]

    if style_mismatches or category_mismatches:
        raise AssertionError(
            f"ALIGNMENT MISMATCH -- style mismatches at indices "
            f"{style_mismatches}, category mismatches at {category_mismatches}. "
            f"The activation rows are not in the same order PROMPTS produces "
            f"when filtered this way. Do not trust any downstream number "
            f"until this is resolved."
        )

    print(f"Alignment verified: all {len(in_dist)} prompts match "
          f"activations_indist.npz in both style and category, in order.")

    texts = [text for (text, _, _) in in_dist]
    y = np.array([1 if s == "testing" else 0 for s in saved_labels])
    word_count = np.array([len(t.split()) for t in texts], dtype=float)
    question_marks = np.array([t.count("?") for t in texts], dtype=float)

    return activations, y, word_count, question_marks


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

if __name__ == "__main__":
    from eval_awareness_prompts import PROMPTS

    styles = sorted(set(p[1] for p in PROMPTS))
    categories = sorted(set(p[2] for p in PROMPTS))
    print("styles found:", styles)
    print("categories found:", categories)
    print("total prompts:", len(PROMPTS), "(expect 70)")

    in_dist = [(text, style) for (text, style, category) in PROMPTS
               if category != "creative_writing"]
    print("in-distribution count:", len(in_dist), "(expect 54)")

    print("\n--- Part A ---")
    df = part_a_descriptive(in_dist)

    print("\n--- Part B ---")
    activations, y, word_count, question_marks = load_indist_data("activations_indist.npz")

    for layer in (0, 20):
        X = activations[layer]  # (54, 4096)

        acc_wc = loo_cv_accuracy(X, y, word_count, C=0.1)
        print(f"Layer {layer:2d}  word-count-residualized accuracy = {acc_wc:.3f}  "
              f"(SANITY CHECK -- compare to your known ~0.815 at layer 0, "
              f"~0.852-0.870 at layer 20 before trusting anything below)")

        acc_qm = loo_cv_accuracy(X, y, question_marks, C=0.1)
        print(f"Layer {layer:2d}  question-mark-residualized accuracy = {acc_qm:.3f}")

    print(
        "\nIf the word-count sanity-check numbers above don't match your known "
        "values, stop here and debug this script before interpreting the "
        "question-mark numbers -- the bug would be in this reimplementation, "
        "not in your original finding."
    )

    print("\n--- Permutation test: layer 0, question-mark covariate ---")
    print("(~100x the cost of one loo_cv_accuracy call -- if this feels slow, "
          "that's expected; let it finish rather than interrupting partway.")
    result = permutation_test(activations[0], y, question_marks, C=0.1, n_shuffles=100)
    print(result)