"""
Apply the existing trained probe, the word-count baseline, and the
word-count-residualized probe to the register-crossed spot-check set, and
compare all three against the true style labels.

The point: in the original dataset, "testing" (eval-style) prompts are short
and "deployment" prompts are long. Here that's deliberately inverted --
testing prompts are the long ones, deployment prompts are the short ones --
while the actual style markers (imperative/decontextualized vs.
personal/casual) stay consistent with the original definition. If a
predictor is really tracking word count rather than style, it should get
these systematically backwards. If the residualized probe does meaningfully
better here than the raw probe, that's evidence the residualization actually
recovered something more robust, not just a diagnostic artifact.

Run from the same directory as both .npz files and both prompt scripts
(does NOT need the GPU/model):
    python spotcheck_analysis.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from eval_awareness_prompts import PROMPTS, IN_DISTRIBUTION_TASK_TYPES

LABEL_TO_INT = {"deployment": 0, "testing": 1}
INT_TO_LABEL = {0: "deployment", 1: "testing"}
LAYER = 0  # the one Bonferroni-significant, headline layer


def load_indist():
    data = np.load("activations_indist.npz", allow_pickle=True)
    activations = data["activations"]
    labels = np.array([LABEL_TO_INT[l] for l in data["labels"]])
    return activations, labels


def load_spotcheck():
    data = np.load("activations_spotcheck.npz", allow_pickle=True)
    activations = data["activations"]
    true_styles = np.array([LABEL_TO_INT[s] for s in data["true_styles"]])
    texts = data["texts"]
    return activations, true_styles, texts


def residualize(X, word_counts, beta=None):
    wc = word_counts.reshape(-1, 1).astype(float)
    design = np.hstack([wc, np.ones_like(wc)])
    if beta is None:
        beta, *_ = np.linalg.lstsq(design, X, rcond=None)
    return X - design @ beta, beta


def main():
    indist_acts, indist_labels = load_indist()
    spot_acts, spot_true, spot_texts = load_spotcheck()

    indist_texts = [text for text, label, task_type in PROMPTS if task_type in IN_DISTRIBUTION_TASK_TYPES]
    indist_wc = np.array([len(t.split()) for t in indist_texts])
    spot_wc = np.array([len(t.split()) for t in spot_texts])

    X_indist = indist_acts[LAYER]
    X_spot = spot_acts[LAYER]

    # --- 1. Raw trained probe, fit on all 54 in-distribution examples ---
    raw_probe = LogisticRegression(max_iter=1000).fit(X_indist, indist_labels)
    raw_preds = raw_probe.predict(X_spot)

    # --- 2. Word-count baseline ---
    wc_baseline = LogisticRegression().fit(indist_wc.reshape(-1, 1), indist_labels)
    wc_preds = wc_baseline.predict(spot_wc.reshape(-1, 1))

    # --- 3. Word-count-residualized probe, beta fit on in-distribution data
    #     and TRANSFERRED to the spot-check activations (not refit there --
    #     14 examples is nowhere near enough to fit a 4096-dim regression) ---
    X_indist_resid, beta = residualize(X_indist, indist_wc)
    X_spot_resid, _ = residualize(X_spot, spot_wc, beta=beta)
    resid_probe = LogisticRegression(max_iter=1000).fit(X_indist_resid, indist_labels)
    resid_preds = resid_probe.predict(X_spot_resid)

    # --- Report ---
    print(f"{'true':<12}{'word_ct':>8}  {'raw_probe':>10}  {'wc_baseline':>12}  {'resid_probe':>12}   text")
    print("-" * 100)
    for i in range(len(spot_true)):
        true_lbl = INT_TO_LABEL[spot_true[i]]
        raw_lbl = INT_TO_LABEL[raw_preds[i]]
        wc_lbl = INT_TO_LABEL[wc_preds[i]]
        resid_lbl = INT_TO_LABEL[resid_preds[i]]
        raw_mark = "" if raw_lbl == true_lbl else " <-- WRONG"
        wc_mark = "" if wc_lbl == true_lbl else " <-- WRONG"
        resid_mark = "" if resid_lbl == true_lbl else " <-- WRONG"
        text_preview = str(spot_texts[i])[:40]
        print(f"{true_lbl:<12}{spot_wc[i]:>8}  {raw_lbl:>10}{raw_mark:<11}{wc_lbl:>12}{wc_mark:<11}{resid_lbl:>12}{resid_mark:<11}  {text_preview}")

    raw_acc = (raw_preds == spot_true).mean()
    wc_acc = (wc_preds == spot_true).mean()
    resid_acc = (resid_preds == spot_true).mean()

    print("\n=== Accuracy on register-crossed spot-check (n=14, chance=0.5) ===")
    print(f"Word-count baseline:      {wc_acc:.3f}")
    print(f"Raw trained probe:        {raw_acc:.3f}")
    print(f"Residualized probe:       {resid_acc:.3f}")
    print("\nIf raw_probe and wc_baseline accuracy are well BELOW chance (i.e.")
    print("systematically backwards, not just noisy), that's a direct")
    print("illustration of the confound: both are tracking word count, and")
    print("word count now points the wrong way. If resid_acc is meaningfully")
    print("higher, that's evidence the residualization recovered something")
    print("more robust than length alone -- not just a diagnostic artifact.")


if __name__ == "__main__":
    main()