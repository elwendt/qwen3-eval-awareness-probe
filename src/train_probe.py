"""
Day B, step 2: train the linear probe, sweep layers, run baselines, and test
generalization on the held-out task type.

This script does NOT need the GPU/model loaded -- it only reads the .npz
files produced by extract_activations.py. Safe to rerun repeatedly while
iterating.

Run from the same directory as the .npz files and eval_awareness_prompts.py:
    python train_probe.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from eval_awareness_prompts import PROMPTS, IN_DISTRIBUTION_TASK_TYPES

LABEL_TO_INT = {"deployment": 0, "testing": 1}
RANDOM_SEED = 42


def load_split(split_name: str):
    data = np.load(f"activations_{split_name}.npz", allow_pickle=True)
    activations = data["activations"]  # [n_layers, n_prompts, d_model]
    labels = np.array([LABEL_TO_INT[l] for l in data["labels"]])
    task_types = data["task_types"]
    return activations, labels, task_types


def word_count_baseline(texts_train, y_train, texts_test, y_test):
    """Cheap baseline: does prompt length alone predict testing vs. deployment?
    Deployment-style prompts tend to be longer (backstory, informal tone);
    testing-style prompts tend to be terse. This replaces a keyword baseline,
    which doesn't apply here since neither class uses literal 'test'/'evaluate'
    vocabulary in this prompt set."""
    X_train = np.array([[len(t.split())] for t in texts_train])
    X_test = np.array([[len(t.split())] for t in texts_test])
    clf = LogisticRegression().fit(X_train, y_train)
    return clf.score(X_test, y_test)


def random_direction_baseline(X_train, y_train, X_test, y_test, d_model, seed=RANDOM_SEED):
    """Project activations onto a single random (untrained) direction and
    see how well a threshold on that projection predicts the label. Tests
    whether ANY arbitrary direction picks up spurious signal."""
    rng = np.random.default_rng(seed)
    random_dir = rng.standard_normal(d_model)
    random_dir /= np.linalg.norm(random_dir)
    proj_train = (X_train @ random_dir).reshape(-1, 1)
    proj_test = (X_test @ random_dir).reshape(-1, 1)
    clf = LogisticRegression().fit(proj_train, y_train)
    return clf.score(proj_test, y_test)


def main():
    indist_acts, indist_labels, indist_task_types = load_split("indist")
    heldout_acts, heldout_labels, heldout_task_types = load_split("heldout")

    n_layers = indist_acts.shape[0]
    d_model = indist_acts.shape[2]
    print(f"Loaded: {n_layers} layers, d_model={d_model}")
    print(f"In-distribution: {indist_acts.shape[1]} prompts, held-out: {heldout_acts.shape[1]} prompts")

    # Recover prompt texts in the same order as extraction, for the length baseline.
    indist_texts = [text for text, label, task_type in PROMPTS if task_type in IN_DISTRIBUTION_TASK_TYPES]

    # Single train/test split (by index) reused across all layers for a fair sweep.
    idx = np.arange(indist_acts.shape[1])
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=RANDOM_SEED, stratify=indist_labels
    )
    y_train, y_test = indist_labels[idx_train], indist_labels[idx_test]
    texts_train = [indist_texts[i] for i in idx_train]
    texts_test = [indist_texts[i] for i in idx_test]

    # --- Layer sweep ---
    layer_accuracies = []
    probes_by_layer = {}
    for layer in range(n_layers):
        X = indist_acts[layer]  # [n_prompts, d_model]
        X_train, X_test = X[idx_train], X[idx_test]
        clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        layer_accuracies.append(acc)
        probes_by_layer[layer] = clf

    best_layer = int(np.argmax(layer_accuracies))
    best_acc = layer_accuracies[best_layer]
    print(f"\nBest layer: {best_layer} (in-distribution test accuracy: {best_acc:.3f})")
    print("\nPer-layer in-distribution test accuracy:")
    for layer, acc in enumerate(layer_accuracies):
        marker = "  <-- best" if layer == best_layer else ""
        print(f"  layer {layer:2d}: {acc:.3f}{marker}")

    # --- Plot layer sweep ---
    plt.figure(figsize=(8, 4))
    plt.plot(range(n_layers), layer_accuracies, marker="o")
    plt.axvline(best_layer, color="red", linestyle="--", label=f"peak (layer {best_layer})")
    plt.xlabel("Layer")
    plt.ylabel("In-distribution test accuracy")
    plt.title("Probe accuracy by layer: eval-awareness (testing vs. deployment framing)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("layer_accuracy.png", dpi=150)
    print("Saved layer_accuracy.png")

    # --- Save the probe direction at the peak layer ---
    best_probe = probes_by_layer[best_layer]
    direction = best_probe.coef_[0] / np.linalg.norm(best_probe.coef_[0])
    np.save(f"probe_direction_layer{best_layer}.npy", direction)
    print(f"Saved probe_direction_layer{best_layer}.npy")

    # --- Baselines, evaluated at the same peak layer for a fair comparison ---
    X_best = indist_acts[best_layer]
    rand_acc = random_direction_baseline(
        X_best[idx_train], y_train, X_best[idx_test], y_test, d_model
    )
    length_acc = word_count_baseline(texts_train, y_train, texts_test, y_test)

    # --- Held-out generalization test, using the peak-layer probe as-is (no retraining) ---
    X_heldout = heldout_acts[best_layer]
    
    heldout_acc = best_probe.score(X_heldout, heldout_labels)
    indist_word_counts = np.array([len(t.split()) for t in indist_texts])
    indist_decision_scores = best_probe.decision_function(X_best)
    indist_corr = np.corrcoef(indist_decision_scores, indist_word_counts)[0, 1]

    heldout_texts = [text for text, label, task_type in PROMPTS if task_type not in IN_DISTRIBUTION_TASK_TYPES]
    heldout_word_counts = np.array([len(t.split()) for t in heldout_texts])
    heldout_decision_scores = best_probe.decision_function(X_heldout)
    heldout_corr = np.corrcoef(heldout_decision_scores, heldout_word_counts)[0, 1]

    print(f"\nCorrelation(probe decision score, word count) -- in-distribution: r={indist_corr:.3f}")
    print(f"Correlation(probe decision score, word count) -- held-out:        r={heldout_corr:.3f}")
    print("(|r| well above ~0.5 here means the trained probe's direction overlaps")
    print(" substantially with plain word count -- treat the accuracy numbers above")
    print(" as evidence for a length/register effect, not eval-awareness specifically,")
    print(" until this is ruled out or the dataset is redesigned with matched pairs.)")

    def residualize_on_wordcount(X, word_counts):
        wc = word_counts.reshape(-1, 1).astype(float)
        design = np.hstack([wc, np.ones_like(wc)])  # [wordcount, intercept]
        beta, *_ = np.linalg.lstsq(design, X, rcond=None)
        return X - design @ beta

    X_best_resid = residualize_on_wordcount(X_best, indist_word_counts)
    X_train_resid, X_test_resid = X_best_resid[idx_train], X_best_resid[idx_test]
    clf_resid = LogisticRegression(max_iter=1000).fit(X_train_resid, y_train)
    resid_acc = clf_resid.score(X_test_resid, y_test)

    print(f"\nProbe accuracy AFTER regressing out word count (layer {best_layer}): {resid_acc:.3f}")
    print(f"(Compare to chance = 0.500 and the original uncontrolled accuracy = {best_acc:.3f}.")
    print(" Near chance means the original result was essentially the length confound;")
    print(" well above chance means something survives beyond length alone.)")

    # --- Summary ---
    print("\n=== Summary ===")
    print(f"{'Metric':<45}{'Accuracy':>10}")
    print(f"{'-'*55}")
    print(f"{'Chance level (balanced classes)':<45}{0.5:>10.3f}")
    print(f"{'Random-direction probe baseline':<45}{rand_acc:>10.3f}")
    print(f"{'Word-count baseline':<45}{length_acc:>10.3f}")
    print(f"{'Trained probe, in-distribution test (layer ' + str(best_layer) + ')':<45}{best_acc:>10.3f}")
    print(f"{'Trained probe, HELD-OUT generalization':<45}{heldout_acc:>10.3f}")


if __name__ == "__main__":
    main()
