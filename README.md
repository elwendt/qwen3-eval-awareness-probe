# Probing Eval-Awareness in Qwen3-8B

MATS Sprint application project (Neel Nanda mentorship stream — "Model
Forensics" open question: *"How well can we probe for eval awareness in
real models?"*).

**Question.** Does a linear probe on Qwen3-8B's residual stream — on the
input prompt itself, single forward pass, no generation — detect
eval-framed vs. deployment-framed prompts, and does that probe generalize
to task categories held out from training?

**Full write-up:** `writeup_scaffold.md` (results, discussion, and
limitations sections; methods and abstract in progress).

---

## Corrected central finding

An early version of this analysis concluded that eval-awareness signal
was real only in layers 0–1 and vanished by layer 2 — reasoning that a
genuine functional representation would persist or strengthen with depth.
**That conclusion rested on a cross-validation leak and does not hold.**

The word-count residualization used to control for a length confound was
originally fit once on the full 54-prompt dataset before leave-one-out
cross-validation, letting each held-out point's own word-count
relationship leak into its own residualization. Correcting this (per-fold
refitting, then regularization strength selected via nested CV) changes
which layers show a defensible signal, not just the reported magnitude:

- **Corrected LOO-CV residualized accuracy is elevated above chance at
  every one of the 36 layers** (0.815–0.889), not just at layers 0–1.
- **Naive per-layer significance tests cannot be trusted at face value**
  — LOO-CV folds are correlated, not independent, which the standard
  binomial test assumes. Permutation testing (100 label shuffles per
  layer, computationally expensive, run at 2 of 36 layers as a targeted
  spot-check) is the trustworthy check:
  - **Layer 20:** cleanest, most robust result — real accuracy clears a
    well-behaved, chance-centered permutation null decisively.
  - **Layer 0:** also real, but its permutation null is heavy-tailed and
    off-center, so the same nominal p-value carries less confidence than
    layer 20's.
- A regularization-sensitivity sweep found **two different stories**
  underlying the correction: layer 0's original discrepancy was fully
  explained by classifier instability under weak regularization (and the
  corrected value, 0.815, is *higher* than the original brief's 0.722);
  layer 20's discrepancy persists even under strong regularization and
  remains mechanistically unexplained — the clearest open question for
  future work.

Full detail, including the falsification chain (regularization sweep,
the `vector_diff_by_layer.py` diagnostic that turned out to be
uninformative by mathematical construction, and the permutation test) is
in `writeup_scaffold.md`.

---

## Repository structure

```
src/
  # Core pipeline
  model_loader.py              Day 1 environment/model sanity check
  eval_awareness_prompts.py    54 in-distribution + 16 held-out prompts
  extract_activations.py       Forward pass -> activations_*.npz
  train_probe.py               Naive single-split baseline + generalization test
  spotcheck_prompts.py         Register-crossed (length-inverted) spot-check prompts
  extract_spotcheck.py         Activation extraction for the spot-check set
  spotcheck_analysis.py        Applies trained/baseline probes to the spot-check set

  # Confound correction and validation (see write-up Results section)
  cv_diagnostics.py                    Per-layer LOO-CV, pooled vs. per-fold residualization
  cv_diagnostics_nested.py             Full 36-layer sweep, nested-CV regularization selection
  regularization_sensitivity_check.py  C-sweep at layers 0 and 20 (falsification chain)
  vector_diff_by_layer.py              Vector-level residualization diagnostic (see write-up
                                        for why its flat output is a mathematical guarantee,
                                        not a finding)
  one_off_diagnostic.py                Point-level precursor to vector_diff_by_layer.py
  permutation_test.py                  100-shuffle permutation validation, layers 0 and 20

  # Not part of the core deliverable
  steering.py                  Probe-gated conditional steering stretch goal (stub,
                                never attempted — explicitly out of scope, see write-up)

  # Outputs (logs cited directly in the write-up)
  train_probe_output.log
  cv_diagnostics_pooled_vs_perfold_comparison.log
  cv_diagnostics_CORRECTED_nested_cv.log
  permutation_test_100shuffles_layers-0-20.log
  spotcheck_analysis_output.log
  vector_diff_by_layer_output.log

data/
  prompts_indist.example.jsonl    Example rows illustrating the matched-pair,
                                   explicit-framing prompt schema (pair_id,
                                   condition: eval/deployment) from an earlier,
                                   abandoned design thread -- superseded by the
                                   unpaired eval_awareness_prompts.py used in
                                   the actual pipeline. Not consumed by any
                                   script. Kept because this schema's
                                   data_schema.py originally flagged the
                                   content/vocabulary confound risk that the
                                   unpaired design's results later confirmed
                                   empirically (see write-up Limitations).
  prompts_heldout.example.jsonl   Same schema, held-out-category example.

.gitignore                     Excludes *.log by default; specific logs cited in the
                                write-up are force-added as exceptions.
```

---

## Environment / model loading notes

- Model loaded via `transformer_lens.model_bridge.TransformerBridge.boot_transformers(...)`
  — **not** the deprecated `HookedTransformer.from_pretrained(...)`.
  Residual-stream hook name is `blocks.{layer}.hook_out`.
- `TransformerBridge.boot_transformers()` does **not** fold LayerNorm or
  center weights by default (unlike legacy `HookedTransformer.from_pretrained`,
  which does both unless `from_pretrained_no_processing` is used). Raw HF
  weights are preserved automatically here — no extra flag needed, and
  `enable_compatibility_mode()` should **not** be called for this project.
- Activations are extracted as last-token residual stream (not
  mean-pooled) at each layer — the last position is what directly
  determines next-token logits, i.e. what the model actually does with
  the eval/deployment framing.

---

## Known limitations (see write-up for full detail)

- Corrected point estimates exist at all 36 layers; permutation
  validation exists at only 2 of them, due to compute cost.
- The original brief's uncorrected p-values (≈0.0009 layer 0, ≈0.005
  layer 1) are confirmed against `cv_diagnostics_pooled_vs_perfold_comparison.log`
  but not cited as evidence — they come from the leaky pooled
  residualization this project's central methods contribution corrects.
- Eval/deployment prompts are unpaired, not matched pairs; word-count
  residualization is a post-hoc statistical correction, not a
  design-level fix.
- `hook_out` / `hook_resid_post` numerical equivalence under the API
  migration is assumed, not independently re-verified this session.
- The register-crossed spot-check's word-count-*residualized* probe
  result is not cited with confidence (shares the pooled-residualization
  leak pattern); its raw-probe and word-count-baseline results are.

---

## Status

Research phase complete. Write-up in progress (Results, Discussion,
Limitations drafted; Methods and Abstract outstanding). Deadline: Sep 4,
2026, 11:59pm PT.