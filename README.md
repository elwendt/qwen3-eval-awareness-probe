# Qwen3 Eval-Awareness Probe

MATS sprint application project, Neel Nanda mentorship stream. Deadline: **Sep 4, 2026, 11:59pm PT**.

## Research question

Neel's "Model Forensics" open question: *"How well can we probe for eval awareness in real models?"*

**Scope:** does a linear probe on Qwen3-8B's residual stream — on the input
prompt alone, single forward pass, no generation — detect eval-framed vs.
deployment-framed prompts? Does it generalize to task categories held out
entirely from training (not just a random split)?

**Deferred, not attempted:** probing token positions inside `<think>` blocks
during generation, for *verbalized* eval-awareness. Genuinely new
engineering (activation capture during generation) beyond this pipeline —
named as future work, not started.

**Explicitly not this project:** CoT faithfulness (whether stated reasoning
matches the model's actual computation) — a different question, out of
scope entirely.

**Note on process:** an earlier version of this README described a
matched-pair prompt design (`pair_id`-linked eval/deployment prompts,
built by a separate never-implemented thread). That design was set aside
early for time reasons in favor of the unpaired dataset below. That was a
known risk, not an oversight — and it turned out to be the dominant effect
in the results (see below).

## Status: research phase complete, write-up in progress

## Repo structure

```
qwen3-eval-awareness-probe/
├── README.md
├── requirements_sprint.txt
└── src/
├── model_loader.py # sanity-check: Qwen3-8B loads via TransformerBridge, 36 layers, d_model=4096
├── eval_awareness_prompts.py # the actual dataset: 54 in-distribution + 16 held-out prompts, unpaired eval/deployment framing
├── extract_activations.py # last-token residual-stream extraction, all 36 layers -> activations_indist.npz, activations_heldout.npz
├── train_probe.py # logistic-regression probe, layer sweep, random-direction + word-count baselines, held-out generalization, decision-score/word-count correlation
├── cv_diagnostics.py # leave-one-out CV (n=54), word-count residualization, per-layer significance
├── analyze_results.py # -> layer_accuracy.csv, layer_sweep.png, baseline_comparison.png
├── spotcheck_prompts.py # register-crossed spot-check set: true style markers held fixed, length deliberately inverted
├── extract_spotcheck.py # activation extraction for the spot-check set -> activations_spotcheck.npz
├── spotcheck_analysis.py # applies the trained/residualized probes + word-count baseline to the spot-check set
├── train_probe_output.log
├── train_probe_output_2.log
├── cv_diagnostics_output.log
├── spotcheck_analysis_output.log
├── layer_accuracy.csv
├── layer_sweep.png
└── baseline_comparison.png
```

## Day-by-day (from the sprint brief)

| Day | Task |
|---|---|
| 1 | Env + hypothesis lock; confirm Qwen3-8B loads via `HookedTransformer`; sanity-check activation shapes |
| 2 | Build in-distribution prompt set (~60-80 eval/deployment pairs across a few task types) |
| 3 | Build held-out task-category set (categories never seen in Day 2) |
| 4 | Activation extraction, probe training, layer sweep |
| 5 | Baselines + held-out generalization test; report honestly, including a clean null |
| 6 | Stretch (steering, or CoT-position variant) if ahead of schedule; buffer/catch-up if not |
| 7 | Buffer + final figures |


`activations_*.npz` files are generated locally by the extraction scripts
and are gitignored (large binary activation caches, not source) — rerun the
relevant extraction script to regenerate them rather than looking for them
in the repo.

## Setup

```bash
pip install -r requirements_sprint.txt
cd src
python model_loader.py            # sanity check -- run this first
```

**Environment gotchas, if rebuilding the instance:** installing
`requirements_sprint.txt` upgrades `torch`, which breaks a Vast.ai
template's preinstalled `torchvision`/`torchaudio` (ABI mismatch, not
needed for this project — `pip uninstall -y torchvision torchaudio` fixes
it). `bfloat16` tensors have no NumPy equivalent — extraction code upcasts
with `.float()` before `.cpu().numpy()`.

## Pipeline

Run from `src/`, in order:

```bash
python extract_activations.py     # -> activations_indist.npz, activations_heldout.npz
python train_probe.py             # headline single-split numbers + baselines + correlation diagnostic
python cv_diagnostics.py          # leave-one-out CV, word-count residualization, per-layer significance
python analyze_results.py         # -> layer_accuracy.csv, layer_sweep.png, baseline_comparison.png

python spotcheck_prompts.py       # sanity-check the spot-check set (word counts, length inversion)
python extract_spotcheck.py       # -> activations_spotcheck.npz
python spotcheck_analysis.py      # apply trained/residualized probes + baseline to the spot-check set
```

## Results

**Naive headline (single 80/20 split, `train_probe.py`): 1.000 in-distribution,
0.938 held-out.** This looked too clean to trust, and it was — a word-count
baseline alone scored 0.727, and the probe's decision score correlated
strongly with raw prompt length (r = -0.598 in-distribution, r = -0.863
held-out).

**After controlling for it:** leave-one-out CV (n=54) with word-count
residualized out per layer, Bonferroni-corrected across all 36 layers, finds
a real signal beyond word count only at **layers 0-1** (0.722 / 0.685
residualized accuracy, layer 0 significant at p≈0.0009). From layer 2
onward, residualized accuracy is indistinguishable from chance (0.42-0.57).
A flat-perfect raw-accuracy curve across all 36 layers (rather than the
low→peak→plateau shape a computed concept would show) is itself consistent
with a signal present from layer 0 that the residual stream simply carries
forward unmodified, not one the model builds up.

**Follow-up spot-check** (`spotcheck_prompts.py` / `spotcheck_analysis.py`):
14 hand-written, register-crossed examples with true style markers held
fixed but length deliberately inverted relative to the training pattern.
Word-count baseline: 0.143 (fooled backwards, as predicted). Raw trained
probe: 0.857. Word-count-residualized probe: 0.786 (the 0.857-vs-0.786 gap
is not itself meaningful at n=14 — 3-vs-2 discordant predictions). Both
hold up well above chance despite the adversarial inversion, corroborating
the layers 0-1 finding as more than a subtler length proxy.

**Headline finding:** not "Qwen3-8B represents eval-awareness" — a naive
probe mostly detects prompt register/length, and the small part that
survives correction doesn't have the shape you'd expect from a persistent,
used representation (appears immediately, decays by layer 2, rather than
persisting or strengthening with depth).

## Notes

- `eval_awareness_prompts.py` is a plain Python list of
  `(text, label, task_type)` tuples — 18 eval + 18 deployment across
  `coding`, `factual_qa`, `agentic_instruction` (in-distribution), 8+8 for
  `creative_writing` (held out entirely). Unpaired by design (see Research
  question note above).
- `spotcheck_prompts.py`'s docstring records a real correction worth
  knowing about if you extend the spot-check further: its first version
  reused 5 of 7 topics/templates already present in the 54-prompt training
  set, which would have let a correct classification reflect memorization
  rather than a genuine register signal. Caught via a term-search against
  the training file before trusting the result; replaced with verified-novel
  topics.
- Three `train_probe`/`cv_diagnostics` logs exist from successive runs as
  the analysis was tightened — `train_probe_output.log` and
  `train_probe_output_2.log` are earlier iterations kept for the record;
  `cv_diagnostics_output.log` has the final, statistically-corrected result
  and is the one to cite.
