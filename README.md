# Qwen3 Eval-Awareness Probe

MATS sprint application project, Neel Nanda mentorship stream. Deadline: **Sep 4, 2026, 11:59pm PT**.

## Research question

Neel's "Model Forensics" open question: *"How well can we probe for eval awareness in real models?"*

**Core scope (Days 1-5):** does a linear probe on Qwen3-8B's residual stream
— on the input prompt alone, single forward pass, no generation — detect
eval-framed vs. deployment-framed prompts? Does it generalize to task
categories held out entirely from training (not just a random split)?

**Hypothesis — lock this before writing prompts:**
`TODO`: write your one-sentence prediction here before Day 2. What
layer-accuracy shape do you expect (rises then plateaus? peaks and falls?
flat?)? Do you expect the held-out-category test to succeed, partially
succeed, or fail? Committing to this in writing is what makes Day 5's result
a real test rather than a post-hoc story.

**Deferred (Day 6 stretch, not core):** probing token positions inside
`<think>` blocks during generation, for *verbalized* eval-awareness. This is
genuinely new engineering (activation capture during generation) not covered
by the core pipeline below.

**Explicitly not this project:** CoT faithfulness (whether stated reasoning
matches the model's actual computation) — a different question, out of
scope here entirely, even for the Day 6 stretch.

## Repo structure

```
qwen3-eval-awareness-probe/
├── README.md
├── requirements_sprint.txt
├── FALLBACK.md                    # what to do if TransformerLens rejects Qwen3-8B
├── src/
│   ├── model_loader.py            # Day 1 -- load + sanity-check Qwen3-8B via HookedTransformer
│   ├── data_schema.py             # prompt-pair contract + validation, used by Days 2-3
│   ├── activation_extraction.py   # Day 4a stub -- run_with_cache across the prompt set
│   ├── probe.py                   # Day 4b stub -- LogisticRegression, layer sweep, direction
│   ├── baselines.py               # Day 5a stub -- random-direction + keyword baselines
│   ├── evaluate_generalization.py # Day 5b stub -- the actual held-out-category test
│   └── steering.py                # Day 6 stretch stub -- probe-gated conditional steering
├── data/
│   ├── prompts_indist.example.jsonl   # schema example -- replace with real Day 2 prompt set
│   └── prompts_heldout.example.jsonl  # schema example -- replace with real Day 3 prompt set
├── results/                       # layer-sweep results, generalization report
└── figures/                       # layer-accuracy plot, baseline comparison, in-dist-vs-held-out chart
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

## Setup

```bash
pip install -r requirements_sprint.txt
python src/model_loader.py   # Day 1 sanity check -- run this first, before anything else
```

## What carries over from the circuit audit, and what's new

| | Circuit audit (Phase 0/4B) | This project |
|---|---|---|
| Method shape | `run_with_cache` → analyze | `run_with_cache` → `LogisticRegression` probe → layer sweep — same shape |
| Causal check | ablation, path patching | steering with probe direction (Day 6 stretch) — same idea as Phase 4B controls |
| What's genuinely new | — | held-out-by-*category* generalization test; no ground-truth circuit to check against — the probe's own generalization behavior *is* the evidence |

## Notes

- The `data/*.example.jsonl` files show the schema (matched eval/deployment
  pairs, sharing a `pair_id`, differing only in framing) — they are not the
  real dataset. Building the real ~60-80 pair in-distribution set (Day 2) and
  the held-out-category set (Day 3) is the actual research work; this
  scaffold deliberately doesn't do that part for you.
- `src/probe.py`, `activation_extraction.py`, `baselines.py`,
  `evaluate_generalization.py`, and `steering.py` are stubs with docstrings
  explaining what each needs to do and why — not full implementations.
  Filling these in, and pre-registering your predictions before running
  them, is Days 4-6.

  Note on src/*.log: three logs, from successive runs as the analysis
  was tightened up. train_probe_output.log and train_probe_output_2.log
  are earlier iterations, kept for the record. cv_diagnostics_output.log
  has the final, statistically-corrected result and is the one to cite.
