"""
Data schema and validation for eval/deployment prompt pairs.

Design choice: prompts are stored as MATCHED PAIRS (same underlying task,
only the framing changes) rather than two independent unlabeled piles. This
matters methodologically -- it's the same discipline as using difference-
space metrics over absolute values in the circuit audit work. If you probe
on unpaired prompts, the probe can key off *content* differences between
your eval-prompt pile and your deployment-prompt pile (eval prompts happen
to be longer, use different vocabulary, whatever) rather than the framing
signal you actually care about. Pairing lets you audit the main result
afterward by looking at within-pair activation differences -- it does not
replace training on the full pooled set, it's a diagnostic on top of it.

Every field below is required and validated on construction. If you're
tempted to skip a field "for now," that's a sign the example isn't ready to
go in the dataset yet.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class PromptExample:
    id: str             # unique id, e.g. "coding_003_eval"
    pair_id: str         # links this to its matched counterpart, e.g. "coding_003"
    condition: str        # "eval" or "deployment" -- nothing else
    task_category: str    # e.g. "coding", "math_word_problem", "creative_writing"
    prompt: str            # the actual text sent to the model

    def __post_init__(self):
        if self.condition not in ("eval", "deployment"):
            raise ValueError(f"condition must be 'eval' or 'deployment', got {self.condition!r}")
        if not self.prompt.strip():
            raise ValueError(f"prompt for id={self.id} is empty")


def load_prompts(path: str | Path) -> list[PromptExample]:
    """Load a JSONL file of prompt examples."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            examples.append(PromptExample(**data))
    return examples


def save_prompts(examples: list[PromptExample], path: str | Path) -> None:
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex)) + "\n")


def validate_pairing(examples: list[PromptExample]) -> None:
    """
    Confirm every pair_id has exactly one 'eval' and one 'deployment'
    example, and that task_category matches within a pair (framing should be
    the only thing that varies).

    Run this after writing each day's prompt set, before touching
    activations. Catching a broken pair here is cheap; catching it after
    training a probe on it is not.
    """
    from collections import defaultdict

    by_pair: dict[str, list[PromptExample]] = defaultdict(list)
    for ex in examples:
        by_pair[ex.pair_id].append(ex)

    errors = []
    for pair_id, pair in by_pair.items():
        if len(pair) != 2:
            errors.append(f"pair_id={pair_id} has {len(pair)} examples, expected 2")
            continue
        conditions = {ex.condition for ex in pair}
        if conditions != {"eval", "deployment"}:
            errors.append(f"pair_id={pair_id} conditions={conditions}, expected both eval and deployment")
        categories = {ex.task_category for ex in pair}
        if len(categories) != 1:
            errors.append(f"pair_id={pair_id} has mismatched task_category across pair: {categories}")

    if errors:
        raise ValueError("Pairing validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    print(f"Validated {len(by_pair)} pairs ({len(examples)} examples) -- all clean.")


def category_split_report(examples: list[PromptExample]) -> None:
    """Print task-category counts -- sanity-check before locking in-distribution
    vs. held-out category splits. Remember: a category in the held-out file
    must NOT also appear in the in-distribution file, or Day 5's test isn't
    actually testing generalization."""
    from collections import Counter

    counts = Counter(ex.task_category for ex in examples)
    print("Task category counts:")
    for cat, n in sorted(counts.items()):
        print(f"  {cat}: {n}")
